from __future__ import annotations

import json
import os

from .investigation import contains_forbidden_ground_truth


REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "transaction_id": {"type": "integer"},
        "risk_tier": {
            "type": "string",
            "enum": ["HIGH", "MEDIUM", "LOW"],
        },
        "executive_summary": {"type": "string"},
        "model_evidence": {
            "type": "array",
            "items": {"type": "string"},
        },
        "contextual_evidence": {
            "type": "array",
            "items": {"type": "string"},
        },
        "autoencoder_evidence": {
            "type": "array",
            "items": {"type": "string"},
        },
        "model_consensus": {"type": "string"},
        "recommended_analyst_checks": {
            "type": "array",
            "items": {"type": "string"},
        },
        "limitations": {
            "type": "array",
            "items": {"type": "string"},
        },
        "conclusion": {"type": "string"},
    },
    "required": [
        "transaction_id",
        "risk_tier",
        "executive_summary",
        "model_evidence",
        "contextual_evidence",
        "autoencoder_evidence",
        "model_consensus",
        "recommended_analyst_checks",
        "limitations",
        "conclusion",
    ],
    "additionalProperties": False,
}


SYSTEM_PROMPT = """
You are a fraud-investigation assistant.

You receive a structured anomaly-investigation packet produced by upstream
machine-learning detectors.

Your role is to summarize and contextualize the evidence for a human analyst.

STRICT RULES:

1. Do not decide whether the transaction is truly fraudulent.
2. Never state or imply that an anomaly score is a probability of fraud.
3. Do not invent facts, customer history, merchant history, locations,
   identities, account information, or external events.
4. Use only information contained in the supplied packet.
5. Copy the upstream risk_tier exactly. Do not change it.
6. Clearly distinguish model evidence, contextual evidence, and Autoencoder
   reconstruction evidence.
7. If models disagree, explicitly state the disagreement.
8. Recommended checks must be actions for a human analyst, not claims that
   those checks already occurred.
9. State important limitations, including missing behavioral or identity
   context where relevant.
10. Never state that fraud is confirmed.
11. Respect the upstream risk tier when writing urgency:
    - LOW: no priority investigation; routine monitoring.
    - MEDIUM: routine analyst review.
    - HIGH: priority analyst review.
12. Do not over-interpret Autoencoder feature errors when the Autoencoder
    anomaly percentile is low.
13. Be concise and factual.
""".strip()


def _model_evidence(packet: dict) -> list[str]:
    evidence = []

    for model_name, info in packet["model_scores"].items():
        evidence.append(
            (
                f"{model_name}: anomaly percentile "
                f"{info['percentile']:.3f}; "
                f"max-F1 alert={info['max_f1_alert']}; "
                f"high-recall alert={info['high_recall_alert']}."
            )
        )

    return evidence


def _autoencoder_evidence(packet: dict) -> list[str]:
    ae_info = packet["model_scores"]["autoencoder"]
    percentile = float(ae_info["percentile"])

    if (
        percentile < 0.95
        and not ae_info["high_recall_alert"]
        and not ae_info["max_f1_alert"]
    ):
        return [
            (
                "Autoencoder did not flag a material reconstruction anomaly "
                f"(anomaly percentile {percentile:.3f}); the largest "
                "feature-level reconstruction errors are therefore not "
                "treated as strong evidence."
            )
        ]

    return [
        (
            f"{feature['feature']}: squared reconstruction "
            f"error {feature['squared_error']:.6f}."
        )
        for feature in packet[
            "autoencoder_top_reconstruction_features"
        ]
    ]


def local_report_from_packet(packet: dict) -> dict:
    risk_tier = packet["risk_tier"]
    agreement = int(packet["models_top_5pct"])

    model_evidence = _model_evidence(packet)
    contextual_evidence = list(packet["context"]["reasons"])
    autoencoder_evidence = _autoencoder_evidence(packet)

    if risk_tier == "HIGH":
        executive_summary = (
            "Transaction routed as HIGH risk by upstream anomaly-detection "
            "and investigation-routing rules. Multiple or high-confidence "
            "signals justify priority analyst review."
        )

        recommended_checks = [
            "Prioritize manual review of the transaction and available account context.",
            "Review nearby transactions for related amount, merchant, or timing anomalies.",
            "Verify whether the transaction is expected for the customer/account before taking action.",
        ]

        conclusion = (
            "The transaction is strongly anomalous according to the supplied "
            "detectors and warrants priority analyst review. "
            "The evidence does not establish fraud."
        )

    elif risk_tier == "MEDIUM":
        executive_summary = (
            "Transaction routed as MEDIUM risk by upstream anomaly-detection "
            "and investigation-routing rules. The available signals justify "
            "routine analyst review."
        )

        recommended_checks = [
            "Review the transaction with available customer/account history.",
            "Compare nearby transactions for unusual merchant, amount, or timing patterns.",
            "Verify whether the observed behavior is expected for the account.",
        ]

        conclusion = (
            "The transaction warrants routine analyst review based on the "
            "supplied anomaly evidence. The evidence does not establish fraud."
        )

    else:
        executive_summary = (
            "Transaction routed as LOW risk. Current detector scores, model "
            "agreement, and operational thresholds do not indicate a strong "
            "anomaly requiring priority investigation."
        )

        recommended_checks = [
            "No priority analyst action is indicated by the current detector evidence.",
            "Retain the transaction for routine monitoring and reassess if additional behavioral context becomes available.",
        ]

        conclusion = (
            "Current detector evidence does not justify priority investigation. "
            "The transaction can remain under routine monitoring."
        )

    return {
        "transaction_id": int(packet["transaction_id"]),
        "risk_tier": risk_tier,
        "executive_summary": executive_summary,
        "model_evidence": model_evidence,
        "contextual_evidence": contextual_evidence,
        "autoencoder_evidence": autoencoder_evidence,
        "model_consensus": (
            f"{agreement} of 3 models place the transaction "
            "in their top 5% anomaly region."
        ),
        "recommended_analyst_checks": recommended_checks,
        "limitations": [
            "Anomaly scores are not fraud probabilities.",
            "The packet contains limited behavioral history.",
            "The report does not establish ground-truth fraud.",
        ],
        "conclusion": conclusion,
    }


def generate_llm_report(
    packet: dict,
    model: str | None = None,
) -> dict:
    if contains_forbidden_ground_truth(packet):
        raise ValueError(
            "Ground-truth field detected in LLM packet."
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI SDK not installed. Run: pip install -U openai"
        ) from exc

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not configured."
        )

    model = (
        model
        or os.getenv("OPENAI_MODEL")
        or "gpt-5.6-luna"
    )

    client = OpenAI()

    response = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=(
            "Create an analyst-facing investigation report "
            "from this packet. Use only the supplied evidence.\n\n"
            + json.dumps(
                packet,
                ensure_ascii=False,
            )
        ),
        text={
            "format": {
                "type": "json_schema",
                "name": "fraud_investigation_report",
                "strict": True,
                "schema": REPORT_SCHEMA,
            }
        },
        store=False,
    )

    report = json.loads(response.output_text)

    if report["transaction_id"] != packet["transaction_id"]:
        raise ValueError("LLM changed transaction_id.")

    if report["risk_tier"] != packet["risk_tier"]:
        raise ValueError("LLM changed upstream risk_tier.")

    return report


def report_to_markdown(report: dict) -> str:
    def bullets(items: list[str]) -> str:
        return "\n".join(
            f"- {item}"
            for item in items
        )

    return f"""# Transaction Investigation Report

**Transaction ID:** {report["transaction_id"]}  
**Risk tier:** {report["risk_tier"]}

## Executive summary

{report["executive_summary"]}

## Model evidence

{bullets(report["model_evidence"])}

## Contextual evidence

{bullets(report["contextual_evidence"])}

## Autoencoder reconstruction evidence

{bullets(report["autoencoder_evidence"])}

## Model consensus

{report["model_consensus"]}

## Recommended analyst actions

{bullets(report["recommended_analyst_checks"])}

## Limitations

{bullets(report["limitations"])}

## Conclusion

{report["conclusion"]}
"""
