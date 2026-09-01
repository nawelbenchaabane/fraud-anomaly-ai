from __future__ import annotations

from src.reporting import local_report_from_packet


def make_packet(
    *,
    transaction_id: int,
    risk_tier: str,
    models_top_5pct: int,
    iforest_percentile: float,
    ocsvm_percentile: float,
    ae_percentile: float,
    iforest_max_f1: bool = False,
    iforest_high_recall: bool = False,
    ocsvm_max_f1: bool = False,
    ocsvm_high_recall: bool = False,
    ae_max_f1: bool = False,
    ae_high_recall: bool = False,
    context_reasons: list[str] | None = None,
) -> dict:
    if context_reasons is None:
        context_reasons = [
            "No strong contextual anomaly was identified."
        ]

    return {
        "transaction_id": transaction_id,
        "risk_tier": risk_tier,
        "models_top_5pct": models_top_5pct,
        "mean_anomaly_percentile": (
            iforest_percentile
            + ocsvm_percentile
            + ae_percentile
        ) / 3,
        "transaction": {
            "step": 170,
            "customer": "C_TEST",
            "merchant": "M_TEST",
            "category": "es_test",
            "amount": 100.0,
            "age": "3",
            "gender": "F",
        },
        "model_scores": {
            "isolation_forest": {
                "score": 0.0,
                "percentile": iforest_percentile,
                "max_f1_alert": iforest_max_f1,
                "high_recall_alert": iforest_high_recall,
            },
            "one_class_svm": {
                "score": 0.0,
                "percentile": ocsvm_percentile,
                "max_f1_alert": ocsvm_max_f1,
                "high_recall_alert": ocsvm_high_recall,
            },
            "autoencoder": {
                "score": 0.0,
                "percentile": ae_percentile,
                "max_f1_alert": ae_max_f1,
                "high_recall_alert": ae_high_recall,
            },
        },
        "context": {
            "global_amount_percentile": 0.50,
            "category_amount_percentile": 0.50,
            "merchant_amount_percentile": 0.50,
            "category_frequency": 0.10,
            "merchant_frequency": 0.10,
            "reasons": context_reasons,
        },
        "autoencoder_top_reconstruction_features": [
            {
                "feature": "num__amount_log",
                "input_value": 1.0,
                "reconstructed_value": 0.9,
                "squared_error": 0.01,
            }
        ],
    }


def report_text(report: dict) -> str:
    parts = [
        report["executive_summary"],
        report["model_consensus"],
        report["conclusion"],
        *report["model_evidence"],
        *report["contextual_evidence"],
        *report["autoencoder_evidence"],
        *report["recommended_analyst_checks"],
        *report["limitations"],
    ]

    return " ".join(parts).lower()


def test_low_report_does_not_escalate():
    packet = make_packet(
        transaction_id=483337,
        risk_tier="LOW",
        models_top_5pct=0,
        iforest_percentile=0.108,
        ocsvm_percentile=0.289,
        ae_percentile=0.124,
    )

    report = local_report_from_packet(packet)
    text = report_text(report)

    assert report["risk_tier"] == "LOW"
    assert report["transaction_id"] == 483337

    assert "routine monitoring" in text
    assert "priority analyst review" not in text
    assert "warrants priority" not in text
    assert "fraud is confirmed" not in text

    assert (
        "did not flag a material reconstruction anomaly"
        in text
    )


def test_medium_report_requests_routine_review_not_priority_review():
    packet = make_packet(
        transaction_id=483339,
        risk_tier="MEDIUM",
        models_top_5pct=2,
        iforest_percentile=0.979,
        ocsvm_percentile=0.971,
        ae_percentile=0.920,
        iforest_max_f1=True,
        iforest_high_recall=True,
        ocsvm_high_recall=True,
        context_reasons=[
            "Amount is above the 95th percentile of training transactions.",
            "Two anomaly detectors place the transaction in their top 5% most anomalous region.",
        ],
    )

    report = local_report_from_packet(packet)
    text = report_text(report)

    assert report["risk_tier"] == "MEDIUM"
    assert report["transaction_id"] == 483339

    assert "routine analyst review" in text
    assert "priority analyst review" not in text
    assert "fraud is confirmed" not in text

    assert (
        "did not flag a material reconstruction anomaly"
        in text
    )


def test_high_report_requests_priority_review_for_three_model_consensus():
    packet = make_packet(
        transaction_id=483406,
        risk_tier="HIGH",
        models_top_5pct=3,
        iforest_percentile=0.995,
        ocsvm_percentile=0.976,
        ae_percentile=0.979,
        iforest_max_f1=True,
        iforest_high_recall=True,
        ocsvm_high_recall=True,
        ae_high_recall=True,
        context_reasons=[
            "All three anomaly detectors place the transaction in their top 5% most anomalous region."
        ],
    )

    report = local_report_from_packet(packet)
    text = report_text(report)

    assert report["risk_tier"] == "HIGH"
    assert report["transaction_id"] == 483406

    assert "priority analyst review" in text
    assert "3 of 3 models" in text

    assert "routine monitoring" not in text
    assert "fraud is confirmed" not in text
    assert "confirmed as fraud" not in text

    assert any(
        "num__amount_log" in item
        for item in report["autoencoder_evidence"]
    )


def test_high_report_can_be_triggered_by_high_confidence_ocsvm():
    packet = make_packet(
        transaction_id=999001,
        risk_tier="HIGH",
        models_top_5pct=1,
        iforest_percentile=0.80,
        ocsvm_percentile=0.99,
        ae_percentile=0.70,
        ocsvm_max_f1=True,
        ocsvm_high_recall=True,
        context_reasons=[
            "One-Class SVM triggered its high-confidence alert threshold."
        ],
    )

    report = local_report_from_packet(packet)
    text = report_text(report)

    assert report["risk_tier"] == "HIGH"
    assert "priority analyst review" in text
    assert "fraud is confirmed" not in text
    assert "confirmed as fraud" not in text


def test_all_reports_preserve_upstream_identity_and_risk_tier():
    cases = [
        ("LOW", 101, 0, 0.10, 0.20, 0.15),
        ("MEDIUM", 102, 2, 0.96, 0.95, 0.60),
        ("HIGH", 103, 3, 0.99, 0.98, 0.97),
    ]

    for (
        risk_tier,
        transaction_id,
        models_top_5pct,
        if_pct,
        svm_pct,
        ae_pct,
    ) in cases:
        packet = make_packet(
            transaction_id=transaction_id,
            risk_tier=risk_tier,
            models_top_5pct=models_top_5pct,
            iforest_percentile=if_pct,
            ocsvm_percentile=svm_pct,
            ae_percentile=ae_pct,
        )

        report = local_report_from_packet(packet)

        assert (
            report["transaction_id"]
            == packet["transaction_id"]
        )

        assert (
            report["risk_tier"]
            == packet["risk_tier"]
        )


def test_reports_never_present_scores_as_probabilities():
    packet = make_packet(
        transaction_id=200,
        risk_tier="HIGH",
        models_top_5pct=3,
        iforest_percentile=0.99,
        ocsvm_percentile=0.99,
        ae_percentile=0.99,
    )

    report = local_report_from_packet(packet)
    text = report_text(report)

    assert "anomaly scores are not fraud probabilities" in text

    positive_probability_claims = [
        "fraud probability is",
        "probability of fraud is",
        "fraud probability:",
    ]

    for phrase in positive_probability_claims:
        assert phrase not in text
