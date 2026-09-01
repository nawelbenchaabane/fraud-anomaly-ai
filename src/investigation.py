from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .explainability import TrainingContext
from .scoring import FraudAnomalyScorer


def _native(
    value: Any,
) -> Any:
    """Convert common numpy/pandas scalar values to JSON-safe Python types."""
    if isinstance(
        value,
        (np.integer,),
    ):
        return int(
            value
        )

    if isinstance(
        value,
        (np.floating,),
    ):
        return float(
            value
        )

    if isinstance(
        value,
        (np.bool_,),
    ):
        return bool(
            value
        )

    if pd.isna(
        value
    ):
        return None

    return value


def build_investigation_packet(
    transaction: pd.Series,
    score_row: pd.Series,
    scorer: FraudAnomalyScorer,
    context: TrainingContext,
    transaction_id: int,
    top_ae_features: int = 5,
) -> dict:
    """
    Build an LLM-safe investigation packet.

    Ground-truth fields such as `fraud` are deliberately not copied.
    """
    row_df = pd.DataFrame(
        [transaction]
    )

    contextual = context.explain(
        transaction,
        score_row,
    )

    ae_features = scorer.autoencoder_feature_explanation(
        row_df,
        top_n=top_ae_features,
    )

    transaction_fields = [
        "step",
        "customer",
        "merchant",
        "category",
        "amount",
        "age",
        "gender",
    ]

    transaction_payload = {}

    for field in transaction_fields:
        if field in transaction.index:
            value = _native(
                transaction[
                    field
                ]
            )

            if field in {
                "customer",
                "merchant",
                "category",
                "age",
                "gender",
            } and value is not None:
                value = str(
                    value
                ).strip("'")

            transaction_payload[
                field
            ] = value

    model_scores = {
        "isolation_forest": {
            "score":
                float(
                    score_row[
                        "iforest_score"
                    ]
                ),

            "percentile":
                float(
                    score_row[
                        "iforest_percentile"
                    ]
                ),

            "max_f1_alert":
                bool(
                    score_row[
                        "iforest_max_f1_alert"
                    ]
                ),

            "high_recall_alert":
                bool(
                    score_row[
                        "iforest_high_recall_alert"
                    ]
                ),
        },

        "one_class_svm": {
            "score":
                float(
                    score_row[
                        "ocsvm_score"
                    ]
                ),

            "percentile":
                float(
                    score_row[
                        "ocsvm_percentile"
                    ]
                ),

            "max_f1_alert":
                bool(
                    score_row[
                        "ocsvm_max_f1_alert"
                    ]
                ),

            "high_recall_alert":
                bool(
                    score_row[
                        "ocsvm_high_recall_alert"
                    ]
                ),
        },

        "autoencoder": {
            "score":
                float(
                    score_row[
                        "autoencoder_score"
                    ]
                ),

            "percentile":
                float(
                    score_row[
                        "autoencoder_percentile"
                    ]
                ),

            "max_f1_alert":
                bool(
                    score_row[
                        "autoencoder_max_f1_alert"
                    ]
                ),

            "high_recall_alert":
                bool(
                    score_row[
                        "autoencoder_high_recall_alert"
                    ]
                ),
        },
    }

    return {
        "transaction_id":
            int(
                transaction_id
            ),

        "risk_tier":
            str(
                score_row[
                    "risk_tier"
                ]
            ),

        "models_top_5pct":
            int(
                score_row[
                    "models_top_5pct"
                ]
            ),

        "mean_anomaly_percentile":
            float(
                score_row[
                    "mean_anomaly_percentile"
                ]
            ),

        "transaction":
            transaction_payload,

        "model_scores":
            model_scores,

        "context":
            contextual,

        "autoencoder_top_reconstruction_features":
            ae_features,
    }


def contains_forbidden_ground_truth(
    obj: Any,
) -> bool:
    forbidden_keys = {
        "fraud",
        "ground_truth_fraud",
        "label",
        "target",
    }

    if isinstance(
        obj,
        dict,
    ):
        for key, value in obj.items():
            if str(
                key
            ).lower() in forbidden_keys:
                return True

            if contains_forbidden_ground_truth(
                value
            ):
                return True

    elif isinstance(
        obj,
        list,
    ):
        return any(
            contains_forbidden_ground_truth(
                item
            )
            for item in obj
        )

    return False
