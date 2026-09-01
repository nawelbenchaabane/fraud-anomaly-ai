from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .preprocessing import clean_banksim_strings


def empirical_percentile(
    sorted_reference: np.ndarray,
    value: float,
) -> float:
    if len(sorted_reference) == 0:
        return float("nan")

    rank = np.searchsorted(
        sorted_reference,
        value,
        side="right",
    )

    return float(
        rank / len(sorted_reference)
    )


@dataclass
class TrainingContext:
    """Training-only contextual statistics used for explanations."""

    global_amount_sorted: np.ndarray
    category_amount_reference: dict[str, np.ndarray]
    merchant_amount_reference: dict[str, np.ndarray]
    category_frequency: dict[str, float]
    merchant_frequency: dict[str, float]

    @classmethod
    def from_raw_dataframe(
        cls,
        raw_df: pd.DataFrame,
        max_train_step: int = 119,
    ) -> "TrainingContext":
        df = clean_banksim_strings(
            raw_df
        )

        train = df[
            df["step"] <= max_train_step
        ].copy()

        if train.empty:
            raise ValueError(
                "Training reference is empty."
            )

        global_amount_sorted = np.sort(
            train[
                "amount"
            ].astype(
                float
            ).to_numpy()
        )

        category_amount_reference = {
            str(category): np.sort(
                group[
                    "amount"
                ].astype(
                    float
                ).to_numpy()
            )
            for category, group
            in train.groupby(
                "category"
            )
        }

        merchant_amount_reference = {
            str(merchant): np.sort(
                group[
                    "amount"
                ].astype(
                    float
                ).to_numpy()
            )
            for merchant, group
            in train.groupby(
                "merchant"
            )
        }

        category_frequency = (
            train[
                "category"
            ].astype(
                str
            ).value_counts(
                normalize=True
            ).to_dict()
        )

        merchant_frequency = (
            train[
                "merchant"
            ].astype(
                str
            ).value_counts(
                normalize=True
            ).to_dict()
        )

        return cls(
            global_amount_sorted=
                global_amount_sorted,

            category_amount_reference=
                category_amount_reference,

            merchant_amount_reference=
                merchant_amount_reference,

            category_frequency=
                category_frequency,

            merchant_frequency=
                merchant_frequency,
        )

    def explain(
        self,
        transaction: pd.Series,
        score_row: pd.Series,
    ) -> dict[str, Any]:
        amount = float(
            transaction[
                "amount"
            ]
        )

        category = str(
            transaction[
                "category"
            ]
        ).strip("'")

        merchant = str(
            transaction[
                "merchant"
            ]
        ).strip("'")

        global_amount_pct = empirical_percentile(
            self.global_amount_sorted,
            amount,
        )

        category_amount_pct = empirical_percentile(
            self.category_amount_reference.get(
                category,
                np.array([]),
            ),
            amount,
        )

        merchant_amount_pct = empirical_percentile(
            self.merchant_amount_reference.get(
                merchant,
                np.array([]),
            ),
            amount,
        )

        category_freq = float(
            self.category_frequency.get(
                category,
                0.0,
            )
        )

        merchant_freq = float(
            self.merchant_frequency.get(
                merchant,
                0.0,
            )
        )

        reasons: list[str] = []

        if global_amount_pct >= 0.99:
            reasons.append(
                "Amount is above the 99th percentile of training transactions."
            )
        elif global_amount_pct >= 0.95:
            reasons.append(
                "Amount is above the 95th percentile of training transactions."
            )

        if category_amount_pct >= 0.99:
            reasons.append(
                "Amount is unusually high for this category."
            )
        elif category_amount_pct >= 0.95:
            reasons.append(
                "Amount is high relative to transactions in the same category."
            )

        if merchant_amount_pct >= 0.99:
            reasons.append(
                "Amount is unusually high for this merchant."
            )
        elif merchant_amount_pct >= 0.95:
            reasons.append(
                "Amount is high relative to transactions for the same merchant."
            )

        if category_freq <= 0.005:
            reasons.append(
                "Transaction category is rare in training history."
            )

        if merchant_freq <= 0.005:
            reasons.append(
                "Merchant is rare in training history."
            )

        agreement = int(
            score_row[
                "models_top_5pct"
            ]
        )

        if agreement == 3:
            reasons.append(
                "All three anomaly detectors place the transaction "
                "in their top 5% most anomalous region."
            )
        elif agreement == 2:
            reasons.append(
                "Two anomaly detectors place the transaction "
                "in their top 5% most anomalous region."
            )

        if bool(
            score_row[
                "ocsvm_max_f1_alert"
            ]
        ):
            reasons.append(
                "One-Class SVM triggered its high-confidence alert threshold."
            )

        if not reasons:
            mean_pct = float(
                score_row[
                    "mean_anomaly_percentile"
                ]
            )

            if mean_pct < 0.50:
                reasons.append(
                    "No strong contextual anomaly was identified, and the "
                    "combined anomaly percentiles are relatively low."
                )
            else:
                reasons.append(
                    "No single contextual rule dominates; the available "
                    "detector signals should be interpreted together."
                )

        def finite_or_none(
            value: float,
        ):
            return (
                float(
                    value
                )
                if np.isfinite(
                    value
                )
                else None
            )

        return {
            "global_amount_percentile":
                finite_or_none(
                    global_amount_pct
                ),

            "category_amount_percentile":
                finite_or_none(
                    category_amount_pct
                ),

            "merchant_amount_percentile":
                finite_or_none(
                    merchant_amount_pct
                ),

            "category_frequency":
                category_freq,

            "merchant_frequency":
                merchant_freq,

            "reasons":
                reasons,
        }
