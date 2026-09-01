from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy import sparse

from .preprocessing import prepare_model_frame


def percentile_from_reference(
    reference_scores: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    """
    Convert scores to empirical percentiles using a fixed reference sample.

    No labels are used. Higher percentile means more anomalous because every
    detector score in this project is oriented so that higher = more anomalous.
    """
    reference_sorted = np.sort(
        np.asarray(reference_scores, dtype=float)
    )

    ranks = np.searchsorted(
        reference_sorted,
        np.asarray(scores, dtype=float),
        side="right",
    )

    return ranks / len(reference_sorted)


class Autoencoder(nn.Module):
    """Architecture used by notebook 06."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim_1: int = 32,
        bottleneck_dim: int = 12,
        hidden_dim_2: int = 32,
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, bottleneck_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(
            self.encoder(x)
        )


@dataclass
class ArtifactPaths:
    root: Path
    models: Path
    scores: Path

    @classmethod
    def from_root(cls, root: Path | str) -> "ArtifactPaths":
        root = Path(root).resolve()

        return cls(
            root=root,
            models=root / "models",
            scores=root / "results" / "scores",
        )


class FraudAnomalyScorer:
    """
    Load final saved artifacts and score new/raw transactions.

    Isolation Forest uses all 79 transformed features.
    One-Class SVM and Autoencoder exclude num__step and use 78 features.
    """

    def __init__(
        self,
        project_root: Path | str,
        device: str = "cpu",
    ) -> None:
        self.paths = ArtifactPaths.from_root(
            project_root
        )

        self.device = torch.device(device)

        self._load_artifacts()

    def _require(self, path: Path) -> Path:
        if not path.exists():
            raise FileNotFoundError(
                f"Required artifact not found: {path}"
            )
        return path

    def _load_json(self, path: Path) -> Any:
        with self._require(path).open(
            "r",
            encoding="utf-8",
        ) as f:
            return json.load(f)

    def _load_artifacts(self) -> None:
        m = self.paths.models
        s = self.paths.scores

        self.preprocessor = joblib.load(
            self._require(
                m / "preprocessor.joblib"
            )
        )

        self.feature_names = self._load_json(
            m / "feature_names.json"
        )

        self.step_feature = "num__step"

        try:
            self.step_index = self.feature_names.index(
                self.step_feature
            )
        except ValueError as exc:
            raise ValueError(
                f"{self.step_feature!r} is absent from feature_names.json"
            ) from exc

        self.no_step_indices = [
            i
            for i in range(len(self.feature_names))
            if i != self.step_index
        ]

        self.no_step_feature_names = [
            self.feature_names[i]
            for i in self.no_step_indices
        ]

        self.isolation_forest = joblib.load(
            self._require(
                m / "isolation_forest.joblib"
            )
        )

        self.ocsvm = joblib.load(
            self._require(
                m / "one_class_svm_no_step.joblib"
            )
        )

        ae_config = self._load_json(
            m / "autoencoder_config.json"
        )

        self.autoencoder = Autoencoder(
            input_dim=int(
                ae_config.get(
                    "input_dim",
                    len(self.no_step_feature_names),
                )
            ),
            hidden_dim_1=int(
                ae_config.get(
                    "hidden_dim_1",
                    32,
                )
            ),
            bottleneck_dim=int(
                ae_config.get(
                    "bottleneck_dim",
                    12,
                )
            ),
            hidden_dim_2=int(
                ae_config.get(
                    "hidden_dim_2",
                    32,
                )
            ),
        ).to(self.device)

        state_dict = torch.load(
            self._require(
                m / "autoencoder_state_dict.pt"
            ),
            map_location=self.device,
        )

        self.autoencoder.load_state_dict(
            state_dict
        )
        self.autoencoder.eval()

        self.iforest_thresholds = self._load_json(
            m / "isolation_forest_thresholds.json"
        )

        self.ocsvm_thresholds = self._load_json(
            m / "one_class_svm_thresholds.json"
        )

        self.ae_thresholds = self._load_json(
            m / "autoencoder_thresholds.json"
        )

        self.iforest_valid_scores = np.load(
            self._require(
                s / "iforest_valid_scores.npy"
            )
        )

        self.ocsvm_valid_scores = np.load(
            self._require(
                s / "ocsvm_valid_scores.npy"
            )
        )

        self.ae_valid_scores = np.load(
            self._require(
                s / "autoencoder_valid_scores.npy"
            )
        )

    @staticmethod
    def _threshold(
        threshold_config: dict,
        strategy: str,
    ) -> float:
        return float(
            threshold_config[
                strategy
            ][
                "threshold"
            ]
        )

    def transform(
        self,
        raw_transactions: pd.DataFrame,
    ):
        model_frame = prepare_model_frame(
            raw_transactions
        )

        transformed = self.preprocessor.transform(
            model_frame
        )

        return transformed

    def _autoencoder_scores(
        self,
        no_step_matrix,
        batch_size: int = 4096,
    ) -> np.ndarray:
        if sparse.issparse(
            no_step_matrix
        ):
            data = no_step_matrix.toarray()
        else:
            data = np.asarray(
                no_step_matrix
            )

        data = data.astype(
            np.float32,
            copy=False,
        )

        scores = []

        self.autoencoder.eval()

        with torch.no_grad():
            for start in range(
                0,
                len(data),
                batch_size,
            ):
                batch = torch.from_numpy(
                    data[
                        start:start + batch_size
                    ]
                ).to(
                    self.device
                )

                reconstructed = self.autoencoder(
                    batch
                )

                mse = torch.mean(
                    (
                        reconstructed
                        - batch
                    ) ** 2,
                    dim=1,
                )

                scores.append(
                    mse.cpu().numpy()
                )

        return np.concatenate(
            scores
        )

    def score(
        self,
        raw_transactions: pd.DataFrame,
    ) -> pd.DataFrame:
        transformed = self.transform(
            raw_transactions
        )

        no_step = transformed[
            :,
            self.no_step_indices,
        ]

        iforest_score = (
            -self.isolation_forest.decision_function(
                transformed
            )
        )

        ocsvm_score = (
            -self.ocsvm.decision_function(
                no_step
            )
        )

        ae_score = self._autoencoder_scores(
            no_step
        )

        result = pd.DataFrame({
            "iforest_score":
                iforest_score,

            "ocsvm_score":
                ocsvm_score,

            "autoencoder_score":
                ae_score,
        })

        result[
            "iforest_percentile"
        ] = percentile_from_reference(
            self.iforest_valid_scores,
            iforest_score,
        )

        result[
            "ocsvm_percentile"
        ] = percentile_from_reference(
            self.ocsvm_valid_scores,
            ocsvm_score,
        )

        result[
            "autoencoder_percentile"
        ] = percentile_from_reference(
            self.ae_valid_scores,
            ae_score,
        )

        result[
            "iforest_max_f1_alert"
        ] = (
            result["iforest_score"]
            >= self._threshold(
                self.iforest_thresholds,
                "max_f1",
            )
        )

        result[
            "iforest_high_recall_alert"
        ] = (
            result["iforest_score"]
            >= self._threshold(
                self.iforest_thresholds,
                "high_recall",
            )
        )

        result[
            "ocsvm_max_f1_alert"
        ] = (
            result["ocsvm_score"]
            >= self._threshold(
                self.ocsvm_thresholds,
                "max_f1",
            )
        )

        result[
            "ocsvm_high_recall_alert"
        ] = (
            result["ocsvm_score"]
            >= self._threshold(
                self.ocsvm_thresholds,
                "high_recall",
            )
        )

        result[
            "autoencoder_max_f1_alert"
        ] = (
            result["autoencoder_score"]
            >= self._threshold(
                self.ae_thresholds,
                "max_f1",
            )
        )

        result[
            "autoencoder_high_recall_alert"
        ] = (
            result["autoencoder_score"]
            >= self._threshold(
                self.ae_thresholds,
                "high_recall",
            )
        )

        result[
            "models_top_5pct"
        ] = (
            (
                result["iforest_percentile"]
                >= 0.95
            ).astype(int)
            + (
                result["ocsvm_percentile"]
                >= 0.95
            ).astype(int)
            + (
                result["autoencoder_percentile"]
                >= 0.95
            ).astype(int)
        )

        result[
            "mean_anomaly_percentile"
        ] = result[
            [
                "iforest_percentile",
                "ocsvm_percentile",
                "autoencoder_percentile",
            ]
        ].mean(
            axis=1
        )

        result[
            "risk_tier"
        ] = result.apply(
            self._assign_risk_tier,
            axis=1,
        )

        return result

    @staticmethod
    def _assign_risk_tier(
        row: pd.Series,
    ) -> str:
        if (
            row["models_top_5pct"] == 3
            or row["ocsvm_max_f1_alert"]
        ):
            return "HIGH"

        if (
            row["models_top_5pct"] >= 2
            or row["iforest_high_recall_alert"]
            or row["autoencoder_high_recall_alert"]
        ):
            return "MEDIUM"

        return "LOW"

    def autoencoder_feature_explanation(
        self,
        raw_transaction: pd.DataFrame,
        top_n: int = 5,
    ) -> list[dict]:
        if len(raw_transaction) != 1:
            raise ValueError(
                "Feature explanation expects exactly one transaction."
            )

        transformed = self.transform(
            raw_transaction
        )

        no_step = transformed[
            :,
            self.no_step_indices,
        ]

        if sparse.issparse(
            no_step
        ):
            x = no_step.toarray()
        else:
            x = np.asarray(
                no_step
            )

        x = x.astype(
            np.float32,
            copy=False,
        )

        tensor = torch.from_numpy(
            x
        ).to(
            self.device
        )

        with torch.no_grad():
            reconstructed = (
                self.autoencoder(
                    tensor
                )
                .cpu()
                .numpy()[0]
            )

        original = x[0]

        errors = (
            original
            - reconstructed
        ) ** 2

        explanation = pd.DataFrame({
            "feature":
                self.no_step_feature_names,

            "input_value":
                original,

            "reconstructed_value":
                reconstructed,

            "squared_error":
                errors,
        })

        top = explanation.sort_values(
            "squared_error",
            ascending=False,
        ).head(
            top_n
        )

        records = top.to_dict(
            orient="records"
        )

        # Convert numpy scalar types so the packet is JSON serializable.
        for record in records:
            record[
                "input_value"
            ] = float(
                record["input_value"]
            )
            record[
                "reconstructed_value"
            ] = float(
                record["reconstructed_value"]
            )
            record[
                "squared_error"
            ] = float(
                record["squared_error"]
            )

        return records
