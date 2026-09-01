from __future__ import annotations

from pathlib import Path

import pandas as pd

from .explainability import TrainingContext
from .investigation import (
    build_investigation_packet,
)
from .preprocessing import (
    clean_banksim_strings,
)
from .scoring import FraudAnomalyScorer


class InvestigationPipeline:
    """
    End-to-end inference pipeline:

    raw transaction
      -> saved preprocessing
      -> three anomaly detectors
      -> risk routing
      -> contextual explanation
      -> structured investigation packet
    """

    def __init__(
        self,
        project_root: Path | str,
        raw_data_path: Path | str | None = None,
        device: str = "cpu",
    ) -> None:
        self.project_root = Path(
            project_root
        ).resolve()

        self.raw_data_path = (
            Path(
                raw_data_path
            ).resolve()
            if raw_data_path
            else (
                self.project_root
                / "data"
                / "raw"
                / "bs140513_032310.csv"
            )
        )

        if not self.raw_data_path.exists():
            raise FileNotFoundError(
                f"Raw BankSim file not found: {self.raw_data_path}"
            )

        self.raw_df = clean_banksim_strings(
            pd.read_csv(
                self.raw_data_path
            )
        )

        self.scorer = FraudAnomalyScorer(
            project_root=self.project_root,
            device=device,
        )

        self.context = TrainingContext.from_raw_dataframe(
            self.raw_df
        )

    def analyze_dataframe(
        self,
        transactions: pd.DataFrame,
        transaction_ids: list[int] | None = None,
        top_ae_features: int = 5,
    ) -> list[dict]:
        transactions = clean_banksim_strings(
            transactions
        ).reset_index(
            drop=True
        )

        scores = self.scorer.score(
            transactions
        )

        if transaction_ids is None:
            transaction_ids = list(
                range(
                    len(
                        transactions
                    )
                )
            )

        if len(
            transaction_ids
        ) != len(
            transactions
        ):
            raise ValueError(
                "transaction_ids length must match transactions."
            )

        packets = []

        for i in range(
            len(
                transactions
            )
        ):
            packet = build_investigation_packet(
                transaction=
                    transactions.iloc[i],

                score_row=
                    scores.iloc[i],

                scorer=
                    self.scorer,

                context=
                    self.context,

                transaction_id=
                    int(
                        transaction_ids[i]
                    ),

                top_ae_features=
                    top_ae_features,
            )

            packets.append(
                packet
            )

        return packets

    def analyze_test_position(
        self,
        test_position: int,
        top_ae_features: int = 5,
    ) -> dict:
        """
        Analyze a positional row from the temporal test set (step >= 150).

        The `fraud` ground-truth column, if present in the raw dataset,
        is never copied into the returned packet.
        """
        test_df = self.raw_df[
            self.raw_df["step"] >= 150
        ]

        if (
            test_position < 0
            or test_position >= len(
                test_df
            )
        ):
            raise IndexError(
                f"test_position must be in [0, {len(test_df) - 1}]"
            )

        original_index = int(
            test_df.index[
                test_position
            ]
        )

        transaction = test_df.iloc[
            [
                test_position
            ]
        ].copy()

        return self.analyze_dataframe(
            transaction,
            transaction_ids=[
                original_index
            ],
            top_ae_features=
                top_ae_features,
        )[0]
