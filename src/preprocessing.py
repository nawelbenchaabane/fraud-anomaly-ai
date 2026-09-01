from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


MODEL_FEATURES = [
    "step",
    "amount_log",
    "age",
    "gender",
    "merchant",
    "category",
]

REQUIRED_TRANSACTION_FIELDS = [
    "step",
    "age",
    "gender",
    "merchant",
    "category",
    "amount",
]


def clean_banksim_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Strip BankSim's surrounding apostrophes from object/string columns."""
    out = df.copy()

    for col in out.select_dtypes(include="object").columns:
        out[col] = out[col].astype(str).str.strip("'")

    return out


def validate_transaction_columns(
    df: pd.DataFrame,
    required: Iterable[str] = REQUIRED_TRANSACTION_FIELDS,
) -> None:
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            "Missing required transaction fields: "
            + ", ".join(missing)
        )


def prepare_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare raw transaction rows for the fitted preprocessor from notebook 02.

    The saved preprocessor expects:
      step, amount_log, age, gender, merchant, category
    """
    validate_transaction_columns(df)

    out = clean_banksim_strings(df)

    out["amount"] = pd.to_numeric(
        out["amount"],
        errors="raise",
    )

    out["step"] = pd.to_numeric(
        out["step"],
        errors="raise",
    )

    if (out["amount"] < 0).any():
        raise ValueError("Transaction amount cannot be negative.")

    out["amount_log"] = np.log1p(
        out["amount"].astype(float)
    )

    return out[MODEL_FEATURES].copy()
