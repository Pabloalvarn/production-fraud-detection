"""Feature engineering using information assumed available before authorisation."""

from __future__ import annotations

import numpy as np
import pandas as pd

RAW_ONLINE_FEATURES = ["step", "type", "amount", "oldbalanceOrg", "oldbalanceDest"]
CATEGORICAL_FEATURES = ["type"]
NUMERIC_FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "log_amount",
    "amount_to_origin_balance",
    "amount_to_destination_balance",
    "origin_balance_is_zero",
    "destination_balance_is_zero",
    "origin_insufficient_balance",
]


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide safely and replace undefined or infinite ratios with zero."""
    denominator_safe = denominator.mask(denominator == 0)
    ratio = numerator / denominator_safe
    return ratio.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_online_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create point-in-time features without post-transaction balances or labels."""
    missing = sorted(set(RAW_ONLINE_FEATURES) - set(frame.columns))
    if missing:
        raise KeyError(f"Missing online feature columns: {missing}")
    result = frame[RAW_ONLINE_FEATURES].copy()
    result["log_amount"] = np.log1p(result["amount"].clip(lower=0))
    result["amount_to_origin_balance"] = safe_ratio(result["amount"], result["oldbalanceOrg"])
    result["amount_to_destination_balance"] = safe_ratio(result["amount"], result["oldbalanceDest"])
    result["origin_balance_is_zero"] = (result["oldbalanceOrg"] == 0).astype("int8")
    result["destination_balance_is_zero"] = (result["oldbalanceDest"] == 0).astype("int8")
    result["origin_insufficient_balance"] = (result["amount"] > result["oldbalanceOrg"]).astype(
        "int8"
    )
    return result
