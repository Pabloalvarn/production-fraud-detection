"""Load a model bundle and produce governed fraud decisions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def load_bundle(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate the serialised model bundle."""
    bundle = joblib.load(path)
    required = {"model", "threshold", "model_version", "raw_features"}
    missing = required - set(bundle)
    if missing:
        raise ValueError(f"Invalid model bundle; missing keys: {sorted(missing)}")
    return bundle


def predict_frame(bundle: dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    """Return probabilities, risk bands and actions for a transaction frame."""
    missing = sorted(set(bundle["raw_features"]) - set(frame.columns))
    if missing:
        raise KeyError(f"Missing prediction fields: {missing}")
    probabilities = bundle["model"].predict_proba(frame[bundle["raw_features"]])[:, 1]
    threshold = float(bundle["threshold"])
    high_threshold = max(float(bundle.get("high_risk_threshold", 0.9)), threshold)
    decisions = []
    risk_levels = []
    for probability in probabilities:
        if probability >= high_threshold:
            risk_levels.append("high")
            decisions.append("enhanced_verification")
        elif probability >= threshold:
            risk_levels.append("medium")
            decisions.append("manual_review")
        else:
            risk_levels.append("low")
            decisions.append("approve")
    return pd.DataFrame(
        {
            "fraud_probability": probabilities,
            "risk_level": risk_levels,
            "decision": decisions,
            "model_version": bundle["model_version"],
        },
        index=frame.index,
    )
