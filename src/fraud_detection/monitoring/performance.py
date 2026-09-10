"""Performance monitoring once delayed fraud labels become available."""

from __future__ import annotations

import pandas as pd

from fraud_detection.evaluation.metrics import classification_metrics


def delayed_label_report(
    frame: pd.DataFrame,
    *,
    threshold: float,
    target_column: str = "y_true",
    probability_column: str = "probability",
    amount_column: str = "amount",
) -> dict[str, float | int]:
    """Recompute governed metrics after confirmed outcomes arrive."""
    required = {target_column, probability_column, amount_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Missing monitoring columns: {missing}")
    return classification_metrics(
        frame[target_column].to_numpy(),
        frame[probability_column].to_numpy(),
        threshold=threshold,
        amounts=frame[amount_column].to_numpy(),
    )
