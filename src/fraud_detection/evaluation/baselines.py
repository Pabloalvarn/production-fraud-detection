"""Operational and no-skill baselines for honest model comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RuleBasedBaseline:
    """Flag large transfers/cash-outs using a training-derived amount cutoff."""

    quantile: float = 0.99
    amount_threshold_: float | None = None

    def fit(self, frame: pd.DataFrame) -> RuleBasedBaseline:
        if not 0 < self.quantile < 1:
            raise ValueError("quantile must be between 0 and 1")
        self.amount_threshold_ = float(frame["amount"].quantile(self.quantile))
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if self.amount_threshold_ is None:
            raise RuntimeError("Baseline must be fitted before prediction")
        high_risk_type = frame["type"].isin(["TRANSFER", "CASH_OUT"])
        score = (high_risk_type & (frame["amount"] >= self.amount_threshold_)).astype(float)
        return np.column_stack([1 - score, score])


def prevalence_probabilities(size: int, prevalence: float) -> np.ndarray:
    """Return a constant no-skill probability baseline."""
    if not 0 <= prevalence <= 1:
        raise ValueError("prevalence must be between 0 and 1")
    return np.full(size, prevalence, dtype=float)
