"""Transparent business-cost assumptions for fraud decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostScenario:
    false_positive_cost: float = 20.0
    false_negative_fixed_cost: float = 500.0
    review_cost: float = 5.0
    recoverable_fraction: float = 0.80

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def decision_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    amounts: np.ndarray | None = None,
    scenario: CostScenario | None = None,
) -> dict[str, float]:
    """Calculate scenario-based costs; values are assumptions, not realised savings."""
    scenario = scenario or CostScenario()
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    amounts = (
        np.ones_like(y_true, dtype=float) if amounts is None else np.asarray(amounts, dtype=float)
    )
    false_positives = (y_true == 0) & (y_pred == 1)
    false_negatives = (y_true == 1) & (y_pred == 0)
    reviewed = y_pred == 1
    missed_fraud_value = float(amounts[false_negatives].sum())
    fp_cost = float(false_positives.sum() * scenario.false_positive_cost)
    review_cost = float(reviewed.sum() * scenario.review_cost)
    fn_cost = float(
        false_negatives.sum() * scenario.false_negative_fixed_cost
        + missed_fraud_value * scenario.recoverable_fraction
    )
    return {
        "false_positive_cost": fp_cost,
        "false_negative_cost": fn_cost,
        "review_cost": review_cost,
        "total_cost": fp_cost + fn_cost + review_cost,
        "missed_fraud_value": missed_fraud_value,
        "reviewed_transactions": int(reviewed.sum()),
    }


def cost_sensitivity_table(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    amounts: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """Evaluate the locked threshold under three explicit cost scenarios."""
    scenarios = {
        "conservative": CostScenario(10, 200, 3, 0.50),
        "medium": CostScenario(20, 500, 5, 0.80),
        "severe": CostScenario(40, 1000, 8, 1.00),
    }
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    rows = []
    for name, scenario in scenarios.items():
        rows.append({"scenario": name, **decision_cost(y_true, predictions, amounts, scenario)})
    return pd.DataFrame(rows)
