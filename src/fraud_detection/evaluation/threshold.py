"""Decision-threshold optimisation performed on validation data only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import precision_score, recall_score

from fraud_detection.evaluation.business_cost import CostScenario, decision_cost


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    total_cost: float
    precision: float
    recall: float
    review_rate: float


def optimise_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    amounts: np.ndarray,
    *,
    scenario: CostScenario | None = None,
    thresholds: np.ndarray | None = None,
) -> ThresholdResult:
    """Select the threshold with the lowest assumed business cost."""
    scenario = scenario or CostScenario()
    thresholds = np.linspace(0.01, 0.99, 99) if thresholds is None else thresholds
    best: ThresholdResult | None = None
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        cost = decision_cost(y_true, predictions, amounts, scenario)["total_cost"]
        candidate = ThresholdResult(
            threshold=float(threshold),
            total_cost=float(cost),
            precision=float(precision_score(y_true, predictions, zero_division=0)),
            recall=float(recall_score(recall_score_true(y_true), predictions, zero_division=0)),
            review_rate=float(predictions.mean()),
        )
        if best is None or candidate.total_cost < best.total_cost:
            best = candidate
    if best is None:
        raise ValueError("No thresholds supplied")
    return best


def recall_score_true(y_true: np.ndarray) -> np.ndarray:
    """Normalise a target array for explicit validation and testing."""
    values = np.asarray(y_true, dtype=int)
    if not set(np.unique(values)).issubset({0, 1}):
        raise ValueError("y_true must be binary")
    return values


def threshold_for_capacity(probabilities: np.ndarray, review_capacity: float) -> float:
    """Return the score cutoff that sends at most the requested fraction to review."""
    if not 0 < review_capacity <= 1:
        raise ValueError("review_capacity must be in (0, 1]")
    probabilities = np.asarray(probabilities, dtype=float)
    count = max(1, int(np.ceil(len(probabilities) * review_capacity)))
    return float(np.sort(probabilities)[::-1][count - 1])
