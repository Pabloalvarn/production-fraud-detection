import numpy as np

from fraud_detection.evaluation.business_cost import CostScenario, decision_cost
from fraud_detection.evaluation.metrics import classification_metrics
from fraud_detection.evaluation.threshold import optimise_threshold, threshold_for_capacity


def test_classification_metrics_are_consistent():
    y_true = np.array([0, 0, 1, 1])
    probability = np.array([0.1, 0.2, 0.8, 0.9])
    result = classification_metrics(y_true, probability, threshold=0.5)
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["average_precision"] == 1.0


def test_cost_penalises_missed_fraud():
    y_true = np.array([0, 1])
    amounts = np.array([10.0, 1000.0])
    result = decision_cost(
        y_true,
        np.array([0, 0]),
        amounts,
        CostScenario(false_negative_fixed_cost=500, recoverable_fraction=1.0),
    )
    assert result["false_negative_cost"] == 1500.0


def test_threshold_optimisation_and_capacity():
    y_true = np.array([0, 0, 0, 1, 1])
    probability = np.array([0.01, 0.10, 0.20, 0.80, 0.95])
    amounts = np.array([10, 20, 30, 500, 1000])
    result = optimise_threshold(y_true, probability, amounts)
    assert 0 < result.threshold < 1
    assert threshold_for_capacity(probability, 0.20) == 0.95
