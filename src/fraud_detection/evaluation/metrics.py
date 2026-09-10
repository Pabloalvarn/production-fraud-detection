"""Metrics for rare-event discrimination, calibration and operations."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
    amounts: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Calculate threshold-free and threshold-dependent rare-event metrics."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    metrics: dict[str, float | int] = {
        "prevalence": float(y_true.mean()),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "f2": float(fbeta_score(y_true, predictions, beta=2, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "matthews_correlation": float(matthews_corrcoef(y_true, predictions)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "threshold": float(threshold),
        "review_rate": float(predictions.mean()),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }
    if amounts is not None:
        amounts = np.asarray(amounts, dtype=float)
        fraud_total = float(amounts[y_true == 1].sum())
        captured = float(amounts[(y_true == 1) & (predictions == 1)].sum())
        metrics["fraud_value_total"] = fraud_total
        metrics["fraud_value_captured"] = captured
        metrics["fraud_value_capture_rate"] = captured / fraud_total if fraud_total else 0.0
    return metrics


def precision_recall_at_capacity(
    y_true: np.ndarray, probabilities: np.ndarray, capacity: float
) -> dict[str, float]:
    """Measure performance when only the highest-risk fraction can be reviewed."""
    if not 0 < capacity <= 1:
        raise ValueError("capacity must be in (0, 1]")
    count = max(1, int(np.ceil(len(probabilities) * capacity)))
    selected = np.argsort(probabilities)[::-1][:count]
    predictions = np.zeros(len(probabilities), dtype=int)
    predictions[selected] = 1
    return {
        "capacity": float(capacity),
        "precision_at_capacity": float(precision_score(y_true, predictions, zero_division=0)),
        "recall_at_capacity": float(recall_score(y_true, predictions, zero_division=0)),
    }
