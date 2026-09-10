"""Reusable model-evaluation plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay


def save_core_plots(
    y_true: np.ndarray, probabilities: np.ndarray, output_dir: str | Path
) -> list[Path]:
    """Save precision-recall, ROC and calibration charts."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    specifications = [
        (
            "precision_recall.png",
            lambda ax: PrecisionRecallDisplay.from_predictions(y_true, probabilities, ax=ax),
        ),
        (
            "roc_curve.png",
            lambda ax: RocCurveDisplay.from_predictions(y_true, probabilities, ax=ax),
        ),
        (
            "calibration.png",
            lambda ax: CalibrationDisplay.from_predictions(
                y_true, probabilities, n_bins=10, strategy="quantile", ax=ax
            ),
        ),
    ]
    for filename, draw in specifications:
        figure, axis = plt.subplots(figsize=(7, 5))
        draw(axis)
        axis.grid(alpha=0.25)
        figure.tight_layout()
        path = output / filename
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)
    return paths
