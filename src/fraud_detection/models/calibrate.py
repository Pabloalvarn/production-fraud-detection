"""Probability calibration utilities."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator


def calibrate_fitted_model(
    fitted_model: Any,
    X_calibration: pd.DataFrame,  # noqa: N803
    y_calibration: pd.Series,
    *,
    method: str = "sigmoid",
) -> CalibratedClassifierCV:
    """Calibrate an already-fitted model on a later chronological partition."""
    if method not in {"sigmoid", "isotonic"}:
        raise ValueError("method must be 'sigmoid' or 'isotonic'")
    calibrated = CalibratedClassifierCV(FrozenEstimator(fitted_model), method=method)
    calibrated.fit(X_calibration, y_calibration)
    return calibrated
