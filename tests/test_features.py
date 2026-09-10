import numpy as np

from fraud_detection.features.build_features import build_online_features, safe_ratio


def test_online_features_exclude_leakage(paysim_frame):
    features = build_online_features(paysim_frame)
    assert "isFraud" not in features
    assert "isFlaggedFraud" not in features
    assert "newbalanceOrig" not in features
    assert "newbalanceDest" not in features
    assert "log_amount" in features


def test_safe_ratio_is_finite(paysim_frame):
    ratios = safe_ratio(paysim_frame["amount"], paysim_frame["oldbalanceDest"])
    assert np.isfinite(ratios).all()
