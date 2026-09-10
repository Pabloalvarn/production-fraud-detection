"""Scikit-learn compatible transformations."""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from fraud_detection.features.build_features import build_online_features


class OnlineFeatureBuilder(TransformerMixin, BaseEstimator):
    """Convert a raw PaySim frame into leakage-safe online features."""

    def fit(self, X: pd.DataFrame, y: object = None) -> OnlineFeatureBuilder:  # noqa: N803
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        return build_online_features(X)

    def get_feature_names_out(self, input_features: object = None) -> list[str]:
        return list(
            build_online_features(
                pd.DataFrame(
                    {
                        "step": [0],
                        "type": ["PAYMENT"],
                        "amount": [0.0],
                        "oldbalanceOrg": [0.0],
                        "oldbalanceDest": [0.0],
                    }
                )
            ).columns
        )
