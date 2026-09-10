from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def paysim_frame() -> pd.DataFrame:
    rows = []
    for step in range(1, 21):
        for index in range(10):
            fraud = int(index == 0 and step % 2 == 0)
            amount = 9000.0 if fraud else float(20 + step * index)
            rows.append(
                {
                    "step": step,
                    "type": "TRANSFER" if fraud else "PAYMENT",
                    "amount": amount,
                    "nameOrig": f"C{step:02d}{index:02d}",
                    "oldbalanceOrg": 10000.0 if fraud else 5000.0,
                    "newbalanceOrig": 1000.0 if fraud else 5000.0 - amount,
                    "nameDest": f"M{index:02d}",
                    "oldbalanceDest": 0.0 if fraud else 1000.0,
                    "newbalanceDest": amount if fraud else 1000.0 + amount,
                    "isFraud": fraud,
                    "isFlaggedFraud": 0,
                }
            )
    return pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)


class FakeProbabilityModel:
    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        probabilities = np.clip(frame["amount"].to_numpy() / 10000.0, 0.01, 0.99)
        return np.column_stack([1 - probabilities, probabilities])
