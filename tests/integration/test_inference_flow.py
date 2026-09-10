import pandas as pd

from fraud_detection.models.predict import predict_frame
from tests.conftest import FakeProbabilityModel


def test_raw_transaction_to_decision():
    bundle = {
        "model": FakeProbabilityModel(),
        "threshold": 0.5,
        "high_risk_threshold": 0.9,
        "model_version": "test",
        "raw_features": ["step", "type", "amount", "oldbalanceOrg", "oldbalanceDest"],
    }
    frame = pd.DataFrame(
        [
            {
                "step": 1,
                "type": "TRANSFER",
                "amount": 9500,
                "oldbalanceOrg": 10000,
                "oldbalanceDest": 0,
            }
        ]
    )
    result = predict_frame(bundle, frame)
    assert result.iloc[0]["decision"] == "enhanced_verification"
