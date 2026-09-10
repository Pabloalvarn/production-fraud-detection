"""Train, calibrate, evaluate and serialise a fraud-detection model."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from fraud_detection.config import load_config, project_path
from fraud_detection.data.split import split_by_time
from fraud_detection.data.validation import validate_dataframe
from fraud_detection.evaluation.baselines import RuleBasedBaseline, prevalence_probabilities
from fraud_detection.evaluation.business_cost import CostScenario, decision_cost
from fraud_detection.evaluation.metrics import (
    classification_metrics,
    precision_recall_at_capacity,
)
from fraud_detection.evaluation.plots import save_core_plots
from fraud_detection.evaluation.threshold import optimise_threshold
from fraud_detection.features.build_features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RAW_ONLINE_FEATURES,
)
from fraud_detection.features.transformers import OnlineFeatureBuilder
from fraud_detection.models.calibrate import calibrate_fitted_model
from fraud_detection.models.registry import log_dictionary, tracked_run


def make_preprocessor(*, scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric = SklearnPipeline(numeric_steps)
    categorical = SklearnPipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def make_estimator(config: dict[str, Any], positive_weight: float) -> Any:
    model_config = config["model"]
    parameters = dict(model_config.get("parameters", {}))
    random_state = int(config["project"]["random_state"])
    if model_config["name"] == "logistic_regression":
        return LogisticRegression(random_state=random_state, **parameters)
    if model_config["name"] == "xgboost":
        parameters.setdefault("scale_pos_weight", positive_weight)
        return XGBClassifier(random_state=random_state, **parameters)
    raise ValueError(f"Unsupported model: {model_config['name']}")


def make_training_pipeline(config: dict[str, Any], positive_weight: float) -> Pipeline:
    model_name = config["model"]["name"]
    steps: list[tuple[str, Any]] = [
        ("features", OnlineFeatureBuilder()),
        ("preprocess", make_preprocessor(scale_numeric=model_name == "logistic_regression")),
    ]
    sampling = config["model"].get("sampling", "none")
    random_state = int(config["project"]["random_state"])
    if sampling == "undersample":
        steps.append(("sampling", RandomUnderSampler(random_state=random_state)))
    elif sampling == "smote":
        steps.append(("sampling", SMOTE(random_state=random_state)))
    elif sampling != "none":
        raise ValueError(f"Unsupported sampling strategy: {sampling}")
    steps.append(("model", make_estimator(config, positive_weight)))
    return Pipeline(steps)


def chronological_validation_halves(
    frame: pd.DataFrame, time_column: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use earlier validation observations for calibration and later ones for thresholding."""
    ordered = frame.sort_values(time_column, kind="stable")
    split_index = len(ordered) // 2
    if split_index == 0 or split_index == len(ordered):
        raise ValueError("Validation partition is too small to divide")
    return ordered.iloc[:split_index].copy(), ordered.iloc[split_index:].copy()


def run_training(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    data_path = project_path(config["paths"]["raw_data"])
    sample_rows = config["training"].get("sample_rows")
    frame = pd.read_csv(data_path, nrows=sample_rows)
    validate_dataframe(frame)
    target = config["data"]["target"]
    time_column = config["data"]["time_column"]
    split = split_by_time(
        frame,
        time_column=time_column,
        train_fraction=float(config["data"]["train_fraction"]),
        validation_fraction=float(config["data"]["validation_fraction"]),
    )
    calibration, threshold_data = chronological_validation_halves(split.validation, time_column)
    X_train = split.train[RAW_ONLINE_FEATURES]
    y_train = split.train[target].astype(int)
    positive_count = int(y_train.sum())
    if positive_count == 0:
        raise ValueError("Training data contains no positive fraud examples")
    positive_weight = float((len(y_train) - positive_count) / positive_count)
    model = make_training_pipeline(config, positive_weight)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    model_name = config["model"]["name"]
    with tracked_run(
        config["training"]["mlflow_experiment"],
        tracking_uri=tracking_uri,
        run_name=model_name,
        tags={"validation": "temporal", "dataset": "PaySim", "use_case": "pre-authorisation"},
    ):
        mlflow.log_params(
            {f"model_{key}": value for key, value in config["model"].get("parameters", {}).items()}
        )
        mlflow.log_param("sampling", config["model"].get("sampling", "none"))
        mlflow.log_param("train_end_step", split.train_end_step)
        mlflow.log_param("validation_end_step", split.validation_end_step)
        model.fit(X_train, y_train)

        calibrated = calibrate_fitted_model(
            model,
            calibration[RAW_ONLINE_FEATURES],
            calibration[target].astype(int),
            method=config["training"]["calibration_method"],
        )
        threshold_probabilities = calibrated.predict_proba(threshold_data[RAW_ONLINE_FEATURES])[
            :, 1
        ]
        scenario = CostScenario(
            false_positive_cost=float(config["decision"]["false_positive_cost"]),
            false_negative_fixed_cost=float(config["decision"]["false_negative_fixed_cost"]),
            review_cost=float(config["decision"]["review_cost"]),
        )
        threshold_result = optimise_threshold(
            threshold_data[target].to_numpy(),
            threshold_probabilities,
            threshold_data["amount"].to_numpy(),
            scenario=scenario,
        )

        test_probabilities = calibrated.predict_proba(split.test[RAW_ONLINE_FEATURES])[:, 1]
        test_metrics = classification_metrics(
            split.test[target].to_numpy(),
            test_probabilities,
            threshold=threshold_result.threshold,
            amounts=split.test["amount"].to_numpy(),
        )
        test_predictions = (test_probabilities >= threshold_result.threshold).astype(int)
        test_metrics.update(
            decision_cost(
                split.test[target].to_numpy(),
                test_predictions,
                split.test["amount"].to_numpy(),
                scenario,
            )
        )
        test_metrics.update(
            precision_recall_at_capacity(
                split.test[target].to_numpy(),
                test_probabilities,
                float(config["decision"]["review_capacity"]),
            )
        )
        rule_baseline = RuleBasedBaseline().fit(split.train)
        rule_probabilities = rule_baseline.predict_proba(split.test)[:, 1]
        baseline_metrics = {
            "constant_prevalence": classification_metrics(
                split.test[target].to_numpy(),
                prevalence_probabilities(len(split.test), float(y_train.mean())),
                threshold=0.5,
                amounts=split.test["amount"].to_numpy(),
            ),
            "large_transfer_rule": classification_metrics(
                split.test[target].to_numpy(),
                rule_probabilities,
                threshold=0.5,
                amounts=split.test["amount"].to_numpy(),
            ),
            "large_transfer_amount_threshold": rule_baseline.amount_threshold_,
        }
        metrics_path = project_path(config["paths"]["metrics"])
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
        baseline_path = project_path(config["paths"]["baseline_metrics"])
        baseline_path.write_text(json.dumps(baseline_metrics, indent=2), encoding="utf-8")
        prediction_path = project_path(config["paths"]["predictions"])
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "step": split.test[time_column].to_numpy(),
                "y_true": split.test[target].to_numpy(),
                "probability": test_probabilities,
                "prediction": test_predictions,
                "amount": split.test["amount"].to_numpy(),
            }
        ).to_csv(prediction_path, index=False)
        figure_paths = save_core_plots(
            split.test[target].to_numpy(),
            test_probabilities,
            project_path(config["paths"]["figures"]),
        )
        for key, value in test_metrics.items():
            if isinstance(value, int | float) and np.isfinite(value):
                mlflow.log_metric(key, float(value))
        log_dictionary(test_metrics, "test_metrics.json")
        log_dictionary(baseline_metrics, "baseline_metrics.json")
        log_dictionary(threshold_result.__dict__, "threshold_selection.json")
        for path in figure_paths:
            mlflow.log_artifact(str(path), artifact_path="figures")

    bundle = {
        "model": calibrated,
        "threshold": threshold_result.threshold,
        "high_risk_threshold": float(config["decision"]["high_risk_threshold"]),
        "model_version": "0.1.0",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "raw_features": RAW_ONLINE_FEATURES,
        "target": target,
        "training_end_step": split.train_end_step,
        "validation_end_step": split.validation_end_step,
        "cost_scenario": scenario.to_dict(),
        "test_metrics": test_metrics,
    }
    model_path = project_path(config["paths"]["model_bundle"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    return {
        "model_path": str(model_path),
        "metrics": test_metrics,
        "threshold": threshold_result.__dict__,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/model_xgboost.yaml")
    args = parser.parse_args()
    print(json.dumps(run_training(args.config), indent=2))


if __name__ == "__main__":
    main()
