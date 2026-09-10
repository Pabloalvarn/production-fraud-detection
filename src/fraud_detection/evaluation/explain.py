"""Model-agnostic permutation importance on the locked test partition."""

from __future__ import annotations

import argparse

import pandas as pd
from sklearn.inspection import permutation_importance

from fraud_detection.config import load_config, project_path
from fraud_detection.data.split import split_by_time
from fraud_detection.features.build_features import RAW_ONLINE_FEATURES
from fraud_detection.models.predict import load_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/model_xgboost.yaml")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    config = load_config(args.config)
    frame = pd.read_csv(project_path(config["paths"]["raw_data"]))
    split = split_by_time(
        frame,
        time_column=config["data"]["time_column"],
        train_fraction=float(config["data"]["train_fraction"]),
        validation_fraction=float(config["data"]["validation_fraction"]),
    )
    bundle = load_bundle(project_path(config["paths"]["model_bundle"]))
    result = permutation_importance(
        bundle["model"],
        split.test[RAW_ONLINE_FEATURES],
        split.test[config["data"]["target"]],
        scoring="average_precision",
        n_repeats=args.repeats,
        random_state=int(config["project"]["random_state"]),
        n_jobs=-1,
    )
    output = pd.DataFrame(
        {
            "feature": RAW_ONLINE_FEATURES,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    destination = project_path("reports/metrics/permutation_importance.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
