"""Data-contract validation for the PaySim transaction table."""

from __future__ import annotations

import argparse
import json
from typing import Any

import pandas as pd

from fraud_detection.config import load_config, project_path

EXPECTED_COLUMNS = {
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
}
VALID_TYPES = {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}


class DataValidationError(ValueError):
    """Raised when a hard data-contract requirement fails."""


def validate_dataframe(frame: pd.DataFrame, *, raise_on_error: bool = True) -> dict[str, Any]:
    """Return a JSON-serialisable validation report."""
    missing = sorted(EXPECTED_COLUMNS - set(frame.columns))
    unexpected_types = (
        sorted(set(frame["type"].dropna().unique()) - VALID_TYPES)
        if "type" in frame.columns
        else []
    )
    target_values = (
        sorted(frame["isFraud"].dropna().unique().tolist()) if "isFraud" in frame.columns else []
    )
    negative_amounts = int((frame["amount"] < 0).sum()) if "amount" in frame.columns else None
    duplicate_rows = int(frame.duplicated().sum())
    null_counts = {column: int(value) for column, value in frame.isna().sum().items()}
    prevalence = float(frame["isFraud"].mean()) if "isFraud" in frame.columns else None
    monotonic_time = (
        bool(frame["step"].is_monotonic_increasing) if "step" in frame.columns else None
    )

    errors: list[str] = []
    if missing:
        errors.append(f"Missing required columns: {missing}")
    if unexpected_types:
        errors.append(f"Unexpected transaction types: {unexpected_types}")
    if target_values and not set(target_values).issubset({0, 1}):
        errors.append(f"Target contains values outside {{0, 1}}: {target_values}")
    if negative_amounts:
        errors.append(f"Found {negative_amounts} negative transaction amounts")

    warnings: list[str] = []
    if duplicate_rows:
        warnings.append(f"Found {duplicate_rows} fully duplicated rows")
    if monotonic_time is False:
        warnings.append(
            "Rows are not ordered by step; sorting is required before temporal splitting"
        )

    report = {
        "valid": not errors,
        "row_count": int(len(frame)),
        "column_count": int(frame.shape[1]),
        "missing_columns": missing,
        "unexpected_transaction_types": unexpected_types,
        "target_values": target_values,
        "fraud_prevalence": prevalence,
        "negative_amounts": negative_amounts,
        "duplicate_rows": duplicate_rows,
        "null_counts": null_counts,
        "time_is_monotonic": monotonic_time,
        "errors": errors,
        "warnings": warnings,
    }
    if errors and raise_on_error:
        raise DataValidationError("; ".join(errors))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--rows", type=int, default=None, help="Optional development sample.")
    args = parser.parse_args()
    config = load_config(args.config)
    raw_path = project_path(config["paths"]["raw_data"])
    report_path = project_path(config["paths"]["validation_report"])
    frame = pd.read_csv(raw_path, nrows=args.rows)
    report = validate_dataframe(frame, raise_on_error=False)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
