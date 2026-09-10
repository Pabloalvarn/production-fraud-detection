"""Population stability and distribution-shift report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Compute PSI using reference quantile bins."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    reference_counts, _ = np.histogram(reference, bins=edges)
    current_counts, _ = np.histogram(current, bins=edges)
    reference_pct = np.clip(reference_counts / reference_counts.sum(), epsilon, None)
    current_pct = np.clip(current_counts / current_counts.sum(), epsilon, None)
    return float(np.sum((current_pct - reference_pct) * np.log(current_pct / reference_pct)))


def numeric_drift_report(
    reference: pd.DataFrame, current: pd.DataFrame, columns: list[str]
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for column in columns:
        ref = reference[column].dropna().to_numpy()
        cur = current[column].dropna().to_numpy()
        statistic, p_value = ks_2samp(ref, cur)
        psi = population_stability_index(ref, cur)
        report[column] = {
            "psi": psi,
            "ks_statistic": float(statistic),
            "ks_p_value": float(p_value),
            "alert": bool(psi >= 0.20),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/metrics/drift.json"))
    parser.add_argument(
        "--columns", nargs="+", default=["amount", "oldbalanceOrg", "oldbalanceDest"]
    )
    args = parser.parse_args()
    report = numeric_drift_report(
        pd.read_csv(args.reference), pd.read_csv(args.current), args.columns
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
