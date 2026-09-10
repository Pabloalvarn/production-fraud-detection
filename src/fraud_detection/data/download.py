"""Acquire the PaySim dataset without committing it to Git."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from fraud_detection.config import project_path

KAGGLE_DATASET = "ealaxi/paysim1"
EXPECTED_FILENAME = "PS_20174392719_1491204439457_log.csv"


def copy_local_dataset(source: Path, destination_dir: Path) -> Path:
    """Copy a user-downloaded PaySim CSV into the expected project location."""
    if not source.exists():
        raise FileNotFoundError(f"Dataset not found: {source}")
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / EXPECTED_FILENAME
    if source.resolve() == destination.resolve():
        return destination
    shutil.copy2(source, destination)
    return destination


def download_from_kaggle(destination_dir: Path) -> Path:
    """Download PaySim using the authenticated Kaggle CLI."""
    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "Kaggle CLI not found. Install it with `pip install kaggle`, configure your "
            "Kaggle credentials, or use --source-path with a manually downloaded CSV."
        )
    destination_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            KAGGLE_DATASET,
            "-p",
            str(destination_dir),
            "--unzip",
        ],
        check=True,
    )
    candidates = sorted(destination_dir.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError("Kaggle download completed but no CSV was found.")
    source = candidates[0]
    destination = destination_dir / EXPECTED_FILENAME
    if source != destination:
        source.rename(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--kaggle", action="store_true", help="Use the Kaggle CLI.")
    source.add_argument("--source-path", type=Path, help="Path to a manually downloaded CSV.")
    parser.add_argument("--destination", default="data/external")
    args = parser.parse_args()

    destination_dir = project_path(args.destination)
    path = (
        download_from_kaggle(destination_dir)
        if args.kaggle
        else copy_local_dataset(args.source_path, destination_dir)
    )
    print(f"Dataset ready at {path}")


if __name__ == "__main__":
    main()
