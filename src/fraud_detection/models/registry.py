"""Small MLflow wrapper that keeps experiment tracking optional and explicit."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow


@contextmanager
def tracked_run(
    experiment_name: str,
    *,
    tracking_uri: str,
    run_name: str,
    tags: dict[str, str] | None = None,
) -> Iterator[Any]:
    """Create a local or remote MLflow run."""
    uri = tracking_uri
    if not uri.startswith(("http://", "https://", "sqlite:", "file:")):
        uri = Path(uri).resolve().as_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        yield run


def log_dictionary(values: dict[str, Any], artifact_file: str) -> None:
    """Log a JSON-serialisable dictionary as an MLflow artifact."""
    mlflow.log_dict(values, artifact_file)
