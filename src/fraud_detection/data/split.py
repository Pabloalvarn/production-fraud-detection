"""Leakage-aware chronological train/validation/test splits."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_end_step: int | float
    validation_end_step: int | float


def split_by_time(
    frame: pd.DataFrame,
    *,
    time_column: str = "step",
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> TemporalSplit:
    """Split whole time steps so a step never appears in multiple partitions."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be below 1")
    if time_column not in frame:
        raise KeyError(f"Missing time column: {time_column}")

    ordered_steps = sorted(frame[time_column].dropna().unique())
    if len(ordered_steps) < 3:
        raise ValueError("At least three unique time steps are required")
    train_index = max(1, int(len(ordered_steps) * train_fraction))
    validation_index = max(
        train_index + 1, int(len(ordered_steps) * (train_fraction + validation_fraction))
    )
    validation_index = min(validation_index, len(ordered_steps) - 1)
    train_end = ordered_steps[train_index - 1]
    validation_end = ordered_steps[validation_index - 1]

    ordered = frame.sort_values(time_column, kind="stable").reset_index(drop=True)
    train = ordered.loc[ordered[time_column] <= train_end].copy()
    validation = ordered.loc[
        (ordered[time_column] > train_end) & (ordered[time_column] <= validation_end)
    ].copy()
    test = ordered.loc[ordered[time_column] > validation_end].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Temporal split produced an empty partition")
    return TemporalSplit(train, validation, test, train_end, validation_end)
