from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from load_data.trial_selection import select_valid_trials


def parse_int_list(value: str) -> list[int]:
  """Parse a comma-separated list of integers."""
  return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_float_list(value: str) -> list[float]:
  """Parse a comma-separated list of floating-point values."""
  return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_lowpass_list(value: str) -> list[float | None]:
  """Parse low-pass values, treating none and zero as no filtering."""
  values = []
  for part in value.split(","):
    part = part.strip().lower()
    if part:
      values.append(None if part in {"none", "0"} else float(part))
  return values


def parse_trials(value: str, n_trials: int, max_trials: int | None = None) -> list[int]:
  """Parse a comma-separated list of trial indices with range validation."""
  trials = [int(part) for part in value.split(",") if part.strip()]
  bad = [t for t in trials if t < 0 or t >= n_trials]
  if bad:
    raise ValueError(f"Trial indices out of range: {bad}")
  return trials if max_trials is None else trials[:max_trials]


def select_trials(
  table: Mapping[str, np.ndarray],
  dataset: str,
  max_trials: int | None = None,
) -> list[int]:
  """Select fixation or non-fixation trials using behavioral table columns."""
  if dataset == "fixation":
    trials = select_valid_trials(table, "fixation")
  elif dataset in ("non-fixation", "non_fixation"):
    trials = select_valid_trials(table, "non_fixation")
  else:
    raise ValueError(f"Unknown dataset: {dataset!r}; expected 'fixation' or 'non-fixation'")
  return trials if max_trials is None else trials[:max_trials]


def split_trials_sequential(
  trials: list[int],
  test_fraction: float,
) -> tuple[list[int], list[int]]:
  """Keep recording order and hold out the final trials for testing."""
  if len(trials) < 2:
    raise ValueError("At least two trials are required for a train/test split.")
  if not 0 < test_fraction < 1:
    raise ValueError("test_fraction must be between 0 and 1.")

  n_test = min(len(trials) - 1, max(1, int(round(len(trials) * test_fraction))))
  return trials[:-n_test], trials[-n_test:]


def split_trials_random(
  trials: list[int],
  test_fraction: float,
  seed: int,
) -> tuple[list[int], list[int]]:
  """Randomly split whole trials using a reproducible seed.

  Args:
    trials: Original integer trial identifiers.
    test_fraction: Unitless fraction assigned to the test set, between 0 and 1.
    seed: NumPy random-generator seed.

  Returns:
    Training and test trial identifiers. Samples within trials are never split.
  """
  if len(trials) < 2:
    raise ValueError("At least two trials are required for a train/test split.")
  if not 0 < test_fraction < 1:
    raise ValueError("test_fraction must be between 0 and 1.")

  shuffled = np.asarray(trials, dtype=int)
  np.random.default_rng(seed).shuffle(shuffled)
  n_test = min(len(shuffled) - 1, max(1, int(round(len(shuffled) * test_fraction))))
  return shuffled[n_test:].tolist(), shuffled[:n_test].tolist()


def best_rows(rows: list[dict[str, object]], limit: int) -> list[dict[str, object]]:
  """Return successful rows ranked by test R2 and then sparsity."""
  valid = [
    row
    for row in rows
    if row["status"] == "ok" and math.isfinite(float(row["test_score_r2"]))
  ]
  return sorted(
    valid,
    key=lambda row: (float(row["test_score_r2"]), -int(row["nonzero_terms"])),
    reverse=True,
  )[:limit]
