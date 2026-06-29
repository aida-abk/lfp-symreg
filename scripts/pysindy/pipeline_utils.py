from __future__ import annotations

import math

import numpy as np

from filter.fixation_filter import fixation_trials, non_fixation_trials
from load_data.preprocessing import channel_traces, preprocess_trace
from load_data.convert import TrialData


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


def select_trials(
  data: TrialData,
  dataset: str,
  max_trials: int | None = None,
) -> list[int]:
  """Select fixation, non-fixation, or all trials in recording order."""
  if dataset == "fixation":
    trials = fixation_trials(data)
  elif dataset == "non-fixation":
    trials = non_fixation_trials(data)
  elif dataset == "all":
    trials = list(range(data.n_trials))
  else:
    raise ValueError(f"Unknown dataset: {dataset}")
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
  """Randomly split whole trials using a reproducible seed."""
  if len(trials) < 2:
    raise ValueError("At least two trials are required for a train/test split.")
  if not 0 < test_fraction < 1:
    raise ValueError("test_fraction must be between 0 and 1.")

  shuffled = np.asarray(trials, dtype=int)
  np.random.default_rng(seed).shuffle(shuffled)
  n_test = min(len(shuffled) - 1, max(1, int(round(len(shuffled) * test_fraction))))
  return shuffled[n_test:].tolist(), shuffled[:n_test].tolist()


def is_contiguous_holdout(
  all_trials: list[int],
  held_out_trials: list[int],
) -> bool:
  """Return whether held-out trials form one contiguous block in time."""
  positions = sorted(all_trials.index(trial) for trial in held_out_trials)
  return positions == list(range(positions[0], positions[-1] + 1))


def split_trials_random_checked(
  trials: list[int],
  test_fraction: float,
  seed: int,
  max_attempts: int = 100,
) -> tuple[list[int], list[int]]:
  """Randomly split trials and reject an accidentally contiguous holdout."""
  if len(trials) < 4:
    raise ValueError("At least four trials are required for a checked random split.")

  for attempt in range(max_attempts):
    train_trials, test_trials = split_trials_random(
      trials,
      test_fraction=test_fraction,
      seed=seed + attempt,
    )
    if not is_contiguous_holdout(trials, test_trials):
      return train_trials, test_trials

  raise RuntimeError(
    "Could not create a random split whose test set is non-contiguous in "
    f"{max_attempts} attempts."
  )


def count_terms(model) -> int:
  """Count nonzero coefficients in a fitted sparse model."""
  return int(np.count_nonzero(np.abs(model.coefficients()) > 1e-12))


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
