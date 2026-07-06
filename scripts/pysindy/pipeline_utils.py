from __future__ import annotations

import math

import numpy as np


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
