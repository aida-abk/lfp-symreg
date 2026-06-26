from __future__ import annotations

import math

import numpy as np
from scipy import signal

from filter.fixation_filter import fixation_trials, non_fixation_trials
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


def preprocess_trace(
  trace: np.ndarray,
  fs: float,
  downsample: int,
  lowpass_hz: float | None,
  normalize: str,
  window_start: float | None = None,
  window_end: float | None = None,
) -> np.ndarray:
  """Detrend, filter, optionally crop, downsample, and normalize one trace."""
  x = np.asarray(trace, dtype=float).squeeze()
  if x.ndim != 1:
    raise ValueError(f"Expected a 1D trace, got shape {x.shape}")
  if downsample < 1:
    raise ValueError("downsample must be >= 1")
  if (window_start is None) != (window_end is None):
    raise ValueError("window_start and window_end must be provided together.")

  x = signal.detrend(x, type="constant")
  if lowpass_hz is not None:
    nyquist = fs / 2
    if not 0 < lowpass_hz < nyquist:
      raise ValueError(f"lowpass_hz must be between 0 and {nyquist}, got {lowpass_hz}")
    sos = signal.butter(4, lowpass_hz, btype="lowpass", fs=fs, output="sos")
    x = signal.sosfiltfilt(sos, x)

  if window_start is not None and window_end is not None:
    if not 0 <= window_start < window_end:
      raise ValueError("Require 0 <= window_start < window_end.")
    start_sample = int(round(window_start * fs))
    end_sample = int(round(window_end * fs))
    if end_sample > x.size:
      raise ValueError(
        f"Window ending at {window_end}s needs {end_sample} samples, "
        f"but the trace contains {x.size}."
      )
    x = x[start_sample:end_sample]

  x = x[::downsample]
  if normalize == "zscore":
    std = np.std(x)
    x = (x - np.mean(x)) / std if std > 0 else x - np.mean(x)
  elif normalize == "center":
    x = x - np.mean(x)
  elif normalize != "none":
    raise ValueError(f"Unknown normalize mode: {normalize}")
  return x


def channel_traces(
  data: TrialData,
  channel: int,
  trials: list[int],
  downsample: int,
  lowpass_hz: float | None,
  normalize: str,
  window_start: float | None = None,
  window_end: float | None = None,
) -> list[np.ndarray]:
  """Preprocess one channel independently for each selected trial."""
  return [
    preprocess_trace(
      data.lfp_trace(trial, channel),
      fs=data.fs,
      downsample=downsample,
      lowpass_hz=lowpass_hz,
      normalize=normalize,
      window_start=window_start,
      window_end=window_end,
    )
    for trial in trials
  ]


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
