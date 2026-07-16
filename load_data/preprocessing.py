from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from load_data.convert import TrialData


@dataclass
class GlobalZScoreStats:
  """Per-channel (mean, std) computed exclusively from training samples.

  Storing these constants makes the transform invertible: multiply by std and
  add mean to recover physical (µV) units. Pass the same object to held-out
  splits to avoid leakage.

  This operates on the state variable fed into the delay-embedding library and
  is strictly separate from STLSQ's normalize_columns, which acts on the
  constructed feature-library columns.
  """
  channel: int
  mean: float
  std: float


def compute_global_zscore_stats(
  traces: list[np.ndarray],
  channel: int,
) -> GlobalZScoreStats:
  """Pool all training samples for one channel and return their (mean, std).

  Args:
    traces: Preprocessed (filtered, not yet normalized) training traces.
      Each trace is 1D; lengths may differ across trials.
    channel: Zero-based channel index stored for identification only.

  Returns:
    GlobalZScoreStats with mean and std computed over all pooled samples.
    The std is guaranteed positive (raises ValueError if zero).
  """
  if not traces:
    raise ValueError("At least one trace is required.")
  all_samples = np.concatenate([np.asarray(t, dtype=float).ravel() for t in traces])
  std = float(np.std(all_samples))
  if std == 0:
    raise ValueError(f"Channel {channel}: pooled training std is zero — cannot z-score.")
  return GlobalZScoreStats(
    channel=channel,
    mean=float(np.mean(all_samples)),
    std=std,
  )


def apply_global_zscore(
  traces: list[np.ndarray],
  stats: GlobalZScoreStats,
) -> list[np.ndarray]:
  """Apply a fixed (mean, std) z-score transform to every trace.

  Uses the same constants for every trial, so trial-to-trial amplitude
  differences are preserved after scaling.

  Args:
    traces: Traces to transform (any split: train, test, validation).
    stats: Pre-computed stats from training data only.

  Returns:
    Transformed traces; same structure as the input.
  """
  return [(np.asarray(t, dtype=float) - stats.mean) / stats.std for t in traces]




def preprocess_trace(
  trace: np.ndarray,
  fs: float,
  downsample: int,
  lowpass_hz: float | None,
  normalize: str,
  window_start: float | None = None,
  window_end: float | None = None,
) -> np.ndarray:
  """Detrend, filter, optionally crop, downsample, and normalize one trace.

  Args:
    trace: One raw LFP trace with shape ``(n_samples,)`` in stored amplitude
      units.
    fs: Sampling frequency in hertz.
    downsample: Keep every Nth sample after filtering.
    lowpass_hz: Optional low-pass cutoff frequency in hertz.
    normalize: One of ``zscore`` or ``none``.
    window_start: Optional crop start in seconds.
    window_end: Optional crop end in seconds.

  Returns:
    A one-dimensional preprocessed LFP trace. ``none`` and ``center`` retain
    the input amplitude scale; ``zscore`` returns per-trial SD units.
  """
  x = np.asarray(trace, dtype=float).squeeze()
  if x.ndim != 1:
    raise ValueError(f"Expected a 1D trace, got shape {x.shape}")
  if downsample < 1:
    raise ValueError("downsample must be >= 1")
  if (window_start is None) != (window_end is None):
    raise ValueError("window_start and window_end must be provided together.")

  x = signal.detrend(x, type="constant") # ==>LOOK UP
  nyquist = fs / 2
  if lowpass_hz is not None and not 0 < lowpass_hz < nyquist:
    raise ValueError(f"lowpass_hz must be between 0 and {nyquist}, got {lowpass_hz}")

  if lowpass_hz is not None:
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
  """Preprocess one channel independently for each selected whole trial.

  Args:
    data: Loaded trial data, including raw sampling frequency in hertz.
    channel: Zero-based channel index.
    trials: Original zero-based trial identifiers.
    downsample: Integer factor applied after filtering.
    lowpass_hz: Optional low-pass cutoff in hertz.
    normalize: ``none`` or ``zscore``.
    window_start: Optional crop start in seconds.
    window_end: Optional crop end in seconds.

  Returns:
    One array with shape ``(processed_samples,)`` per selected trial. Unequal
    trial lengths remain unequal.
  """
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


# RMS
def pooled_trace_rms(traces: list[np.ndarray]) -> float:
  """Return RMS across every sample in a collection of unequal-length traces.

  Args:
    traces: One-dimensional traces with a shared amplitude unit. Trials may
      contain different numbers of samples.

  Returns:
    Root mean square in the input amplitude unit. Each sample has equal weight.
  """
  if not traces:
    raise ValueError("At least one trace is required to calculate RMS.")
  squared_sum = 0.0
  sample_count = 0
  for trace in traces:
    values = np.asarray(trace, dtype=float).squeeze()
    if values.ndim != 1:
      raise ValueError(f"Expected a 1D trace, got shape {values.shape}.")
    squared_sum += float(np.sum(values**2))
    sample_count += values.size
  if sample_count == 0:
    raise ValueError("Cannot calculate RMS from empty traces.")
  return float(np.sqrt(squared_sum / sample_count))