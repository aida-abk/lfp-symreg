from __future__ import annotations

import numpy as np
from scipy import signal

from load_data.convert import TrialData


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
    normalize: One of ``zscore``, ``center``, or ``none``.
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

  x = signal.detrend(x, type="constant")
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
  """Preprocess one channel independently for each selected whole trial.

  Args:
    data: Loaded trial data, including raw sampling frequency in hertz.
    channel: Zero-based channel index.
    trials: Original zero-based trial identifiers.
    downsample: Integer factor applied after filtering.
    lowpass_hz: Optional low-pass cutoff in hertz.
    normalize: ``none``, ``center``, or ``zscore``.
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
