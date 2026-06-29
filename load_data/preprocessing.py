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
  highpass_hz: float | None = None,
) -> np.ndarray:
  """Detrend, filter, optionally crop, downsample, and normalize one trace.

  Args:
    trace: One raw LFP trace.
    fs: Sampling frequency in Hz.
    downsample: Keep every Nth sample after filtering.
    lowpass_hz: Optional low-pass cutoff frequency.
    normalize: One of ``zscore``, ``center``, or ``none``.
    window_start: Optional crop start in seconds.
    window_end: Optional crop end in seconds.
    highpass_hz: Optional high-pass cutoff frequency.

  Returns:
    A one-dimensional preprocessed LFP trace.
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
  if highpass_hz is not None and not 0 < highpass_hz < nyquist:
    raise ValueError(f"highpass_hz must be between 0 and {nyquist}, got {highpass_hz}")
  if lowpass_hz is not None and not 0 < lowpass_hz < nyquist:
    raise ValueError(f"lowpass_hz must be between 0 and {nyquist}, got {lowpass_hz}")
  if highpass_hz is not None and lowpass_hz is not None and highpass_hz >= lowpass_hz:
    raise ValueError("highpass_hz must be lower than lowpass_hz")

  if highpass_hz is not None or lowpass_hz is not None:
    if highpass_hz is None:
      cutoff, btype = lowpass_hz, "lowpass"
    elif lowpass_hz is None:
      cutoff, btype = highpass_hz, "highpass"
    else:
      cutoff, btype = [highpass_hz, lowpass_hz], "bandpass"
    sos = signal.butter(4, cutoff, btype=btype, fs=fs, output="sos")
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
  highpass_hz: float | None = None,
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
      highpass_hz=highpass_hz,
    )
    for trial in trials
  ]
