from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import hdf5storage
import numpy as np

MAT_FILE = Path(__file__).resolve().parents[1] / (
    "raw_data/trialdata_v03_buzz_20231106_pre-0.100_post0.100.mat"
)


def _scalar(value) -> float | int | str:
  """Extract a Python scalar from nested MATLAB arrays."""
  arr = np.asarray(value).squeeze()
  if arr.dtype.kind in {"U", "S", "O"}:
    return str(arr)
  if arr.ndim != 0:
    raise ValueError(f"Expected scalar value, got shape {arr.shape}")
  if np.issubdtype(arr.dtype, np.floating):
    return float(arr)
  return int(arr)


def _string_array(value) -> list[str]:
  """Extract a list of strings from a MATLAB cell/char array."""
  arr = np.asarray(value).squeeze()
  if arr.ndim == 0:
    return [str(arr)]
  return [str(x.squeeze()) for x in arr.ravel()]


@dataclass
class TrialData:
  fs: float
  pre_pad: float
  post_pad: float
  sessname: str
  channels: np.ndarray
  grouping_vars: list[str]
  trial_definition: str
  trial_rows: list[np.ndarray]
  lfp: np.ndarray

  @classmethod
  def load(cls, path: str | Path = MAT_FILE) -> TrialData:
    raw = hdf5storage.loadmat(str(path))
    td = raw["trialdata"]
    info = td["info"][0, 0]

    trial_rows = [
      np.atleast_1d(np.asarray(x, dtype=float).squeeze())
      for x in td["trial_rows"][0, :, 0]
    ]

    return cls(
      fs=_scalar(info["Fs"]),
      pre_pad=_scalar(info["pre_pad"]),
      post_pad=_scalar(info["post_pad"]),
      sessname=_scalar(info["sessname"]),
      channels=np.asarray(info["channels"], dtype=float).squeeze(),
      grouping_vars=_string_array(info["grouping_vars"]),
      trial_definition=_scalar(info["trial_definition"]),
      trial_rows=trial_rows,
      lfp=td["lfp"],
    )

  @property
  def n_trials(self) -> int:
    return self.lfp.shape[1]

  @property
  def n_channels(self) -> int:
    return self.lfp.shape[2]

  def trial_row(self, trial: int) -> np.ndarray:
    """Return behavioral row metadata for a trial (scalar or vector)."""
    if not (0 <= trial < self.n_trials):
      raise IndexError(f"trial {trial} out of range (0..{self.n_trials - 1})")
    return self.trial_rows[trial]

  def lfp_trace(self, trial: int, channel: int) -> np.ndarray:
    """Return one LFP trace as a 1D float array.

    Args:
      trial: 0-based trial index.
      channel: 0-based channel index.
    """
    if not (0 <= trial < self.n_trials):
      raise IndexError(f"trial {trial} out of range (0..{self.n_trials - 1})")
    if not (0 <= channel < self.n_channels):
      raise IndexError(f"channel {channel} out of range (0..{self.n_channels - 1})")

    return np.asarray(self.lfp[0, trial, channel], dtype=float).squeeze()

  def time_axis(self, n_samples: int | None = None, trial: int = 0, channel: int = 0) -> np.ndarray:
    """Return time in seconds for an LFP trace."""
    if n_samples is None:
      n_samples = self.lfp_trace(trial, channel).size
    return np.arange(n_samples) / self.fs


if __name__ == "__main__":
  data = TrialData.load()

  print(f"session: {data.sessname}")
  print(f"trials: {data.n_trials}, channels: {data.n_channels}, fs: {data.fs} Hz")
  print(f"pre_pad: {data.pre_pad}s, post_pad: {data.post_pad}s")
  print(f"grouping vars: {data.grouping_vars}")

  trial = 0
  channel = 0
  x = data.lfp_trace(trial, channel)
  t = data.time_axis(len(x))
  print(f"lfp shape for trial={trial}, channel={channel}: {x.shape}")
  print(f"time range: {t[0]:.3f}s to {t[-1]:.3f}s")
  print(f"first samples: {x[:5]}")
  print(f"trial row metadata: {data.trial_row(trial)}")
