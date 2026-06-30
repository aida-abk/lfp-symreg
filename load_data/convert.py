from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
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


def _decode_matlab_char(dataset: h5py.Dataset) -> str:
  """Decode a MATLAB uint16 char dataset."""
  values = np.asarray(dataset[()]).squeeze().ravel()
  return "".join(chr(int(value)) for value in values if int(value) != 0)


def _read_cell_refs(file: h5py.File, ref_path: str) -> list[h5py.Dataset]:
  """Return the referenced objects from a MATLAB HDF5 cell array."""
  refs = np.asarray(file[ref_path][()]).ravel()
  return [file[ref] for ref in refs]


def _load_matlab_table_from_refs(
  file: h5py.File,
  data_ref_path: str,
  variable_name_ref_path: str,
) -> dict[str, np.ndarray]:
  """Decode a MATLAB table stored as separate data and name reference cells."""
  data_columns = _read_cell_refs(file, data_ref_path)
  variable_names = [
    _decode_matlab_char(dataset)
    for dataset in _read_cell_refs(file, variable_name_ref_path)
  ]
  if len(data_columns) != len(variable_names):
    raise ValueError(
      "MATLAB table data columns and variable names have different lengths: "
      f"{len(data_columns)} vs {len(variable_names)}"
    )

  table = {}
  for name, dataset in zip(variable_names, data_columns):
    table[name] = np.asarray(dataset[()]).squeeze()
  return table


def load_bhv_trial_table(path: str | Path = MAT_FILE) -> dict[str, np.ndarray]:
  """Load the MATLAB ``trialdata.bhvTrialTbl`` table as NumPy arrays.

  This function is intentionally separate from ``TrialData.load`` because the
  behavioral table uses MATLAB's HDF5 table/MCOS encoding and is slower/more
  specialized to decode. Call this only when a script needs trial metadata such
  as ``goodFix``, ``is_fixation_trial``, ``duration``, or sequence columns.

  Args:
    path: Path to the MATLAB ``trialdata`` file.

  Returns:
    A dictionary mapping each ``bhvTrialTbl`` variable name to a NumPy array.
    Each array is indexed by the 0-based LFP trial index.
  """
  with h5py.File(path, "r") as file:
    # hdf5storage can load the LFP cell array, but it leaves MATLAB table
    # objects as opaque references. For this dataset, these hidden refs are the
    # bhvTrialTbl data-cell and variable-name-cell entries.
    return _load_matlab_table_from_refs(
      file,
      data_ref_path="#refs#/eun",
      variable_name_ref_path="#refs#/kvn",
    )


def load_lfp_shape(path: str | Path = MAT_FILE) -> tuple[int, int]:
  """Return ``(n_channels, n_trials)`` without loading all LFP traces."""
  with h5py.File(path, "r") as file:
    n_channels, n_trials = file["trialdata/lfp"].shape
  return int(n_channels), int(n_trials)


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
