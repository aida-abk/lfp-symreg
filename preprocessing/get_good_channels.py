from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from load_data.convert import MAT_FILE


DEFAULT_OUT_CSV = Path("outputs/channel_analysis/good_fixation_trials_by_channel.csv")


def _decode_matlab_string(dataset: h5py.Dataset) -> str:
  """Decode a MATLAB uint16 char dataset as a Python string."""
  values = np.asarray(dataset[()]).squeeze().ravel()
  return "".join(chr(int(value)) for value in values if int(value) != 0)


def _read_cell_refs(file: h5py.File, ref_path: str) -> list[h5py.Dataset]:
  """Return the referenced objects from a MATLAB cell array dataset."""
  refs = np.asarray(file[ref_path][()]).ravel()
  return [file[ref] for ref in refs]


def _read_matlab_table(file: h5py.File) -> dict[str, np.ndarray]:
  """Read the decoded columns from trialdata.bhvTrialTbl.

  The MATLAB table is stored as an HDF5 object with separate hidden cell arrays
  for data columns and variable names. For this dataset:

    #refs#/eun stores bhvTrialTbl data column references.
    #refs#/kvn stores bhvTrialTbl variable-name references.

  This helper reconstructs a Python dict like:

    {"goodFix": array(...), "is_fixation_trial": array(...), ...}
  """
  data_columns = _read_cell_refs(file, "#refs#/eun")
  variable_names = [
    _decode_matlab_string(dataset)
    for dataset in _read_cell_refs(file, "#refs#/kvn")
  ]
  if len(data_columns) != len(variable_names):
    raise ValueError(
      "bhvTrialTbl data columns and variable names have different lengths: "
      f"{len(data_columns)} vs {len(variable_names)}"
    )

  table = {}
  for name, dataset in zip(variable_names, data_columns):
    table[name] = np.asarray(dataset[()]).squeeze()
  return table


def good_fixation_trial_indices(mat_file: Path = MAT_FILE) -> tuple[np.ndarray, np.ndarray]:
  """Return fixation trial indices and goodFix fixation trial indices."""
  with h5py.File(mat_file, "r") as file:
    table = _read_matlab_table(file)

  if "goodFix" not in table:
    raise KeyError("bhvTrialTbl does not contain a goodFix column.")
  if "is_fixation_trial" not in table:
    raise KeyError("bhvTrialTbl does not contain an is_fixation_trial column.")

  is_fixation = table["is_fixation_trial"].astype(bool)
  good_fix = table["goodFix"].astype(float) == 1
  return np.where(is_fixation)[0], np.where(is_fixation & good_fix)[0]


def summarize_good_fixation_by_channel(mat_file: Path = MAT_FILE) -> list[dict[str, float | int]]:
  """Count goodFix fixation trials for every LFP channel.

  In this dataset, goodFix is trial-level behavioral metadata. It is not a
  separate channel-quality label, so every channel receives the same trial
  count. The per-channel CSV is still useful because downstream analyses often
  choose channels one at a time.
  """
  fixation_trials, good_fixation_trials = good_fixation_trial_indices(mat_file)

  with h5py.File(mat_file, "r") as file:
    n_channels = file["trialdata/lfp"].shape[0]

  total = int(fixation_trials.size)
  good = int(good_fixation_trials.size)
  bad = total - good
  percent_good = round(100 * good / total, 2) if total else float("nan")

  return [
    {
      "channel": channel,
      "fixation_trials": total,
      "goodFix_fixation_trials": good,
      "bad_or_not_goodFix_fixation_trials": bad,
      "percent_goodFix_fixation": percent_good,
    }
    for channel in range(n_channels)
  ]


def save_summary(rows: list[dict[str, float | int]], out_csv: Path) -> None:
  """Save the per-channel goodFix fixation summary as a CSV file."""
  out_csv.parent.mkdir(parents=True, exist_ok=True)
  with out_csv.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)


def main() -> None:
  """CLI entry point for counting goodFix fixation trials."""
  parser = argparse.ArgumentParser(
    description="Count bhvTrialTbl.goodFix fixation trials for each LFP channel."
  )
  parser.add_argument("--mat-file", type=Path, default=MAT_FILE)
  parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
  args = parser.parse_args()

  fixation_trials, good_fixation_trials = good_fixation_trial_indices(args.mat_file)
  rows = summarize_good_fixation_by_channel(args.mat_file)
  save_summary(rows, args.out_csv)

  print(f"fixation trials: {fixation_trials.size}")
  print(f"goodFix fixation trials: {good_fixation_trials.size}")
  print(f"bad/not-goodFix fixation trials: {fixation_trials.size - good_fixation_trials.size}")
  print(f"goodFix fixation trial indices: {good_fixation_trials.tolist()}")
  print(f"saved: {args.out_csv}")


if __name__ == "__main__":
  main()
