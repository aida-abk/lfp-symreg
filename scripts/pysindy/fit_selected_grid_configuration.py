from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Project imports
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from load_data.convert import MAT_FILE
from raw_grid_sweep import run_raw_grid


def read_selection(path: Path, selection_index: int) -> dict[str, str]:
  """Read one zero-based parameter row from a selection manifest.

  Args:
    path: CSV manifest with one selected parameter configuration per row.
    selection_index: Zero-based data-row index.

  Returns:
    Selected string-valued parameter row.
  """
  with path.open(newline="") as file:
    rows = list(csv.DictReader(file))
  if not 0 <= selection_index < len(rows):
    raise IndexError(
      f"Selection index {selection_index} is outside 0..{len(rows) - 1}."
    )
  return rows[selection_index]


def main() -> None:
  """Fit one selected parameter configuration using the raw-grid pipeline."""
  parser = argparse.ArgumentParser(
    description="Fit one manifest-selected PySINDy configuration."
  )
  parser.add_argument("--selection-csv", type=Path, required=True)
  parser.add_argument("--selection-index", type=int, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--mat-file", type=Path, default=MAT_FILE)
  parser.add_argument("--threshold", type=float, default=0.1)
  parser.add_argument(
    "--normalize-columns",
    action=argparse.BooleanOptionalAction,
    default=True,
  )
  args = parser.parse_args()

  selected = read_selection(args.selection_csv, args.selection_index)
  stem = f"part_selected_{args.selection_index + 1:02d}"
  run_args = argparse.Namespace(
    mat_file=args.mat_file,
    trial_type="fixation",
    channel=0,
    max_trials=None,
    test_fraction=0.25,
    seed=0,
    downsample=2,
    lowpass_list=selected["lowpass_hz"],
    degree_list=selected["degree"],
    n_delays_list=selected["n_delays"],
    delay_list=selected["delay_samples"],
    smooth_window_list=selected["smooth_window_samples"],
    threshold=args.threshold,
    normalize_columns=args.normalize_columns,
    out_csv=args.output_dir / "parts" / f"{stem}.csv",
    equations_out=args.output_dir / "parts" / f"{stem}_equations.txt",
    metadata_out=args.output_dir / "parts" / f"{stem}_metadata.json",
  )
  rows = run_raw_grid(run_args)
  if len(rows) != 1 or rows[0]["fit_status"] != "success":
    raise RuntimeError(f"Selected configuration {args.selection_index} did not fit.")
  print(f"saved selected configuration: {args.selection_index + 1}")


if __name__ == "__main__":
  main()
