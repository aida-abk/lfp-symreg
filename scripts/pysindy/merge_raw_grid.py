from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Project imports
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from models.sindy import maximum_polynomial_terms

DERIVED_FIELDS = ["possible_terms", "term_utilization_percent"]


def configuration_key(row: dict[str, str]) -> tuple[float, int, int, int, int]:
  """Return a sortable identity for one raw-grid configuration."""
  return (
    float(row["lowpass_hz"]),
    int(row["degree"]),
    int(row["n_delays"]),
    int(row["delay_samples"]),
    int(row["smooth_window_samples"]),
  )


def add_term_utilization(row: dict[str, str]) -> None:
  """Add polynomial-library capacity and percent utilization to one row."""
  possible_terms = maximum_polynomial_terms(
    n_states=int(row["n_delays"]),
    degree=int(row["degree"]),
  )
  row["possible_terms"] = str(possible_terms)
  row["term_utilization_percent"] = (
    str(100 * int(row["nonzero_terms"]) / possible_terms)
    if row["fit_status"] == "success"
    else ""
  )


def merge_raw_grid(input_dir: Path, output_csv: Path, expected: int) -> list[dict[str, str]]:
  """Merge Slurm part CSVs and verify unique parameter combinations.

  Args:
    input_dir: Directory containing files named ``part_*.csv``.
    output_csv: Destination merged CSV.
    expected: Required number of unique configurations.

  Returns:
    Sorted merged rows with reassigned global configuration indices.
  """
  paths = sorted(input_dir.glob("part_*.csv"))
  if not paths:
    raise FileNotFoundError(f"No part_*.csv files found in {input_dir}")

  rows = []
  fieldnames = None
  for path in paths:
    with path.open(newline="") as file:
      reader = csv.DictReader(file)
      if fieldnames is None:
        fieldnames = reader.fieldnames
      elif reader.fieldnames != fieldnames:
        raise ValueError(f"CSV header differs in {path}")
      rows.extend(reader)

  identities = [configuration_key(row) for row in rows]
  if len(set(identities)) != len(identities):
    raise ValueError("Raw-grid parts contain duplicate parameter configurations.")
  if len(rows) != expected:
    raise ValueError(f"Expected {expected} configurations, found {len(rows)}.")

  rows.sort(key=configuration_key)
  for index, row in enumerate(rows, start=1):
    row["configuration_index"] = str(index)
    add_term_utilization(row)

  output_fieldnames = [*fieldnames]
  for derived_field in DERIVED_FIELDS:
    if derived_field not in output_fieldnames:
      insert_at = output_fieldnames.index("fit_runtime_s")
      output_fieldnames.insert(insert_at, derived_field)

  output_csv.parent.mkdir(parents=True, exist_ok=True)
  with output_csv.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=output_fieldnames)
    writer.writeheader()
    writer.writerows(rows)
  return rows


def main() -> None:
  """Run the raw-grid merge CLI."""
  parser = argparse.ArgumentParser(description="Merge and validate Oscar raw-grid parts.")
  parser.add_argument(
    "--input-dir",
    type=Path,
    default=Path("outputs/pysindy/raw_grid/parts"),
  )
  parser.add_argument(
    "--output-csv",
    type=Path,
    default=Path("outputs/pysindy/raw_grid/raw_grid_merged.csv"),
  )
  parser.add_argument("--expected", type=int, default=216)
  args = parser.parse_args()

  rows = merge_raw_grid(args.input_dir, args.output_csv, expected=args.expected)
  successful = sum(row["fit_status"] == "success" for row in rows)
  print(f"merged configurations: {len(rows)}")
  print(f"successful fits: {successful}/{len(rows)}")
  print(f"saved: {args.output_csv}")


if __name__ == "__main__":
  main()
