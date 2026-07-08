from __future__ import annotations

import argparse
import csv
import sys

csv.field_size_limit(sys.maxsize)
from pathlib import Path

csv.field_size_limit(sys.maxsize)

# Project imports
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from models.sindy import maximum_fourier_terms, maximum_polynomial_terms

DERIVED_FIELDS = ["possible_terms", "term_utilization_percent"]


def configuration_key(
    row: dict[str, str], degree_field: str = "degree"
) -> tuple[float, int, int, int, int]:
  """Return a sortable identity for one raw-grid configuration."""
  return (
    float(row["lowpass_hz"]),
    int(row[degree_field]),
    int(row["n_delays"]),
    int(row["delay_samples"]),
    int(row["smooth_window_samples"]),
  )


def add_term_utilization(row: dict[str, str], degree_field: str = "degree") -> None:
  """Add library capacity and percent utilization to one row."""
  if degree_field == "n_frequencies":
    possible_terms = maximum_fourier_terms(
      n_states=int(row["n_delays"]),
      n_frequencies=int(row["n_frequencies"]),
    )
  else:
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


def merge_raw_grid(
    input_dir: Path,
    output_csv: Path,
    expected: int,
    degree_field: str = "degree",
) -> list[dict[str, str]]:
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
  fieldnames = []
  for path in paths:
    with path.open(newline="") as file:
      reader = csv.DictReader(file)
      if reader.fieldnames is None:
        raise ValueError(f"CSV file has no header: {path}")
      for fieldname in reader.fieldnames:
        if fieldname not in fieldnames:
          fieldnames.append(fieldname)
      rows.extend(reader)

  identities = [configuration_key(row, degree_field) for row in rows]
  if len(set(identities)) != len(identities):
    raise ValueError("Raw-grid parts contain duplicate parameter configurations.")
  if len(rows) != expected:
    raise ValueError(f"Expected {expected} configurations, found {len(rows)}.")

  rows.sort(key=lambda r: configuration_key(r, degree_field))
  for index, row in enumerate(rows, start=1):
    row["configuration_index"] = str(index)
    add_term_utilization(row, degree_field)

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
  parser.add_argument(
    "--degree-field",
    default="degree",
    choices=("degree", "n_frequencies"),
    help=(
      "Column name representing model order for deduplication and sorting. "
      "Use 'n_frequencies' when merging Fourier-library parts."
    ),
  )
  args = parser.parse_args()

  rows = merge_raw_grid(
      args.input_dir, args.output_csv,
      expected=args.expected, degree_field=args.degree_field,
  )
  successful = sum(row["fit_status"] == "success" for row in rows)
  print(f"merged configurations: {len(rows)}")
  print(f"successful fits: {successful}/{len(rows)}")
  print(f"saved: {args.output_csv}")


if __name__ == "__main__":
  main()
