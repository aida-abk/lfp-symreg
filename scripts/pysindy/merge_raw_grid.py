from __future__ import annotations

import argparse
import csv
from pathlib import Path


def configuration_key(row: dict[str, str]) -> tuple[float, int, int, int, int]:
  """Return a sortable identity for one raw-grid configuration."""
  return (
    float(row["lowpass_hz"]),
    int(row["degree"]),
    int(row["n_delays"]),
    int(row["delay_samples"]),
    int(row["smooth_window_samples"]),
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

  output_csv.parent.mkdir(parents=True, exist_ok=True)
  with output_csv.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
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
