from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

# Project paths
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GRID = ROOT / "outputs" / "pysindy" / "raw_grid" / "raw_grid_merged.csv"
DEFAULT_STATUS = (
  ROOT
  / "outputs"
  / "pysindy"
  / "raw_grid"
  / "simulations"
  / "simulation_status_merged.csv"
)
RMS_FIELD = "successful_simulation_x0_pooled_rms_uv"
COUNT_FIELD = "simulations_used_for_rms"


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
  """Read a CSV while preserving its column order.

  Args:
    path: Input CSV path.

  Returns:
    Column names and string-valued rows.
  """
  with path.open(newline="") as file:
    reader = csv.DictReader(file)
    rows = list(reader)
    fieldnames = list(reader.fieldnames or [])
  if not rows or not fieldnames:
    raise ValueError(f"No rows found in {path}.")
  return fieldnames, rows


def configuration_rms(
  status_rows: list[dict[str, str]],
) -> dict[int, tuple[float, int]]:
  """Pool simulated x0 RMS across completed held-out simulations.

  The per-trial status stores RMS and sample count. Pooling reconstructs
  ``sqrt(sum(x0**2) / total_samples)`` without storing full trajectories.
  Failed or partial simulations are excluded to keep duration coverage
  comparable across equations.

  Args:
    status_rows: Trial-level simulation status rows.

  Returns:
    Mapping from configuration ID to pooled RMS in microvolts and the number
    of completed held-out simulations used.
  """
  required = {
    "configuration_index",
    "simulation_status",
    "simulated_samples",
    "simulated_x0_rms_uv",
  }
  missing = required - set(status_rows[0])
  if missing:
    raise ValueError(
      "Simulation statuses do not contain RMS data. Rerun "
      "visualize_raw_grid_simulations.py, merge the status files, and retry. "
      f"Missing columns: {sorted(missing)}"
    )

  squared_sums: dict[int, float] = defaultdict(float)
  sample_counts: dict[int, int] = defaultdict(int)
  simulation_counts: dict[int, int] = defaultdict(int)
  for row in status_rows:
    if row["simulation_status"] != "success":
      continue
    configuration_id = int(row["configuration_index"])
    samples = int(row["simulated_samples"])
    rms_uv = float(row["simulated_x0_rms_uv"])
    if samples < 1 or not math.isfinite(rms_uv):
      raise ValueError(
        f"Invalid simulation RMS for configuration {configuration_id}."
      )
    squared_sums[configuration_id] += samples * rms_uv**2
    sample_counts[configuration_id] += samples
    simulation_counts[configuration_id] += 1

  return {
    configuration_id: (
      math.sqrt(squared_sum / sample_counts[configuration_id]),
      simulation_counts[configuration_id],
    )
    for configuration_id, squared_sum in squared_sums.items()
  }


def write_enriched_grid(
  output_path: Path,
  fieldnames: list[str],
  grid_rows: list[dict[str, str]],
  rms_by_configuration: dict[int, tuple[float, int]],
) -> None:
  """Write simulation RMS and its contributing simulation count to the grid.

  Args:
    output_path: Destination grid CSV.
    fieldnames: Existing grid column order.
    grid_rows: Existing grid rows.
    rms_by_configuration: Pooled RMS and count by configuration ID.
  """
  output_fields = [
    field for field in fieldnames if field not in {RMS_FIELD, COUNT_FIELD}
  ]
  insert_at = output_fields.index("fit_runtime_s") + 1
  output_fields[insert_at:insert_at] = [RMS_FIELD, COUNT_FIELD]

  for row in grid_rows:
    values = rms_by_configuration.get(int(row["configuration_index"]))
    row[RMS_FIELD] = "" if values is None else str(values[0])
    row[COUNT_FIELD] = "0" if values is None else str(values[1])

  output_path.parent.mkdir(parents=True, exist_ok=True)
  temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
  with temporary_path.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=output_fields)
    writer.writeheader()
    writer.writerows(grid_rows)
  temporary_path.replace(output_path)


def main() -> None:
  """Parse CLI arguments and add completed-simulation x0 RMS to a grid."""
  parser = argparse.ArgumentParser(
    description=(
      "Add pooled x0 RMS from completed held-out simulations to a raw grid."
    )
  )
  parser.add_argument("--grid-csv", type=Path, default=DEFAULT_GRID)
  parser.add_argument("--status-csv", type=Path, default=DEFAULT_STATUS)
  parser.add_argument("--output-csv", type=Path, default=None)
  args = parser.parse_args()

  grid_fields, grid_rows = read_csv(args.grid_csv)
  _, status_rows = read_csv(args.status_csv)
  rms_by_configuration = configuration_rms(status_rows)
  output_path = args.output_csv or args.grid_csv
  write_enriched_grid(
    output_path,
    grid_fields,
    grid_rows,
    rms_by_configuration,
  )
  print(f"configurations with completed simulations: {len(rms_by_configuration)}")
  print(f"saved: {output_path}")


if __name__ == "__main__":
  main()
