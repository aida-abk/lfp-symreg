from __future__ import annotations

import argparse
import csv
import html
import math
from collections import defaultdict
from pathlib import Path

# Default simulation outputs
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "pysindy" / "raw_grid" / "simulations"
DEFAULT_GRID = ROOT / "outputs" / "pysindy" / "raw_grid" / "raw_grid_merged.csv"

CONFIGURATION_METRIC_FIELDS = [
  "configuration_index",
  "stlsq_threshold",
  "lowpass_hz",
  "degree",
  "n_delays",
  "delay_samples",
  "smooth_window_samples",
  "nonzero_terms",
  "term_utilization_percent",
  "simulation_attempts",
  "successful_simulations",
  "failed_simulations",
  "simulations_used_for_rmse",
  "pooled_x0_test_rmse_uv",
  "pooled_trajectory_test_rmse_uv",
]


def merge_status_files(
  status_dir: Path,
  output_csv: Path,
  expected_configurations: int,
) -> list[dict[str, str]]:
  """Merge per-configuration simulation status files.

  Args:
    status_dir: Directory containing one ``config_*.csv`` per equation.
    output_csv: Destination for all trial-level simulation outcomes.
    expected_configurations: Required number of configuration files.

  Returns:
    Merged trial-level status rows sorted by configuration and trial order.
  """
  paths = sorted(status_dir.glob("config_*.csv"))
  if len(paths) != expected_configurations:
    raise ValueError(
      f"Found {len(paths)} status files; expected {expected_configurations}."
    )

  rows = []
  fieldnames = None
  for path in paths:
    with path.open(newline="") as file:
      reader = csv.DictReader(file)
      if fieldnames is None:
        fieldnames = reader.fieldnames
      elif reader.fieldnames != fieldnames:
        raise ValueError(f"CSV header mismatch in {path}.")
      rows.extend(reader)
  if not fieldnames:
    raise ValueError("Simulation status files have no CSV header.")

  rows.sort(key=lambda row: (int(row["configuration_index"]), int(row["test_trial_id"])))
  output_csv.parent.mkdir(parents=True, exist_ok=True)
  with output_csv.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
  return rows


def write_html_index(
  figures_dir: Path,
  output_html: Path,
  expected_configurations: int,
) -> None:
  """Write a clickable thumbnail index for all configuration figures."""
  paths = sorted(figures_dir.glob("config_*.png"))
  if len(paths) != expected_configurations:
    raise ValueError(
      f"Found {len(paths)} figures; expected {expected_configurations}."
    )

  cards = []
  for path in paths:
    relative = path.relative_to(output_html.parent)
    label = path.stem.replace("_", " ").title()
    source = html.escape(relative.as_posix())
    cards.append(
      f'<a class="card" href="{source}">'
      f'<img src="{source}" loading="lazy" alt="{html.escape(label)}">'
      f'<span>{html.escape(label)}</span></a>'
    )
  document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Raw-grid simulation figures</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; color: #202124; }}
    h1 {{ font-size: 24px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; }}
    .card {{ color: inherit; text-decoration: none; border: 1px solid #dadce0; padding: 8px; }}
    .card img {{ display: block; width: 100%; height: 180px; object-fit: contain; background: #fff; }}
    .card span {{ display: block; margin-top: 8px; font-size: 14px; }}
  </style>
</head>
<body>
  <h1>Measured vs simulated x0: all configurations</h1>
  <div class="grid">{''.join(cards)}</div>
</body>
</html>
"""
  output_html.parent.mkdir(parents=True, exist_ok=True)
  output_html.write_text(document)


def write_configuration_metrics(
  status_rows: list[dict[str, str]],
  output_csv: Path,
  grid_by_configuration: dict[int, dict[str, str]],
) -> list[dict[str, object]]:
  """Aggregate completed held-out simulation RMSE by configuration.

  Failed and partial trajectories remain in the trial-level status table but
  are excluded from pooled RMSE so every included trial covers its full
  requested duration.

  Args:
    status_rows: Trial-level simulation status rows with RMSE in microvolts.
    output_csv: Destination configuration-level CSV.
    grid_by_configuration: Fitted parameter-grid row by configuration ID.

  Returns:
    One summary row per configuration.
  """
  required = {
    "configuration_index",
    "simulation_status",
    "compared_samples",
    "x0_rmse_uv",
    "trajectory_rmse_uv",
  }
  missing = required - set(status_rows[0])
  if missing:
    raise ValueError(f"Simulation status is missing RMSE columns: {sorted(missing)}")

  grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
  for row in status_rows:
    grouped[int(row["configuration_index"])].append(row)

  summaries = []
  for configuration_id, trials in sorted(grouped.items()):
    if configuration_id not in grid_by_configuration:
      raise ValueError(f"Configuration {configuration_id} is absent from the grid.")
    configuration = grid_by_configuration[configuration_id]
    successful = [
      row
      for row in trials
      if row["simulation_status"] == "success"
      and row["x0_rmse_uv"]
      and row["trajectory_rmse_uv"]
    ]
    total_samples = sum(int(row["compared_samples"]) for row in successful)
    if total_samples:
      pooled_x0_rmse = math.sqrt(
        sum(
          int(row["compared_samples"]) * float(row["x0_rmse_uv"]) ** 2
          for row in successful
        )
        / total_samples
      )
      pooled_trajectory_rmse = math.sqrt(
        sum(
          int(row["compared_samples"])
          * float(row["trajectory_rmse_uv"]) ** 2
          for row in successful
        )
        / total_samples
      )
    else:
      pooled_x0_rmse = ""
      pooled_trajectory_rmse = ""
    summaries.append(
      {
        "configuration_index": configuration_id,
        "stlsq_threshold": configuration.get("stlsq_threshold", ""),
        "lowpass_hz": configuration["lowpass_hz"],
        "degree": configuration["degree"],
        "n_delays": configuration["n_delays"],
        "delay_samples": configuration["delay_samples"],
        "smooth_window_samples": configuration["smooth_window_samples"],
        "nonzero_terms": configuration["nonzero_terms"],
        "term_utilization_percent": configuration["term_utilization_percent"],
        "simulation_attempts": len(trials),
        "successful_simulations": sum(
          row["simulation_status"] == "success" for row in trials
        ),
        "failed_simulations": sum(
          row["simulation_status"] != "success" for row in trials
        ),
        "simulations_used_for_rmse": len(successful),
        "pooled_x0_test_rmse_uv": pooled_x0_rmse,
        "pooled_trajectory_test_rmse_uv": pooled_trajectory_rmse,
      }
    )

  output_csv.parent.mkdir(parents=True, exist_ok=True)
  with output_csv.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=CONFIGURATION_METRIC_FIELDS)
    writer.writeheader()
    writer.writerows(summaries)
  return summaries


def main() -> None:
  """Merge Oscar outputs and build the visual index."""
  parser = argparse.ArgumentParser(
    description="Merge raw-grid simulation statuses and index all figures."
  )
  parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
  parser.add_argument("--grid-csv", type=Path, default=DEFAULT_GRID)
  parser.add_argument("--expected", type=int, default=216)
  args = parser.parse_args()

  status_output = args.output_dir / "simulation_status_merged.csv"
  rows = merge_status_files(
    args.output_dir / "status",
    status_output,
    expected_configurations=args.expected,
  )
  with args.grid_csv.open(newline="") as file:
    grid_rows = list(csv.DictReader(file))
  grid_by_configuration = {
    int(row["configuration_index"]): row for row in grid_rows
  }
  rmse_fields = {"compared_samples", "x0_rmse_uv", "trajectory_rmse_uv"}
  metrics_output = args.output_dir / "configuration_test_metrics.csv"
  if rmse_fields.issubset(rows[0]):
    metrics = write_configuration_metrics(
      rows,
      metrics_output,
      grid_by_configuration,
    )
  else:
    metrics = []
  index_output = args.output_dir / "index.html"
  write_html_index(
    args.output_dir / "figures",
    index_output,
    expected_configurations=args.expected,
  )
  successful = sum(row["simulation_status"] == "success" for row in rows)
  print(f"merged trial simulations: {len(rows)}")
  print(f"successful trial simulations: {successful}/{len(rows)}")
  if metrics:
    print(f"configuration metric rows: {len(metrics)}")
  else:
    print("configuration RMSE skipped: legacy statuses contain no RMSE columns")
  print(f"saved: {status_output}")
  if metrics:
    print(f"saved: {metrics_output}")
  print(f"saved: {index_output}")


if __name__ == "__main__":
  main()
