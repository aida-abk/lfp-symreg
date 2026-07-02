from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from scipy.stats import spearmanr

# Project paths
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GRID = ROOT / "outputs" / "pysindy" / "raw_grid" / "raw_grid_merged.csv"
DEFAULT_SIMULATION_DIR = (
  ROOT / "outputs" / "pysindy" / "raw_grid" / "simulations"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "pysindy" / "raw_grid" / "analysis"

CONFIGURATION_FIELDS = [
  "configuration_index",
  "lowpass_hz",
  "degree",
  "n_delays",
  "delay_samples",
  "delay_ms",
  "embedding_span_ms",
  "smooth_window_samples",
  "smooth_window_ms",
  "derivative_method",
  "nonzero_terms",
  "possible_terms",
  "term_utilization_percent",
  "fit_runtime_s",
  "simulation_attempts",
  "completed_simulations",
  "failed_simulations",
  "completion_fraction",
  "all_held_out_completed",
  "failure_reasons",
  "figure_exists",
]

PARAMETER_FIELDS = [
  "parameter",
  "value",
  "configurations",
  "all_held_out_completed_count",
  "all_held_out_completed_fraction",
  "simulation_attempts",
  "completed_simulations",
  "simulation_completion_fraction",
]

INTERACTION_FIELDS = [
  "degree",
  *PARAMETER_FIELDS,
]

GROUP_PARAMETERS = [
  "degree",
  "lowpass_hz",
  "n_delays",
  "delay_samples",
  "smooth_window_samples",
  "nonzero_terms",
]

DEGREE_INTERACTION_PARAMETERS = [
  "lowpass_hz",
  "n_delays",
  "delay_samples",
  "smooth_window_samples",
]


def read_csv(path: Path) -> list[dict[str, str]]:
  """Read a CSV file into string-valued dictionaries."""
  with path.open(newline="") as file:
    rows = list(csv.DictReader(file))
  if not rows:
    raise ValueError(f"No rows found in {path}.")
  return rows


def write_csv(
  path: Path,
  rows: list[dict[str, object]],
  fieldnames: list[str],
) -> None:
  """Write analysis rows with an explicit stable column order."""
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def validate_inputs(
  grid_rows: list[dict[str, str]],
  status_rows: list[dict[str, str]],
  expected_configurations: int,
  expected_trials: int,
) -> None:
  """Validate complete, unique configuration and trial-level records."""
  grid_ids = [int(row["configuration_index"]) for row in grid_rows]
  if len(grid_ids) != expected_configurations:
    raise ValueError(
      f"Expected {expected_configurations} grid rows, found {len(grid_ids)}."
    )
  if len(set(grid_ids)) != len(grid_ids):
    raise ValueError("The parameter grid contains duplicate configuration IDs.")

  status_by_configuration: dict[int, list[dict[str, str]]] = defaultdict(list)
  for row in status_rows:
    status_by_configuration[int(row["configuration_index"])].append(row)
  extra = sorted(set(status_by_configuration) - set(grid_ids))
  if extra:
    raise ValueError(f"Status rows contain unknown configurations: {extra}.")

  for configuration_id in grid_ids:
    rows = status_by_configuration[configuration_id]
    if len(rows) != expected_trials:
      raise ValueError(
        f"Configuration {configuration_id} has {len(rows)} trial rows; "
        f"expected {expected_trials}."
      )
    trial_ids = [int(row["test_trial_id"]) for row in rows]
    if len(set(trial_ids)) != len(trial_ids):
      raise ValueError(
        f"Configuration {configuration_id} has duplicate test trial IDs."
      )


def configuration_summary(
  grid_rows: list[dict[str, str]],
  status_rows: list[dict[str, str]],
  figures_dir: Path,
) -> list[dict[str, object]]:
  """Create one numerical-completion summary row per fitted equation."""
  status_by_configuration: dict[int, list[dict[str, str]]] = defaultdict(list)
  for row in status_rows:
    status_by_configuration[int(row["configuration_index"])].append(row)

  summaries = []
  for grid_row in sorted(grid_rows, key=lambda row: int(row["configuration_index"])):
    configuration_id = int(grid_row["configuration_index"])
    trials = status_by_configuration[configuration_id]
    completed = sum(row["simulation_status"] == "success" for row in trials)
    failures = sorted(
      {
        row["failure_reason"]
        for row in trials
        if row["simulation_status"] != "success" and row["failure_reason"]
      }
    )
    attempts = len(trials)
    summaries.append(
      {
        "configuration_index": configuration_id,
        "lowpass_hz": float(grid_row["lowpass_hz"]),
        "degree": int(grid_row["degree"]),
        "n_delays": int(grid_row["n_delays"]),
        "delay_samples": int(grid_row["delay_samples"]),
        "delay_ms": float(grid_row["delay_ms"]),
        "embedding_span_ms": float(grid_row["embedding_span_ms"]),
        "smooth_window_samples": int(grid_row["smooth_window_samples"]),
        "smooth_window_ms": float(grid_row["smooth_window_ms"]),
        "derivative_method": grid_row["derivative_method"],
        "nonzero_terms": int(grid_row["nonzero_terms"]),
        "possible_terms": int(grid_row["possible_terms"]),
        "term_utilization_percent": float(grid_row["term_utilization_percent"]),
        "fit_runtime_s": float(grid_row["fit_runtime_s"]),
        "simulation_attempts": attempts,
        "completed_simulations": completed,
        "failed_simulations": attempts - completed,
        "completion_fraction": completed / attempts,
        "all_held_out_completed": completed == attempts,
        "failure_reasons": "; ".join(failures),
        "figure_exists": (
          figures_dir / f"config_{configuration_id:04d}.png"
        ).exists(),
      }
    )
  return summaries


def grouped_summary(
  rows: list[dict[str, object]],
  parameter: str,
) -> list[dict[str, object]]:
  """Summarize equation- and trial-level completion by one parameter."""
  groups: dict[object, list[dict[str, object]]] = defaultdict(list)
  for row in rows:
    groups[row[parameter]].append(row)

  summaries = []
  for value, group in sorted(groups.items(), key=lambda item: item[0]):
    configurations = len(group)
    all_completed = sum(bool(row["all_held_out_completed"]) for row in group)
    attempts = sum(int(row["simulation_attempts"]) for row in group)
    completed = sum(int(row["completed_simulations"]) for row in group)
    summaries.append(
      {
        "parameter": parameter,
        "value": value,
        "configurations": configurations,
        "all_held_out_completed_count": all_completed,
        "all_held_out_completed_fraction": all_completed / configurations,
        "simulation_attempts": attempts,
        "completed_simulations": completed,
        "simulation_completion_fraction": completed / attempts,
      }
    )
  return summaries


def degree_interaction_summary(
  rows: list[dict[str, object]],
) -> list[dict[str, object]]:
  """Summarize parameter effects separately within each polynomial degree."""
  output = []
  degrees = sorted({int(row["degree"]) for row in rows})
  for degree in degrees:
    degree_rows = [row for row in rows if int(row["degree"]) == degree]
    for parameter in DEGREE_INTERACTION_PARAMETERS:
      for summary in grouped_summary(degree_rows, parameter):
        output.append({"degree": degree, **summary})
  return output


def overall_summary(
  configuration_rows: list[dict[str, object]],
  status_rows: list[dict[str, str]],
) -> dict[str, object]:
  """Calculate overall numerical-completion and artifact-audit statistics."""
  all_completed = sum(
    bool(row["all_held_out_completed"]) for row in configuration_rows
  )
  completed_simulations = sum(
    int(row["completed_simulations"]) for row in configuration_rows
  )
  attempts = sum(int(row["simulation_attempts"]) for row in configuration_rows)
  correlation, p_value = spearmanr(
    [int(row["nonzero_terms"]) for row in configuration_rows],
    [int(row["completed_simulations"]) for row in configuration_rows],
  )
  failure_reasons = Counter(
    row["failure_reason"]
    for row in status_rows
    if row["simulation_status"] != "success"
  )
  missing_figures = [
    int(row["configuration_index"])
    for row in configuration_rows
    if not bool(row["figure_exists"])
  ]
  return {
    "configurations": len(configuration_rows),
    "all_held_out_completed_count": all_completed,
    "all_held_out_completed_fraction": all_completed / len(configuration_rows),
    "simulation_attempts": attempts,
    "completed_simulations": completed_simulations,
    "failed_simulations": attempts - completed_simulations,
    "simulation_completion_fraction": completed_simulations / attempts,
    "completed_trials_per_configuration_distribution": dict(
      sorted(
        Counter(
          int(row["completed_simulations"]) for row in configuration_rows
        ).items()
      )
    ),
    "nonzero_terms_vs_completed_trials_spearman_rho": float(correlation),
    "nonzero_terms_vs_completed_trials_spearman_p": float(p_value),
    "failure_reason_counts": dict(sorted(failure_reasons.items())),
    "missing_figure_configuration_ids": missing_figures,
  }


def run(args: argparse.Namespace) -> dict[str, object]:
  """Validate, analyze, and save the simulation-grid summaries."""
  grid_rows = read_csv(args.grid_csv)
  status_rows = read_csv(args.status_csv)
  validate_inputs(
    grid_rows,
    status_rows,
    expected_configurations=args.expected_configurations,
    expected_trials=args.expected_trials,
  )
  configurations = configuration_summary(
    grid_rows,
    status_rows,
    figures_dir=args.figures_dir,
  )
  parameters = [
    summary
    for parameter in GROUP_PARAMETERS
    for summary in grouped_summary(configurations, parameter)
  ]
  interactions = degree_interaction_summary(configurations)
  overall = overall_summary(configurations, status_rows)

  write_csv(
    args.output_dir / "configuration_simulation_summary.csv",
    configurations,
    CONFIGURATION_FIELDS,
  )
  write_csv(
    args.output_dir / "parameter_simulation_summary.csv",
    parameters,
    PARAMETER_FIELDS,
  )
  write_csv(
    args.output_dir / "degree_parameter_interactions.csv",
    interactions,
    INTERACTION_FIELDS,
  )
  args.output_dir.mkdir(parents=True, exist_ok=True)
  (args.output_dir / "overall_summary.json").write_text(
    json.dumps(overall, indent=2) + "\n"
  )
  return overall


def main() -> None:
  """Parse CLI arguments and analyze the completed simulation grid."""
  parser = argparse.ArgumentParser(
    description="Analyze numerical completion across a PySINDy simulation grid."
  )
  parser.add_argument("--grid-csv", type=Path, default=DEFAULT_GRID)
  parser.add_argument(
    "--status-csv",
    type=Path,
    default=DEFAULT_SIMULATION_DIR / "simulation_status_merged.csv",
  )
  parser.add_argument(
    "--figures-dir",
    type=Path,
    default=DEFAULT_SIMULATION_DIR / "figures",
  )
  parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
  parser.add_argument("--expected-configurations", type=int, default=216)
  parser.add_argument("--expected-trials", type=int, default=9)
  args = parser.parse_args()

  summary = run(args)
  print(
    "all held-out trials completed: "
    f"{summary['all_held_out_completed_count']}/"
    f"{summary['configurations']} "
    f"({100 * summary['all_held_out_completed_fraction']:.1f}%)"
  )
  print(
    "individual simulations completed: "
    f"{summary['completed_simulations']}/{summary['simulation_attempts']} "
    f"({100 * summary['simulation_completion_fraction']:.1f}%)"
  )
  print(
    "missing figures: "
    f"{summary['missing_figure_configuration_ids']}"
  )
  print(f"saved analyses: {args.output_dir}")


if __name__ == "__main__":
  main()
