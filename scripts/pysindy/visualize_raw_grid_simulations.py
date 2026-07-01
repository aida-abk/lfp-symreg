from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

# Project imports
ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
PYSINDY_SCRIPTS = SCRIPTS / "pysindy"
for path in (ROOT, SCRIPTS, PYSINDY_SCRIPTS):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

from load_data.convert import LFP_AMPLITUDE_UNIT, MAT_FILE, TrialData
from load_data.preprocessing import channel_traces
from models.sindy import StoredPolynomialModel, delay_embed_trajectories
from models.validation import SimulationResult, simulate_model_detailed

# Default raw-grid artifacts
DEFAULT_GRID = ROOT / "outputs" / "pysindy" / "raw_grid" / "raw_grid_merged.csv"
DEFAULT_METADATA = (
  ROOT
  / "outputs"
  / "pysindy"
  / "raw_grid"
  / "parts"
  / "part_lp35_degree1_metadata.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "pysindy" / "raw_grid" / "simulations"

STATUS_FIELDS = [
  "configuration_index",
  "test_trial_id",
  "simulation_status",
  "failure_reason",
  "requested_duration_s",
  "reached_duration_s",
  "simulation_runtime_s",
  "rhs_evaluations",
]


def load_grid(path: Path) -> list[dict[str, str]]:
  """Load successful stored-equation rows from the merged raw-grid CSV."""
  with path.open(newline="") as file:
    rows = list(csv.DictReader(file))
  if not rows:
    raise ValueError(f"No configurations found in {path}.")
  failed = [row for row in rows if row["fit_status"] != "success"]
  if failed:
    raise ValueError(f"The grid contains {len(failed)} unsuccessful fits.")
  return rows


def select_benchmark_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
  """Select one fixed middle-sized configuration per degree/filter pair.

  The benchmark fixes four delay coordinates, two-sample spacing, and a
  five-sample smoothing window. This selection estimates runtime only and does
  not rank the scientific quality of those settings.
  """
  selected = []
  pairs = sorted({(float(row["lowpass_hz"]), int(row["degree"])) for row in rows})
  for lowpass_hz, degree in pairs:
    matches = [
      row
      for row in rows
      if float(row["lowpass_hz"]) == lowpass_hz
      and int(row["degree"]) == degree
      and int(row["n_delays"]) == 4
      and int(row["delay_samples"]) == 2
      and int(row["smooth_window_samples"]) == 5
    ]
    if len(matches) != 1:
      raise ValueError(
        "Expected one benchmark row for "
        f"lowpass={lowpass_hz}, degree={degree}; found {len(matches)}."
      )
    selected.append(matches[0])
  return selected


def select_rows(
  rows: list[dict[str, str]],
  configuration_index: int | None,
  benchmark: bool,
) -> list[dict[str, str]]:
  """Select all rows, one global configuration, or six benchmark rows."""
  if configuration_index is not None and benchmark:
    raise ValueError("Use either --configuration-index or --benchmark, not both.")
  if benchmark:
    return select_benchmark_rows(rows)
  if configuration_index is None:
    return rows
  matches = [
    row for row in rows if int(row["configuration_index"]) == configuration_index
  ]
  if len(matches) != 1:
    raise ValueError(
      f"Expected one row for configuration {configuration_index}; found {len(matches)}."
    )
  return matches


def reconstruct_model(row: dict[str, str]) -> StoredPolynomialModel:
  """Reconstruct one fitted polynomial ODE without fitting it again."""
  return StoredPolynomialModel(
    degree=int(row["degree"]),
    coefficients=np.asarray(json.loads(row["coefficients_json"]), dtype=float),
    feature_names=list(json.loads(row["feature_names_json"])),
  )


def plot_configuration(
  path: Path,
  row: dict[str, str],
  trial_ids: list[int],
  measured_trials: list[np.ndarray],
  results: list[SimulationResult],
  dt: float,
) -> None:
  """Plot measured and simulated x0 for every held-out trial.

  Args:
    path: Destination PNG path.
    row: Raw-grid configuration row.
    trial_ids: Original zero-based trial identifiers.
    measured_trials: Embedded measured trajectories in microvolts.
    results: Numerical simulation results corresponding to the trials.
    dt: Processed sample interval in seconds.
  """
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  columns = 3
  n_rows = math.ceil(len(trial_ids) / columns)
  figure, axes = plt.subplots(
    n_rows,
    columns,
    figsize=(5 * columns, 2.7 * n_rows),
    sharex=False,
    sharey=True,
    squeeze=False,
  )
  for axis, trial_id, measured, result in zip(
    axes.ravel(), trial_ids, measured_trials, results
  ):
    measured_time = np.arange(measured.shape[0]) * dt
    axis.plot(measured_time, measured[:, 0], label="Measured", linewidth=0.9)
    if result.trajectory is not None and result.trajectory.size:
      axis.plot(
        result.time,
        result.trajectory[:, 0],
        label="Simulated",
        linewidth=0.9,
      )
    status = "complete" if result.completed else "failed"
    axis.set_title(
      f"Trial {trial_id}: {status}, reached {result.reached_horizon_s:.2f} s",
      fontsize=9,
    )
    axis.set_xlabel("Time from embedded initial state (s)")
    axis.set_ylabel(f"x0 ({LFP_AMPLITUDE_UNIT})")

  for axis in axes.ravel()[len(trial_ids):]:
    axis.set_visible(False)
  axes.ravel()[0].legend(loc="upper right")
  figure.suptitle(
    f"Configuration {row['configuration_index']}: "
    f"LP={row['lowpass_hz']} Hz, degree={row['degree']}, "
    f"delays={row['n_delays']}, spacing={row['delay_samples']} samples, "
    f"smoothing={row['smooth_window_samples']} samples"
  )
  figure.tight_layout(rect=(0, 0, 1, 0.97))
  path.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(path, dpi=160)
  plt.close(figure)


def simulate_configuration(
  row: dict[str, str],
  test_raw: list[np.ndarray],
  test_trial_ids: list[int],
  dt: float,
  output_dir: Path,
) -> dict[str, object]:
  """Simulate one stored equation from every held-out initial condition."""
  configuration_index = int(row["configuration_index"])
  measured_trials = delay_embed_trajectories(
    test_raw,
    n_delays=int(row["n_delays"]),
    delay=int(row["delay_samples"]),
  )
  model = reconstruct_model(row)
  results = []
  status_rows = []
  configuration_started = time.perf_counter()

  for trial_id, measured in zip(test_trial_ids, measured_trials):
    requested_duration_s = (measured.shape[0] - 1) * dt
    started = time.perf_counter()
    result = simulate_model_detailed(
      model,
      initial_state=measured[0],
      dt=dt,
      horizon_s=requested_duration_s,
    )
    runtime_s = time.perf_counter() - started
    results.append(result)
    status_rows.append(
      {
        "configuration_index": configuration_index,
        "test_trial_id": trial_id,
        "simulation_status": "success" if result.completed else "failed",
        "failure_reason": result.failure_reason,
        "requested_duration_s": requested_duration_s,
        "reached_duration_s": result.reached_horizon_s,
        "simulation_runtime_s": runtime_s,
        "rhs_evaluations": result.rhs_evaluations,
      }
    )
    print(
      f"config={configuration_index} trial={trial_id} "
      f"status={status_rows[-1]['simulation_status']} "
      f"reached={result.reached_horizon_s:.2f}/{requested_duration_s:.2f}s "
      f"runtime={runtime_s:.1f}s",
      flush=True,
    )

  stem = f"config_{configuration_index:04d}"
  status_path = output_dir / "status" / f"{stem}.csv"
  status_path.parent.mkdir(parents=True, exist_ok=True)
  with status_path.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=STATUS_FIELDS)
    writer.writeheader()
    writer.writerows(status_rows)

  figure_path = output_dir / "figures" / f"{stem}.png"
  plot_configuration(
    figure_path,
    row=row,
    trial_ids=test_trial_ids,
    measured_trials=measured_trials,
    results=results,
    dt=dt,
  )
  return {
    "configuration_index": configuration_index,
    "lowpass_hz": float(row["lowpass_hz"]),
    "degree": int(row["degree"]),
    "n_delays": int(row["n_delays"]),
    "delay_samples": int(row["delay_samples"]),
    "smooth_window_samples": int(row["smooth_window_samples"]),
    "test_trials": len(test_trial_ids),
    "successful_simulations": sum(result.completed for result in results),
    "configuration_runtime_s": time.perf_counter() - configuration_started,
    "figure": str(figure_path),
    "status_csv": str(status_path),
  }


def run(args: argparse.Namespace) -> list[dict[str, object]]:
  """Run the selected stored-equation simulations and visualizations."""
  rows = select_rows(
    load_grid(args.grid_csv),
    configuration_index=args.configuration_index,
    benchmark=args.benchmark,
  )
  metadata = json.loads(args.metadata_json.read_text())
  if metadata["preprocessing"]["normalization"] != "none":
    raise ValueError("This visualizer requires the raw grid's normalization='none'.")

  data = TrialData.load(args.mat_file)
  if float(metadata["raw_sampling_hz"]) != float(data.fs):
    raise ValueError("Local data sampling frequency does not match the sweep metadata.")
  test_trial_ids = [int(value) for value in metadata["split"]["test_trial_ids"]]
  if args.max_test_trials is not None:
    test_trial_ids = test_trial_ids[: args.max_test_trials]
  downsample = int(metadata["downsample_factor"])
  channel = int(metadata["channel"])
  dt = downsample / data.fs

  # Cache measured traces because configurations reuse the same two cutoffs.
  traces_by_lowpass: dict[float, list[np.ndarray]] = {}
  summaries = []
  for index, row in enumerate(rows, start=1):
    lowpass_hz = float(row["lowpass_hz"])
    if lowpass_hz not in traces_by_lowpass:
      traces_by_lowpass[lowpass_hz] = channel_traces(
        data,
        channel=channel,
        trials=test_trial_ids,
        downsample=downsample,
        lowpass_hz=lowpass_hz,
        normalize="none",
      )
    print(
      f"[{index}/{len(rows)}] configuration={row['configuration_index']} "
      f"degree={row['degree']} lowpass={row['lowpass_hz']}",
      flush=True,
    )
    summaries.append(
      simulate_configuration(
        row,
        test_raw=traces_by_lowpass[lowpass_hz],
        test_trial_ids=test_trial_ids,
        dt=dt,
        output_dir=args.output_dir,
      )
    )

  summary_name = (
    "benchmark_summary.json"
    if args.benchmark
    else f"config_{args.configuration_index:04d}_summary.json"
    if args.configuration_index is not None
    else "all_configurations_summary.json"
  )
  summary_path = args.output_dir / summary_name
  summary_path.parent.mkdir(parents=True, exist_ok=True)
  summary_path.write_text(json.dumps(summaries, indent=2) + "\n")
  print(f"saved: {summary_path}")
  return summaries


def main() -> None:
  """Parse CLI arguments and visualize raw-grid simulations."""
  parser = argparse.ArgumentParser(
    description=(
      "Reconstruct stored raw-grid equations and compare full held-out "
      "simulations with measured x0 in microvolts."
    )
  )
  parser.add_argument("--mat-file", type=Path, default=MAT_FILE)
  parser.add_argument("--grid-csv", type=Path, default=DEFAULT_GRID)
  parser.add_argument("--metadata-json", type=Path, default=DEFAULT_METADATA)
  parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
  parser.add_argument("--configuration-index", type=int, default=None)
  parser.add_argument("--benchmark", action="store_true")
  parser.add_argument(
    "--max-test-trials",
    type=int,
    default=None,
    help="Optional computational smoke-test limit; default uses every held-out trial.",
  )
  args = parser.parse_args()
  if args.max_test_trials is not None and args.max_test_trials < 1:
    parser.error("--max-test-trials must be at least 1.")
  run(args)


if __name__ == "__main__":
  main()
