from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

# Project imports
ROOT = Path(__file__).resolve().parents[3]
PYSINDY_SCRIPTS = ROOT / "scripts" / "pysindy"
for path in (ROOT, PYSINDY_SCRIPTS):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

from load_data.convert import MAT_FILE, TrialData
from load_data.preprocessing import channel_traces
from models.sindy import StoredPolynomialModel, delay_embed_trajectories
from models.validation import SimulationResult, simulate_model_detailed
from raw_grid_io import write_csv_checkpoint
from plot_threshold_path import (
  DEFAULT_GRID,
  DEFAULT_REFIT_CSVS,
  collect_threshold_models,
)
from visualize_raw_grid_simulations import plot_configuration

# Saved-run defaults
DEFAULT_METADATA = (
  ROOT
  / "outputs"
  / "pysindy"
  / "raw_grid"
  / "parts"
  / "part_lp35_degree1_metadata.json"
)
DEFAULT_OUTPUT_DIR = (
  ROOT
  / "outputs"
  / "pysindy"
  / "raw_grid"
  / "analysis"
  / "configuration_0216_threshold_simulations"
)

STATUS_FIELDS = [
  "configuration_index",
  "stlsq_threshold",
  "test_trial_id",
  "simulation_status",
  "failure_reason",
  "requested_duration_s",
  "reached_duration_s",
  "simulation_runtime_s",
  "rhs_evaluations",
]


def threshold_label(threshold: float) -> str:
  """Return a filesystem-safe threshold label."""
  return f"{threshold:g}".replace(".", "p").replace("-", "m")


def select_threshold_model(
  models: list[dict[str, object]],
  threshold: float,
) -> dict[str, object]:
  """Select exactly one stored equation at the requested threshold."""
  matches = [
    model
    for model in models
    if abs(float(model["threshold"]) - threshold) < 1e-12
  ]
  if len(matches) != 1:
    available = [float(model["threshold"]) for model in models]
    raise ValueError(
      f"Expected one model at threshold {threshold}; found {len(matches)}. "
      f"Available thresholds: {available}."
    )
  return matches[0]


def simulate_trials(
  model: StoredPolynomialModel,
  measured_trials: list,
  trial_ids: list[int],
  configuration_index: int,
  threshold: float,
  dt: float,
  status_path: Path,
  trial_timeout_s: float | None,
) -> tuple[list[SimulationResult], list[dict[str, object]]]:
  """Simulate held-out trials with immediate status checkpointing.

  Args:
    model: Stored polynomial ODE model.
    measured_trials: Embedded held-out trajectories in microvolts.
    trial_ids: Original zero-based trial identifiers.
    configuration_index: Raw-grid configuration identifier.
    threshold: STLSQ coefficient threshold used for the model.
    dt: Processed sample interval in seconds.
    status_path: Trial-level CSV checkpoint path.
    trial_timeout_s: Optional operational wall-time limit per simulation in
      seconds. This is not a scientific rejection threshold.

  Returns:
    Simulation results and trial-level status rows.
  """
  results = []
  rows = []
  for trial_id, measured in zip(trial_ids, measured_trials):
    requested_duration_s = (measured.shape[0] - 1) * dt
    started = time.perf_counter()
    result = simulate_model_detailed(
      model,
      initial_state=measured[0],
      dt=dt,
      horizon_s=requested_duration_s,
      wall_timeout_s=trial_timeout_s,
    )
    runtime_s = time.perf_counter() - started
    results.append(result)
    rows.append(
      {
        "configuration_index": configuration_index,
        "stlsq_threshold": threshold,
        "test_trial_id": trial_id,
        "simulation_status": "success" if result.completed else "failed",
        "failure_reason": result.failure_reason,
        "requested_duration_s": requested_duration_s,
        "reached_duration_s": result.reached_horizon_s,
        "simulation_runtime_s": runtime_s,
        "rhs_evaluations": result.rhs_evaluations,
      }
    )
    write_csv_checkpoint(status_path, STATUS_FIELDS, rows)
    print(
      f"threshold={threshold:g} trial={trial_id} "
      f"status={rows[-1]['simulation_status']} "
      f"reached={result.reached_horizon_s:.2f}/{requested_duration_s:.2f}s "
      f"runtime={runtime_s:.1f}s",
      flush=True,
    )
  return results, rows


def run(args: argparse.Namespace) -> None:
  """Reconstruct and simulate one thresholded configuration-216 equation."""
  stored_models = collect_threshold_models(
    args.grid_csv,
    refit_csvs=args.refit_csv or DEFAULT_REFIT_CSVS,
    configuration_index=args.configuration_index,
  )
  selected = select_threshold_model(stored_models, args.threshold)
  configuration = selected["configuration"]
  metadata = json.loads(args.metadata_json.read_text())
  if metadata["preprocessing"]["normalization"] != "none":
    raise ValueError("The source raw grid must use normalization='none'.")

  data = TrialData.load(args.mat_file)
  test_trial_ids = [int(value) for value in metadata["split"]["test_trial_ids"]]
  if args.max_test_trials is not None:
    test_trial_ids = test_trial_ids[: args.max_test_trials]
  downsample = int(metadata["downsample_factor"])
  dt = downsample / data.fs
  test_raw = channel_traces(
    data,
    channel=int(metadata["channel"]),
    trials=test_trial_ids,
    downsample=downsample,
    lowpass_hz=float(configuration["lowpass_hz"]),
    normalize="none",
  )
  measured_trials = delay_embed_trajectories(
    test_raw,
    n_delays=int(configuration["n_delays"]),
    delay=int(configuration["delay_samples"]),
  )
  model = StoredPolynomialModel(
    degree=int(configuration["degree"]),
    coefficients=selected["coefficients"],
    feature_names=selected["feature_names"],
  )
  label = threshold_label(args.threshold)
  status_path = args.output_dir / f"threshold_{label}_status.csv"
  results, status_rows = simulate_trials(
    model,
    measured_trials,
    test_trial_ids,
    configuration_index=args.configuration_index,
    threshold=args.threshold,
    dt=dt,
    status_path=status_path,
    trial_timeout_s=args.trial_timeout_s,
  )

  figure_path = args.output_dir / f"threshold_{label}_simulations.png"
  summary_path = args.output_dir / f"threshold_{label}_summary.json"
  args.output_dir.mkdir(parents=True, exist_ok=True)
  plot_configuration(
    figure_path,
    row=configuration,
    trial_ids=test_trial_ids,
    measured_trials=measured_trials,
    results=results,
    dt=dt,
    title=(
      f"Configuration {args.configuration_index}, STLSQ threshold={args.threshold:g}: "
      "measured vs simulated x0"
    ),
  )
  summary = {
    "configuration_index": args.configuration_index,
    "stlsq_threshold": args.threshold,
    "nonzero_terms": int(selected["nonzero_terms"]),
    "possible_terms": int(selected["coefficients"].size),
    "test_trials": len(test_trial_ids),
    "successful_simulations": sum(result.completed for result in results),
    "signal_units": "microvolts (uV)",
    "comparison_target": "low-pass-only held-out x0; no z-scoring or SG smoothing",
  }
  summary_path.write_text(json.dumps(summary, indent=2) + "\n")
  print(f"saved: {status_path}")
  print(f"saved: {figure_path}")
  print(f"saved: {summary_path}")


def main() -> None:
  """Parse CLI arguments and simulate one thresholded equation."""
  parser = argparse.ArgumentParser(
    description=(
      "Simulate one configuration-216 threshold refit on the fixed held-out trials."
    )
  )
  parser.add_argument("--threshold", type=float, required=True)
  parser.add_argument("--configuration-index", type=int, default=216)
  parser.add_argument("--mat-file", type=Path, default=MAT_FILE)
  parser.add_argument("--grid-csv", type=Path, default=DEFAULT_GRID)
  parser.add_argument("--metadata-json", type=Path, default=DEFAULT_METADATA)
  parser.add_argument("--refit-csv", type=Path, action="append", default=None)
  parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
  parser.add_argument("--max-test-trials", type=int, default=None)
  parser.add_argument(
    "--trial-timeout-s",
    type=float,
    default=None,
    help=(
      "Optional operational wall-time limit per held-out simulation in seconds; "
      "timeouts are recorded as numerical failures."
    ),
  )
  args = parser.parse_args()
  if args.max_test_trials is not None and args.max_test_trials < 1:
    parser.error("--max-test-trials must be at least 1.")
  if args.trial_timeout_s is not None and args.trial_timeout_s <= 0:
    parser.error("--trial-timeout-s must be positive.")
  run(args)


if __name__ == "__main__":
  main()
