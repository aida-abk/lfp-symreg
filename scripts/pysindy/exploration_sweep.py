from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_squared_error

# Project imports
ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
PYSINDY_SCRIPTS = SCRIPTS / "pysindy"
for path in (ROOT, SCRIPTS, PYSINDY_SCRIPTS):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

from load_data.convert import (
  LFP_AMPLITUDE_UNIT,
  MAT_FILE,
  TrialData,
  load_bhv_trial_table,
)
from load_data.preprocessing import channel_traces
from load_data.synthetic import make_lorenz_dataset
from load_data.trial_selection import select_valid_trials
from models.sindy import (
  SINDyConfig,
  count_terms,
  delay_embed_trajectories,
  equation_text,
  fit_sindy_model,
)
from models.validation import (
  SimulationConfig,
  evaluate_simulation,
  simulate_model_detailed,
)
from pipeline_utils import parse_float_list, parse_int_list, split_trials_random

# Default output locations
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "pysindy"
DEFAULT_OUT = DEFAULT_OUTPUT_DIR / "exploration_sweep.csv"
DEFAULT_CHECKPOINTS = DEFAULT_OUTPUT_DIR / "exploration_sweep_checkpoints.csv"
DEFAULT_TRIAL_METRICS = DEFAULT_OUTPUT_DIR / "exploration_sweep_trial_metrics.csv"
DEFAULT_EQUATIONS = DEFAULT_OUTPUT_DIR / "exploration_sweep_equations.txt"

# Metrics reported without acceptance thresholds
METRIC_FIELDS = [
  "trajectory_rmse",
  "x0_rmse",
  "x0_correlation",
  "max_amplitude_ratio",
  "collapse_std_ratio",
  "psd_similarity",
  "distribution_ks",
  "divergence_time_s",
]


def optional_float(value: str) -> float | None:
  """Parse a floating-point value or the text ``none``."""
  if value.lower() in {"none", "null"}:
    return None
  return float(value)


def parse_optional_float_list(value: str) -> list[float | None]:
  """Parse comma-separated filter cutoffs in hertz, allowing ``none``."""
  return [optional_float(part.strip()) for part in value.split(",") if part.strip()]


def evaluation_horizons(args: argparse.Namespace) -> list[float]:
  """Return sorted evaluation times in seconds, including the maximum horizon."""
  requested = (
    parse_float_list(args.evaluation_horizons)
    if args.evaluation_horizons
    else [args.simulation_horizon]
  )
  horizons = sorted(set([*requested, args.simulation_horizon]))
  if any(horizon <= 0 for horizon in horizons):
    raise ValueError("Evaluation horizons must be positive seconds.")
  if any(horizon > args.simulation_horizon for horizon in horizons):
    raise ValueError("Evaluation horizons cannot exceed --simulation-horizon.")
  return horizons


def validate_diagnostic_options(args: argparse.Namespace) -> None:
  """Validate paired optional diagnostics without choosing scientific values."""
  threshold = args.divergence_threshold_std
  persistence = args.divergence_persistence_s
  if (threshold is None) != (persistence is None):
    raise ValueError(
      "Provide both divergence options or neither; no default definition is imposed."
    )
  if threshold is not None and threshold <= 0:
    raise ValueError("--divergence-threshold-std must be positive.")
  if persistence is not None and persistence <= 0:
    raise ValueError("--divergence-persistence-s must be positive seconds.")


def write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
  """Write CSV rows and create the parent directory when necessary."""
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def append_row(path: Path, row: dict[str, object], fieldnames: list[str]) -> None:
  """Append one row to an initialized CSV file."""
  with path.open("a", newline="") as file:
    csv.DictWriter(file, fieldnames=fieldnames).writerow(row)


def write_equations(path: Path, rows: list[dict[str, object]]) -> None:
  """Write every successfully fitted equation without ranking models."""
  path.parent.mkdir(parents=True, exist_ok=True)
  sections = []
  for row in rows:
    if row["fit_status"] != "success":
      continue
    header = (
      f"Configuration {row['configuration_index']}: lowpass={row['lowpass_hz']}, "
      f"degree={row['degree']}, n_delays={row['n_delays']}, "
      f"delay_samples={row['delay_samples']}"
    )
    sections.append(f"{header}\n{row['equations']}")
  path.write_text("\n\n".join(sections) + ("\n" if sections else ""))


def finite_summary(values: list[float]) -> tuple[float, float]:
  """Return the mean and median of finite values, or NaN when none exist."""
  array = np.asarray(values, dtype=float)
  finite = array[np.isfinite(array)]
  if not finite.size:
    return float("nan"), float("nan")
  return float(np.mean(finite)), float(np.median(finite))


def summarize_trial_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
  """Summarize one horizon while retaining all trial rows in a separate CSV."""
  successful = [row for row in rows if row["simulation_status"] == "success"]
  failures = [row for row in rows if row["simulation_status"] != "success"]
  summary: dict[str, object] = {
    "simulation_status": (
      "all_success"
      if not failures
      else "all_failed"
      if not successful
      else "partial_failure"
    ),
    "simulation_success_fraction": len(successful) / len(rows) if rows else float("nan"),
    "successful_test_trials": len(successful),
    "failed_test_trials": len(failures),
    "simulation_failure_reasons": "; ".join(
      sorted({str(row["simulation_failure_reason"]) for row in failures})
    ),
  }
  for field in METRIC_FIELDS:
    mean, median = finite_summary([float(row[field]) for row in successful])
    summary[f"{field}_mean"] = mean
    summary[f"{field}_median"] = median

  known_divergence = [row["diverged"] for row in successful if row["diverged"] != ""]
  summary["diverged_fraction"] = (
    sum(bool(value) for value in known_divergence) / len(known_divergence)
    if known_divergence
    else float("nan")
  )
  rhs_mean, rhs_median = finite_summary(
    [float(row["rhs_evaluations"]) for row in rows]
  )
  summary["rhs_evaluations_mean"] = rhs_mean
  summary["rhs_evaluations_median"] = rhs_median
  return summary


def failed_trial_metrics(
  trial_index: int,
  trial_id: int,
  horizon_s: float,
  reached_horizon_s: float,
  rhs_evaluations: int,
  reason: str,
) -> dict[str, object]:
  """Create one explicit trial-level simulation failure row."""
  return {
    "test_trial_index": trial_index,
    "test_trial_id": trial_id,
    "evaluation_horizon_s": horizon_s,
    "reached_horizon_s": reached_horizon_s,
    "rhs_evaluations": rhs_evaluations,
    "simulation_status": "failed",
    "simulation_failure_reason": reason,
    **{field: float("nan") for field in METRIC_FIELDS},
    "diverged": "",
  }


def evaluate_model_on_trials(
  model,
  measured_trials: list[np.ndarray],
  trial_ids: list[int],
  dt: float,
  config: SimulationConfig,
  horizons: list[float],
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
  """Simulate each held-out trial once and report every requested checkpoint.

  Args:
    model: Fitted PySINDy model.
    measured_trials: Held-out state trajectories with shape ``(time, state)``.
    trial_ids: Original dataset identifiers corresponding to measured trials.
    dt: Processed sample interval in seconds.
    config: Numerical simulation and optional diagnostic settings.
    horizons: Evaluation checkpoints in seconds.

  Returns:
    Maximum-horizon summary, all checkpoint summaries, and raw per-trial rows.
  """
  trial_rows: list[dict[str, object]] = []
  maximum_samples = int(round(config.simulation_horizon_s / dt)) + 1

  for trial_index, (trial_id, measured) in enumerate(zip(trial_ids, measured_trials)):
    reference_samples = min(measured.shape[0], maximum_samples)
    divergence_reference_std = float(np.std(measured[:reference_samples, 0]))
    result = simulate_model_detailed(
      model,
      initial_state=measured[0],
      dt=dt,
      horizon_s=config.simulation_horizon_s,
    )
    for horizon in horizons:
      required_samples = int(round(horizon / dt)) + 1
      if measured.shape[0] < required_samples:
        trial_rows.append(
          failed_trial_metrics(
            trial_index,
            trial_id,
            horizon,
            result.reached_horizon_s,
            result.rhs_evaluations,
            (
              f"measured trajectory has {measured.shape[0]} samples; "
              f"{required_samples} required"
            ),
          )
        )
        continue
      if result.trajectory is None or result.trajectory.shape[0] < required_samples:
        trial_rows.append(
          failed_trial_metrics(
            trial_index,
            trial_id,
            horizon,
            result.reached_horizon_s,
            result.rhs_evaluations,
            result.failure_reason,
          )
        )
        continue

      metrics = evaluate_simulation(
        measured=measured[:required_samples],
        simulated=result.trajectory[:required_samples],
        fs=1.0 / dt,
        config=config,
        divergence_reference_std=divergence_reference_std,
      )
      trial_rows.append(
        {
          "test_trial_index": trial_index,
          "test_trial_id": trial_id,
          "evaluation_horizon_s": horizon,
          "reached_horizon_s": result.reached_horizon_s,
          "rhs_evaluations": result.rhs_evaluations,
          "simulation_status": "success",
          "simulation_failure_reason": "",
          **metrics,
          "diverged": "" if metrics["diverged"] is None else metrics["diverged"],
        }
      )

  checkpoint_rows = []
  for horizon in horizons:
    rows_at_horizon = [
      row for row in trial_rows if row["evaluation_horizon_s"] == horizon
    ]
    checkpoint_rows.append(
      {
        "evaluation_horizon_s": horizon,
        **summarize_trial_metrics(rows_at_horizon),
      }
    )
  maximum_summary = dict(checkpoint_rows[-1])
  maximum_summary.pop("evaluation_horizon_s")
  return maximum_summary, checkpoint_rows, trial_rows


def prepare_lfp_trials(
  args: argparse.Namespace,
) -> tuple[TrialData, list[int], list[int]]:
  """Load valid trial identifiers and make one reproducible whole-trial split."""
  data = TrialData.load(args.mat_file)
  table = load_bhv_trial_table(args.mat_file)
  trials = select_valid_trials(table, args.trial_type)
  if args.max_trials is not None:
    trials = trials[: args.max_trials]
  train_ids, test_ids = split_trials_random(
    trials,
    test_fraction=args.test_fraction,
    seed=args.seed,
  )
  return data, train_ids, test_ids


def preprocess_lfp_trials(
  data: TrialData,
  args: argparse.Namespace,
  train_ids: list[int],
  test_ids: list[int],
  lowpass_hz: float | None,
) -> tuple[list[np.ndarray], list[np.ndarray], float]:
  """Preprocess selected LFP trials while retaining the stored amplitude scale."""
  train = channel_traces(
    data,
    channel=args.channel,
    trials=train_ids,
    downsample=args.downsample,
    lowpass_hz=lowpass_hz,
    normalize=args.normalize,
  )
  test = channel_traces(
    data,
    channel=args.channel,
    trials=test_ids,
    downsample=args.downsample,
    lowpass_hz=lowpass_hz,
    normalize=args.normalize,
  )
  return train, test, args.downsample / data.fs


def prepare_lorenz_trials(
  args: argparse.Namespace,
) -> tuple[list[np.ndarray], list[np.ndarray], float, list[int], list[int]]:
  """Create known Lorenz trajectories and make a reproducible random split."""
  dataset = make_lorenz_dataset(
    n_trajectories=args.synthetic_trajectories,
    duration=args.synthetic_duration,
    dt=args.synthetic_dt,
    seed=args.seed,
  )
  trial_ids = list(range(len(dataset.trajectories)))
  train_ids, test_ids = split_trials_random(
    trial_ids,
    test_fraction=args.test_fraction,
    seed=args.seed,
  )
  train = [dataset.trajectories[index] for index in train_ids]
  test = [dataset.trajectories[index] for index in test_ids]
  return train, test, dataset.dt, train_ids, test_ids


def signal_units(source: str, normalization: str) -> str:
  """Describe the amplitude units retained after preprocessing."""
  if source == "lorenz":
    return "Lorenz state units"
  if normalization == "zscore":
    return "per-trial standard deviations"
  return LFP_AMPLITUDE_UNIT


def model_row_defaults() -> dict[str, object]:
  """Return empty fit and simulation fields for one model configuration."""
  summary_fields = {
    f"{metric}_{statistic}": float("nan")
    for metric in METRIC_FIELDS
    for statistic in ("mean", "median")
  }
  return {
    "fit_status": "failed",
    "fit_failure_reason": "model fitting did not complete",
    "train_derivative_r2": float("nan"),
    "test_derivative_r2": float("nan"),
    "test_derivative_rmse": float("nan"),
    "nonzero_terms": 0,
    "simulation_status": "not_run",
    "simulation_success_fraction": float("nan"),
    "successful_test_trials": 0,
    "failed_test_trials": 0,
    "simulation_failure_reasons": "",
    "diverged_fraction": float("nan"),
    "rhs_evaluations_mean": float("nan"),
    "rhs_evaluations_median": float("nan"),
    "fit_runtime_s": float("nan"),
    "simulation_runtime_s": float("nan"),
    "configuration_runtime_s": float("nan"),
    "equations": "",
    **summary_fields,
  }


def _build_row_metadata(
  configuration_index: int,
  args: argparse.Namespace,
  train_ids: list[int],
  test_ids: list[int],
  dt: float,
  lowpass_hz: float | None,
  degree: int,
  n_delays: int,
  delay: int,
  sindy_config: SINDyConfig,
  horizons: list[float],
) -> dict[str, object]:
  """Assemble the static configuration fields for one sweep row."""
  return {
    "configuration_index": configuration_index,
    "source": args.source,
    "trial_type": args.trial_type if args.source == "lfp" else "synthetic",
    "channel": args.channel if args.source == "lfp" else "",
    "random_seed": args.seed,
    "train_trials": len(train_ids),
    "test_trials": len(test_ids),
    "train_trial_ids": " ".join(map(str, train_ids)),
    "test_trial_ids": " ".join(map(str, test_ids)),
    "downsample_factor": args.downsample if args.source == "lfp" else 1,
    "processed_sampling_hz": 1.0 / dt,
    "sample_interval_s": dt,
    "signal_units": signal_units(args.source, args.normalize),
    "lowpass_hz": lowpass_hz if lowpass_hz is not None else "none",
    "normalization": args.normalize if args.source == "lfp" else "none",
    "degree": degree,
    "n_delays": n_delays,
    "delay_samples": delay,
    "delay_s": delay * dt,
    "embedding_span_s": (n_delays - 1) * delay * dt,
    "stlsq_threshold": sindy_config.threshold,
    "stlsq_alpha": sindy_config.alpha,
    "derivative_method": "PySINDy default finite difference",
    "simulation_solver": "LSODA",
    "simulation_horizon_s": args.simulation_horizon,
    "evaluation_horizons_s": " ".join(f"{value:g}" for value in horizons),
    "divergence_threshold_std": (
      args.divergence_threshold_std
      if args.divergence_threshold_std is not None
      else "disabled"
    ),
    "divergence_persistence_s": (
      args.divergence_persistence_s
      if args.divergence_persistence_s is not None
      else "disabled"
    ),
    **model_row_defaults(),
  }


def _fit_and_evaluate(
  train_raw: list[np.ndarray],
  test_raw: list[np.ndarray],
  dt: float,
  n_delays: int,
  delay: int,
  sindy_config: SINDyConfig,
  simulation: SimulationConfig,
  horizons: list[float],
  test_ids: list[int],
  train_ids: list[int],
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
  """Fit one SINDy model and evaluate it on held-out trials.

  Returns (row_updates, checkpoint_rows, trial_rows).
  """
  train = delay_embed_trajectories(train_raw, n_delays=n_delays, delay=delay)
  test = delay_embed_trajectories(test_raw, n_delays=n_delays, delay=delay)

  fit_started = time.perf_counter()
  model = fit_sindy_model(train, dt=dt, config=sindy_config)
  fit_runtime = time.perf_counter() - fit_started

  updates: dict[str, object] = {
    "fit_runtime_s": fit_runtime,
    "fit_status": "success",
    "fit_failure_reason": "",
    "train_derivative_r2": float(model.score(train, t=dt)),
    "test_derivative_r2": float(model.score(test, t=dt)),
    "test_derivative_rmse": math.sqrt(
      float(model.score(test, t=dt, metric=mean_squared_error))
    ),
    "nonzero_terms": count_terms(model),
    "equations": equation_text(model),
  }

  simulation_started = time.perf_counter()
  summary, checkpoint_rows, trial_rows = evaluate_model_on_trials(
    model,
    test,
    trial_ids=test_ids,
    dt=dt,
    config=simulation,
    horizons=horizons,
  )
  updates["simulation_runtime_s"] = time.perf_counter() - simulation_started
  updates.update(summary)
  return updates, checkpoint_rows, trial_rows


def run_sweep(args: argparse.Namespace) -> list[dict[str, object]]:
  """Fit the requested grid and save model, checkpoint, and per-trial results."""
  validate_diagnostic_options(args)
  horizons = evaluation_horizons(args)
  simulation = SimulationConfig(
    simulation_horizon_s=args.simulation_horizon,
    divergence_threshold_std=args.divergence_threshold_std,
    divergence_persistence_s=args.divergence_persistence_s,
  )
  lowpass_values = parse_optional_float_list(args.lowpass_list)
  if args.source == "lorenz":
    lowpass_values = [None]
  degree_values = parse_int_list(args.degree_list)
  n_delay_values = parse_int_list(args.n_delays_list)
  delay_values = parse_int_list(args.delay_list)
  total = math.prod(
    [len(lowpass_values), len(degree_values), len(n_delay_values), len(delay_values)]
  )

  write_rows(args.out_csv, [], MODEL_FIELDNAMES)
  write_rows(args.checkpoint_csv, [], CHECKPOINT_FIELDNAMES)
  write_rows(args.trial_metrics_csv, [], TRIAL_FIELDNAMES)

  lfp_data = None
  if args.source == "lfp":
    lfp_data, train_ids, test_ids = prepare_lfp_trials(args)
  else:
    synthetic = prepare_lorenz_trials(args)
    synthetic_train, synthetic_test, synthetic_dt, train_ids, test_ids = synthetic

  rows = []
  configuration_index = 0
  for lowpass_hz in lowpass_values:
    if args.source == "lfp":
      train_raw, test_raw, dt = preprocess_lfp_trials(
        lfp_data,
        args,
        train_ids,
        test_ids,
        lowpass_hz,
      )
    else:
      train_raw, test_raw, dt = synthetic_train, synthetic_test, synthetic_dt

    for degree, n_delays, delay in itertools.product(
      degree_values,
      n_delay_values,
      delay_values,
    ):
      configuration_index += 1
      sindy_config = SINDyConfig(degree=degree)
      started = time.perf_counter()
      row = _build_row_metadata(
        configuration_index=configuration_index,
        args=args,
        train_ids=train_ids,
        test_ids=test_ids,
        dt=dt,
        lowpass_hz=lowpass_hz,
        degree=degree,
        n_delays=n_delays,
        delay=delay,
        sindy_config=sindy_config,
        horizons=horizons,
      )
      checkpoint_rows: list[dict[str, object]] = []
      trial_rows: list[dict[str, object]] = []
      try:
        updates, checkpoint_rows, trial_rows = _fit_and_evaluate(
          train_raw=train_raw,
          test_raw=test_raw,
          dt=dt,
          n_delays=n_delays,
          delay=delay,
          sindy_config=sindy_config,
          simulation=simulation,
          horizons=horizons,
          test_ids=test_ids,
          train_ids=train_ids,
        )
        row.update(updates)
      except Exception as exc:
        row["fit_failure_reason"] = str(exc)

      row["configuration_runtime_s"] = time.perf_counter() - started
      rows.append(row)
      append_row(args.out_csv, row, MODEL_FIELDNAMES)

      config_fields = {field: row[field] for field in CONFIG_ID_FIELDS}
      for checkpoint in checkpoint_rows:
        append_row(
          args.checkpoint_csv,
          {**config_fields, **checkpoint},
          CHECKPOINT_FIELDNAMES,
        )
      for trial_row in trial_rows:
        append_row(
          args.trial_metrics_csv,
          {**config_fields, **trial_row},
          TRIAL_FIELDNAMES,
        )

      print(
        f"[{configuration_index}/{total}] fit={row['fit_status']} "
        f"simulation={row['simulation_status']} lowpass={row['lowpass_hz']} "
        f"degree={degree} delays={n_delays} delay={delay} "
        f"success_fraction={float(row['simulation_success_fraction']):.3f} "
        f"runtime={float(row['configuration_runtime_s']):.1f}s",
        flush=True,
      )

  return rows


# Output schemas
CONFIG_ID_FIELDS = [
  "configuration_index",
  "source",
  "trial_type",
  "channel",
  "random_seed",
  "lowpass_hz",
  "degree",
  "n_delays",
  "delay_samples",
]

SUMMARY_FIELDS = [
  item
  for metric in METRIC_FIELDS
  for item in (f"{metric}_mean", f"{metric}_median")
]

MODEL_FIELDNAMES = [
  "configuration_index",
  "source",
  "trial_type",
  "channel",
  "random_seed",
  "train_trials",
  "test_trials",
  "train_trial_ids",
  "test_trial_ids",
  "downsample_factor",
  "processed_sampling_hz",
  "sample_interval_s",
  "signal_units",
  "lowpass_hz",
  "normalization",
  "degree",
  "n_delays",
  "delay_samples",
  "delay_s",
  "embedding_span_s",
  "stlsq_threshold",
  "stlsq_alpha",
  "derivative_method",
  "simulation_solver",
  "simulation_horizon_s",
  "evaluation_horizons_s",
  "divergence_threshold_std",
  "divergence_persistence_s",
  "fit_status",
  "fit_failure_reason",
  "train_derivative_r2",
  "test_derivative_r2",
  "test_derivative_rmse",
  "nonzero_terms",
  "simulation_status",
  "simulation_success_fraction",
  "successful_test_trials",
  "failed_test_trials",
  "simulation_failure_reasons",
  *SUMMARY_FIELDS,
  "diverged_fraction",
  "rhs_evaluations_mean",
  "rhs_evaluations_median",
  "fit_runtime_s",
  "simulation_runtime_s",
  "configuration_runtime_s",
  "equations",
]

CHECKPOINT_FIELDNAMES = [
  *CONFIG_ID_FIELDS,
  "evaluation_horizon_s",
  "simulation_status",
  "simulation_success_fraction",
  "successful_test_trials",
  "failed_test_trials",
  "simulation_failure_reasons",
  *SUMMARY_FIELDS,
  "diverged_fraction",
  "rhs_evaluations_mean",
  "rhs_evaluations_median",
]

TRIAL_FIELDNAMES = [
  *CONFIG_ID_FIELDS,
  "test_trial_index",
  "test_trial_id",
  "evaluation_horizon_s",
  "reached_horizon_s",
  "rhs_evaluations",
  "simulation_status",
  "simulation_failure_reason",
  *METRIC_FIELDS,
  "diverged",
]


def main() -> None:
  """Run the documented PySINDy exploration CLI."""
  parser = argparse.ArgumentParser(
    description="Fit PySINDy models and record raw held-out simulation diagnostics."
  )
  parser.add_argument("--source", choices=("lfp", "lorenz"), default="lfp")
  parser.add_argument("--mat-file", type=Path, default=MAT_FILE)
  parser.add_argument("--trial-type", choices=("fixation", "non_fixation"), default="fixation")
  parser.add_argument("--channel", type=int, default=0)
  parser.add_argument("--max-trials", type=int, default=None)
  parser.add_argument("--test-fraction", type=float, default=0.25)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--downsample", type=int, default=2)
  parser.add_argument(
    "--lowpass-list",
    default="80",
    help="Comma-separated low-pass cutoffs in Hz; use 'none' for no low-pass.",
  )
  parser.add_argument(
    "--normalize",
    choices=("none", "center", "zscore"),
    default="none",
    help="Amplitude normalization; default 'none' preserves stored LFP scale.",
  )
  parser.add_argument("--degree-list", default="1,2,3", help="Polynomial degrees.")
  parser.add_argument("--n-delays-list", default="2,4,6,8", help="Delay-coordinate counts.")
  parser.add_argument(
    "--delay-list",
    default="1,2,5",
    help="Coordinate spacings in processed samples.",
  )
  parser.add_argument(
    "--simulation-horizon",
    type=float,
    required=True,
    help="Maximum autonomous simulation duration in seconds.",
  )
  parser.add_argument(
    "--evaluation-horizons",
    default=None,
    help="Optional comma-separated checkpoints in seconds; defaults to the maximum only.",
  )
  parser.add_argument(
    "--divergence-threshold-std",
    type=optional_float,
    default=None,
    help="Error threshold in measured-signal SD; requires persistence option.",
  )
  parser.add_argument(
    "--divergence-persistence-s",
    type=optional_float,
    default=None,
    help="Continuous threshold-exceedance duration in seconds.",
  )
  parser.add_argument("--synthetic-trajectories", type=int, default=8)
  parser.add_argument("--synthetic-duration", type=float, default=8.0)
  parser.add_argument("--synthetic-dt", type=float, default=0.01)
  parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT)
  parser.add_argument("--checkpoint-csv", type=Path, default=DEFAULT_CHECKPOINTS)
  parser.add_argument("--trial-metrics-csv", type=Path, default=DEFAULT_TRIAL_METRICS)
  parser.add_argument("--equations-out", type=Path, default=DEFAULT_EQUATIONS)
  args = parser.parse_args()

  rows = run_sweep(args)
  write_rows(args.out_csv, rows, MODEL_FIELDNAMES)
  write_equations(args.equations_out, rows)
  print(f"saved: {args.out_csv}")
  print(f"saved: {args.checkpoint_csv}")
  print(f"saved: {args.trial_metrics_csv}")
  print(f"saved: {args.equations_out}")


if __name__ == "__main__":
  main()
