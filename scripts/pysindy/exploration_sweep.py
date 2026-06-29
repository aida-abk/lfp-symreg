from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
PYSINDY_SCRIPTS = SCRIPTS / "pysindy"
for path in (ROOT, SCRIPTS, PYSINDY_SCRIPTS):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

from load_data.convert import MAT_FILE, TrialData, load_bhv_trial_table
from load_data.preprocessing import channel_traces
from load_data.synthetic import make_lorenz_dataset
from load_data.trial_selection import (
  DEFAULT_TRIAL_VALIDITY,
  TrialValidityConfig,
  select_valid_trials,
)
from models.sindy import (
  SINDyConfig,
  count_terms,
  delay_embed_trajectories,
  equation_text,
  fit_sindy_model,
)
from models.validation import (
  ValidationConfig,
  evaluate_simulation,
  finite_difference_jacobian,
  simulate_model,
)
from pipeline_utils import (
  parse_float_list,
  parse_int_list,
  split_trials_random_checked,
)


DEFAULT_OUT = ROOT / "outputs" / "pysindy" / "exploration_sweep.csv"
DEFAULT_TOP = ROOT / "outputs" / "pysindy" / "exploration_sweep_top.csv"
DEFAULT_EQUATIONS = ROOT / "outputs" / "pysindy" / "exploration_sweep_equations.txt"


def optional_float(value: str) -> float | None:
  """Parse optional floats from CLI input."""
  if value.lower() in {"none", "null"}:
    return None
  return float(value)


def parse_optional_float_list(value: str) -> list[float | None]:
  """Parse comma-separated optional floats, allowing none/null entries."""
  return [optional_float(part.strip()) for part in value.split(",") if part.strip()]


def write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
  """Write rows to CSV and create parent directories as needed."""
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def write_equations(path: Path, rows: list[dict[str, object]]) -> None:
  """Save readable equations for ranked sweep results."""
  path.parent.mkdir(parents=True, exist_ok=True)
  sections = []
  for rank, row in enumerate(rows, start=1):
    header = (
      f"Rank {rank}: source={row['source']}, trial_type={row['trial_type']}, "
      f"degree={row['degree']}, n_delays={row['n_delays']}, "
      f"delay={row['delay_samples']}, threshold={row['threshold']}, "
      f"status={row['dynamic_status']}, psd={float(row['psd_similarity']):.4f}, "
      f"rmse={float(row['trajectory_rmse']):.4f}"
    )
    sections.append(header + "\n" + str(row["equations"]))
  path.write_text("\n\n".join(sections) + ("\n" if sections else ""))


def ranked_rows(rows: list[dict[str, object]], limit: int) -> list[dict[str, object]]:
  """Rank viable rows by PSD similarity, trajectory error, and sparsity."""
  viable = [row for row in rows if row["dynamic_status"] == "viable"]
  return sorted(
    viable,
    key=lambda row: (
      float(row["psd_similarity"]),
      -float(row["trajectory_rmse"]),
      -int(row["nonzero_terms"]),
    ),
    reverse=True,
  )[:limit]


def aggregate_metric_rows(metric_rows: list[dict[str, float | str]]) -> dict[str, object]:
  """Aggregate per-trial simulation metrics into one sweep row fragment."""
  numeric_fields = [
    "trajectory_rmse",
    "x0_rmse",
    "x0_correlation",
    "max_amplitude_ratio",
    "collapse_std_ratio",
    "psd_similarity",
    "autocorrelation_similarity",
    "distribution_ks",
  ]
  aggregated: dict[str, object] = {}
  for field in numeric_fields:
    values = [float(row[field]) for row in metric_rows]
    aggregated[field] = float(np.nanmean(values))

  reasons = [str(row["rejection_reason"]) for row in metric_rows if row["rejection_reason"]]
  aggregated["dynamic_status"] = "rejected" if reasons else "viable"
  aggregated["rejection_reason"] = "; ".join(sorted(set(reasons)))
  return aggregated


def evaluate_model_on_trials(
  model,
  measured_trials: list[np.ndarray],
  dt: float,
  validation: ValidationConfig,
) -> dict[str, object]:
  """Simulate held-out trajectories and aggregate dynamic suitability checks."""
  metric_rows = []
  for index, measured in enumerate(measured_trials):
    measured_scale = max(1.0, float(np.max(np.abs(measured))))
    simulated, reason = simulate_model(
      model,
      initial_state=measured[0],
      dt=dt,
      horizon_s=validation.simulation_horizon,
      amplitude_limit=validation.amplitude_factor * measured_scale,
      max_rhs_evaluations=validation.max_rhs_evaluations,
    )
    if simulated is None:
      return {
        "dynamic_status": "rejected",
        "rejection_reason": f"held-out trajectory {index}: {reason}",
        "simulated_test_trials": index,
        "trajectory_rmse": float("nan"),
        "x0_rmse": float("nan"),
        "x0_correlation": float("nan"),
        "max_amplitude_ratio": float("nan"),
        "collapse_std_ratio": float("nan"),
        "psd_similarity": float("nan"),
        "autocorrelation_similarity": float("nan"),
        "distribution_ks": float("nan"),
      }
    metric_rows.append(
      evaluate_simulation(
        measured=measured,
        simulated=simulated,
        fs=1.0 / dt,
        config=validation,
      )
    )

  aggregated = aggregate_metric_rows(metric_rows)
  aggregated["simulated_test_trials"] = len(metric_rows)
  return aggregated


def jacobian_diagnostics(model, trajectory: np.ndarray) -> dict[str, object]:
  """Report Jacobian eigenvalue diagnostics at the first measured state."""
  try:
    jacobian = finite_difference_jacobian(model, trajectory[0])
    eigenvalues = np.linalg.eigvals(jacobian)
  except Exception:
    return {
      "jacobian_max_real": float("nan"),
      "jacobian_eigenvalues": "",
    }
  return {
    "jacobian_max_real": float(np.max(eigenvalues.real)),
    "jacobian_eigenvalues": ";".join(
      f"{value.real:.6g}{value.imag:+.6g}j" for value in eigenvalues
    ),
  }


def trial_validity_config(args: argparse.Namespace) -> TrialValidityConfig:
  """Build trial-validity rules from defaults plus CLI overrides."""
  type_columns = dict(DEFAULT_TRIAL_VALIDITY.trial_type_columns)
  validity_columns = dict(DEFAULT_TRIAL_VALIDITY.validity_columns)
  if args.trial_type_column:
    type_columns[args.trial_type] = args.trial_type_column
  if args.validity_columns:
    validity_columns[args.trial_type] = tuple(
      column.strip() for column in args.validity_columns.split(",") if column.strip()
    )
  return TrialValidityConfig(
    trial_type_columns=type_columns,
    validity_columns=validity_columns,
  )


def load_lfp_source(
  args: argparse.Namespace,
  lowpass_hz: float | None,
) -> tuple[list[np.ndarray], list[np.ndarray], float, list[int], list[int]]:
  """Load, filter, split, and preprocess LFP trajectories for one channel."""
  data = TrialData.load(args.mat_file)
  table = load_bhv_trial_table(args.mat_file)
  trials = select_valid_trials(table, args.trial_type, config=trial_validity_config(args))
  if args.max_trials is not None:
    trials = trials[: args.max_trials]
  train_trials, test_trials = split_trials_random_checked(
    trials,
    test_fraction=args.test_fraction,
    seed=args.seed,
  )
  train = channel_traces(
    data,
    channel=args.channel,
    trials=train_trials,
    downsample=args.downsample,
    lowpass_hz=lowpass_hz,
    normalize=args.normalize,
    window_start=args.window_start,
    window_end=args.window_end,
  )
  test = channel_traces(
    data,
    channel=args.channel,
    trials=test_trials,
    downsample=args.downsample,
    lowpass_hz=lowpass_hz,
    normalize=args.normalize,
    window_start=args.window_start,
    window_end=args.window_end,
  )
  return train, test, args.downsample / data.fs, train_trials, test_trials


def load_synthetic_source(args: argparse.Namespace) -> tuple[list[np.ndarray], list[np.ndarray], float, list[int], list[int]]:
  """Load a synthetic dataset and make the same random non-contiguous split."""
  dataset = make_lorenz_dataset(
    n_trajectories=args.synthetic_trajectories,
    duration=args.synthetic_duration,
    dt=args.synthetic_dt,
    seed=args.seed,
  )
  trials = list(range(len(dataset.trajectories)))
  train_indices, test_indices = split_trials_random_checked(
    trials,
    test_fraction=args.test_fraction,
    seed=args.seed,
  )
  train = [dataset.trajectories[index] for index in train_indices]
  test = [dataset.trajectories[index] for index in test_indices]
  return train, test, dataset.dt, train_indices, test_indices


def base_row(
  args: argparse.Namespace,
  train_trials: list[int],
  test_trials: list[int],
  lowpass_hz: float | None,
) -> dict[str, object]:
  """Create row fields that are shared by all sweep configurations."""
  return {
    "source": args.source,
    "trial_type": args.trial_type,
    "channel": args.channel,
    "random_seed": args.seed,
    "train_trials": len(train_trials),
    "test_trials": len(test_trials),
    "train_trial_ids": " ".join(map(str, train_trials)),
    "test_trial_ids": " ".join(map(str, test_trials)),
    "downsample": args.downsample if args.source == "lfp" else 1,
    "lowpass_hz": lowpass_hz if args.source == "lfp" else "",
    "normalize": args.normalize if args.source == "lfp" else "",
    "window_start": args.window_start if args.source == "lfp" else "",
    "window_end": args.window_end if args.source == "lfp" else "",
  }


def run_sweep(args: argparse.Namespace) -> list[dict[str, object]]:
  """Run the configured SINDy exploration sweep."""
  rows = []
  validation = ValidationConfig(
    simulation_horizon=args.simulation_horizon,
    amplitude_factor=args.amplitude_factor,
    collapse_std_fraction=args.collapse_std_fraction,
    min_psd_similarity=args.min_psd_similarity,
    min_autocorrelation_similarity=args.min_autocorrelation_similarity,
    max_distribution_ks=args.max_distribution_ks,
    max_rhs_evaluations=args.max_rhs_evaluations,
  )
  lowpass_values = (
    parse_optional_float_list(args.lowpass_list)
    if args.lowpass_list
    else [args.lowpass_hz]
  )

  for lowpass_hz in lowpass_values:
    if args.source == "lfp":
      train_raw, test_raw, dt, train_trials, test_trials = load_lfp_source(
        args,
        lowpass_hz=lowpass_hz,
      )
    elif args.source == "lorenz":
      train_raw, test_raw, dt, train_trials, test_trials = load_synthetic_source(args)
    else:
      raise ValueError(f"Unknown source: {args.source}")

    common = base_row(args, train_trials, test_trials, lowpass_hz=lowpass_hz)
    grid = itertools.product(
      parse_int_list(args.degree_list),
      parse_int_list(args.n_delays_list),
      parse_int_list(args.delay_list),
      parse_float_list(args.threshold_list),
      parse_int_list(args.smooth_window_list),
    )

    for degree, n_delays, delay, threshold, smooth_window in grid:
      row = {
        **common,
        "degree": degree,
        "n_delays": n_delays,
        "delay_samples": delay,
        "delay_ms": 1000 * delay * dt,
        "embedding_span_ms": 1000 * (n_delays - 1) * delay * dt,
        "threshold": threshold,
        "smooth_window": smooth_window,
        "dt": dt,
        "train_score_r2": float("nan"),
        "test_score_r2": float("nan"),
        "test_derivative_rmse": float("nan"),
        "nonzero_terms": 0,
        "jacobian_max_real": float("nan"),
        "jacobian_eigenvalues": "",
        "dynamic_status": "rejected",
        "rejection_reason": "model fitting did not complete",
        "simulated_test_trials": 0,
        "trajectory_rmse": float("nan"),
        "x0_rmse": float("nan"),
        "x0_correlation": float("nan"),
        "max_amplitude_ratio": float("nan"),
        "collapse_std_ratio": float("nan"),
        "psd_similarity": float("nan"),
        "autocorrelation_similarity": float("nan"),
        "distribution_ks": float("nan"),
        "equations": "",
        "error": "",
      }
      try:
        train = delay_embed_trajectories(train_raw, n_delays=n_delays, delay=delay)
        test = delay_embed_trajectories(test_raw, n_delays=n_delays, delay=delay)
        model = fit_sindy_model(
          train,
          dt=dt,
          config=SINDyConfig(
            threshold=threshold,
            degree=degree,
            smooth_window=smooth_window,
          ),
        )
        row["train_score_r2"] = float(model.score(train, t=dt))
        row["test_score_r2"] = float(model.score(test, t=dt))
        row["test_derivative_rmse"] = math.sqrt(
          float(model.score(test, t=dt, metric=mean_squared_error))
        )
        row["nonzero_terms"] = count_terms(model)
        row["equations"] = equation_text(model)
        row.update(jacobian_diagnostics(model, test[0]))
        row.update(evaluate_model_on_trials(model, test, dt=dt, validation=validation))
        print(
          f"{row['dynamic_status']:8} lowpass={row['lowpass_hz']} "
          f"degree={degree} delays={n_delays} delay={delay} "
          f"threshold={threshold:g} psd={float(row['psd_similarity']):.3f} "
          f"rmse={float(row['trajectory_rmse']):.3f} terms={row['nonzero_terms']}",
          flush=True,
        )
      except Exception as exc:
        row["error"] = str(exc)
        row["rejection_reason"] = f"model error: {exc}"
        print(
          f"failed   lowpass={lowpass_hz} degree={degree} delays={n_delays} "
          f"delay={delay} threshold={threshold:g}: {exc}",
          flush=True,
        )
      rows.append(row)
      write_rows(args.out_csv, rows, FIELDNAMES)

  return rows


FIELDNAMES = [
  "source",
  "trial_type",
  "channel",
  "random_seed",
  "train_trials",
  "test_trials",
  "train_trial_ids",
  "test_trial_ids",
  "downsample",
  "lowpass_hz",
  "normalize",
  "window_start",
  "window_end",
  "degree",
  "n_delays",
  "delay_samples",
  "delay_ms",
  "embedding_span_ms",
  "threshold",
  "smooth_window",
  "dt",
  "train_score_r2",
  "test_score_r2",
  "test_derivative_rmse",
  "nonzero_terms",
  "jacobian_max_real",
  "jacobian_eigenvalues",
  "dynamic_status",
  "rejection_reason",
  "simulated_test_trials",
  "trajectory_rmse",
  "x0_rmse",
  "x0_correlation",
  "max_amplitude_ratio",
  "collapse_std_ratio",
  "psd_similarity",
  "autocorrelation_similarity",
  "distribution_ks",
  "equations",
  "error",
]


def main() -> None:
  """CLI entry point for modular PySINDy exploration sweeps."""
  parser = argparse.ArgumentParser(
    description="Run modular PySINDy sweeps with dynamic simulation checks."
  )
  parser.add_argument("--source", choices=("lfp", "lorenz"), default="lfp")
  parser.add_argument("--mat-file", type=Path, default=MAT_FILE)
  parser.add_argument("--trial-type", choices=("fixation", "non_fixation"), default="fixation")
  parser.add_argument("--trial-type-column", default=None)
  parser.add_argument("--validity-columns", default=None)
  parser.add_argument("--channel", type=int, default=0)
  parser.add_argument("--max-trials", type=int, default=None)
  parser.add_argument("--test-fraction", type=float, default=0.25)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--downsample", type=int, default=2)
  parser.add_argument("--lowpass-hz", type=optional_float, default=80.0)
  parser.add_argument("--lowpass-list", default=None)
  parser.add_argument("--normalize", choices=("zscore", "center", "none"), default="zscore")
  parser.add_argument("--window-start", type=optional_float, default=None)
  parser.add_argument("--window-end", type=optional_float, default=None)
  parser.add_argument("--degree-list", default="1,2")
  parser.add_argument("--n-delays-list", default="2,4,6")
  parser.add_argument("--delay-list", default="1,2,5")
  parser.add_argument("--threshold-list", default="0.01,0.05,0.1,0.2")
  parser.add_argument("--smooth-window-list", default="0")
  parser.add_argument("--simulation-horizon", type=float, default=2.0)
  parser.add_argument("--amplitude-factor", type=float, default=10.0)
  parser.add_argument("--collapse-std-fraction", type=float, default=0.05)
  parser.add_argument("--min-psd-similarity", type=float, default=0.25)
  parser.add_argument("--min-autocorrelation-similarity", type=float, default=0.25)
  parser.add_argument("--max-distribution-ks", type=float, default=0.75)
  parser.add_argument("--max-rhs-evaluations", type=int, default=20_000)
  parser.add_argument("--synthetic-trajectories", type=int, default=8)
  parser.add_argument("--synthetic-duration", type=float, default=8.0)
  parser.add_argument("--synthetic-dt", type=float, default=0.01)
  parser.add_argument("--top-n", type=int, default=10)
  parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT)
  parser.add_argument("--top-csv", type=Path, default=DEFAULT_TOP)
  parser.add_argument("--equations-out", type=Path, default=DEFAULT_EQUATIONS)
  args = parser.parse_args()

  rows = run_sweep(args)
  top = ranked_rows(rows, limit=args.top_n)
  write_rows(args.out_csv, rows, FIELDNAMES)
  write_rows(args.top_csv, top, FIELDNAMES)
  write_equations(args.equations_out, top)
  print(f"saved: {args.out_csv}")
  print(f"saved: {args.top_csv}")
  print(f"saved: {args.equations_out}")
  print(f"viable models: {len(top)} shown, {sum(row['dynamic_status'] == 'viable' for row in rows)}/{len(rows)} total")


if __name__ == "__main__":
  main()
