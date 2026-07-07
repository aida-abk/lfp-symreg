"""PySR parameter sweep: one Slurm array task per configuration.

Grid (36 configurations total):
    n_delays  ∈ {2, 4, 6}
    delay     ∈ {1, 2, 5}  processed samples
    lowpass   ∈ {35, 80}   Hz
    smooth    ∈ {0, 9}     samples (0 = finite difference)

Each task fits PySR on training trials, simulates all held-out trials,
and saves a metrics CSV, equation text, and a 2×2 simulation plot.

Run on Oscar via scripts/hpc/run_pysr_array.slurm.
Run locally:
    .venv/bin/python scripts/pysr/pysr_sweep.py --configuration-index 1
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[2]
PYSINDY_SCRIPTS = ROOT / "scripts" / "pysindy"
for path in (ROOT, PYSINDY_SCRIPTS):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

from load_data.convert import MAT_FILE
from load_data.preprocessing import channel_traces
from models.pysr import (
  build_regression_arrays,
  cap_rows,
  metric_rows,
  selected_equations,
)
from models.sindy import delay_embed_trajectories
from sweep_io import prepare_lfp_trials

# 36 configurations: product order is n_delays × delay × lowpass × smooth
GRID = list(itertools.product(
  [2, 4, 6],       # n_delays
  [1, 2, 5],       # delay_samples (processed samples after downsampling)
  [35.0, 80.0],    # lowpass_hz
  [0, 9],          # smooth_window_samples
))

DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "pysr"
N_PLOT_TRIALS = 4  # held-out trials shown in the simulation grid figure

METRIC_FIELDS = [
  "configuration_index",
  "n_delays",
  "delay_samples",
  "delay_ms",
  "embedding_span_ms",
  "lowpass_hz",
  "smooth_window_samples",
  "dt_s",
  "processed_hz",
  "train_trials",
  "test_trials",
  "train_samples_used",
  "niterations",
  "maxsize",
  "populations",
  "fit_status",
  "fit_failure_reason",
  "fit_runtime_s",
  "train_derivative_r2",
  "test_derivative_r2",
  "equations_json",
  "simulation_success_fraction",
  "mean_x0_rmse",
  "median_x0_rmse",
  "simulation_runtime_s",
]


def build_rhs(model, n_delays: int):
  """Return a scipy-compatible RHS callable from PySR symbolic output.

  Args:
    model: Fitted ``PySRRegressor``.
    n_delays: Number of delay coordinates (state dimension).

  Returns:
    Function ``rhs(t, state) -> np.ndarray`` for use with ``solve_ivp``.
  """
  import sympy  # PySR dependency; available on Oscar when pysr is installed

  exprs = model.sympy()
  if not isinstance(exprs, list):
    exprs = [exprs]

  variable_names = [f"x{i}" for i in range(n_delays)]
  syms = sympy.symbols(" ".join(variable_names))
  if not isinstance(syms, tuple):
    syms = (syms,)

  fns = [sympy.lambdify(syms, expr, "numpy") for expr in exprs]

  def rhs(t: float, state: np.ndarray) -> np.ndarray:
    return np.array([float(fn(*state)) for fn in fns])

  return rhs


def simulate_trial(
  rhs,
  initial_state: np.ndarray,
  n_samples: int,
  dt: float,
) -> np.ndarray | None:
  """Simulate from initial_state for n_samples steps.

  Args:
    rhs: ODE right-hand side from ``build_rhs``.
    initial_state: Starting state vector.
    n_samples: Number of output time points.
    dt: Sample interval in seconds.

  Returns:
    Trajectory array with shape ``(n_samples, n_states)``, or ``None`` on failure.
  """
  horizon_s = (n_samples - 1) * dt
  t_eval = np.linspace(0, horizon_s, n_samples)
  try:
    result = solve_ivp(
      rhs,
      t_span=(0, horizon_s),
      y0=initial_state,
      t_eval=t_eval,
      method="RK45",
      max_step=dt * 2,
      rtol=1e-6,
      atol=1e-6,
    )
    if not result.success or result.y.shape[1] < n_samples:
      return None
    traj = result.y.T
    if not np.all(np.isfinite(traj)):
      return None
    return traj
  except Exception:
    return None


def plot_simulation_grid(
  trial_results: list[tuple],
  config_index: int,
  n_delays: int,
  delay_samples: int,
  lowpass_hz: float,
  smooth_window: int,
  dt: float,
  output_path: Path,
  raw_ref_x0: list[np.ndarray] | None = None,
) -> None:
  """Save a 2×2 figure of measured vs simulated x0 for up to 4 held-out trials.

  Args:
    trial_results: List of ``(trial_id, measured, simulated, status_label)`` tuples.
    config_index: Configuration index used in the figure title.
    n_delays: Number of delay coordinates.
    delay_samples: Delay spacing in processed samples.
    lowpass_hz: Low-pass filter cutoff in Hz.
    smooth_window: Derivative smoothing window in samples.
    dt: Sample interval in seconds.
    output_path: Where to save the PNG.
    raw_ref_x0: Optional lowpass-filtered (not z-scored) x0 reference traces, one
      per trial, already cropped to align with the delay-embedded x0. Shown in grey
      on a secondary y-axis to give µV-scale context of what the model is matching.
  """
  fig, axes = plt.subplots(2, 2, figsize=(12, 8))
  n = min(len(trial_results), N_PLOT_TRIALS)

  for i, ax in enumerate(axes.flat):
    if i >= n:
      ax.set_visible(False)
      continue
    trial_id, measured, simulated, status = trial_results[i]
    t_meas = np.arange(len(measured)) * dt

    # Grey secondary axis: lowpass-filtered signal in µV (not z-scored)
    ax2 = None
    if raw_ref_x0 is not None and i < len(raw_ref_x0):
      ax2 = ax.twinx()
      raw = raw_ref_x0[i][:len(measured)]
      ax2.plot(t_meas[:len(raw)], raw, color="grey", linewidth=0.8, alpha=0.45,
               label="lowpass (µV)")
      ax2.set_ylabel("µV", fontsize=8, color="grey")
      ax2.tick_params(axis="y", labelcolor="grey", labelsize=7)
      ax2.spines["right"].set_color("grey")
      ax2.spines["right"].set_alpha(0.5)
      # Keep primary axis on top so blue/orange lines render over the grey trace
      ax.set_zorder(ax2.get_zorder() + 1)
      ax.patch.set_visible(False)

    ax.plot(t_meas, measured[:, 0], color="steelblue", linewidth=1, label="measured")
    if simulated is not None:
      t_sim = np.arange(len(simulated)) * dt
      ax.plot(t_sim, simulated[:, 0], color="darkorange", linewidth=1,
              linestyle="--", label="simulated")
    ax.set_title(f"Trial {trial_id}  [{status}]", fontsize=10)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("x₀ (z-score)", fontsize=9)

    # Combine handles so grey reference appears in the single legend
    lines, labels = ax.get_legend_handles_labels()
    if ax2 is not None:
      lines2, labels2 = ax2.get_legend_handles_labels()
      ax.legend(lines2 + lines, labels2 + labels, fontsize=8)
    else:
      ax.legend(lines, labels, fontsize=8)

  fig.suptitle(
    f"Config {config_index} — n_delays={n_delays}, delay={delay_samples} smp, "
    f"lowpass={lowpass_hz} Hz, smooth={smooth_window} smp",
    fontsize=11,
  )
  fig.tight_layout()
  output_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(output_path, dpi=150, bbox_inches="tight")
  plt.close(fig)


def run_configuration(args: argparse.Namespace) -> dict[str, object]:
  """Fit PySR and simulate one configuration. Returns one metrics row.

  Args:
    args: Parsed CLI namespace.

  Returns:
    Dict with keys matching ``METRIC_FIELDS``.
  """
  config_idx = args.configuration_index
  if config_idx < 1 or config_idx > len(GRID):
    raise ValueError(f"--configuration-index must be 1–{len(GRID)}, got {config_idx}")

  n_delays, delay_samples, lowpass_hz, smooth_window = GRID[config_idx - 1]
  data, train_ids, test_ids = prepare_lfp_trials(args)
  dt = args.downsample / data.fs
  delay_ms = delay_samples * dt * 1000
  embedding_span_ms = (n_delays - 1) * delay_samples * dt * 1000

  row: dict[str, object] = {
    "configuration_index": config_idx,
    "n_delays": n_delays,
    "delay_samples": delay_samples,
    "delay_ms": round(delay_ms, 3),
    "embedding_span_ms": round(embedding_span_ms, 3),
    "lowpass_hz": lowpass_hz,
    "smooth_window_samples": smooth_window,
    "dt_s": dt,
    "processed_hz": round(1.0 / dt, 3),
    "train_trials": len(train_ids),
    "test_trials": len(test_ids),
    "train_samples_used": 0,
    "niterations": args.niterations,
    "maxsize": args.maxsize,
    "populations": args.populations,
    "fit_status": "failed",
    "fit_failure_reason": "",
    "fit_runtime_s": float("nan"),
    "train_derivative_r2": float("nan"),
    "test_derivative_r2": float("nan"),
    "equations_json": "[]",
    "simulation_success_fraction": float("nan"),
    "mean_x0_rmse": float("nan"),
    "median_x0_rmse": float("nan"),
    "simulation_runtime_s": float("nan"),
  }

  # Preprocess
  train_raw = channel_traces(
    data,
    channel=args.channel,
    trials=train_ids,
    downsample=args.downsample,
    lowpass_hz=lowpass_hz,
    normalize="zscore",
  )
  test_raw = channel_traces(
    data,
    channel=args.channel,
    trials=test_ids,
    downsample=args.downsample,
    lowpass_hz=lowpass_hz,
    normalize="zscore",
  )
  # Lowpass-filtered reference for the grey overlay: load only the trials that
  # will be plotted, no z-scoring so the signal stays in µV.
  test_raw_ref = channel_traces(
    data,
    channel=args.channel,
    trials=test_ids[:N_PLOT_TRIALS],
    downsample=args.downsample,
    lowpass_hz=lowpass_hz,
    normalize="none",
  )
  ref_offset = (n_delays - 1) * delay_samples
  raw_ref_x0 = [tr[ref_offset:] for tr in test_raw_ref]

  # Delay embed
  train_embedded = delay_embed_trajectories(train_raw, n_delays=n_delays, delay=delay_samples)
  test_embedded = delay_embed_trajectories(test_raw, n_delays=n_delays, delay=delay_samples)

  # Build regression arrays
  x_train_full, y_train_full = build_regression_arrays(
    train_embedded, dt=dt, smooth_window=smooth_window
  )
  x_test, y_test = build_regression_arrays(
    test_embedded, dt=dt, smooth_window=smooth_window
  )
  x_train, y_train = cap_rows(x_train_full, y_train_full, max_samples=args.max_train_samples)
  row["train_samples_used"] = x_train.shape[0]

  # Fit PySR
  try:
    from pysr import PySRRegressor
  except ImportError as exc:
    raise ImportError(
      "PySR is not installed. Run: pip install pysr, then pysr.install()"
    ) from exc

  variable_names = [f"x{i}" for i in range(n_delays)]
  config_dir = args.output_dir / f"config_{config_idx:04d}"

  model = PySRRegressor(
    niterations=args.niterations,
    maxsize=args.maxsize,
    populations=args.populations,
    binary_operators=["+", "-", "*"],
    unary_operators=[],
    model_selection="best",
    random_state=args.seed,
    verbosity=0,
  )

  fit_started = time.perf_counter()
  try:
    model.fit(x_train, y_train, variable_names=variable_names)
    row["fit_runtime_s"] = time.perf_counter() - fit_started
    row["fit_status"] = "success"
  except Exception as exc:
    row["fit_runtime_s"] = time.perf_counter() - fit_started
    row["fit_failure_reason"] = str(exc)
    return row

  # Derivative fit metrics
  try:
    train_pred = np.atleast_2d(model.predict(x_train))
    test_pred = np.atleast_2d(model.predict(x_test))
    if train_pred.shape[0] == 1:
      train_pred = train_pred.T
      test_pred = test_pred.T
    row["train_derivative_r2"] = float(metric_rows(y_train, train_pred)[-1]["r2"])
    row["test_derivative_r2"] = float(metric_rows(y_test, test_pred)[-1]["r2"])
  except Exception:
    pass

  # Save equations
  try:
    eqs = selected_equations(model)
    row["equations_json"] = json.dumps(eqs)
    eq_path = config_dir / "equations.txt"
    eq_path.parent.mkdir(parents=True, exist_ok=True)
    eq_path.write_text(
      "\n".join(f"dx{i}/dt = {eq}" for i, eq in enumerate(eqs)) + "\n"
    )
  except Exception:
    pass

  # Simulate all test trials and plot first N_PLOT_TRIALS
  sim_started = time.perf_counter()
  try:
    rhs = build_rhs(model, n_delays)
    trial_results = []
    x0_rmse_values = []

    for trial_id, measured in zip(test_ids, test_embedded):
      simulated = simulate_trial(rhs, measured[0], n_samples=len(measured), dt=dt)
      if simulated is not None:
        x0_rmse = float(np.sqrt(np.mean((measured[:, 0] - simulated[:, 0]) ** 2)))
        x0_rmse_values.append(x0_rmse)
        status = f"ok  RMSE={x0_rmse:.3f}"
      else:
        status = "diverged"
      trial_results.append((trial_id, measured, simulated, status))

    n_success = sum(1 for _, _, sim, _ in trial_results if sim is not None)
    row["simulation_success_fraction"] = (
      n_success / len(trial_results) if trial_results else float("nan")
    )
    if x0_rmse_values:
      row["mean_x0_rmse"] = float(np.mean(x0_rmse_values))
      row["median_x0_rmse"] = float(np.median(x0_rmse_values))
    row["simulation_runtime_s"] = time.perf_counter() - sim_started

    plot_simulation_grid(
      trial_results[:N_PLOT_TRIALS],
      config_index=config_idx,
      n_delays=n_delays,
      delay_samples=delay_samples,
      lowpass_hz=lowpass_hz,
      smooth_window=smooth_window,
      dt=dt,
      output_path=config_dir / "simulation.png",
      raw_ref_x0=raw_ref_x0,
    )
  except Exception as exc:
    row["simulation_runtime_s"] = time.perf_counter() - sim_started
    row["fit_failure_reason"] = f"simulation error: {exc}"

  return row


def main() -> None:
  """Fit PySR for one configuration and write metrics, equations, and simulation plot."""
  parser = argparse.ArgumentParser(
    description=(
      f"Fit PySR for one of {len(GRID)} configurations and simulate held-out trials."
    )
  )
  parser.add_argument(
    "--configuration-index", type=int, required=True,
    help=f"1-based configuration index (1–{len(GRID)}).",
  )
  parser.add_argument("--mat-file", type=Path, default=MAT_FILE)
  parser.add_argument(
    "--trial-type", choices=("fixation", "non_fixation"), default="fixation"
  )
  parser.add_argument("--channel", type=int, default=0)
  parser.add_argument("--max-trials", type=int, default=None)
  parser.add_argument("--test-fraction", type=float, default=0.25)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--downsample", type=int, default=2)
  parser.add_argument(
    "--niterations", type=int, default=100,
    help="Number of PySR evolutionary iterations.",
  )
  parser.add_argument(
    "--maxsize", type=int, default=20,
    help="Maximum symbolic expression size.",
  )
  parser.add_argument(
    "--populations", type=int, default=8,
    help="Number of independent PySR populations.",
  )
  parser.add_argument(
    "--max-train-samples", type=int, default=10000,
    help="Maximum training samples passed to PySR (evenly subsampled if exceeded).",
  )
  parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
  parser.add_argument(
    "--out-csv", type=Path, required=True,
    help="Output path for this configuration's single-row metrics CSV.",
  )
  args = parser.parse_args()

  row = run_configuration(args)

  args.out_csv.parent.mkdir(parents=True, exist_ok=True)
  with args.out_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
    writer.writeheader()
    writer.writerow(row)

  print(
    f"config {args.configuration_index}/{len(GRID)}: "
    f"fit={row['fit_status']} "
    f"train_r2={row['train_derivative_r2']} "
    f"test_r2={row['test_derivative_r2']} "
    f"sim_fraction={row['simulation_success_fraction']} "
    f"mean_x0_rmse={row['mean_x0_rmse']}"
  )
  print(f"saved: {args.out_csv}")


if __name__ == "__main__":
  main()
