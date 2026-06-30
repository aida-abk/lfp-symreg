from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import signal
from sklearn.metrics import mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
PYSINDY_SCRIPTS = SCRIPTS / "pysindy"
for path in (ROOT, SCRIPTS, PYSINDY_SCRIPTS):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

from load_data.convert import MAT_FILE, TrialData, load_bhv_trial_table
from load_data.preprocessing import preprocess_trace
from load_data.trial_selection import select_valid_trials
from models.sindy import delay_embed_trajectories
from scripts.pysindy.pipeline_utils import parse_trials, split_trials_sequential


def optional_float(value: str) -> float | None:
  return None if value.lower() in {"none", "off", "0"} else float(value)


def optional_int(value: str) -> int | None:
  return None if value.lower() in {"none", "off", "0"} else int(value)


def estimate_derivative(
  trajectory: np.ndarray,
  dt: float,
  smooth_window: int,
  polynomial_order: int = 3,
) -> np.ndarray:
  """Estimate derivatives without crossing trial boundaries."""
  if smooth_window <= 2:
    return np.gradient(trajectory, dt, axis=0, edge_order=2)

  window = smooth_window + 1 if smooth_window % 2 == 0 else smooth_window
  if window > trajectory.shape[0]:
    raise ValueError(
      f"smooth_window={window} exceeds trajectory length={trajectory.shape[0]}"
    )
  polyorder = min(polynomial_order, window - 1)
  return signal.savgol_filter(
    trajectory,
    window_length=window,
    polyorder=polyorder,
    deriv=1,
    delta=dt,
    axis=0,
  )


# def build_regression_arrays(
#   embedded_trials: list[np.ndarray],
#   dt: float,
#   smooth_window: int,
# ) -> tuple[np.ndarray, np.ndarray]:
#   """Stack delay states and their derivatives after per-trial estimation."""
#   derivatives = [
#     estimate_derivative(trial, dt=dt, smooth_window=smooth_window)
#     for trial in embedded_trials
#   ]
#   return np.vstack(embedded_trials), np.vstack(derivatives)


# def cap_rows(
#   x: np.ndarray,
#   y: np.ndarray,
#   max_samples: int | None,
# ) -> tuple[np.ndarray, np.ndarray]:
#   """Keep evenly spaced rows when a deterministic training cap is requested."""
#   if max_samples is None or x.shape[0] <= max_samples:
#     return x, y
#   indices = np.linspace(0, x.shape[0] - 1, num=max_samples, dtype=int)
#   return x[indices], y[indices]


# def selected_equations(model) -> list[str]:
#   equations = model.sympy()
#   if isinstance(equations, list):
#     return [str(equation) for equation in equations]
#   return [str(equations)]


# def metric_rows(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict[str, float | int | str]]:
#   rows: list[dict[str, float | int | str]] = []
#   for target in range(y_true.shape[1]):
#     mse = float(mean_squared_error(y_true[:, target], y_pred[:, target]))
#     rows.append(
#       {
#         "target": f"dx{target}_dt",
#         "target_index": target,
#         "r2": float(r2_score(y_true[:, target], y_pred[:, target])),
#         "mse": mse,
#         "rmse": float(np.sqrt(mse)),
#       }
#     )

#   mse = float(mean_squared_error(y_true, y_pred))
#   rows.append(
#     {
#       "target": "all_uniform_average",
#       "target_index": -1,
#       "r2": float(r2_score(y_true, y_pred, multioutput="uniform_average")),
#       "mse": mse,
#       "rmse": float(np.sqrt(mse)),
#     }
#   )
#   return rows


# def parse_operators(value: str) -> list[str]:
#   return [operator.strip() for operator in value.split(",") if operator.strip()]


# def choose_trials(
#   data: TrialData,
#   table: dict,
#   dataset: str,
#   value: str | None,
#   max_trials: int | None,
# ) -> list[int]:
#   if value:
#     return parse_trials(value, data.n_trials, max_trials=None)
#   if dataset == "fixation":
#     trials = select_valid_trials(table, "fixation")
#   elif dataset == "non-fixation":
#     trials = select_valid_trials(table, "non_fixation")
#   else:
#     trials = list(range(data.n_trials))
#   return trials if max_trials is None else trials[:max_trials]


# def main() -> None:
#   parser = argparse.ArgumentParser(
#     description="Fit PySR equations for derivatives of delay-embedded LFP trajectories."
#   )
#   parser.add_argument("--mat-file", type=Path, default=MAT_FILE)
#   parser.add_argument("--channel", type=int, default=0, help="0-based LFP channel.")
#   parser.add_argument(
#     "--dataset",
#     choices=("fixation", "non-fixation", "all"),
#     default="fixation",
#   )
#   parser.add_argument("--trials", default=None, help="Optional comma-separated trial indices.")
#   parser.add_argument("--max-trials", type=int, default=None)
#   parser.add_argument("--test-fraction", type=float, default=0.25)
#   parser.add_argument("--seed", type=int, default=0)
#   parser.add_argument("--n-delays", type=int, default=6)
#   parser.add_argument("--delay", type=int, default=1, help="Samples after downsampling.")
#   parser.add_argument("--downsample", type=int, default=2)
#   parser.add_argument("--lowpass-hz", type=optional_float, default=80.0)
#   parser.add_argument("--normalize", choices=("zscore", "center", "none"), default="zscore")
#   parser.add_argument(
#     "--smooth-window",
#     type=int,
#     default=9,
#     help="Savitzky-Golay derivative window; 0 disables smoothing.",
#   )
#   parser.add_argument("--max-train-samples", type=optional_int, default=10000)
#   parser.add_argument("--niterations", type=int, default=100)
#   parser.add_argument("--maxsize", type=int, default=20)
#   parser.add_argument("--populations", type=int, default=8)
#   parser.add_argument("--binary-operators", default="+,-,*")
#   parser.add_argument("--unary-operators", default="", help="For example: square,sin")
#   parser.add_argument(
#     "--model-selection",
#     choices=("best", "accuracy", "score"),
#     default="best",
#   )
#   parser.add_argument("--timeout-seconds", type=float, default=None)
#   parser.add_argument("--output-dir", type=Path, default=Path("outputs/pysr"))
#   parser.add_argument("--run-id", default="lfp_delay_dynamics")
#   parser.add_argument("--prepare-only", action="store_true")
#   parser.add_argument("--save-dataset", type=Path, default=None)
#   args = parser.parse_args()

#   data = TrialData.load(args.mat_file)
#   table = load_bhv_trial_table(args.mat_file)
#   trials = choose_trials(data, table, args.dataset, args.trials, args.max_trials)
#   train_trials, test_trials = split_trials_sequential(trials, args.test_fraction)
#   dt = args.downsample / data.fs

#   def prepare(selected_trials: list[int]) -> tuple[np.ndarray, np.ndarray]:
#     traces = [
#       preprocess_trace(
#         data.lfp_trace(trial, args.channel),
#         fs=data.fs,
#         downsample=args.downsample,
#         lowpass_hz=args.lowpass_hz,
#         normalize=args.normalize,
#       )
#       for trial in selected_trials
#     ]
#     embedded = delay_embed_trajectories(traces, n_delays=args.n_delays, delay=args.delay)
#     return build_regression_arrays(embedded, dt=dt, smooth_window=args.smooth_window)

#   x_train_full, y_train_full = prepare(train_trials)
#   x_test, y_test = prepare(test_trials)
#   x_train, y_train = cap_rows(
#     x_train_full,
#     y_train_full,
#     max_samples=args.max_train_samples,
#   )
#   variable_names = [f"x{index}" for index in range(args.n_delays)]

#   summary = {
#     "dataset": args.dataset,
#     "channel": args.channel,
#     "sampling_rate_hz": data.fs,
#     "dt_seconds": dt,
#     "n_delays": args.n_delays,
#     "delay_samples_after_downsampling": args.delay,
#     "train_trials": len(train_trials),
#     "test_trials": len(test_trials),
#     "train_samples_before_cap": x_train_full.shape[0],
#     "train_samples_used": x_train.shape[0],
#     "test_samples": x_test.shape[0],
#     "features": variable_names,
#     "targets": [f"dx{index}_dt" for index in range(args.n_delays)],
#   }
#   print(json.dumps(summary, indent=2))

#   if args.save_dataset:
#     args.save_dataset.parent.mkdir(parents=True, exist_ok=True)
#     np.savez_compressed(
#       args.save_dataset,
#       x_train=x_train,
#       y_train=y_train,
#       x_test=x_test,
#       y_test=y_test,
#       train_trials=np.asarray(train_trials),
#       test_trials=np.asarray(test_trials),
#       variable_names=np.asarray(variable_names),
#       metadata=json.dumps(summary),
#     )
#     print(f"saved dataset: {args.save_dataset}")

#   if args.prepare_only:
#     return

#   try:
#     from pysr import PySRRegressor
#   except ImportError as exc:
#     raise ImportError(
#       "PySR is not installed. Run `.venv/bin/python -m pip install pysr`, "
#       "then rerun this command."
#     ) from exc

#   args.output_dir.mkdir(parents=True, exist_ok=True)
#   model = PySRRegressor(
#     niterations=args.niterations,
#     populations=args.populations,
#     maxsize=args.maxsize,
#     binary_operators=parse_operators(args.binary_operators),
#     unary_operators=parse_operators(args.unary_operators),
#     model_selection=args.model_selection,
#     timeout_in_seconds=args.timeout_seconds,
#     random_state=args.seed,
#     output_directory=str(args.output_dir),
#     run_id=args.run_id,
#   )
#   model.fit(x_train, y_train, variable_names=variable_names)

#   train_prediction = np.asarray(model.predict(x_train))
#   test_prediction = np.asarray(model.predict(x_test))
#   if train_prediction.ndim == 1:
#     train_prediction = train_prediction[:, None]
#     test_prediction = test_prediction[:, None]

#   train_metrics = [{**row, "split": "train"} for row in metric_rows(y_train, train_prediction)]
#   test_metrics = [{**row, "split": "test"} for row in metric_rows(y_test, test_prediction)]
#   metrics = train_metrics + test_metrics
#   metrics_path = args.output_dir / f"{args.run_id}_metrics.csv"
#   with metrics_path.open("w", newline="") as file:
#     writer = csv.DictWriter(file, fieldnames=list(metrics[0]))
#     writer.writeheader()
#     writer.writerows(metrics)

#   equations = selected_equations(model)
#   equations_path = args.output_dir / f"{args.run_id}_equations.txt"
#   equations_path.write_text(
#     "\n".join(
#       f"dx{index}/dt = {equation}" for index, equation in enumerate(equations)
#     )
#     + "\n"
#   )
#   summary_path = args.output_dir / f"{args.run_id}_config.json"
#   summary_path.write_text(json.dumps({**summary, **vars(args)}, indent=2, default=str) + "\n")

#   overall_test = test_metrics[-1]
#   print("\nSelected equations:")
#   print(equations_path.read_text(), end="")
#   print(
#     f"\ntest R2={overall_test['r2']:.4f}, "
#     f"test RMSE={overall_test['rmse']:.4f}"
#   )
#   print(f"saved metrics: {metrics_path}")
#   print(f"saved equations: {equations_path}")
#   print(f"PySR run files: {args.output_dir / args.run_id}")


# if __name__ == "__main__":
#   main()
