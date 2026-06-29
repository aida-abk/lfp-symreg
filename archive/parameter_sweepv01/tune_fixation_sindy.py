from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import eigvals
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
PYSINDY_SCRIPTS = SCRIPTS / "pysindy"
for path in (ROOT, SCRIPTS, PYSINDY_SCRIPTS):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

from load_data.convert import MAT_FILE, TrialData
from load_data.preprocessing import channel_traces
from models.sindy import SINDyConfig, delay_embed_trajectories, fit_sindy_model
from pipeline_utils import (
  count_terms,
  parse_float_list,
  parse_int_list,
  parse_lowpass_list,
  select_trials,
  split_trials_random,
)


def linear_system_matrix(model) -> np.ndarray:
  """Extract A from a fitted degree-1 model x_dot = A x + b."""
  feature_names = model.get_feature_names()
  expected = [f"x{index}" for index in range(model.n_features_in_)]
  feature_indices = []
  for name in expected:
    if name not in feature_names:
      raise ValueError(f"Missing linear feature {name}; got {feature_names}")
    feature_indices.append(feature_names.index(name))
  return np.asarray(model.coefficients(), dtype=float)[:, feature_indices]


def format_eigenvalues(values: np.ndarray) -> str:
  """Serialize complex eigenvalues for storage in a CSV cell."""
  return "; ".join(f"{value.real:.12g}{value.imag:+.12g}j" for value in values)


def eligible_top_rows(rows: list[dict[str, object]], limit: int) -> list[dict[str, object]]:
  """Rank successful rows, excluding dynamically unstable linear models."""
  eligible = [
    row
    for row in rows
    if not row["error"]
    and math.isfinite(float(row["test_score_r2"]))
    and row["linear_stability_status"] != "unstable"
  ]
  return sorted(
    eligible,
    key=lambda row: (float(row["test_score_r2"]), -int(row["nonzero_terms"])),
    reverse=True,
  )[:limit]


def main() -> None:
  parser = argparse.ArgumentParser(description="Tune fixation-only LFP SINDy preprocessing and model parameters.")
  parser.add_argument("--mat-file", type=Path, default=MAT_FILE)
  parser.add_argument("--channel", type=int, default=0)
  parser.add_argument("--max-trials", type=int, default=None)
  parser.add_argument("--test-fraction", type=float, default=0.25)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--n-delays-list", default="2,4,6")
  parser.add_argument("--delay-list", default="1,2,5")
  parser.add_argument("--downsample-list", default="2")
  parser.add_argument("--lowpass-list", default="80")
  parser.add_argument("--normalize-list", default="zscore")
  parser.add_argument("--threshold-list", default="0.05,0.1,0.2")
  parser.add_argument("--degree-list", default="1,2")
  parser.add_argument("--smooth-window-list", default="0")
  parser.add_argument(
    "--max-real-eigenvalue",
    type=float,
    default=0.0,
    help="A degree-1 model is stable only when max Re(lambda) is below this value.",
  )
  parser.add_argument("--out-csv", type=Path, default=Path("outputs/fixation_tuning_results.csv"))
  parser.add_argument("--top-csv", type=Path, default=Path("outputs/fixation_tuning_top_results.csv"))
  parser.add_argument("--top-n", type=int, default=15)
  parser.add_argument(
    "--print-best-model",
    action="store_true",
    help="Refit the best configuration on the training trials and print its equations.",
  )
  parser.add_argument(
    "--equations-out",
    type=Path,
    default=None,
    help="Optional text file for the best model equations.",
  )
  args = parser.parse_args()

  data = TrialData.load(args.mat_file)
  trials = select_trials(data, dataset="fixation", max_trials=args.max_trials)
  train_trials, test_trials = split_trials_random(
    trials,
    test_fraction=args.test_fraction,
    seed=args.seed,
  )
  print(f"random split seed: {args.seed}")
  print(f"training trials: {train_trials}")
  print(f"test trials: {test_trials}")

  rows = []
  grid = itertools.product(
    parse_int_list(args.n_delays_list),
    parse_int_list(args.delay_list),
    parse_int_list(args.downsample_list),
    parse_lowpass_list(args.lowpass_list),
    [part.strip() for part in args.normalize_list.split(",") if part.strip()],
    parse_float_list(args.threshold_list),
    parse_int_list(args.degree_list),
    parse_int_list(args.smooth_window_list),
  )

  for n_delays, delay, downsample, lowpass_hz, normalize, threshold, degree, smooth_window in grid:
    row = {
      "channel": args.channel,
      "n_delays": n_delays,
      "delay_samples_after_downsample": delay,
      "threshold": threshold,
      "degree": degree,
      "random_seed": args.seed,
      "train_trials": len(train_trials),
      "test_trials": len(test_trials),
      "train_score_r2": float("nan"),
      "test_score_r2": float("nan"),
      "train_rmse": float("nan"),
      "test_rmse": float("nan"),
      "nonzero_terms": 0,
      "eigenvalues": "",
      "max_real_eigenvalue": float("nan"),
      "linear_stability_status": "not_applicable",
      "error": "",
    }

    try:
      train_traces = channel_traces(
        data,
        channel=args.channel,
        trials=train_trials,
        downsample=downsample,
        lowpass_hz=lowpass_hz,
        normalize=normalize,
      )
      test_traces = channel_traces(
        data,
        channel=args.channel,
        trials=test_trials,
        downsample=downsample,
        lowpass_hz=lowpass_hz,
        normalize=normalize,
      )
      train_embedded = delay_embed_trajectories(train_traces, n_delays=n_delays, delay=delay)
      test_embedded = delay_embed_trajectories(test_traces, n_delays=n_delays, delay=delay)
      model = fit_sindy_model(
        train_embedded,
        dt=downsample / data.fs,
        config=SINDyConfig(
          threshold=threshold,
          degree=degree,
          smooth_window=smooth_window,
        ),
      )
      dt = downsample / data.fs
      row["train_score_r2"] = float(model.score(train_embedded, t=dt))
      row["test_score_r2"] = float(model.score(test_embedded, t=dt))
      train_mse = float(
        model.score(train_embedded, t=dt, metric=mean_squared_error)
      )
      test_mse = float(
        model.score(test_embedded, t=dt, metric=mean_squared_error)
      )
      row["train_rmse"] = math.sqrt(train_mse)
      row["test_rmse"] = math.sqrt(test_mse)
      row["nonzero_terms"] = count_terms(model)
      if degree == 1:
        eigenvalues = eigvals(linear_system_matrix(model))
        max_real_eigenvalue = float(np.max(eigenvalues.real))
        row["eigenvalues"] = format_eigenvalues(eigenvalues)
        row["max_real_eigenvalue"] = max_real_eigenvalue
        row["linear_stability_status"] = (
          "stable"
          if max_real_eigenvalue < args.max_real_eigenvalue
          else "unstable"
        )
      print(
        f"ok n_delays={n_delays} delay={delay} ds={downsample} "
        f"lp={lowpass_hz} norm={normalize} thr={threshold} deg={degree} "
        f"smooth={smooth_window} test_r2={row['test_score_r2']:.4f} "
        f"test_rmse={row['test_rmse']:.4f} "
        f"stability={row['linear_stability_status']} "
        f"terms={row['nonzero_terms']}"
      )
    except Exception as exc:
      row["error"] = str(exc)
      print(
        f"failed n_delays={n_delays} delay={delay} ds={downsample} "
        f"lp={lowpass_hz} norm={normalize} thr={threshold} deg={degree} "
        f"smooth={smooth_window}: {exc}"
      )

    rows.append(row)

  args.out_csv.parent.mkdir(parents=True, exist_ok=True)
  with args.out_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

  top = eligible_top_rows(rows, limit=args.top_n)
  with args.top_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(top)

  print(f"saved: {args.out_csv}")
  print(f"saved: {args.top_csv}")
  if top:
    best = top[0]
    print(
      "best: "
      f"test_r2={float(best['test_score_r2']):.4f}, "
      f"test_rmse={float(best['test_rmse']):.4f}, "
      f"n_delays={best['n_delays']}, delay={best['delay_samples_after_downsample']}, "
      f"threshold={best['threshold']}, degree={best['degree']}, "
      f"stability={best['linear_stability_status']}, "
      f"max_real_eigenvalue={float(best['max_real_eigenvalue']):.6g}, "
      f"terms={best['nonzero_terms']}"
    )

    if args.print_best_model or args.equations_out:
      best_traces = channel_traces(
        data,
        channel=args.channel,
        trials=train_trials,
        downsample=2,
        lowpass_hz=80.0,
        normalize="zscore",
      )
      best_embedded = delay_embed_trajectories(
        best_traces,
        n_delays=int(best["n_delays"]),
        delay=int(best["delay_samples_after_downsample"]),
      )
      best_model = fit_sindy_model(
        best_embedded,
        dt=2 / data.fs,
        config=SINDyConfig(
          threshold=float(best["threshold"]),
          degree=int(best["degree"]),
          smooth_window=0,
        ),
      )

      if args.print_best_model:
        print("\nbest model equations:")
        best_model.print()

      if args.equations_out:
        equations = [
          f"(x{index})' = {equation}"
          for index, equation in enumerate(best_model.equations())
        ]
        args.equations_out.parent.mkdir(parents=True, exist_ok=True)
        args.equations_out.write_text("\n".join(equations) + "\n")
        print(f"saved: {args.equations_out}")


if __name__ == "__main__":
  main()
