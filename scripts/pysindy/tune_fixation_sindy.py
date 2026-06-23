from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
from pathlib import Path

import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

from load_data.convert import MAT_FILE, TrialData
from filter.get_fixed_trials import fixation_trials
from lfp_sindy import delay_embed_trials, fit_pysindy


def parse_int_list(value: str) -> list[int]:
  return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_float_list(value: str) -> list[float]:
  return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_lowpass_list(value: str) -> list[float | None]:
  values = []
  for part in value.split(","):
    part = part.strip().lower()
    if not part:
      continue
    values.append(None if part in {"none", "0"} else float(part))
  return values


def preprocess_trace(
  trace: np.ndarray,
  fs: float,
  downsample: int,
  lowpass_hz: float | None,
  normalize: str,
) -> np.ndarray:
  x = np.asarray(trace, dtype=float).squeeze()
  x = signal.detrend(x, type="constant")

  if lowpass_hz is not None:
    nyquist = fs / 2
    if not 0 < lowpass_hz < nyquist:
      raise ValueError(f"lowpass_hz must be between 0 and {nyquist}, got {lowpass_hz}")
    sos = signal.butter(4, lowpass_hz / nyquist, btype="lowpass", output="sos")
    x = signal.sosfiltfilt(sos, x)

  x = x[::downsample]

  if normalize == "zscore":
    std = np.std(x)
    if std > 0:
      x = (x - np.mean(x)) / std
  elif normalize == "center":
    x = x - np.mean(x)
  elif normalize != "none":
    raise ValueError(f"Unknown normalize mode: {normalize}")

  return x


def channel_traces(
  data: TrialData,
  channel: int,
  trials: list[int],
  downsample: int,
  lowpass_hz: float | None,
  normalize: str,
) -> list[np.ndarray]:
  return [
    preprocess_trace(
      data.lfp_trace(trial, channel),
      fs=data.fs,
      downsample=downsample,
      lowpass_hz=lowpass_hz,
      normalize=normalize,
    )
    for trial in trials
  ]


def count_terms(model) -> int:
  return int(np.count_nonzero(np.abs(model.coefficients()) > 1e-12))


def split_trials(trials: list[int], test_fraction: float, seed: int) -> tuple[list[int], list[int]]:
  rng = np.random.default_rng(seed)
  shuffled = np.asarray(trials)
  rng.shuffle(shuffled)
  n_test = max(1, int(round(len(shuffled) * test_fraction)))
  test = shuffled[:n_test].tolist()
  train = shuffled[n_test:].tolist()
  if not train:
    raise ValueError("Need at least one training trial.")
  return train, test


def best_rows(rows: list[dict[str, object]], limit: int) -> list[dict[str, object]]:
  ok = [row for row in rows if row["status"] == "ok" and math.isfinite(float(row["test_score_r2"]))]
  return sorted(ok, key=lambda row: (float(row["test_score_r2"]), -int(row["nonzero_terms"])), reverse=True)[:limit]


def main() -> None:
  parser = argparse.ArgumentParser(description="Tune fixation-only LFP SINDy preprocessing and model parameters.")
  parser.add_argument("--mat-file", type=Path, default=MAT_FILE)
  parser.add_argument("--channel", type=int, default=0)
  parser.add_argument("--max-trials", type=int, default=None)
  parser.add_argument("--test-fraction", type=float, default=0.25)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--n-delays-list", default="2,4,6")
  parser.add_argument("--delay-list", default="2,5")
  parser.add_argument("--downsample-list", default="5,10")
  parser.add_argument("--lowpass-list", default="none,20")
  parser.add_argument("--normalize-list", default="zscore,center")
  parser.add_argument("--threshold-list", default="0.05,0.1,0.2")
  parser.add_argument("--degree-list", default="1,2")
  parser.add_argument("--smooth-window-list", default="0,9")
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
  trials = fixation_trials(data)
  if args.max_trials is not None:
    trials = trials[: args.max_trials]
  train_trials, test_trials = split_trials(trials, test_fraction=args.test_fraction, seed=args.seed)

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
      "dataset": "fixation_only",
      "channel": args.channel,
      "n_delays": n_delays,
      "delay_samples_after_downsample": delay,
      "downsample": downsample,
      "lowpass_hz": "none" if lowpass_hz is None else lowpass_hz,
      "normalize": normalize,
      "threshold": threshold,
      "degree": degree,
      "smooth_window": smooth_window,
      "train_trials": len(train_trials),
      "test_trials": len(test_trials),
      "dt": downsample / data.fs,
      "train_score_r2": float("nan"),
      "test_score_r2": float("nan"),
      "nonzero_terms": 0,
      "min_train_embedded_samples": 0,
      "min_test_embedded_samples": 0,
      "status": "ok",
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
      train_embedded = delay_embed_trials(train_traces, n_delays=n_delays, delay=delay)
      test_embedded = delay_embed_trials(test_traces, n_delays=n_delays, delay=delay)
      model = fit_pysindy(
        train_embedded,
        dt=row["dt"],
        threshold=threshold,
        degree=degree,
        smooth_window=smooth_window,
      )
      row["train_score_r2"] = float(model.score(train_embedded, t=row["dt"]))
      row["test_score_r2"] = float(model.score(test_embedded, t=row["dt"]))
      row["nonzero_terms"] = count_terms(model)
      row["min_train_embedded_samples"] = min(x.shape[0] for x in train_embedded)
      row["min_test_embedded_samples"] = min(x.shape[0] for x in test_embedded)
      print(
        f"ok n_delays={n_delays} delay={delay} ds={downsample} "
        f"lp={row['lowpass_hz']} norm={normalize} thr={threshold} deg={degree} "
        f"smooth={smooth_window} test_r2={row['test_score_r2']:.4f} "
        f"terms={row['nonzero_terms']}"
      )
    except Exception as exc:
      row["status"] = "failed"
      row["error"] = str(exc)
      print(
        f"failed n_delays={n_delays} delay={delay} ds={downsample} "
        f"lp={row['lowpass_hz']} norm={normalize} thr={threshold} deg={degree} "
        f"smooth={smooth_window}: {exc}"
      )

    rows.append(row)

  args.out_csv.parent.mkdir(parents=True, exist_ok=True)
  with args.out_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

  top = best_rows(rows, limit=args.top_n)
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
      f"n_delays={best['n_delays']}, delay={best['delay_samples_after_downsample']}, "
      f"downsample={best['downsample']}, lowpass={best['lowpass_hz']}, "
      f"normalize={best['normalize']}, threshold={best['threshold']}, "
      f"degree={best['degree']}, smooth={best['smooth_window']}, "
      f"terms={best['nonzero_terms']}"
    )

    if args.print_best_model or args.equations_out:
      best_lowpass = None if best["lowpass_hz"] == "none" else float(best["lowpass_hz"])
      best_downsample = int(best["downsample"])
      best_traces = channel_traces(
        data,
        channel=args.channel,
        trials=train_trials,
        downsample=best_downsample,
        lowpass_hz=best_lowpass,
        normalize=str(best["normalize"]),
      )
      best_embedded = delay_embed_trials(
        best_traces,
        n_delays=int(best["n_delays"]),
        delay=int(best["delay_samples_after_downsample"]),
      )
      best_model = fit_pysindy(
        best_embedded,
        dt=best_downsample / data.fs,
        threshold=float(best["threshold"]),
        degree=int(best["degree"]),
        smooth_window=int(best["smooth_window"]),
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
