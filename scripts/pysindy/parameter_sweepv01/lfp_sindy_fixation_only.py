from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

from filter.fixation_filter import fixation_trials, limit_trials
from load_data.convert import MAT_FILE, TrialData
from lfp_sindy import channel_lfp_traces, delay_embed_trials, fit_pysindy
from pipeline_utils import count_terms, parse_int_list


def save_trial_filter_csv(data: TrialData, path: Path) -> None:
  fixation = set(fixation_trials(data))
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["trial", "is_included_fixation", "trial_rows", "channel0_samples"])
    for trial in range(data.n_trials):
      writer.writerow(
        [
          trial,
          trial in fixation,
          " ".join(str(int(x)) for x in data.trial_row(trial)),
          data.lfp_trace(trial, 0).size,
        ]
      )


def run_single_fit(args, data: TrialData, trials: list[int]) -> None:
  traces = channel_lfp_traces(data, channel=args.channel, trials=trials, downsample=args.downsample)
  embedded_trials = delay_embed_trials(traces, n_delays=args.n_delays, delay=args.delay)
  dt = args.downsample / data.fs
  trace_lengths = [trace.size for trace in traces]
  embedded_lengths = [embedded.shape[0] for embedded in embedded_trials]

  print(f"channel: {args.channel}")
  print(f"fixation trials used: {len(trials)}")
  print(f"trace lengths after downsampling: min={min(trace_lengths)}, max={max(trace_lengths)}")
  print(f"embedded lengths: min={min(embedded_lengths)}, max={max(embedded_lengths)}")
  print(f"first embedded trial shape: {embedded_trials[0].shape}")
  print(f"dt after downsampling: {dt:.6f} s")

  if args.save_npz:
    args.save_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
      args.save_npz,
      traces=np.asarray(traces, dtype=object),
      embedded_trials=np.asarray(embedded_trials, dtype=object),
      trials=np.asarray(trials),
      channel=args.channel,
      fs=data.fs,
      dt=dt,
      n_delays=args.n_delays,
      delay=args.delay,
      downsample=args.downsample,
    )
    print(f"saved: {args.save_npz}")

  if args.fit:
    model = fit_pysindy(
      embedded_trials,
      dt=dt,
      threshold=args.threshold,
      degree=args.degree,
      smooth_window=args.smooth_window,
    )
    model.print()


def run_delay_sweep(args, data: TrialData, trials: list[int]) -> None:
  traces = channel_lfp_traces(data, channel=args.channel, trials=trials, downsample=args.downsample)
  dt = args.downsample / data.fs
  n_test = max(1, int(round(len(traces) * args.test_fraction)))
  n_train = len(traces) - n_test
  if n_train < 1:
    raise ValueError("Need at least one training trial. Increase --max-trials or lower --test-fraction.")

  train_traces = traces[:n_train]
  test_traces = traces[n_train:]
  rows = []
  for n_delays in parse_int_list(args.n_delays_list):
    train_embedded = delay_embed_trials(train_traces, n_delays=n_delays, delay=args.delay)
    test_embedded = delay_embed_trials(test_traces, n_delays=n_delays, delay=args.delay)
    model = fit_pysindy(
      train_embedded,
      dt=dt,
      threshold=args.threshold,
      degree=args.degree,
      smooth_window=args.smooth_window,
    )
    train_score = float(model.score(train_embedded, t=dt))
    test_score = float(model.score(test_embedded, t=dt))
    terms = count_terms(model)
    row = {
      "channel": args.channel,
      "included_fixation_trials": len(trials),
      "n_delays": n_delays,
      "delay_samples_after_downsample": args.delay,
      "downsample": args.downsample,
      "dt": dt,
      "train_trials": len(train_embedded),
      "test_trials": len(test_embedded),
      "min_train_embedded_samples": min(x.shape[0] for x in train_embedded),
      "min_test_embedded_samples": min(x.shape[0] for x in test_embedded),
      "threshold": args.threshold,
      "degree": args.degree,
      "smooth_window": args.smooth_window,
      "train_score_r2": train_score,
      "test_score_r2": test_score,
      "nonzero_terms": terms,
    }
    rows.append(row)
    print(
      f"n_delays={n_delays:2d} "
      f"train_r2={train_score:8.4f} test_r2={test_score:8.4f} terms={terms:4d}"
    )

  args.out_csv.parent.mkdir(parents=True, exist_ok=True)
  with args.out_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
  print(f"saved: {args.out_csv}")


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Keep only likely fixation trials, delay-embed LFP, and fit PySINDy."
  )
  parser.add_argument("--mat-file", type=Path, default=MAT_FILE)
  parser.add_argument("--channel", type=int, default=0)
  parser.add_argument("--max-trials", type=int, default=None)
  parser.add_argument("--n-delays", type=int, default=6)
  parser.add_argument("--n-delays-list", default=None, help="If set, run a delay sweep instead of one fit.")
  parser.add_argument("--delay", type=int, default=5)
  parser.add_argument("--downsample", type=int, default=10)
  parser.add_argument("--threshold", type=float, default=0.05)
  parser.add_argument("--degree", type=int, default=2)
  parser.add_argument("--smooth-window", type=int, default=9)
  parser.add_argument("--test-fraction", type=float, default=0.25)
  parser.add_argument("--fit", action="store_true")
  parser.add_argument("--save-npz", type=Path, default=None)
  parser.add_argument(
    "--trial-filter-csv",
    type=Path,
    default=Path("outputs/pysindy/fixation_only_trial_filter.csv"),
  )
  parser.add_argument(
    "--out-csv",
    type=Path,
    default=Path("outputs/pysindy/fixation_only_delay_experiment_results.csv"),
  )
  args = parser.parse_args()

  data = TrialData.load(args.mat_file)
  all_fixation_trials = fixation_trials(data)
  trials = limit_trials(all_fixation_trials, args.max_trials)
  save_trial_filter_csv(data, args.trial_filter_csv)

  print(f"total trials: {data.n_trials}")
  print(f"available likely fixation trials: {len(all_fixation_trials)}")
  print(f"using fixation trials: {len(trials)}")
  print(f"saved: {args.trial_filter_csv}")

  if args.n_delays_list:
    run_delay_sweep(args, data=data, trials=trials)
  else:
    run_single_fit(args, data=data, trials=trials)


if __name__ == "__main__":
  main()
