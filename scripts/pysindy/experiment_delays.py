from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

from load_data.convert import MAT_FILE, TrialData
from lfp_sindy import channel_lfp_traces, delay_embed_trials, fit_pysindy, parse_trials
from pipeline_utils import count_terms, parse_int_list


def main() -> None:
  parser = argparse.ArgumentParser(description="Try multiple delay-embedding dimensions for LFP SINDy.")
  parser.add_argument("--mat-file", type=Path, default=MAT_FILE)
  parser.add_argument("--channel", type=int, default=0)
  parser.add_argument("--trials", default=None, help="Comma-separated 0-based trial indices.")
  parser.add_argument("--max-trials", type=int, default=30)
  parser.add_argument("--test-fraction", type=float, default=0.25)
  parser.add_argument("--n-delays-list", default="2,4,6,8,10,12")
  parser.add_argument("--delay", type=int, default=5, help="Delay in samples after downsampling.")
  parser.add_argument("--downsample", type=int, default=10)
  parser.add_argument("--threshold", type=float, default=0.05)
  parser.add_argument("--degree", type=int, default=2)
  parser.add_argument("--smooth-window", type=int, default=9)
  parser.add_argument("--out-csv", type=Path, default=Path("outputs/pysindy/delay_experiment_results.csv"))
  args = parser.parse_args()

  data = TrialData.load(args.mat_file)
  trials = parse_trials(args.trials, data.n_trials, args.max_trials)
  traces = channel_lfp_traces(data, channel=args.channel, trials=trials, downsample=args.downsample)
  dt = args.downsample / data.fs

  n_test = max(1, int(round(len(traces) * args.test_fraction)))
  n_train = len(traces) - n_test
  if n_train < 1:
    raise ValueError("Need at least one training trial. Increase --max-trials or lower --test-fraction.")

  train_traces = traces[:n_train]
  test_traces = traces[n_train:]
  args.out_csv.parent.mkdir(parents=True, exist_ok=True)

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
    min_train_len = min(x.shape[0] for x in train_embedded)
    min_test_len = min(x.shape[0] for x in test_embedded)
    row = {
      "channel": args.channel,
      "n_delays": n_delays,
      "delay_samples_after_downsample": args.delay,
      "downsample": args.downsample,
      "dt": dt,
      "train_trials": len(train_embedded),
      "test_trials": len(test_embedded),
      "min_train_embedded_samples": min_train_len,
      "min_test_embedded_samples": min_test_len,
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

  with args.out_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
  print(f"saved: {args.out_csv}")


if __name__ == "__main__":
  main()
