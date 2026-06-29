from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

from load_data.convert import MAT_FILE, TrialData
from models.sindy import (
  SINDyConfig,
  delay_embed_trace as model_delay_embed_trace,
  delay_embed_trajectories,
  fit_sindy_model,
)


def channel_lfp_table(
  data: TrialData,
  channel: int,
  trials: list[int] | None = None,
  downsample: int = 1,
) -> np.ndarray:
  """Return one LFP channel as a trial-by-time NumPy array.

  Use this only when all selected trials have the same sample count.
  """
  traces = channel_lfp_traces(data, channel=channel, trials=trials, downsample=downsample)
  lengths = {trace.size for trace in traces}
  if len(lengths) != 1:
    raise ValueError(f"All traces must have equal length, got lengths {sorted(lengths)}")

  return np.vstack(traces)


def channel_lfp_traces(
  data: TrialData,
  channel: int,
  trials: list[int] | None = None,
  downsample: int = 1,
) -> list[np.ndarray]:
  """Return one LFP channel as a list of per-trial traces."""
  if trials is None:
    trials = list(range(data.n_trials))
  if downsample < 1:
    raise ValueError("downsample must be >= 1")

  return [data.lfp_trace(trial, channel)[::downsample] for trial in trials]


def delay_embed_trace(trace: np.ndarray, n_delays: int, delay: int) -> np.ndarray:
  """Build delay coordinates [x(t), x(t-tau), ..., x(t-(m-1)tau)]."""
  return model_delay_embed_trace(trace, n_delays=n_delays, delay=delay)


def delay_embed_trials(traces: list[np.ndarray], n_delays: int, delay: int) -> list[np.ndarray]:
  """Apply delay embedding to every per-trial trace."""
  return delay_embed_trajectories(traces, n_delays=n_delays, delay=delay)


def fit_pysindy(
  embedded_trials: list[np.ndarray],
  dt: float,
  threshold: float = 0.05,
  degree: int = 2,
  smooth_window: int = 9,
):
  """Fit a SINDy model to a list of embedded trajectories."""
  return fit_sindy_model(
    embedded_trials,
    dt=dt,
    config=SINDyConfig(
      threshold=threshold,
      degree=degree,
      smooth_window=smooth_window,
    ),
  )


def parse_trials(value: str | None, n_trials: int, max_trials: int | None) -> list[int]:
  if value:
    trials = [int(part) for part in value.split(",")]
  else:
    stop = n_trials if max_trials is None else min(max_trials, n_trials)
    trials = list(range(stop))

  bad = [trial for trial in trials if trial < 0 or trial >= n_trials]
  if bad:
    raise ValueError(f"Trial indices out of range: {bad}")
  return trials


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Load one LFP channel, build delay embeddings, and optionally fit PySINDy."
  )
  parser.add_argument("--mat-file", type=Path, default=MAT_FILE)
  parser.add_argument("--channel", type=int, default=0, help="0-based channel index.")
  parser.add_argument("--trials", default=None, help="Comma-separated 0-based trial indices.")
  parser.add_argument("--max-trials", type=int, default=20, help="Used when --trials is omitted.")
  parser.add_argument("--n-delays", type=int, default=8, help="Embedding dimension.")
  parser.add_argument("--delay", type=int, default=5, help="Delay in samples after downsampling.")
  parser.add_argument("--downsample", type=int, default=5, help="Keep every Nth sample.")
  parser.add_argument("--threshold", type=float, default=0.05, help="STLSQ sparsity threshold.")
  parser.add_argument("--degree", type=int, default=2, help="Polynomial library degree.")
  parser.add_argument(
    "--smooth-window",
    type=int,
    default=9,
    help="Odd Savitzky-Golay window for smoothed finite differences. Use 0 to disable.",
  )
  parser.add_argument("--save-npz", type=Path, default=None, help="Optional output .npz path.")
  parser.add_argument("--fit", action="store_true", help="Fit a PySINDy model.")
  args = parser.parse_args()

  data = TrialData.load(args.mat_file)
  trials = parse_trials(args.trials, data.n_trials, args.max_trials)
  traces = channel_lfp_traces(
    data,
    channel=args.channel,
    trials=trials,
    downsample=args.downsample,
  )
  embedded_trials = delay_embed_trials(
    traces,
    n_delays=args.n_delays,
    delay=args.delay,
  )
  dt = args.downsample / data.fs
  trace_lengths = [trace.size for trace in traces]
  embedded_lengths = [embedded.shape[0] for embedded in embedded_trials]

  print(f"channel: {args.channel}, trials used: {len(trials)}")
  print(
    "trace lengths after downsampling: "
    f"min={min(trace_lengths)}, max={max(trace_lengths)}, unique={sorted(set(trace_lengths))[:10]}"
  )
  print(
    "embedded lengths: "
    f"min={min(embedded_lengths)}, max={max(embedded_lengths)}, n_delays={args.n_delays}"
  )
  print(f"first embedded trial shape: {embedded_trials[0].shape}  # samples x delay variables")
  print(f"dt after downsampling: {dt:.6f} s")

  if args.save_npz:
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


if __name__ == "__main__":
  main()
