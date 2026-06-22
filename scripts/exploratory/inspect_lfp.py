from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from convert import MAT_FILE, TrialData


def channel_stats(data: TrialData, trial: int) -> list[dict[str, float]]:
  rows = []
  for channel in range(data.n_channels):
    x = data.lfp_trace(trial, channel)
    rows.append(
      {
        "channel": channel,
        "n_samples": x.size,
        "duration_s": x.size / data.fs,
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "rms": float(np.sqrt(np.mean(x**2))),
      }
    )
  return rows


def print_summary(data: TrialData, trial: int, max_trials: int) -> None:
  print(f"session: {data.sessname}")
  print(f"fs: {data.fs} Hz")
  print(f"trials: {data.n_trials}")
  print(f"channels: {data.n_channels}")
  print(f"lfp MATLAB/Python container shape: {data.lfp.shape} ({data.lfp.dtype})")
  print(f"grouping vars: {data.grouping_vars}")
  print()

  print(f"first {max_trials} trial lengths for channel 0:")
  for trial_idx in range(min(max_trials, data.n_trials)):
    x = data.lfp_trace(trial_idx, 0)
    print(f"  trial {trial_idx:4d}: {x.size:5d} samples, {x.size / data.fs:7.3f} s")
  print()

  print(f"across-channel stats for trial {trial}:")
  print("  ch  samples  duration_s      mean       std       rms")
  for row in channel_stats(data, trial):
    print(
      f"  {row['channel']:2.0f}  {row['n_samples']:7.0f}  {row['duration_s']:10.3f}"
      f"  {row['mean']:8.2f}  {row['std']:8.2f}  {row['rms']:8.2f}"
    )


def save_plots(data: TrialData, trial: int, out_dir: Path, shared_y: bool) -> None:
  try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
  except ImportError as exc:
    raise ImportError(
      "Plotting needs matplotlib. Install it with: python3 -m pip install matplotlib"
    ) from exc

  out_dir.mkdir(parents=True, exist_ok=True)
  traces = [data.lfp_trace(trial, channel) for channel in range(data.n_channels)]
  min_len = min(trace.size for trace in traces)
  traces = [trace[:min_len] for trace in traces]
  t = np.arange(min_len) / data.fs

  fig, axes = plt.subplots(8, 4, figsize=(16, 12), sharex=True, sharey=shared_y)
  for channel, ax in enumerate(axes.ravel()):
    ax.plot(t, traces[channel], linewidth=0.7)
    ax.set_title(f"Ch {channel}", fontsize=9)
  fig.supxlabel("Time (s)")
  fig.supylabel("LFP")
  fig.tight_layout()
  suffix = "_shared_y" if shared_y else ""
  grid_path = out_dir / f"trial_{trial}_all_channels{suffix}.png"
  fig.savefig(grid_path, dpi=180)
  plt.close(fig)

  corr = np.corrcoef(np.vstack(traces))
  fig, ax = plt.subplots(figsize=(8, 7))
  im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
  ax.set_title(f"Channel correlation, trial {trial}")
  ax.set_xlabel("Channel")
  ax.set_ylabel("Channel")
  fig.colorbar(im, ax=ax, label="Correlation")
  fig.tight_layout()
  corr_path = out_dir / f"trial_{trial}_channel_correlation.png"
  fig.savefig(corr_path, dpi=180)
  plt.close(fig)

  print(f"saved: {grid_path}")
  print(f"saved: {corr_path}")


def main() -> None:
  parser = argparse.ArgumentParser(description="Inspect LFP trial/channel structure.")
  parser.add_argument("--mat-file", type=Path, default=MAT_FILE)
  parser.add_argument("--trial", type=int, default=0)
  parser.add_argument("--max-trials", type=int, default=20)
  parser.add_argument("--plot", action="store_true")
  parser.add_argument("--shared-y", action="store_true", help="Use the same y-axis for all channels.")
  parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
  args = parser.parse_args()

  data = TrialData.load(args.mat_file)
  print_summary(data, trial=args.trial, max_trials=args.max_trials)
  if args.plot:
    save_plots(data, trial=args.trial, out_dir=args.out_dir, shared_y=args.shared_y)


if __name__ == "__main__":
  main()
