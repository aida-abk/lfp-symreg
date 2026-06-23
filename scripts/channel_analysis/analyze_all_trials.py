from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from load_data.convert import MAT_FILE, TrialData


def summarize_trials(data: TrialData, corr_downsample: int, min_samples: int):
  n_trials = data.n_trials
  n_channels = data.n_channels
  lengths = np.zeros(n_trials, dtype=int)
  means = np.zeros((n_trials, n_channels), dtype=float)
  stds = np.zeros((n_trials, n_channels), dtype=float)
  rms = np.zeros((n_trials, n_channels), dtype=float)
  ptp = np.zeros((n_trials, n_channels), dtype=float)
  corr_sum = np.zeros((n_channels, n_channels), dtype=float)
  corr_count = 0

  for trial in range(n_trials):
    traces = [data.lfp_trace(trial, channel) for channel in range(n_channels)]
    length = min(trace.size for trace in traces)
    lengths[trial] = length

    for channel, trace in enumerate(traces):
      means[trial, channel] = np.mean(trace)
      stds[trial, channel] = np.std(trace)
      rms[trial, channel] = np.sqrt(np.mean(trace**2))
      ptp[trial, channel] = np.ptp(trace)

      trimmed = np.vstack([trace[:length:corr_downsample] for trace in traces])
      corr = np.corrcoef(trimmed)
      if np.all(np.isfinite(corr)):
        corr_sum += corr
        corr_count += 1


  avg_corr = corr_sum / corr_count if corr_count else np.full((n_channels, n_channels), np.nan)
  return {
    "lengths": lengths,
    "means": means,
    "stds": stds,
    "rms": rms,
    "ptp": ptp,
    "avg_corr": avg_corr,
    "corr_count": corr_count,
  }


def save_channel_summary(summary: dict, data: TrialData, out_dir: Path) -> Path:
  path = out_dir / "all_trials_channel_summary.csv"
  with path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
      [
        "channel",
        "mean_rms",
        "median_rms",
        "mean_std",
        "median_std",
        "mean_peak_to_peak",
        "median_peak_to_peak",
      ]
    )
    for channel in range(data.n_channels):
      writer.writerow(
        [
          channel,
          np.mean(summary["rms"][:, channel]),
          np.median(summary["rms"][:, channel]),
          np.mean(summary["stds"][:, channel]),
          np.median(summary["stds"][:, channel]),
          np.mean(summary["ptp"][:, channel]),
          np.median(summary["ptp"][:, channel]),
        ]
      )
  return path


def save_plots(summary: dict, data: TrialData, out_dir: Path) -> list[Path]:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  paths = []
  channels = np.arange(data.n_channels)

  fig, ax = plt.subplots(figsize=(10, 4))
  ax.hist(summary["lengths"] / data.fs, bins=40)
  ax.set_title("Trial Duration Distribution")
  ax.set_xlabel("Duration (s)")
  ax.set_ylabel("Number of trials")
  fig.tight_layout()
  path = out_dir / "all_trials_duration_hist.png"
  fig.savefig(path, dpi=180)
  plt.close(fig)
  paths.append(path)

  fig, ax = plt.subplots(figsize=(11, 4))
  ax.bar(channels, np.mean(summary["rms"], axis=0))
  ax.set_title("Mean RMS by Channel Across All Trials")
  ax.set_xlabel("Channel")
  ax.set_ylabel("Mean RMS")
  ax.set_xticks(channels)
  fig.tight_layout()
  path = out_dir / "all_trials_mean_rms_by_channel.png"
  fig.savefig(path, dpi=180)
  plt.close(fig)
  paths.append(path)

  fig, ax = plt.subplots(figsize=(11, 4))
  ax.boxplot([summary["rms"][:, channel] for channel in channels], positions=channels, showfliers=False)
  ax.set_title("RMS Distribution by Channel Across All Trials")
  ax.set_xlabel("Channel")
  ax.set_ylabel("RMS")
  ax.set_xticks(channels)
  fig.tight_layout()
  path = out_dir / "all_trials_rms_boxplot_by_channel.png"
  fig.savefig(path, dpi=180)
  plt.close(fig)
  paths.append(path)

  fig, ax = plt.subplots(figsize=(8, 7))
  im = ax.imshow(summary["avg_corr"], vmin=-1, vmax=1, cmap="coolwarm")
  ax.set_title(f"Average Channel Correlation Across {summary['corr_count']} Trials")
  ax.set_xlabel("Channel")
  ax.set_ylabel("Channel")
  fig.colorbar(im, ax=ax, label="Correlation")
  fig.tight_layout()
  path = out_dir / "all_trials_average_channel_correlation.png"
  fig.savefig(path, dpi=180)
  plt.close(fig)
  paths.append(path)

  return paths


def main() -> None:
  parser = argparse.ArgumentParser(description="Analyze LFP channel structure across all trials.")
  parser.add_argument("--mat-file", type=Path, default=MAT_FILE)
  parser.add_argument("--out-dir", type=Path, default=Path("outputs/channel_analysis"))
  parser.add_argument("--corr-downsample", type=int, default=10)
  parser.add_argument("--min-samples", type=int, default=100)
  args = parser.parse_args()

  data = TrialData.load(args.mat_file)
  args.out_dir.mkdir(parents=True, exist_ok=True)
  summary = summarize_trials(
    data,
    corr_downsample=args.corr_downsample,
    min_samples=args.min_samples,
  )

  npz_path = args.out_dir / "all_trials_channel_analysis.npz"
  np.savez_compressed(npz_path, **summary, fs=data.fs)
  csv_path = save_channel_summary(summary, data=data, out_dir=args.out_dir)

  print(f"trials: {data.n_trials}, channels: {data.n_channels}, fs: {data.fs} Hz", flush=True)
  print(
    "duration range: "
    f"{summary['lengths'].min() / data.fs:.3f}s to {summary['lengths'].max() / data.fs:.3f}s",
    flush=True,
  )
  print(f"correlation trials used: {summary['corr_count']}", flush=True)
  print(f"saved: {npz_path}", flush=True)
  print(f"saved: {csv_path}", flush=True)

  mean_rms = np.mean(summary["rms"], axis=0)
  top = np.argsort(mean_rms)[-5:][::-1]
  print("top 5 channels by mean RMS:", flush=True)
  for channel in top:
    print(f"  channel {channel}: mean RMS {mean_rms[channel]:.3f}", flush=True)



if __name__ == "__main__":
  main()
