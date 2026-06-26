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

from filter.fixation_filter import fixation_trials
from load_data.convert import MAT_FILE, TrialData


def fixation_trace_matrix(
  data: TrialData,
  channel: int,
  trials: list[int],
  downsample: int,
) -> tuple[np.ndarray, int]:
  """Return equal-length fixation traces for one channel."""
  traces = [data.lfp_trace(trial, channel)[::downsample] for trial in trials]
  min_len = min(trace.size for trace in traces)
  cropped = [trace[:min_len] for trace in traces]
  return np.vstack(cropped), min_len


def save_trial_summary(
  path: Path,
  data: TrialData,
  trials: list[int],
  original_lengths: list[int],
  cropped_samples: int,
  downsample: int,
) -> None:
  """Save the trial indices and lengths represented in a matrix."""
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
      [
        "matrix_row",
        "trial",
        "trial_rows",
        "original_samples",
        "downsampled_samples",
        "cropped_samples_used",
        "cropped_duration_s",
      ]
    )
    for matrix_row, (trial, original_samples) in enumerate(zip(trials, original_lengths)):
      writer.writerow(
        [
          matrix_row,
          trial,
          " ".join(str(int(x)) for x in data.trial_row(trial)),
          original_samples,
          int(np.ceil(original_samples / downsample)),
          cropped_samples,
          cropped_samples * downsample / data.fs,
        ]
      )


def save_plots(
  corr: np.ndarray,
  out_dir: Path,
  label: str,
  filename: str,
) -> list[Path]:
  """Save a fixation-trial correlation heatmap."""
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  out_dir.mkdir(parents=True, exist_ok=True)
  paths = []

  fig, ax = plt.subplots(figsize=(8, 7))
  im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
  ax.set_title(label)
  ax.set_xlabel("Fixation trial index")
  ax.set_ylabel("Fixation trial index")
  fig.colorbar(im, ax=ax, label="Correlation")
  fig.tight_layout()
  path = out_dir / filename
  fig.savefig(path, dpi=180)
  plt.close(fig)
  paths.append(path)

  return paths


def off_diagonal_values(corr: np.ndarray) -> np.ndarray:
  """Return matrix entries excluding the self-correlation diagonal."""
  return corr[~np.eye(corr.shape[0], dtype=bool)]


def save_channel_summary(path: Path, rows: list[dict[str, float | int]]) -> None:
  """Save per-channel fixation similarity summaries."""
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="") as f:
    writer = csv.DictWriter(
      f,
      fieldnames=[
        "channel",
        "cropped_samples",
        "mean_off_diagonal_correlation",
        "median_off_diagonal_correlation",
      ],
    )
    writer.writeheader()
    writer.writerows(rows)


def run_one_channel(args, data: TrialData, trials: list[int]) -> None:
  """Run fixation-trial similarity analysis for one channel."""
  original_lengths = [data.lfp_trace(trial, args.channel).size for trial in trials]
  matrix, cropped_samples = fixation_trace_matrix(
    data,
    channel=args.channel,
    trials=trials,
    downsample=args.downsample,
  )
  corr = np.corrcoef(matrix)

  npz_path = args.out_dir / f"fixation_trial_similarity_ch{args.channel}.npz"
  csv_path = args.out_dir / f"fixation_trial_similarity_ch{args.channel}.csv"
  if args.save_npz:
    np.savez_compressed(
      npz_path,
      trials=np.asarray(trials),
      matrix=matrix,
      corr=corr,
      channel=args.channel,
      fs=data.fs,
      downsample=args.downsample,
      cropped_samples=cropped_samples,
    )
  save_trial_summary(
    csv_path,
    data=data,
    trials=trials,
    original_lengths=original_lengths,
    cropped_samples=cropped_samples,
    downsample=args.downsample,
  )

  off_diag = off_diagonal_values(corr)
  print(f"fixation trials used: {len(trials)}")
  print(f"channel: {args.channel}")
  print(f"original sample range: {min(original_lengths)} to {max(original_lengths)}")
  print(f"cropped/downsampled samples used: {cropped_samples}")
  print(f"mean off-diagonal correlation: {np.mean(off_diag):.4f}")
  print(f"median off-diagonal correlation: {np.median(off_diag):.4f}")
  print(f"saved: {csv_path}")
  if args.save_npz:
    print(f"saved: {npz_path}")

  if not args.no_plots:
    for path in save_plots(
      corr,
      out_dir=args.out_dir,
      label=f"Fixation Trial Similarity, Ch {args.channel}",
      filename=f"fixation_trial_correlation_ch{args.channel}.png",
    ):
      print(f"saved: {path}")


def run_all_channels(args, data: TrialData, trials: list[int]) -> None:
  """Run fixation-trial similarity analysis for every channel."""
  corrs = []
  rows = []
  cropped_samples_by_channel = []

  for channel in range(data.n_channels):
    matrix, cropped_samples = fixation_trace_matrix(
      data,
      channel=channel,
      trials=trials,
      downsample=args.downsample,
    )
    corr = np.corrcoef(matrix)
    corrs.append(corr)
    cropped_samples_by_channel.append(cropped_samples)
    off_diag = off_diagonal_values(corr)
    rows.append(
      {
        "channel": channel,
        "cropped_samples": cropped_samples,
        "mean_off_diagonal_correlation": float(np.mean(off_diag)),
        "median_off_diagonal_correlation": float(np.median(off_diag)),
      }
    )

  corr_stack = np.stack(corrs, axis=0)
  avg_corr = np.mean(corr_stack, axis=0)
  summary_path = args.out_dir / "fixation_trial_similarity_all_channels_summary.csv"
  npz_path = args.out_dir / "fixation_trial_similarity_all_channels.npz"
  save_channel_summary(summary_path, rows)
  if args.save_npz:
    np.savez_compressed(
      npz_path,
      trials=np.asarray(trials),
      corr_by_channel=corr_stack,
      avg_corr=avg_corr,
      fs=data.fs,
      downsample=args.downsample,
      cropped_samples_by_channel=np.asarray(cropped_samples_by_channel),
    )

  avg_off_diag = off_diagonal_values(avg_corr)
  print(f"fixation trials used: {len(trials)}")
  print(f"channels used: {data.n_channels}")
  print(f"mean off-diagonal correlation in channel-averaged matrix: {np.mean(avg_off_diag):.4f}")
  print(f"median off-diagonal correlation in channel-averaged matrix: {np.median(avg_off_diag):.4f}")
  print(f"saved: {summary_path}")
  if args.save_npz:
    print(f"saved: {npz_path}")

  if not args.no_plots:
    for path in save_plots(
      avg_corr,
      out_dir=args.out_dir,
      label="Fixation Trial Similarity, Averaged Across Channels",
      filename="fixation_trial_correlation_all_channels_average.png",
    ):
      print(f"saved: {path}")


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Compare LFP similarity across fixation trials."
  )
  parser.add_argument("--mat-file", type=Path, default=MAT_FILE)
  parser.add_argument("--channel", type=int, default=0)
  parser.add_argument("--all-channels", action="store_true")
  parser.add_argument("--max-trials", type=int, default=None)
  parser.add_argument("--downsample", type=int, default=1)
  parser.add_argument("--out-dir", type=Path, default=Path("outputs/channel_analysis"))
  parser.add_argument("--no-plots", action="store_true")
  parser.add_argument(
    "--save-npz",
    action="store_true",
    help="Also cache full trace/correlation arrays in compressed NumPy format.",
  )
  args = parser.parse_args()

  data = TrialData.load(args.mat_file)
  trials = fixation_trials(data)
  if args.max_trials is not None:
    trials = trials[: args.max_trials]

  args.out_dir.mkdir(parents=True, exist_ok=True)
  if args.all_channels:
    run_all_channels(args, data=data, trials=trials)
  else:
    run_one_channel(args, data=data, trials=trials)


if __name__ == "__main__":
  main()
