from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
PYSINDY_SCRIPTS = SCRIPTS / "pysindy"
for path in (ROOT, SCRIPTS, PYSINDY_SCRIPTS):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

from filter.fixation_filter import fixation_trials
from load_data.convert import MAT_FILE, TrialData
from scripts.pysindy.pipeline_utils import preprocess_trace


def visualize_fixation_trials_80hz(
  channel: int,
  mat_file: str | Path = MAT_FILE,
  out_dir: str | Path = Path("outputs/filter"),
  downsample: int = 2,
  normalize: str = "zscore",
  window_start: float | None = None,
  window_end: float | None = None,
  columns: int = 5,
  max_trials: int | None = None,
) -> Path:
  """Plot 80 Hz low-pass-filtered fixation trials for one LFP channel.

  Args:
    channel: Zero-based LFP channel index.
    mat_file: Source MATLAB file.
    out_dir: Directory in which to save the figure.
    downsample: Keep every Nth sample after filtering.
    normalize: Per-trial normalization: zscore, center, or none.
    window_start: Optional window start in seconds.
    window_end: Optional window end in seconds.
    columns: Number of subplot columns.
    max_trials: Optional number of fixation trials to plot.

  Returns:
    Path to the saved PNG figure.
  """
  if columns < 1:
    raise ValueError("columns must be >= 1")

  data = TrialData.load(mat_file)
  if not 0 <= channel < data.n_channels:
    raise ValueError(f"channel must be between 0 and {data.n_channels - 1}")

  trials = fixation_trials(data)
  if max_trials is not None:
    trials = trials[:max_trials]
  if not trials:
    raise ValueError("No fixation trials were selected.")

  traces = [
    preprocess_trace(
      data.lfp_trace(trial, channel),
      fs=data.fs,
      downsample=downsample,
      lowpass_hz=80.0,
      normalize=normalize,
      window_start=window_start,
      window_end=window_end,
    )
    for trial in trials
  ]

  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  rows = math.ceil(len(trials) / columns)
  fig, axes = plt.subplots(
    rows,
    columns,
    figsize=(3.2 * columns, 1.8 * rows),
    sharex=True,
    sharey=True,
    squeeze=False,
  )
  processed_fs = data.fs / downsample
  time_offset = 0.0 if window_start is None else window_start

  for axis, trial, trace in zip(axes.ravel(), trials, traces):
    time = time_offset + np.arange(trace.size) / processed_fs
    axis.plot(time, trace, linewidth=0.7)
    axis.set_title(f"Trial {trial}", fontsize=8)

  for axis in axes.ravel()[len(trials):]:
    axis.set_visible(False)

  y_label = "Z-scored LFP" if normalize == "zscore" else "LFP"
  fig.supxlabel("Time (s)")
  fig.supylabel(y_label)
  fig.suptitle(
    f"Fixation Trials, Channel {channel}: 80 Hz Low-Pass, "
    f"{processed_fs:g} Hz Sampling",
    fontsize=13,
  )
  fig.tight_layout(rect=(0.02, 0.02, 1, 0.98))

  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  window_label = (
    ""
    if window_start is None
    else f"_{window_start:g}_{window_end:g}s"
  )
  output_path = out_dir / (
    f"fixation_ch{channel}_lowpass80_ds{downsample}_{normalize}"
    f"{window_label}.png"
  )
  fig.savefig(output_path, dpi=180)
  plt.close(fig)
  return output_path


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Visualize 80 Hz low-pass-filtered fixation trials for one channel."
  )
  parser.add_argument("--mat-file", type=Path, default=MAT_FILE)
  parser.add_argument("--channel", type=int, required=True)
  parser.add_argument("--out-dir", type=Path, default=Path("outputs/filter"))
  parser.add_argument("--downsample", type=int, default=2)
  parser.add_argument(
    "--normalize",
    choices=("zscore", "center", "none"),
    default="zscore",
  )
  parser.add_argument("--window-start", type=float, default=None)
  parser.add_argument("--window-end", type=float, default=None)
  parser.add_argument("--columns", type=int, default=5)
  parser.add_argument("--max-trials", type=int, default=None)
  args = parser.parse_args()

  output_path = visualize_fixation_trials_80hz(
    channel=args.channel,
    mat_file=args.mat_file,
    out_dir=args.out_dir,
    downsample=args.downsample,
    normalize=args.normalize,
    window_start=args.window_start,
    window_end=args.window_end,
    columns=args.columns,
    max_trials=args.max_trials,
  )
  print(f"saved: {output_path}")


if __name__ == "__main__":
  main()
