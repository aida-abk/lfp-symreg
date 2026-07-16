from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

# Project imports
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from load_data.convert import MAT_FILE, TrialData, load_bhv_trial_table
from load_data.preprocessing import channel_traces
from load_data.trial_selection import select_valid_trials


def plot_lowpass_trials(
  traces: list[np.ndarray],
  trial_ids: list[int],
  sampling_hz: float,
  lowpass_hz: float,
  channel: int,
  normalize: str,
  output_path: Path,
  columns: int = 5,
) -> None:
  """Plot low-pass-filtered fixation trials in a grid.

  Args:
    traces: Already preprocessed trial traces. No filtering is done here.
    trial_ids: Original zero-based trial identifiers corresponding to traces.
    sampling_hz: Sampling rate after any downsampling, in hertz.
    lowpass_hz: Low-pass cutoff used before plotting, in hertz.
    channel: Zero-based LFP channel index.
    normalize: Normalization mode used before plotting.
    output_path: Destination PNG path.
    columns: Number of subplot columns.
  """
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  if len(traces) != len(trial_ids):
    raise ValueError("traces and trial_ids must have the same length.")
  if not traces:
    raise ValueError("At least one trace is required.")
  if columns < 1:
    raise ValueError("columns must be at least 1.")

  rows = math.ceil(len(traces) / columns)
  fig, axes = plt.subplots(
    rows,
    columns,
    figsize=(3.15 * columns, 1.45 * rows),
    sharex=True,
    sharey=True,
    squeeze=False,
  )
  ylabel = "Z-scored LFP" if normalize == "zscore" else "LFP (microvolts)"

  for axis in axes.ravel():
    axis.set_visible(False)

  for axis, trial_id, trace in zip(axes.ravel(), trial_ids, traces):
    time_s = np.arange(trace.size) / sampling_hz
    axis.plot(time_s, trace, linewidth=0.75)
    axis.set_title(f"Trial {trial_id}", fontsize=8)
    axis.set_visible(True)

  fig.suptitle(
    (
      f"Fixation Trials, Channel {channel}: {lowpass_hz:g} Hz Low-Pass, "
      f"{sampling_hz:g} Hz Sampling"
    ),
    fontsize=14,
  )
  fig.supxlabel("Time (s)")
  fig.supylabel(ylabel)
  fig.tight_layout(rect=(0.02, 0.03, 1, 0.97))

  output_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(output_path, dpi=180)
  plt.close(fig)


def main() -> None:
  """Filter selected fixation trials and save a grid visualization."""
  parser = argparse.ArgumentParser(
    description="Visualize low-pass-filtered valid fixation LFP trials."
  )
  parser.add_argument("--mat-file", type=Path, default=MAT_FILE)
  parser.add_argument("--channel", type=int, default=0)
  parser.add_argument("--lowpass-hz", type=float, required=True)
  parser.add_argument("--downsample", type=int, default=2)
  parser.add_argument("--normalize", choices=("none", "center", "zscore"), default="zscore")
  parser.add_argument("--max-trials", type=int, default=None)
  parser.add_argument(
    "--trial-ids",
    type=str,
    default=None,
    help="Optional comma-separated trial IDs. Overrides automatic valid fixation selection.",
  )
  parser.add_argument("--columns", type=int, default=5)
  parser.add_argument("--output", type=Path, default=None)
  args = parser.parse_args()

  data = TrialData.load(args.mat_file)
  if args.trial_ids is not None:
    trial_ids = [int(value) for value in args.trial_ids.split(",") if value.strip()]
  else:
    table = load_bhv_trial_table(args.mat_file)
    trial_ids = select_valid_trials(table, "fixation")
  if args.max_trials is not None and args.trial_ids is None:
    if args.max_trials < 1:
      raise ValueError("--max-trials must be at least 1.")
    trial_ids = trial_ids[: args.max_trials]

  traces = channel_traces(
    data,
    channel=args.channel,
    trials=trial_ids,
    downsample=args.downsample,
    lowpass_hz=args.lowpass_hz,
    normalize=args.normalize,
  )
  sampling_hz = data.fs / args.downsample
  output_path = args.output or Path(
    "outputs/filter/"
    f"fixation_ch{args.channel}_lowpass{args.lowpass_hz:g}_"
    f"ds{args.downsample}_{args.normalize}.png"
  )
  plot_lowpass_trials(
    traces,
    trial_ids,
    sampling_hz=sampling_hz,
    lowpass_hz=args.lowpass_hz,
    channel=args.channel,
    normalize=args.normalize,
    output_path=output_path,
    columns=args.columns,
  )
  print(f"valid fixation trials plotted: {len(trial_ids)}")
  print(f"raw sampling rate: {data.fs:g} Hz")
  print(f"processed sampling rate: {sampling_hz:g} Hz")
  print(f"saved: {output_path}")


if __name__ == "__main__":
  main()
