from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Project imports
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from load_data.convert import (
  LFP_AMPLITUDE_UNIT,
  MAT_FILE,
  TrialData,
  load_bhv_trial_table,
)
from load_data.preprocessing import channel_traces
from load_data.trial_selection import select_valid_trials


def plot_filter_comparison(
  trial_ids: list[int],
  traces_35hz: list[np.ndarray],
  traces_80hz: list[np.ndarray],
  sampling_hz: float,
  channel: int,
  seed: int,
  output_path: Path,
) -> None:
  """Plot paired, already-filtered 35 Hz and 80 Hz traces.

  Args:
    trial_ids: Original MATLAB trial identifiers.
    traces_35hz: Traces low-pass filtered at 35 Hz, in microvolts.
    traces_80hz: Traces low-pass filtered at 80 Hz, in microvolts.
    sampling_hz: Processed sampling frequency in hertz.
    channel: Zero-based LFP channel index.
    seed: Random seed used to select the displayed trials.
    output_path: Destination PNG path.
  """
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  if not (len(trial_ids) == len(traces_35hz) == len(traces_80hz)):
    raise ValueError("Trial IDs and filtered trace lists must have equal lengths.")
  if not trial_ids:
    raise ValueError("At least one trial is required for plotting.")

  fig, axes = plt.subplots(
    len(trial_ids),
    2,
    figsize=(15, max(3.2, 2.5 * len(trial_ids))),
    sharex="col",
    sharey=True,
    squeeze=False,
  )

  for row, (trial_id, trace_35hz, trace_80hz) in enumerate(
    zip(trial_ids, traces_35hz, traces_80hz)
  ):
    n_samples = min(trace_35hz.size, trace_80hz.size)
    filtered_35hz = trace_35hz[:n_samples]
    filtered_80hz = trace_80hz[:n_samples]
    time_s = np.arange(n_samples) / sampling_hz

    axis_35hz, axis_80hz = axes[row]
    axis_35hz.plot(
      time_s,
      filtered_35hz,
      color="#c13d4a",
      linewidth=0.75,
    )
    axis_80hz.plot(
      time_s,
      filtered_80hz,
      color="#187c78",
      linewidth=0.75,
    )
    axis_35hz.set_ylabel(f"Trial {trial_id}\n{LFP_AMPLITUDE_UNIT}")
    axis_35hz.grid(alpha=0.16, linewidth=0.5)
    axis_80hz.grid(alpha=0.16, linewidth=0.5)

  axes[0, 0].set_title("35 Hz low-pass")
  axes[0, 1].set_title("80 Hz low-pass")
  axes[-1, 0].set_xlabel("Time (s)")
  axes[-1, 1].set_xlabel("Time (s)")
  fig.suptitle(
    f"Random Valid Fixation Trials, Channel {channel}: "
    f"35 Hz vs 80 Hz Low-Pass (seed {seed})",
    fontsize=14,
  )
  fig.tight_layout(rect=(0, 0, 1, 0.98))

  output_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(output_path, dpi=180)
  plt.close(fig)


def main() -> None:
  """Prepare filtered fixation traces and save their visual comparison."""
  parser = argparse.ArgumentParser(
    description="Compare 35 Hz and 80 Hz low-pass-filtered fixation LFP traces."
  )
  parser.add_argument("--mat-file", type=Path, default=MAT_FILE)
  parser.add_argument("--channel", type=int, default=0)
  parser.add_argument("--downsample", type=int, default=2)
  parser.add_argument("--max-trials", type=int, default=4)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument(
    "--output",
    type=Path,
    default=None,
  )
  args = parser.parse_args()

  data = TrialData.load(args.mat_file)
  table = load_bhv_trial_table(args.mat_file)
  valid_trial_ids = select_valid_trials(table, "fixation")
  if args.max_trials < 1:
    raise ValueError("--max-trials must be at least 1.")
  rng = np.random.default_rng(args.seed)
  trial_ids = rng.choice(
    valid_trial_ids,
    size=min(args.max_trials, len(valid_trial_ids)),
    replace=False,
  ).tolist()
  traces_35hz = channel_traces(
    data,
    channel=args.channel,
    trials=trial_ids,
    downsample=args.downsample,
    lowpass_hz=35.0,
    normalize="none",
  )
  traces_80hz = channel_traces(
    data,
    channel=args.channel,
    trials=trial_ids,
    downsample=args.downsample,
    lowpass_hz=80.0,
    normalize="none",
  )
  output_path = args.output or Path(
    f"outputs/channel_analysis/fixation_ch{args.channel}_random_lowpass35_vs80.png"
  )
  plot_filter_comparison(
    trial_ids,
    traces_35hz,
    traces_80hz,
    sampling_hz=data.fs / args.downsample,
    channel=args.channel,
    seed=args.seed,
    output_path=output_path,
  )
  print(f"valid fixation trials plotted: {trial_ids}")
  print(f"processed sampling rate: {data.fs / args.downsample:g} Hz")
  print(f"saved: {output_path}")


if __name__ == "__main__":
  main()
