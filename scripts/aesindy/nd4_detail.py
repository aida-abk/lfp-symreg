"""Resolve nd4's tracking horizon, and show what periodic reinitialisation buys.

Three questions the 500 ms windowed analysis could not answer:

1. Where does tracking actually end? Lead-time scoring put it between 20 ms
   (+0.639) and 50 ms (-0.468), and a 500 ms window averages straight over
   that. Windows of 100 ms stepped every 20 ms resolve it.

2. What does the model look like if it is corrected periodically? A free run
   is the hardest test: one initial condition, then nothing. Restarting from
   the measured state every few seconds is the regime a short-horizon model
   would actually be used in, and it separates "the dynamics are wrong" from
   "the dynamics are right but phase drifts".

3. Does any of this hold across trials, or is it one trajectory? Every
   held-out trial is drawn, because single-trial conclusions have already
   reversed once in this project.

Spectral concentration is omitted from the fine-grained pass: a 100 ms window
is 25 samples at 250 Hz, too short for a meaningful periodogram. It is
reported in the 500 ms analysis instead.

Run with:

    .venv/bin/python scripts/aesindy/nd4_detail.py --case nd4

"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(_PROJECT_ROOT))
_INTERNALS_DIR = _PROJECT_ROOT / "scripts" / "pysindy" / "internals_probes"
if str(_INTERNALS_DIR) not in sys.path:
  sys.path.insert(0, str(_INTERNALS_DIR))
_AESINDY_DIR = _PROJECT_ROOT / "scripts" / "aesindy"
if str(_AESINDY_DIR) not in sys.path:
  sys.path.insert(0, str(_AESINDY_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from load_data.archived_split import (  # noqa: E402
  CHANNEL,
  DOWNSAMPLE,
  load_archived_split,
)
from load_data.convert import MAT_FILE, TrialData  # noqa: E402
from load_data.preprocessing import channel_traces  # noqa: E402
from models.validation import simulate_model_detailed  # noqa: E402

LOWPASS_HZ = 35.0
FINE_WINDOW_S = 0.1
FINE_STEP_S = 0.02
OUTPUT_DIR = _PROJECT_ROOT / "outputs/nd4_horizon"


def fine_metrics(
  simulated: np.ndarray, measured: np.ndarray, fs: float, limit_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Correlation and amplitude in short windows over the start of a run.

  Args:
    simulated: Free-running simulation.
    measured: The measurement it was launched from.
    fs: Processed sampling frequency in hertz.
    limit_s: How far into the run to measure, in seconds.

  Returns:
    ``(centres_s, correlation, amplitude_ratio)`` per window.
  """
  n = min(len(simulated), len(measured), int(round(limit_s * fs)))
  width = max(int(round(FINE_WINDOW_S * fs)), 4)
  step = max(int(round(FINE_STEP_S * fs)), 1)
  centres, corr, amp = [], [], []
  for start in range(0, n - width + 1, step):
    a, b = simulated[start:start + width], measured[start:start + width]
    centres.append((start + width / 2) / fs)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
      corr.append(np.nan)
    else:
      corr.append(float(np.corrcoef(a, b)[0, 1]))
    amp.append(float(np.std(a)) / max(float(np.std(b)), 1e-12))
  return np.array(centres), np.array(corr), np.array(amp)


def reinitialised_run(
  method, trace: np.ndarray, dt: float, horizon_s: float, reinit_s: float,
) -> tuple[np.ndarray, np.ndarray]:
  """Simulate in blocks, restarting from the measured state at each block.

  Each block is an independent free run of ``reinit_s`` seconds launched from
  the measured embedded state at the block boundary, so the model is corrected
  periodically rather than left to drift for the whole trial.

  Args:
    method: Fitted method with ``project``, ``model`` and ``signal_from_state``.
    trace: One held-out preprocessed trace.
    dt: Processed sample interval in seconds.
    horizon_s: Total duration to cover, in seconds.
    reinit_s: Block length between restarts, in seconds.

  Returns:
    ``(simulated, boundaries_s)``. Samples that failed to integrate are ``nan``.
  """
  states = method.project(trace)
  total = min(int(round(horizon_s / dt)), len(states))
  block = max(int(round(reinit_s / dt)), 2)
  simulated = np.full(total, np.nan)
  boundaries = []
  for start in range(0, total, block):
    length = min(block, total - start)
    if length < 2:
      break
    boundaries.append(start * dt)
    simulation = simulate_model_detailed(
      method.model, initial_state=states[start], dt=dt,
      horizon_s=(length - 1) * dt, wall_timeout_s=60.0,
    )
    if not (simulation.completed and simulation.trajectory is not None):
      continue
    signal = method.signal_from_state(simulation.trajectory)
    m = min(len(signal), length)
    simulated[start:start + m] = signal[:m]
  return simulated, np.array(boundaries)


def plot_fine(
  centres: np.ndarray, corr: np.ndarray, amp: np.ndarray, case_label: str,
  output_dir: Path,
) -> Path:
  """Plot correlation and amplitude over the first part of the run.

  Args:
    centres: Window centres in seconds.
    corr: Correlation per trial per window, shape ``(n_trials, n_windows)``.
    amp: Amplitude ratio with the same shape.
    case_label: Configuration label.
    output_dir: Directory to write into.

  Returns:
    Path of the written figure.
  """
  fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
  for ax, values, ylabel, guide in (
    (axes[0], corr, "correlation with this trial", 0.0),
    (axes[1], amp, "amplitude ratio (1.0 = correct)", 1.0),
  ):
    for row in values:
      ax.plot(centres * 1000, row, lw=0.7, alpha=0.35, color="tab:gray")
    median = np.nanmedian(values, axis=0)
    ax.plot(centres * 1000, median, lw=2.2, color="tab:orange", label="median")
    ax.axhline(guide, color="black", ls="--", lw=1.0)
    ax.set_xlabel("time from initial condition (ms)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
  axes[0].set_ylim(-1, 1)
  crossing = None
  median = np.nanmedian(corr, axis=0)
  below = np.where(median < 0.2)[0]
  if below.size:
    crossing = centres[below[0]] * 1000
    axes[0].axvline(crossing, color="tab:red", ls=":", lw=1.4)
    axes[0].annotate(f"drops below 0.2\nat {crossing:.0f} ms",
                     (crossing, 0.6), fontsize=8, color="tab:red",
                     xytext=(8, 0), textcoords="offset points")
  fig.suptitle(f"{case_label}: tracking over the first second, "
               f"{FINE_WINDOW_S*1000:.0f} ms windows, thin lines are trials",
               fontsize=11)
  fig.tight_layout()
  path = output_dir / f"fine_first_second_{case_label}.png"
  fig.savefig(path, dpi=150)
  plt.close(fig)
  return path


def plot_per_trial(
  series: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]], dt: float,
  case_label: str, output_dir: Path, title: str, filename: str,
) -> Path:
  """Draw measured against simulated for every trial, one panel each.

  Args:
    series: Tuples of ``(trial_index, measured, simulated, boundaries_s)``.
      ``boundaries_s`` may be empty for a plain free run.
    dt: Processed sample interval in seconds.
    case_label: Configuration label.
    output_dir: Directory to write into.
    title: Figure title.
    filename: Output filename.

  Returns:
    Path of the written figure.
  """
  n_rows = len(series)
  fig, axes = plt.subplots(n_rows, 1, figsize=(13, 1.9 * n_rows),
                           squeeze=False, sharex=True)
  for row, (index, measured, simulated, boundaries) in enumerate(series):
    ax = axes[row][0]
    n = min(len(measured), len(simulated))
    time_s = np.arange(n) * dt
    ax.plot(time_s, measured[:n], color="tab:blue", lw=0.7, label="measured")
    ax.plot(time_s, simulated[:n], color="tab:orange", lw=0.7, ls="--",
            label="simulated")
    for boundary in boundaries:
      ax.axvline(boundary, color="tab:green", lw=0.6, alpha=0.55)
    valid = np.isfinite(simulated[:n])
    if valid.sum() > 10:
      r = float(np.corrcoef(simulated[:n][valid], measured[:n][valid])[0, 1])
      ratio = float(np.std(simulated[:n][valid])) / max(
        float(np.std(measured[:n][valid])), 1e-12)
      ax.text(0.995, 0.06, f"r={r:+.3f}  amp={ratio:.2f}",
              transform=ax.transAxes, ha="right", fontsize=8)
    span = float(np.max(np.abs(measured[:n]))) * 1.8
    ax.set_ylim(-span, span)
    ax.set_ylabel(f"trial {index}\nx0 (uV)", fontsize=8)
    ax.grid(alpha=0.3)
    if row == 0:
      ax.legend(fontsize=8, loc="upper left")
  axes[-1][0].set_xlabel("time (s)")
  fig.suptitle(title, fontsize=12)
  fig.tight_layout()
  path = output_dir / filename
  fig.savefig(path, dpi=150)
  plt.close(fig)
  return path


def main() -> None:
  """Measure the tracking horizon and the effect of periodic reinitialisation."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--case", default="nd4")
  parser.add_argument("--horizon", type=float, default=14.0)
  parser.add_argument("--reinit", type=float, default=2.0)
  parser.add_argument("--fine-limit", type=float, default=1.0)
  parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
  args = parser.parse_args()

  from forecast_skill import build_delay_method
  from unbias_comparison import CASES

  args.out_dir.mkdir(parents=True, exist_ok=True)
  train_ids, test_ids = load_archived_split()
  print(f"Loading {MAT_FILE} ...", flush=True)
  data = TrialData.load(MAT_FILE)
  dt = DOWNSAMPLE / data.fs
  fs = 1.0 / dt
  train_traces = channel_traces(data, channel=CHANNEL, trials=train_ids,
                                downsample=DOWNSAMPLE, lowpass_hz=LOWPASS_HZ,
                                normalize="none")
  test_traces = channel_traces(data, channel=CHANNEL, trials=test_ids,
                               downsample=DOWNSAMPLE, lowpass_hz=LOWPASS_HZ,
                               normalize="none")
  case = next(c for c in CASES if c.label == args.case)
  print(f"case {case.label}: n_delays={case.n_delays}, delay={case.delay}, "
        f"degree={case.degree}, threshold={case.threshold:g}")
  method = build_delay_method(case, train_traces, dt)

  centres = None
  fine_corr, fine_amp = [], []
  free_series, reinit_series = [], []

  for index, trace in enumerate(test_traces):
    states = method.project(trace)
    measured = states[:, 0]

    free = simulate_model_detailed(
      method.model, initial_state=states[0], dt=dt,
      horizon_s=min(args.horizon, (len(states) - 1) * dt), wall_timeout_s=180.0,
    )
    if free.completed and free.trajectory is not None:
      simulated = method.signal_from_state(free.trajectory)
      if np.all(np.isfinite(simulated)):
        free_series.append((index, measured, simulated, np.array([])))
        c, corr, amp = fine_metrics(simulated, measured, fs, args.fine_limit)
        centres = c if centres is None else centres
        fine_corr.append(corr[:len(centres)])
        fine_amp.append(amp[:len(centres)])
    else:
      print(f"  trial {index}: free run failed to integrate")

    simulated, boundaries = reinitialised_run(
      method, trace, dt, args.horizon, args.reinit
    )
    reinit_series.append((index, measured, simulated, boundaries))
    valid = np.isfinite(simulated)
    n = min(len(simulated), len(measured))
    r = (float(np.corrcoef(simulated[:n][valid[:n]], measured[:n][valid[:n]])[0, 1])
         if valid[:n].sum() > 10 else float("nan"))
    print(f"  trial {index}: reinit every {args.reinit:g}s -> r={r:+.3f} "
          f"({int(valid.sum())}/{len(simulated)} samples integrated)", flush=True)

  if not fine_corr:
    print("No free run integrated; cannot measure the tracking horizon.")
  else:
    length = min(len(v) for v in fine_corr)
    centres = centres[:length]
    corr_stack = np.vstack([v[:length] for v in fine_corr])
    amp_stack = np.vstack([v[:length] for v in fine_amp])
    median = np.nanmedian(corr_stack, axis=0)
    print(f"\n{'time':>9} {'corr median':>13} {'amp median':>12}")
    for probe in (0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0):
      if probe > centres[-1]:
        continue
      i = int(np.argmin(np.abs(centres - probe)))
      print(f"{centres[i]*1000:>7.0f}ms {median[i]:>+13.3f} "
            f"{np.nanmedian(amp_stack[:, i]):>12.3g}")
    below = np.where(median < 0.2)[0]
    print(f"\ntracking (corr > 0.2) ends at "
          f"{centres[below[0]]*1000:.0f} ms" if below.size
          else "\ncorrelation stays above 0.2 across the measured window")

    with open(args.out_dir / f"fine_first_second_{case.label}.csv", "w",
              newline="") as handle:
      writer = csv.writer(handle)
      writer.writerow(["time_s", "corr_median", "amp_median"])
      for i, t in enumerate(centres):
        writer.writerow([f"{t:.4g}", f"{median[i]:.4g}",
                         f"{np.nanmedian(amp_stack[:, i]):.4g}"])
    print(f"\nwrote {plot_fine(centres, corr_stack, amp_stack, case.label, args.out_dir)}")

  if free_series:
    print(f"wrote {plot_per_trial(free_series, dt, case.label, args.out_dir, f'{case.label}: free run from one initial condition, every held-out trial', f'free_run_per_trial_{case.label}.png')}")
  if reinit_series:
    print(f"wrote {plot_per_trial(reinit_series, dt, case.label, args.out_dir, f'{case.label}: reinitialised from the measurement every {args.reinit:g} s (green lines mark restarts)', f'reinit_{args.reinit:g}s_per_trial_{case.label}.png')}")


if __name__ == "__main__":
  main()
