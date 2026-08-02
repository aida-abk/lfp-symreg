"""When does a delay-coordinate model stop tracking and become an oscillator?

The poster's three regimes -- fixed frequency and amplitude oscillators,
weakly nonlinear models with a stable dominant frequency, and unstable models
with diverged trajectories -- describe where a fitted model *ends up*. They do
not say when it gets there. A model can follow the measurement for a second
and then settle onto its own limit cycle, and a single correlation number, or
a single 14 s trajectory plot, shows neither the tracking nor the transition.

This script resolves that transition in time. A free run is launched from the
first embedded state of each held-out trial, and three quantities are measured
in sliding windows along it:

    correlation             does the simulation still follow this trial?
    amplitude ratio         is it the right size? 1.0 is correct.
    spectral concentration  fraction of power in the strongest frequency bin.
                            A broadband signal is low; an oscillator is high,
                            so this is what separates regime (ii) from (i).

The measured trace is put through the identical windowing, so the simulated
concentration is read against the signal's own value rather than an absolute
threshold.

Every held-out trial is used and results are reported as a median with an
inter-quartile band, because single-trial trajectories have already proved
misleading in this project: one method's 14 s amplitude ratio read 0.32 on
trial 0 and 0.94 as a median over all nine.

Run with:

    .venv/bin/python scripts/aesindy/nd4_horizon_analysis.py --case nd4

"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import signal as sig

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
WINDOW_S = 0.5
STEP_S = 0.1
OUTPUT_DIR = _PROJECT_ROOT / "outputs/nd4_horizon"


def windowed_metrics(
  simulated: np.ndarray, measured: np.ndarray, fs: float,
  window_s: float, step_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Measure tracking, size and narrowbandness along a trajectory.

  Args:
    simulated: Free-running simulation in microvolts.
    measured: The measurement it was launched from, same units.
    fs: Processed sampling frequency in hertz.
    window_s: Window length in seconds.
    step_s: Step between window starts in seconds.

  Returns:
    ``(centres_s, correlation, amplitude_ratio, concentration_sim,
    concentration_measured)``, one entry per window.
  """
  n = min(len(simulated), len(measured))
  width = int(round(window_s * fs))
  step = max(int(round(step_s * fs)), 1)
  nperseg = min(width, 128)

  centres, corr, amp, conc_s, conc_m = [], [], [], [], []
  for start in range(0, n - width + 1, step):
    a = simulated[start:start + width]
    b = measured[start:start + width]
    centres.append((start + width / 2) / fs)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
      corr.append(np.nan)
    else:
      corr.append(float(np.corrcoef(a, b)[0, 1]))
    amp.append(float(np.std(a)) / max(float(np.std(b)), 1e-12))
    for series, store in ((a, conc_s), (b, conc_m)):
      f, power = sig.welch(series, fs=fs, nperseg=nperseg)
      band = (f > 1) & (f < 40)
      store.append(float(power[band].max() / power[band].sum())
                   if band.any() and power[band].sum() > 0 else np.nan)
  return (np.array(centres), np.array(corr), np.array(amp),
          np.array(conc_s), np.array(conc_m))


def plot_transition(
  centres: np.ndarray, stacks: dict[str, np.ndarray], case_label: str,
  output_dir: Path, zoom_s: float,
) -> Path:
  """Plot the three quantities against time, median and inter-quartile band.

  Args:
    centres: Window centre times in seconds.
    stacks: Metric name to array of shape ``(n_trials, n_windows)``.
    case_label: Configuration label for the title.
    output_dir: Directory to write into.
    zoom_s: Upper limit of the zoomed row, in seconds.

  Returns:
    Path of the written figure.
  """
  panels = [
    ("correlation", "correlation with this trial", (-1, 1), 0.0),
    ("amplitude", "amplitude ratio (1.0 = correct)", None, 1.0),
    ("concentration", "power in strongest bin", (0, 1), None),
  ]
  fig, axes = plt.subplots(2, 3, figsize=(16, 7), squeeze=False)
  for row, limit in enumerate((centres[-1], zoom_s)):
    for column, (key, ylabel, ylim, guide) in enumerate(panels):
      ax = axes[row][column]
      values = stacks[key]
      median = np.nanmedian(values, axis=0)
      lower = np.nanpercentile(values, 25, axis=0)
      upper = np.nanpercentile(values, 75, axis=0)
      ax.plot(centres, median, lw=1.6, color="tab:orange", label="simulated")
      ax.fill_between(centres, lower, upper, alpha=0.20, color="tab:orange")
      if key == "concentration":
        reference = np.nanmedian(stacks["concentration_measured"], axis=0)
        ax.plot(centres, reference, lw=1.6, color="tab:blue", label="measured")
        ax.legend(fontsize=8)
      if guide is not None:
        ax.axhline(guide, color="black", ls="--", lw=1.0)
      if ylim:
        ax.set_ylim(*ylim)
      ax.set_xlim(0, limit)
      ax.set_xlabel("time from initial condition (s)", fontsize=9)
      ax.set_ylabel(ylabel, fontsize=9)
      ax.grid(alpha=0.3)
      if row == 0 and column == 1:
        ax.set_title(f"{case_label}: full horizon", fontsize=10)
      if row == 1 and column == 1:
        ax.set_title(f"first {zoom_s:g} s", fontsize=10)
  fig.suptitle(
    f"{case_label}: when does the model stop tracking and become an oscillator?\n"
    f"median over held-out trials, shaded band is the inter-quartile range",
    fontsize=12,
  )
  fig.tight_layout()
  path = output_dir / f"horizon_transition_{case_label}.png"
  fig.savefig(path, dpi=150)
  plt.close(fig)
  return path


def main() -> None:
  """Resolve in time where a delay-coordinate model stops tracking."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--case", default="nd4")
  parser.add_argument("--horizon", type=float, default=14.0)
  parser.add_argument("--zoom", type=float, default=2.0)
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
        f"degree={case.degree}, lowpass={case.lowpass:g}, "
        f"threshold={case.threshold:g}")
  method = build_delay_method(case, train_traces, dt)

  collected: dict[str, list[np.ndarray]] = {
    "correlation": [], "amplitude": [], "concentration": [],
    "concentration_measured": [],
  }
  centres = None
  for index, trace in enumerate(test_traces):
    states = method.project(trace)
    simulation = simulate_model_detailed(
      method.model, initial_state=states[0], dt=dt,
      horizon_s=min(args.horizon, (len(states) - 1) * dt), wall_timeout_s=180.0,
    )
    if not (simulation.completed and simulation.trajectory is not None):
      print(f"  trial {index}: integration failed")
      continue
    simulated = method.signal_from_state(simulation.trajectory)
    if not np.all(np.isfinite(simulated)):
      print(f"  trial {index}: non-finite trajectory")
      continue
    measured = states[:, 0]
    c, corr, amp, conc_s, conc_m = windowed_metrics(
      simulated, measured, fs, WINDOW_S, STEP_S
    )
    centres = c if centres is None else centres
    length = min(len(centres), len(corr))
    collected["correlation"].append(corr[:length])
    collected["amplitude"].append(amp[:length])
    collected["concentration"].append(conc_s[:length])
    collected["concentration_measured"].append(conc_m[:length])
    print(f"  trial {index}: corr@0.5s={corr[0]:+.3f} "
          f"amp@0.5s={amp[0]:.3g} conc@0.5s={conc_s[0]:.1%}")

  if not collected["correlation"]:
    print("No trial produced a usable trajectory.")
    return

  length = min(len(v) for v in collected["correlation"])
  centres = centres[:length]
  stacks = {k: np.vstack([v[:length] for v in vals])
            for k, vals in collected.items()}

  path = args.out_dir / f"horizon_transition_{case.label}.csv"
  with open(path, "w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["time_s", "corr_median", "corr_q25", "corr_q75",
                     "amp_median", "conc_sim_median", "conc_measured_median"])
    for i, t in enumerate(centres):
      writer.writerow([
        f"{t:.4g}",
        f"{np.nanmedian(stacks['correlation'][:, i]):.4g}",
        f"{np.nanpercentile(stacks['correlation'][:, i], 25):.4g}",
        f"{np.nanpercentile(stacks['correlation'][:, i], 75):.4g}",
        f"{np.nanmedian(stacks['amplitude'][:, i]):.4g}",
        f"{np.nanmedian(stacks['concentration'][:, i]):.4g}",
        f"{np.nanmedian(stacks['concentration_measured'][:, i]):.4g}",
      ])

  print(f"\n{'time':>8} {'corr':>8} {'amp':>8} {'conc sim':>10} {'conc meas':>10}")
  for probe in (0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 14.0):
    if probe > centres[-1]:
      continue
    i = int(np.argmin(np.abs(centres - probe)))
    print(f"{centres[i]:>7.2f}s "
          f"{np.nanmedian(stacks['correlation'][:, i]):>+8.3f} "
          f"{np.nanmedian(stacks['amplitude'][:, i]):>8.3g} "
          f"{np.nanmedian(stacks['concentration'][:, i]):>9.1%} "
          f"{np.nanmedian(stacks['concentration_measured'][:, i]):>9.1%}")

  # Where the simulation stops resembling the trial and starts resembling an
  # oscillator, stated as two crossing times rather than left to the eye.
  corr_median = np.nanmedian(stacks["correlation"], axis=0)
  conc_median = np.nanmedian(stacks["concentration"], axis=0)
  conc_reference = np.nanmedian(stacks["concentration_measured"], axis=0)
  below = np.where(corr_median < 0.2)[0]
  above = np.where(conc_median > 2 * conc_reference)[0]
  print(f"\ncorrelation falls below 0.2 at "
        f"{centres[below[0]]:.2f} s" if below.size else
        "\ncorrelation stays above 0.2 throughout")
  print(f"narrowbandness exceeds 2x the measurement at "
        f"{centres[above[0]]:.2f} s" if above.size else
        "narrowbandness stays within 2x the measurement throughout")

  print(f"\nwrote {path}")
  print(f"wrote {plot_transition(centres, stacks, case.label, args.out_dir, args.zoom)}")


if __name__ == "__main__":
  main()
