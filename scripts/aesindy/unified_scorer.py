"""Score every method on shape, amplitude, and long free-running simulation.

Each method in this project has so far been judged on a different yardstick,
which makes the results unrankable. Delay-coordinate SINDy free-runs for a
full 14 s trial at a plausible amplitude while scoring -0.47 correlation at
50 ms. The autoencoder scores +0.469 at 50 ms while its amplitude grows by
eight orders of magnitude. Both facts are real; neither alone decides which
model is better.

The reason is that correlation is scale-invariant. It cannot separate a
forecast that tracks the signal from one that tracks its *shape* while
exploding or collapsing, so a model can score well and be useless as a
simulation -- which is what the trajectory figures in this project have shown
all along.

This script applies one yardstick to every method:

    shape       correlation against the measurement at each lead time, and
                against persistence, which says how far ahead the signal is
                self-predictable at all (about 350 ms here).
    amplitude   forecast spread over measured spread. 1.0 is correct;
                far above is exploding, far below is collapsing to a point.
    endurance   a single free run from one initial condition over 1, 2 and
                14 s, which is the protocol behind this project's existing
                trajectory figures and the test that short forecasts hide.

A model has to pass all three to be worth anything. Passing only the first is
the failure mode this script exists to expose.

The PySINDy and Hankel-SVD baselines are refitted here, which takes seconds.
The autoencoder is not retrained: its forecasts are read from the ``.npz``
that ``scripts/aesindy/rescore_saved_model.py`` writes.

Run in the PySINDy environment:

    .venv/bin/python scripts/aesindy/unified_scorer.py

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
from models.forecast_metrics import persistence_by_lead, skill_by_lead  # noqa: E402
from models.validation import simulate_model_detailed  # noqa: E402

LOWPASS_HZ = 35.0
PROBES_S = (0.02, 0.05, 0.1, 0.2, 0.35)
LONG_HORIZONS_S = (1.0, 2.0, 14.0)
OUTPUT_DIR = _PROJECT_ROOT / "outputs/unified_comparison"


def amplitude_ratio(predicted: np.ndarray, measured: np.ndarray) -> np.ndarray:
  """Return forecast spread over measured spread at each lead.

  Args:
    predicted: Forecasts with shape ``(n_forecasts, n_leads)``.
    measured: Measurements with the same shape.

  Returns:
    Ratio per lead. 1.0 means the forecast has the right size.
  """
  spread = np.std(measured, axis=0)
  spread[spread < 1e-12] = np.nan
  return np.std(predicted, axis=0) / spread


def long_free_run(
  method, trace: np.ndarray, dt: float, horizons_s: tuple[float, ...],
) -> dict[float, np.ndarray | None]:
  """Free-run one simulation from a single initial condition.

  Args:
    method: A ``forecast_skill.Method`` with ``project``, ``model`` and
      ``signal_from_state``.
    trace: One held-out preprocessed trace.
    dt: Processed sample interval in seconds.
    horizons_s: Durations to attempt, in seconds.

  Returns:
    Horizon to simulated signal, or ``None`` where integration failed.
  """
  states = method.project(trace)
  results: dict[float, np.ndarray | None] = {}
  for horizon in horizons_s:
    n_steps = min(int(round(horizon / dt)) + 1, len(states))
    simulation = simulate_model_detailed(
      method.model, initial_state=states[0], dt=dt,
      horizon_s=(n_steps - 1) * dt, wall_timeout_s=120.0,
    )
    if not (simulation.completed and simulation.trajectory is not None):
      results[horizon] = None
      continue
    signal = method.signal_from_state(simulation.trajectory)
    results[horizon] = signal if np.all(np.isfinite(signal)) else None
  return results


def build_methods(train_traces: list[np.ndarray], dt: float) -> dict:
  """Fit the delay-coordinate and Hankel-SVD baselines.

  Args:
    train_traces: Training traces in signal units.
    dt: Processed sample interval in seconds.

  Returns:
    Mapping from label to a fitted ``forecast_skill.Method``.
  """
  from forecast_skill import build_delay_method, build_havok_method
  from unbias_comparison import CASES

  methods = {}
  for case in CASES:
    if case.label in ("nd2", "nd4"):
      methods[f"delay {case.label}"] = build_delay_method(case, train_traces, dt)
  for n_modes in (3, 12):
    methods[f"HAVOK r={n_modes}"] = build_havok_method(
      train_traces, dt, n_delays=80, n_modes=n_modes, degree=1, threshold=1000.0
    )
  return methods


def plot_summary(rows: list[dict], output_dir: Path) -> Path:
  """Plot correlation against amplitude ratio for every method.

  A model is only usable in the region where correlation is positive *and*
  the amplitude ratio is near 1. Plotting the two together makes that region
  visible rather than leaving it implicit across two separate tables.

  Args:
    rows: Result rows holding ``method``, ``corr_50ms`` and ``amp_50ms``.
    output_dir: Directory to write into.

  Returns:
    Path of the written figure.
  """
  fig, ax = plt.subplots(figsize=(8.5, 5.5))
  for row in rows:
    amp = row["amp_50ms"]
    if not np.isfinite(amp) or amp <= 0:
      continue
    ax.scatter(amp, row["corr_50ms"], s=90)
    ax.annotate(row["method"], (amp, row["corr_50ms"]),
                textcoords="offset points", xytext=(7, 4), fontsize=8)
  ax.axvline(1.0, color="black", ls="--", lw=1.4)
  ax.axhline(0.0, color="gray", lw=0.8)
  ax.axvspan(0.5, 2.0, color="tab:green", alpha=0.10)
  ax.set_xscale("log")
  ax.set_xlabel("amplitude ratio at 50 ms  (1.0 = correct size)")
  ax.set_ylabel("correlation at 50 ms  (shape)")
  ax.set_title("A usable model needs both: correct shape and correct amplitude\n"
               "shaded band marks a plausible amplitude", fontsize=11)
  ax.grid(alpha=0.3)
  fig.tight_layout()
  path = output_dir / "shape_vs_amplitude.png"
  fig.savefig(path, dpi=150)
  plt.close(fig)
  return path


def plot_free_runs(
  free_runs: dict[str, dict[float, np.ndarray | None]], reference: np.ndarray,
  dt: float, output_dir: Path, horizon: float,
) -> Path:
  """Plot every method's free run against the measurement at one horizon.

  Args:
    free_runs: Method label to horizon-keyed simulations.
    reference: The measured signal in microvolts.
    dt: Processed sample interval in seconds.
    output_dir: Directory to write into.
    horizon: Which horizon to draw.

  Returns:
    Path of the written figure.
  """
  labels = list(free_runs)
  fig, axes = plt.subplots(len(labels), 1, figsize=(12, 2.3 * len(labels)),
                           squeeze=False, sharex=True)
  n = min(int(round(horizon / dt)) + 1, reference.size)
  time_s = np.arange(n) * dt
  for row, label in enumerate(labels):
    ax = axes[row][0]
    ax.plot(time_s, reference[:n], color="tab:blue", lw=0.8, label="measured")
    simulated = free_runs[label].get(horizon)
    if simulated is None:
      ax.text(0.5, 0.5, "integration failed / non-finite",
              transform=ax.transAxes, ha="center", color="tab:red", fontsize=9)
    else:
      m = min(len(simulated), n)
      ax.plot(time_s[:m], simulated[:m], color="tab:orange", lw=0.8, ls="--",
              label="simulated")
      ratio = float(np.std(simulated[:m])) / max(float(np.std(reference[:m])), 1e-12)
      ax.text(0.995, 0.06, f"amplitude ratio {ratio:.3g}",
              transform=ax.transAxes, ha="right", fontsize=8,
              color="tab:red" if (ratio > 3 or ratio < 0.3) else "black")
    span = float(np.max(np.abs(reference[:n]))) * 1.8
    ax.set_ylim(-span, span)
    ax.set_ylabel(f"{label}\nx0 (uV)", fontsize=8)
    ax.grid(alpha=0.3)
    if row == 0:
      ax.legend(fontsize=8, loc="upper left")
  axes[-1][0].set_xlabel("time from embedded initial state (s)")
  fig.suptitle(f"Free-running simulation, {horizon:g} s horizon, "
               f"one initial condition (y-axis fixed to the measurement)",
               fontsize=12)
  fig.tight_layout()
  path = output_dir / f"free_run_{horizon:g}s.png"
  fig.savefig(path, dpi=150)
  plt.close(fig)
  return path


def main() -> None:
  """Score every available method on shape, amplitude and endurance."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--max-lead", type=float, default=1.0)
  parser.add_argument("--origin-stride", type=float, default=0.5)
  parser.add_argument(
    "--aesindy-npz", type=Path, nargs="*", default=[],
    help="rescore_traces.npz files from autoencoder runs, added as extra rows.",
  )
  parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
  args = parser.parse_args()

  from forecast_skill import collect_forecasts

  args.out_dir.mkdir(parents=True, exist_ok=True)
  train_ids, test_ids = load_archived_split()
  print(f"Loading {MAT_FILE} ...", flush=True)
  data = TrialData.load(MAT_FILE)
  dt = DOWNSAMPLE / data.fs
  train_traces = channel_traces(data, channel=CHANNEL, trials=train_ids,
                                downsample=DOWNSAMPLE, lowpass_hz=LOWPASS_HZ,
                                normalize="none")
  test_traces = channel_traces(data, channel=CHANNEL, trials=test_ids,
                               downsample=DOWNSAMPLE, lowpass_hz=LOWPASS_HZ,
                               normalize="none")

  print("fitting baselines ...", flush=True)
  methods = build_methods(train_traces, dt)

  rows: list[dict] = []
  free_runs: dict[str, dict[float, np.ndarray | None]] = {}
  persistence = None

  for label, method in methods.items():
    predicted, measured = collect_forecasts(
      method, test_traces, dt, args.max_lead, args.origin_stride,
      sim_timeout_s=30.0,
    )
    if predicted.shape[0] == 0:
      print(f"  {label:16} no usable forecasts")
      rows.append({"method": label, "n": 0, "corr_50ms": float("nan"),
                   "amp_50ms": float("nan")})
      continue
    skill = skill_by_lead(predicted, measured)
    amplitude = amplitude_ratio(predicted, measured)
    if persistence is None:
      persistence = persistence_by_lead(measured)
    # Every held-out trial, so the endurance number is a median rather than
    # one initial condition that might be unrepresentative.
    per_trial = [long_free_run(method, trace, dt, LONG_HORIZONS_S)
                 for trace in test_traces]
    free_runs[label] = per_trial[0]
    for horizon in LONG_HORIZONS_S:
      ratios = []
      for trial_index, runs in enumerate(per_trial):
        simulated = runs.get(horizon)
        if simulated is None:
          continue
        reference = test_traces[trial_index]
        n = min(len(simulated), reference.size)
        ratios.append(float(np.std(simulated[:n]))
                      / max(float(np.std(reference[:n])), 1e-12))
      row_key = f"free_{horizon:g}s_amp_median"
      globals().setdefault("_endurance", {}).setdefault(label, {})[horizon] = (
        float(np.median(ratios)) if ratios else float("nan"),
        len(ratios), len(per_trial),
      )
    row = {"method": label, "n": predicted.shape[0]}
    for probe in PROBES_S:
      i = int(round(probe / dt))
      row[f"corr_{int(probe*1000)}ms"] = float(skill[i])
      row[f"amp_{int(probe*1000)}ms"] = float(amplitude[i])
    for horizon, simulated in free_runs[label].items():
      row[f"free_run_{horizon:g}s"] = (
        "failed" if simulated is None
        else f"{np.std(simulated)/max(np.std(test_traces[0]),1e-12):.3g}"
      )
    rows.append(row)
    print(f"  {label:16} n={predicted.shape[0]:<4} "
          f"corr50={row['corr_50ms']:+.3f} amp50={row['amp_50ms']:.3g}")

  for npz_path in args.aesindy_npz:
    if not npz_path.exists():
      print(f"  (skipping missing {npz_path})")
      continue
    archive = np.load(npz_path)
    predicted, measured = archive["predicted"], archive["measured"]
    label = f"AE {npz_path.parent.name.replace('aesindy_', '')}"
    skill = skill_by_lead(predicted, measured)
    amplitude = amplitude_ratio(predicted, measured)
    row = {"method": label, "n": predicted.shape[0]}
    for probe in PROBES_S:
      i = int(round(probe / dt))
      row[f"corr_{int(probe*1000)}ms"] = float(skill[i])
      row[f"amp_{int(probe*1000)}ms"] = float(amplitude[i])
    rows.append(row)
    print(f"  {label:16} n={predicted.shape[0]:<4} "
          f"corr50={row['corr_50ms']:+.3f} amp50={row['amp_50ms']:.3g}")

  fieldnames = sorted({k for r in rows for k in r},
                      key=lambda k: (k != "method", k))
  path = args.out_dir / "unified_comparison.csv"
  with open(path, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

  print("\n===== shape and amplitude, held-out trials =====")
  print(f"{'method':18} {'n':>5} {'corr@50ms':>10} {'amp@50ms':>10} "
        f"{'corr@200ms':>11} {'amp@200ms':>10}  verdict")
  for row in rows:
    corr = row.get("corr_50ms", float("nan"))
    amp = row.get("amp_50ms", float("nan"))
    # Amplitude has to stay plausible at every lead, not merely at 50 ms.
    # Checking one lead reported a model whose amplitude reaches 20x by 200 ms
    # and 1e8 by a second as "plausible", which is the exact failure this
    # column exists to catch.
    amps = [(probe, row.get(f"amp_{int(probe*1000)}ms", float("nan")))
            for probe in PROBES_S]
    finite = [(probe, a) for probe, a in amps if np.isfinite(a)]
    broken = [(probe, a) for probe, a in finite if not 0.3 < a < 3.0]
    if not np.isfinite(corr):
      verdict = "no forecast"
    elif broken:
      probe, a = broken[0]
      verdict = (f"wrong amplitude by {int(probe*1000)}ms "
                 f"({'explodes' if a >= 3 else 'collapses'}, {a:.3g}x)")
    elif corr <= 0:
      verdict = "wrong shape"
    else:
      verdict = "plausible to 350ms"
    print(f"{row['method']:18} {row.get('n',0):>5} {corr:>+10.3f} {amp:>10.3g} "
          f"{row.get('corr_200ms', float('nan')):>+11.3f} "
          f"{row.get('amp_200ms', float('nan')):>10.3g}  {verdict}")
  if persistence is not None:
    i = int(round(0.05 / dt))
    j = int(round(0.20 / dt))
    print(f"{'persistence':18} {'-':>5} {persistence[i]:>+10.3f} {1.0:>10.3g} "
          f"{persistence[j]:>+11.3f} {1.0:>10.3g}  reference")

  endurance = globals().get("_endurance", {})
  if endurance:
    print("\n===== endurance: free-run amplitude ratio, median over held-out trials =====")
    print(f"{'method':18}" + "".join(f"{h:>12g}s" for h in LONG_HORIZONS_S)
          + "   (1.0 = correct, completed/total)")
    for label, per_horizon in endurance.items():
      line = f"{label:18}"
      for horizon in LONG_HORIZONS_S:
        median, done, total = per_horizon.get(horizon, (float("nan"), 0, 0))
        line += f"{median:>10.3g}[{done}/{total}]"
      print(line)

  print(f"\nwrote {path}")
  print(f"wrote {plot_summary(rows, args.out_dir)}")
  reference = test_traces[0]
  for horizon in LONG_HORIZONS_S:
    print(f"wrote {plot_free_runs(free_runs, reference, dt, args.out_dir, horizon)}")


if __name__ == "__main__":
  main()
