"""Measure forecast skill as a function of lead time, against persistence.

Every result so far scored a single free-running simulation per held-out
trial, integrated to 6 seconds. At roughly 10 Hz that is about 60 oscillation
cycles, and a *correct* model of a noisy or chaotic oscillator would also
score near zero there -- phase drift alone destroys the correlation. So a
correlation of zero at 6 s cannot distinguish "the model is wrong" from "this
signal is not predictable that far ahead".

This script separates the two by resolving skill against lead time:

* Forecasts are launched from many origins inside each held-out trial, not
  just its first sample, so each lead time is estimated from ~100 forecasts
  instead of 9.
* Skill at lead tau is the correlation, across all (trial, origin) pairs,
  between the predicted signal at tau and the measured signal at tau. This is
  the standard anomaly-correlation construction, and it is a much sharper
  instrument than correlating a whole 6 s window.
* Persistence -- correlating the measured signal at lead tau against the
  measured signal at the origin -- is computed on the same origins. This is
  the signal's own autocorrelation, and it is the reference that says how far
  ahead the signal remains self-predictable at all.

Reading the result:

* Model skill above persistence at short lead times means the fitted dynamics
  carry genuine information, and the 6 s metric was hiding it.
* Model skill at zero while persistence is still high means the models fail
  where the signal is demonstrably predictable -- a real modelling failure.
* Both decaying together to zero within a few hundred milliseconds means the
  signal itself is not deterministically predictable, and no coordinate
  transformation, learned or otherwise, will change that.

Run with:

    .venv/bin/python scripts/pysindy/forecast_skill.py

"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(_PROJECT_ROOT))

# unbias_comparison.py supplies the archived case definitions, the trial split
# and the shared constants. It lives in scripts/pysindy/internals_probes/,
# which is not this script's own directory, so `from unbias_comparison import`
# cannot rely on it being a sibling on sys.path[0]. The location is resolved
# explicitly instead, which keeps these scripts runnable from anywhere.
_INTERNALS_DIR = _PROJECT_ROOT / "scripts" / "pysindy" / "internals_probes"
if str(_INTERNALS_DIR) not in sys.path:
  sys.path.insert(0, str(_INTERNALS_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pysindy as ps  # noqa: E402

from load_data.convert import MAT_FILE, TrialData  # noqa: E402
from load_data.preprocessing import channel_traces  # noqa: E402
from models.havok import fit_hankel_basis  # noqa: E402
from models.sindy import delay_embed_trace, delay_embed_trajectories  # noqa: E402
from models.forecast_metrics import (  # noqa: E402
  persistence_by_lead,
  skill_by_lead,
)
from models.validation import simulate_model_detailed  # noqa: E402

from unbias_comparison import (  # noqa: E402
  ALPHA,
  CHANNEL,
  DOWNSAMPLE,
  NORMALIZE_COLUMNS,
  CASES,
  fit_case,
  load_split,
)

DEFAULT_MAX_LEAD_S = 2.0
DEFAULT_ORIGIN_STRIDE_S = 1.0
DEFAULT_SIM_TIMEOUT_S = 30.0
LOWPASS_HZ = 35.0
OUTPUT_DIR = _PROJECT_ROOT / "outputs/pysindy/forecast_skill"


@dataclass
class Method:
  """One fitted approach whose forecast skill is measured.

  Attributes:
    label: Short name used in output rows and the figure legend.
    description: Human-readable summary of the configuration.
    project: Maps a preprocessed trace to the state trajectory the model was
      fitted in, with shape ``(time, state)``.
    signal_from_state: Maps a state trajectory back to the scalar signal.
    model: The fitted PySINDy model, assigned during setup.
  """

  label: str
  description: str
  project: object
  signal_from_state: object
  model: object = field(default=None)


def build_delay_method(case, train_traces: list[np.ndarray], dt: float) -> Method:
  """Fit an archived delay-coordinate configuration.

  Args:
    case: Archived configuration from ``unbias_comparison.CASES``.
    train_traces: Training traces in signal units.
    dt: Processed sample interval in seconds.

  Returns:
    A :class:`Method` whose state space is raw delay coordinates.
  """
  embedded = delay_embed_trajectories(
    train_traces, n_delays=case.n_delays, delay=case.delay
  )
  model = fit_case(embedded, dt, case, unbias=True)
  return Method(
    label=f"delay {case.label}",
    description=(
      f"delay coordinates, n_delays={case.n_delays}, delay={case.delay}, "
      f"degree={case.degree}"
    ),
    project=lambda trace: delay_embed_trace(
      trace, n_delays=case.n_delays, delay=case.delay
    ),
    signal_from_state=lambda state: np.asarray(state)[:, 0],
    model=model,
  )


def build_havok_method(
  train_traces: list[np.ndarray],
  dt: float,
  n_delays: int,
  n_modes: int,
  degree: int,
  threshold: float,
) -> Method:
  """Fit a Hankel-SVD (HAVOK) configuration.

  Args:
    train_traces: Training traces in signal units.
    dt: Processed sample interval in seconds.
    n_delays: Embedding dimension for the Hankel matrix.
    n_modes: Retained singular vectors.
    degree: Polynomial library degree.
    threshold: STLSQ threshold.

  Returns:
    A :class:`Method` whose state space is mode coordinates, with the signal
    recovered by reconstructing into delay space.
  """
  basis = fit_hankel_basis(
    train_traces, n_delays=n_delays, delay=1, n_modes=n_modes
  )
  mode_train = [basis.project(t) for t in train_traces]
  model = ps.SINDy(
    optimizer=ps.STLSQ(
      threshold=threshold, alpha=ALPHA,
      normalize_columns=NORMALIZE_COLUMNS, verbose=False, max_iter=20,
    ),
    feature_library=ps.PolynomialLibrary(degree=degree),
  )
  model.fit(mode_train, t=dt)
  return Method(
    label=f"havok r={n_modes} deg={degree}",
    description=(
      f"Hankel-SVD, n_delays={n_delays}, modes={n_modes}, degree={degree}, "
      f"threshold={threshold:g}"
    ),
    project=basis.project,
    signal_from_state=lambda state: basis.reconstruct(np.asarray(state))[:, 0],
    model=model,
  )


def collect_forecasts(
  method: Method,
  test_traces: list[np.ndarray],
  dt: float,
  max_lead_s: float,
  origin_stride_s: float,
  sim_timeout_s: float,
) -> tuple[np.ndarray, np.ndarray]:
  """Launch forecasts from many origins and collect predictions by lead time.

  Args:
    method: Fitted approach to forecast with.
    test_traces: Held-out preprocessed traces.
    dt: Processed sample interval in seconds.
    max_lead_s: Longest lead time to forecast, in seconds.
    origin_stride_s: Spacing between forecast origins within a trial.
    sim_timeout_s: Wall-clock cap per simulation, in seconds.

  Returns:
    ``(predicted, measured)`` arrays with shape ``(n_forecasts, n_leads)``.
    Failed or truncated simulations are omitted.
  """
  n_leads = int(round(max_lead_s / dt)) + 1
  origin_stride = max(int(round(origin_stride_s / dt)), 1)
  predicted_rows, measured_rows = [], []

  for trace in test_traces:
    states = method.project(trace)
    signal = np.asarray(states)[:, 0] if states.shape[1] > 0 else None
    # The measured signal is column zero of the delay embedding, which is the
    # current sample. For HAVOK the state is in mode coordinates, so the
    # measured signal is taken from the delay embedding instead.
    measured_signal = delay_embed_trace(trace, n_delays=1, delay=1)[:, 0]
    # Align: projected states start at the same sample index as the embedding
    # used to build them, so index i of `states` corresponds to index
    # i + (n_delays-1)*delay of the raw trace. Working entirely in the
    # projected index space keeps the two consistent.
    usable = len(states) - n_leads
    if usable <= 0:
      continue
    offset = len(measured_signal) - len(states)
    for origin in range(0, usable, origin_stride):
      simulation = simulate_model_detailed(
        method.model, initial_state=states[origin], dt=dt,
        horizon_s=max_lead_s, wall_timeout_s=sim_timeout_s,
      )
      if not (simulation.completed and simulation.trajectory is not None):
        continue
      if len(simulation.trajectory) < n_leads:
        continue
      predicted = method.signal_from_state(simulation.trajectory[:n_leads])
      truth = measured_signal[offset + origin : offset + origin + n_leads]
      if len(truth) < n_leads or not np.all(np.isfinite(predicted)):
        continue
      predicted_rows.append(predicted)
      measured_rows.append(truth)

  if not predicted_rows:
    return np.empty((0, n_leads)), np.empty((0, n_leads))
  return np.vstack(predicted_rows), np.vstack(measured_rows)


def plot_skill(rows: list[dict], output_dir: Path) -> Path:
  """Plot skill against lead time for every method plus persistence.

  Args:
    rows: Result rows holding label, lead_s, and skill.
    output_dir: Directory to write the figure into.

  Returns:
    Path of the written figure.
  """
  labels = []
  for row in rows:
    if row["label"] not in labels:
      labels.append(row["label"])
  fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
  for label in labels:
    leads = [r["lead_s"] for r in rows if r["label"] == label]
    skill = [r["skill"] for r in rows if r["label"] == label]
    style = dict(lw=2.2, color="black", ls="--") if label == "persistence" else dict(lw=1.5)
    for ax in axes:
      ax.plot(leads, skill, label=label, **style)
  for ax, limit, title in (
    (axes[0], None, "full lead range"),
    (axes[1], 0.5, "first 500 ms"),
  ):
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlabel("lead time (s)")
    ax.set_ylabel("correlation with measured signal")
    ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=10)
    if limit is not None:
      ax.set_xlim(0, limit)
  axes[0].legend(fontsize=8)
  fig.suptitle(
    "Forecast skill vs lead time on held-out trials "
    "(persistence = signal autocorrelation)",
    fontsize=11,
  )
  fig.tight_layout()
  path = output_dir / "forecast_skill.png"
  fig.savefig(path, dpi=150)
  plt.close(fig)
  return path


def main() -> None:
  """Fit each method, collect forecasts, and report skill against lead time."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--max-lead", type=float, default=DEFAULT_MAX_LEAD_S)
  parser.add_argument("--origin-stride", type=float, default=DEFAULT_ORIGIN_STRIDE_S)
  parser.add_argument("--sim-timeout", type=float, default=DEFAULT_SIM_TIMEOUT_S)
  parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
  parser.add_argument(
    "--n-delays-list", default="80",
    help="Comma-separated Hankel embedding dimensions to sweep.",
  )
  parser.add_argument(
    "--modes-list", default="3,12",
    help="Comma-separated retained-mode counts. Skipped where they exceed "
         "the embedding dimension.",
  )
  parser.add_argument(
    "--degree-list", default="1",
    help="Comma-separated polynomial degrees for the mode-space library.",
  )
  parser.add_argument("--threshold", type=float, default=1000.0)
  parser.add_argument(
    "--lowpass", type=float, default=LOWPASS_HZ,
    help="Low-pass cutoff in hertz applied during preprocessing.",
  )
  parser.add_argument(
    "--delay-baseline", default="nd4",
    help="Archived delay-coordinate case to include as a reference, or "
         "'none' to skip it.",
  )
  args = parser.parse_args()

  args.out_dir.mkdir(parents=True, exist_ok=True)
  train_ids, test_ids = load_split()
  print(f"Loading {MAT_FILE} ...", flush=True)
  data = TrialData.load(MAT_FILE)
  dt = DOWNSAMPLE / data.fs

  train_traces = channel_traces(
    data, channel=CHANNEL, trials=train_ids, downsample=DOWNSAMPLE,
    lowpass_hz=args.lowpass, normalize="none",
  )
  test_traces = channel_traces(
    data, channel=CHANNEL, trials=test_ids, downsample=DOWNSAMPLE,
    lowpass_hz=args.lowpass, normalize="none",
  )
  print(f"train {len(train_traces)}, held-out {len(test_traces)}, "
        f"dt={dt*1000:.0f} ms, max lead {args.max_lead:g} s, "
        f"lowpass {args.lowpass:g} Hz")

  n_delays_list = [int(n) for n in args.n_delays_list.split(",")]
  modes_list = [int(m) for m in args.modes_list.split(",")]
  degrees = [int(d) for d in args.degree_list.split(",")]

  print("\nfitting methods ...", flush=True)
  methods = []
  if args.delay_baseline != "none":
    case = next(c for c in CASES if c.label == args.delay_baseline)
    methods.append(build_delay_method(case, train_traces, dt))
  for n_delays in n_delays_list:
    window_ms = n_delays * dt * 1000.0
    for n_modes in modes_list:
      # A truncation cannot retain more modes than the embedding has
      # dimensions; such combinations are skipped rather than clamped, so
      # that a swept grid never silently collapses two settings into one.
      if n_modes > n_delays:
        continue
      for degree in degrees:
        method = build_havok_method(
          train_traces, dt, n_delays=n_delays, n_modes=n_modes,
          degree=degree, threshold=args.threshold,
        )
        method.label = f"havok q={n_delays} r={n_modes} deg={degree}"
        method.description += f" ({window_ms:.0f} ms window)"
        methods.append(method)
  for method in methods:
    print(f"  {method.label:>26}: {method.description}")

  rows: list[dict] = []
  persistence_written = False
  for method in methods:
    predicted, measured = collect_forecasts(
      method, test_traces, dt, args.max_lead, args.origin_stride,
      args.sim_timeout,
    )
    print(f"\n{method.label}: {predicted.shape[0]} usable forecasts", flush=True)
    if predicted.shape[0] == 0:
      print("  no usable forecasts")
      continue
    skill = skill_by_lead(predicted, measured)
    leads = np.arange(skill.size) * dt
    for lead, value in zip(leads, skill):
      rows.append({"label": method.label, "lead_s": float(lead),
                   "skill": float(value), "n_forecasts": predicted.shape[0]})
    if not persistence_written:
      reference = persistence_by_lead(measured)
      for lead, value in zip(leads, reference):
        rows.append({"label": "persistence", "lead_s": float(lead),
                     "skill": float(value), "n_forecasts": measured.shape[0]})
      persistence_written = True
    for probe in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0):
      index = int(round(probe / dt))
      if index < skill.size:
        print(f"  lead {probe*1000:>6.0f} ms: skill {skill[index]:+.3f}")

  path = args.out_dir / "forecast_skill.csv"
  with open(path, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["label", "lead_s", "skill",
                                                "n_forecasts"])
    writer.writeheader()
    writer.writerows(rows)
  print(f"\nwrote {path}")
  print(f"wrote {plot_skill(rows, args.out_dir)}")
  _summary(rows, dt)


def _summary(rows: list[dict], dt: float) -> None:
  """Print a compact skill table at selected lead times."""
  labels = []
  for row in rows:
    if row["label"] not in labels:
      labels.append(row["label"])
  # Probes beyond the longest lead actually simulated are dropped rather than
  # snapped to the nearest available lead, which would print the final value
  # twice and read as though a longer forecast had been made.
  max_lead = max(r["lead_s"] for r in rows)
  probes = [p for p in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0) if p <= max_lead + 1e-9]
  print("\n===== skill vs lead time =====")
  print(f"{'method':>20}", end="")
  for probe in probes:
    print(f"{probe*1000:>8.0f}ms", end="")
  print()
  for label in labels:
    series = {r["lead_s"]: r["skill"] for r in rows if r["label"] == label}
    leads = sorted(series)
    print(f"{label:>20}", end="")
    for probe in probes:
      nearest = min(leads, key=lambda x: abs(x - probe))
      print(f"{series[nearest]:>+10.3f}", end="")
    print()


if __name__ == "__main__":
  main()
