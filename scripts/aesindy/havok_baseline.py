"""Does rotating into Hankel-SVD coordinates lift held-out prediction off zero?

The learning curve (``scripts/pysindy/learning_curve.py``) established that
delay-coordinate polynomial SINDy produces held-out waveform correlation
indistinguishable from zero at every training-set size and at both embedding
dimensions tested, while retaining anywhere from 4 to 74 terms. Neither data
volume nor model capacity is the binding constraint.

The remaining suspect is the coordinate system itself. This script tests it
directly: build the Hankel matrix, take its SVD on the *training* trials
only, and fit the same SINDy machinery in the leading singular-vector
coordinates instead of raw delay coordinates. Everything else -- channel,
preprocessing, trial split, simulation, metrics -- is held identical to the
learning-curve runs so the numbers are directly comparable.

Two outcomes, both decisive:

* Correlation lifts off zero for some configuration. The coordinate
  hypothesis is supported, and a learned (autoencoder) transformation is
  well motivated -- HAVOK's fixed linear rotation would be a floor, not a
  ceiling.
* Correlation stays at zero everywhere. A linear change of coordinates does
  not help, which is evidence that the obstacle is not the basis but the
  signal, and worth knowing before investing weeks in a deep autoencoder.

This is deliberately a generous test. It sweeps mode count, polynomial
degree, and threshold, and reports the best configuration found. Because the
best is chosen by looking at held-out scores, a positive result here is a
lead to confirm on a clean split, not a finished measurement. A negative
result under these generous conditions is strong.

Run with:

    .venv/bin/python scripts/pysindy/havok_baseline.py

"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pysindy as ps  # noqa: E402

from load_data.convert import MAT_FILE, TrialData  # noqa: E402
from load_data.preprocessing import channel_traces  # noqa: E402
from models.havok import HankelBasis, fit_hankel_basis, modes_for_variance  # noqa: E402
from models.validation import (  # noqa: E402
  SimulationConfig,
  evaluate_simulation,
  simulate_model_detailed,
)

# Held identical to the learning-curve and unbias-comparison runs so that the
# HAVOK numbers can be read directly against the nd2 / nd4 rows.
from unbias_comparison import (  # noqa: E402
  ALPHA,
  CHANNEL,
  DOWNSAMPLE,
  MAX_HORIZON_S,
  NORMALIZE_COLUMNS,
  SIM_WALL_TIMEOUT_S,
  load_split,
)

# The paper's real-data example uses input_dim=80 with unit delay spacing;
# that is the embedding this baseline mirrors.
DEFAULT_N_DELAYS = 80
DEFAULT_DELAY = 1
DEFAULT_MODES = (3, 5, 8, 12)
# Degree 1 is the central test: HAVOK's claim is that the leading Hankel-SVD
# coordinates evolve near-linearly. Degrees 2 and 3 check whether polynomial
# structure buys anything the linear model misses.
DEFAULT_DEGREES = (1, 2, 3)
# Coordinate scale after projection differs from raw microvolts, so the
# archived thresholds do not transfer. Zero is included as the most generous
# case: no thresholding at all, i.e. plain ridge regression.
DEFAULT_THRESHOLDS = (0.0, 100.0, 1000.0, 10000.0)

OUTPUT_DIR = _PROJECT_ROOT / "outputs/pysindy/havok_baseline"


@dataclass(frozen=True)
class HavokRun:
  """One HAVOK configuration to fit and score.

  Attributes:
    n_modes: Retained singular vectors. Unitless count.
    degree: Maximum polynomial degree in the SINDy library. Unitless.
    threshold: STLSQ coefficient-removal threshold in fitted-equation units.
  """

  n_modes: int
  degree: int
  threshold: float


def fit_in_mode_coordinates(
  mode_trajectories: list[np.ndarray], dt: float, run: HavokRun
):
  """Fit SINDy in Hankel-SVD coordinates.

  Optimizer settings mirror the archived configurations so that only the
  coordinate system differs from the delay-coordinate results.

  Args:
    mode_trajectories: Training trajectories with shape ``(time, n_modes)``.
    dt: Processed sample interval in seconds.
    run: Library and threshold settings.

  Returns:
    A fitted ``pysindy.SINDy`` model in mode coordinates.
  """
  model = ps.SINDy(
    optimizer=ps.STLSQ(
      threshold=run.threshold,
      alpha=ALPHA,
      normalize_columns=NORMALIZE_COLUMNS,
      verbose=False,
      max_iter=20,
    ),
    feature_library=ps.PolynomialLibrary(degree=run.degree),
  )
  model.fit(mode_trajectories, t=dt)
  return model


def score_run(
  run: HavokRun,
  basis: HankelBasis,
  test_traces: list[np.ndarray],
  train_traces: list[np.ndarray],
  dt: float,
) -> dict:
  """Fit one configuration and score it on the held-out trials.

  The simulated mode trajectory is reconstructed back into delay coordinates
  before scoring, so column zero is the predicted signal and every metric is
  computed on the same quantity as the delay-coordinate runs.

  Args:
    run: Configuration to fit.
    basis: Shared Hankel-SVD coordinates fitted on training trials only.
    test_traces: Held-out preprocessed traces.
    train_traces: Training preprocessed traces.
    dt: Processed sample interval in seconds.

  Returns:
    A result row holding the configuration, the fitted term count, and the
    median of each metric over held-out trials that reached the horizon.
  """
  fs = 1.0 / dt
  mode_train = [basis.project(t) for t in train_traces]
  model = fit_in_mode_coordinates(mode_train, dt, run)
  nonzero_terms = int(np.count_nonzero(np.asarray(model.coefficients())))

  metrics_list = []
  for trace in test_traces:
    measured_delay = basis.embed(trace)
    measured_modes = measured_delay @ basis.modes
    horizon = min((len(measured_modes) - 1) * dt, MAX_HORIZON_S)
    sim = simulate_model_detailed(
      model, initial_state=measured_modes[0], dt=dt, horizon_s=horizon,
      wall_timeout_s=SIM_WALL_TIMEOUT_S,
    )
    if not (sim.completed and sim.trajectory is not None):
      continue
    reconstructed = basis.reconstruct(sim.trajectory)
    n = min(len(measured_delay), len(reconstructed))
    metrics_list.append(
      evaluate_simulation(
        measured_delay[:n], reconstructed[:n], fs=fs,
        config=SimulationConfig(simulation_horizon_s=horizon),
      )
    )

  def median_metric(key: str) -> float:
    values = [float(m[key]) for m in metrics_list if np.isfinite(float(m[key]))]
    return float(np.median(values)) if values else float("nan")

  return {
    "n_modes": run.n_modes,
    "degree": run.degree,
    "threshold": run.threshold,
    "explained_variance": basis.explained_variance_ratio,
    "nonzero_terms": nonzero_terms,
    "n_completed": len(metrics_list),
    "n_test": len(test_traces),
    "psd_similarity": median_metric("psd_similarity"),
    "x0_correlation": median_metric("x0_correlation"),
    "collapse_std_ratio": median_metric("collapse_std_ratio"),
    "x0_rmse": median_metric("x0_rmse"),
  }


def plot_spectrum(basis: HankelBasis, output_dir: Path) -> Path:
  """Plot the Hankel singular value spectrum and cumulative variance.

  Args:
    basis: Fitted basis whose full spectrum is plotted.
    output_dir: Directory to write the figure into.

  Returns:
    Path of the written figure.
  """
  energy = basis.singular_values**2
  cumulative = np.cumsum(energy) / np.sum(energy)
  fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
  axes[0].semilogy(np.arange(1, len(basis.singular_values) + 1),
                   basis.singular_values, marker="o", ms=3)
  axes[0].set_xlabel("mode")
  axes[0].set_ylabel("singular value")
  axes[0].set_title("Hankel singular values")
  axes[0].grid(alpha=0.3)
  axes[1].plot(np.arange(1, len(cumulative) + 1), cumulative, marker="o", ms=3)
  for level in (0.9, 0.99):
    axes[1].axhline(level, color="tab:red", ls="--", lw=0.8)
    axes[1].text(len(cumulative) * 0.6, level + 0.005, f"{level:.0%}",
                 color="tab:red", fontsize=8)
  axes[1].set_xlabel("modes retained")
  axes[1].set_ylabel("cumulative variance")
  axes[1].set_title("Cumulative explained variance")
  axes[1].grid(alpha=0.3)
  fig.tight_layout()
  path = output_dir / "hankel_spectrum.png"
  fig.savefig(path, dpi=150)
  plt.close(fig)
  return path


def main() -> None:
  """Sweep HAVOK configurations and report the best held-out scores."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--n-delays", type=int, default=DEFAULT_N_DELAYS)
  parser.add_argument("--delay", type=int, default=DEFAULT_DELAY)
  parser.add_argument("--lowpass", type=float, default=35.0)
  parser.add_argument(
    "--modes-list", default=",".join(str(m) for m in DEFAULT_MODES)
  )
  parser.add_argument(
    "--degree-list", default=",".join(str(d) for d in DEFAULT_DEGREES)
  )
  parser.add_argument(
    "--threshold-list", default=",".join(str(t) for t in DEFAULT_THRESHOLDS)
  )
  parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
  args = parser.parse_args()

  modes_list = [int(m) for m in args.modes_list.split(",")]
  degrees = [int(d) for d in args.degree_list.split(",")]
  thresholds = [float(t) for t in args.threshold_list.split(",")]

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
  print(f"train {len(train_traces)} trials, held-out {len(test_traces)} trials")

  # One SVD at the largest requested mode count supplies every smaller
  # truncation, since the leading modes are nested.
  full_basis = fit_hankel_basis(
    train_traces, n_delays=args.n_delays, delay=args.delay,
    n_modes=max(modes_list),
  )
  print(f"\nHankel SVD on training trials only "
        f"(n_delays={args.n_delays}, delay={args.delay}):")
  for target in (0.9, 0.99, 0.999):
    print(f"  modes for {target:.1%} variance: "
          f"{modes_for_variance(full_basis.singular_values, target)}")
  print(f"wrote {plot_spectrum(full_basis, args.out_dir)}")

  rows: list[dict] = []
  total = len(modes_list) * len(degrees) * len(thresholds)
  index = 0
  for n_modes in modes_list:
    basis = HankelBasis(
      modes=full_basis.modes[:, :n_modes],
      singular_values=full_basis.singular_values,
      n_delays=args.n_delays,
      delay=args.delay,
    )
    for degree in degrees:
      for threshold in thresholds:
        index += 1
        run = HavokRun(n_modes=n_modes, degree=degree, threshold=threshold)
        row = score_run(run, basis, test_traces, train_traces, dt)
        rows.append(row)
        print(
          f"  [{index:>3}/{total}] modes={n_modes:>3} deg={degree} "
          f"thr={threshold:>8.0f} terms={row['nonzero_terms']:>5} "
          f"done={row['n_completed']}/{row['n_test']} "
          f"psd={row['psd_similarity']:+.3f} "
          f"corr={row['x0_correlation']:+.3f} "
          f"rmse={row['x0_rmse']:.1f}",
          flush=True,
        )

  path = args.out_dir / "havok_baseline.csv"
  with open(path, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
  print(f"\nwrote {path}")
  _report(rows)


def _report(rows: list[dict]) -> None:
  """Print the best configurations and the comparison against delay coordinates."""
  usable = [r for r in rows if r["n_completed"] > 0 and np.isfinite(r["x0_correlation"])]
  print(f"\n{len(usable)}/{len(rows)} configurations produced a usable simulation.")
  if not usable:
    print("No configuration simulated to the horizon on any held-out trial.")
    return

  best = max(usable, key=lambda r: abs(r["x0_correlation"]))
  print("\nbest |x0_correlation| found:")
  print(f"  modes={best['n_modes']} degree={best['degree']} "
        f"threshold={best['threshold']:g} terms={best['nonzero_terms']}")
  print(f"  x0_correlation = {best['x0_correlation']:+.4f}   "
        f"psd = {best['psd_similarity']:+.3f}   rmse = {best['x0_rmse']:.1f}")

  correlations = np.array([r["x0_correlation"] for r in usable])
  print(f"\nacross all {len(usable)} configurations: "
        f"mean {correlations.mean():+.4f}, sd {correlations.std():.4f}, "
        f"max |corr| {np.abs(correlations).max():.4f}")
  print("\ndelay-coordinate reference (learning curve, pooled over 31 fits):")
  print("  nd2  x0_correlation = +0.007 +/- 0.016")
  print("  nd4  x0_correlation = -0.000 +/- 0.022")


if __name__ == "__main__":
  main()
