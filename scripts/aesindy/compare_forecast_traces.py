"""Draw measured-versus-forecast traces for every method on one figure.

The skill-versus-lead curves are the quantitative result, but they summarise
each forecast to a single correlation and hide what the models actually
produce. This script plots the forecasts themselves, in microvolts, with one
row per method and the same held-out trial and forecast origin in every row,
so the comparison is like for like.

The autoencoder's forecasts are read from the ``.npz`` files that
``scripts/aesindy/run_lfp.py`` writes. The baselines are recomputed here,
which is cheap: fitting Hankel-SVD coordinates and a linear SINDy model takes
seconds, against roughly an hour to retrain an autoencoder.

Run with:

    python scripts/aesindy/compare_forecast_traces.py \\
      --aesindy-npz outputs/aesindy/aesindy_forecast_traces_fixation.npz

"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(_PROJECT_ROOT))
_INTERNALS_DIR = _PROJECT_ROOT / "scripts" / "pysindy" / "internals_probes"
if str(_INTERNALS_DIR) not in sys.path:
  sys.path.insert(0, str(_INTERNALS_DIR))

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

LOWPASS_HZ = 35.0
# The persistence curve reaches zero here; forecasts beyond it are not
# expected to track the signal, and the line makes that explicit.
PREDICTABILITY_LIMIT_MS = 350.0
OUTPUT_DIR = _PROJECT_ROOT / "outputs/aesindy"


def baseline_forecasts(
  n_leads: int, origin_stride_s: float, dt: float
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
  """Recompute delay-coordinate and Hankel-SVD forecasts on the same origins.

  Args:
    n_leads: Forecast length in samples, matching the autoencoder arrays.
    origin_stride_s: Spacing between forecast origins, in seconds.
    dt: Processed sample interval in seconds.

  Returns:
    A mapping from method label to ``(predicted, measured)`` arrays with shape
    ``(n_forecasts, n_leads)``.
  """
  # Imported lazily: these pull in PySINDy, which is deliberately absent from
  # the TensorFlow environment used to train the autoencoder.
  from forecast_skill import build_delay_method, build_havok_method, collect_forecasts
  from unbias_comparison import CASES

  train_ids, test_ids = load_archived_split()
  data = TrialData.load(MAT_FILE)
  train_traces = channel_traces(
    data, channel=CHANNEL, trials=train_ids, downsample=DOWNSAMPLE,
    lowpass_hz=LOWPASS_HZ, normalize="none",
  )
  test_traces = channel_traces(
    data, channel=CHANNEL, trials=test_ids, downsample=DOWNSAMPLE,
    lowpass_hz=LOWPASS_HZ, normalize="none",
  )

  max_lead_s = (n_leads - 1) * dt
  methods = {
    "delay SINDy (nd4)": build_delay_method(
      next(c for c in CASES if c.label == "nd4"), train_traces, dt
    ),
    "HAVOK r=3, linear": build_havok_method(
      train_traces, dt, n_delays=80, n_modes=3, degree=1, threshold=1000.0
    ),
  }
  results = {}
  for label, method in methods.items():
    predicted, measured = collect_forecasts(
      method, test_traces, dt, max_lead_s, origin_stride_s, sim_timeout_s=30.0
    )
    if predicted.shape[0]:
      results[label] = (predicted, measured)
  return results


def plot_comparison(
  series: dict[str, tuple[np.ndarray, np.ndarray]], dt: float,
  output_dir: Path, n_columns: int = 4,
) -> Path:
  """Plot one row per method, with matched forecast origins down the columns.

  Args:
    series: Mapping from method label to ``(predicted, measured)``.
    dt: Processed sample interval in seconds.
    output_dir: Directory to write the figure into.
    n_columns: Number of forecast origins to show.

  Returns:
    Path of the written figure.
  """
  labels = list(series)
  n_available = min(p.shape[0] for p, _ in series.values())
  picks = np.linspace(0, n_available - 1, n_columns).astype(int)

  fig, axes = plt.subplots(
    len(labels), n_columns, figsize=(3.6 * n_columns, 2.3 * len(labels)),
    squeeze=False, sharex=True,
  )
  for row, label in enumerate(labels):
    predicted, measured = series[label]
    time_ms = np.arange(predicted.shape[1]) * dt * 1000.0
    for column, pick in enumerate(picks):
      ax = axes[row][column]
      ax.plot(time_ms, measured[pick], color="tab:blue", lw=1.0)
      ax.plot(time_ms, predicted[pick], color="tab:orange", lw=1.0, ls="--")
      ax.axvline(PREDICTABILITY_LIMIT_MS, color="gray", ls=":", lw=0.9)
      ax.tick_params(labelsize=7)
      # Held to the measured signal's range. Forecasts that diverge leave the
      # frame, which is the honest picture -- rescaling to fit them would
      # flatten the measurement into a straight line.
      span = float(np.max(np.abs(measured[pick]))) * 1.6
      ax.set_ylim(-span, span)
      if column == 0:
        ax.set_ylabel(f"{label}\nx0 (uV)", fontsize=8)
      if row == 0:
        ax.set_title(f"forecast {pick}", fontsize=8)
      if row == len(labels) - 1:
        ax.set_xlabel("lead time (ms)", fontsize=8)
  fig.suptitle(
    "Measured (blue) versus forecast (orange dashed) on held-out trials\n"
    f"dotted line: ~{PREDICTABILITY_LIMIT_MS:.0f} ms predictability limit; "
    "y-axis fixed to the measured range",
    fontsize=11,
  )
  fig.tight_layout()
  path = output_dir / "forecast_traces_comparison.png"
  fig.savefig(path, dpi=150)
  plt.close(fig)
  return path


def main() -> None:
  """Combine autoencoder and baseline forecasts into one comparison figure."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--aesindy-npz", type=Path,
    default=OUTPUT_DIR / "aesindy_forecast_traces_fixation.npz",
    help="Traces written by run_lfp.py for a held-out group.",
  )
  parser.add_argument("--origin-stride", type=float, default=0.5)
  parser.add_argument("--columns", type=int, default=4)
  parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
  args = parser.parse_args()

  if not args.aesindy_npz.exists():
    raise SystemExit(
      f"{args.aesindy_npz} not found. It is written by a run of "
      f"scripts/aesindy/run_lfp.py that produced usable forecasts; runs whose "
      f"latent ODE diverged write no traces."
    )
  archive = np.load(args.aesindy_npz)
  ae_predicted, ae_measured = archive["predicted"], archive["measured"]
  dt = float(archive["dt"])
  print(f"autoencoder: {ae_predicted.shape[0]} forecasts, "
        f"{ae_predicted.shape[1]} leads, dt={dt*1000:.0f} ms")

  args.out_dir.mkdir(parents=True, exist_ok=True)
  series = {"AE-SINDy": (ae_predicted, ae_measured)}
  print("recomputing baselines ...", flush=True)
  series.update(
    baseline_forecasts(ae_predicted.shape[1], args.origin_stride, dt)
  )
  for label, (predicted, _) in series.items():
    print(f"  {label:22} {predicted.shape[0]} forecasts")

  print(f"\nwrote {plot_comparison(series, dt, args.out_dir, args.columns)}")


if __name__ == "__main__":
  main()
