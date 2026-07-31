"""Measure the reconstruction ceiling of truncated Hankel-SVD coordinates.

``scripts/pysindy/havok_baseline.py`` fits SINDy in the leading ``r``
Hankel-SVD coordinates, simulates there, then reconstructs back into delay
space and scores column zero against the measured signal. That scoring is
directly comparable to the delay-coordinate runs, but it omits a control:
projecting onto ``r`` modes and back is itself lossy. Even a *perfect*
simulation cannot score better than the round trip

    signal -> delay embed -> project to r modes -> reconstruct -> signal

allows. That round trip is the ceiling, and it involves no fitting, no
optimizer, and no integration -- only the projection.

Without it, a mediocre score is ambiguous: the dynamics may have failed, or
the coordinates may simply be unable to represent the signal. This script
separates the two by measuring the ceiling on held-out trials for a range of
mode counts, using the same metrics as the fitted runs.

It also writes the leading mode shapes, since a mode is otherwise an abstract
object: each is a characteristic waveform the length of the embedding window.

Run with:

    .venv/bin/python scripts/pysindy/havok_mode_probe.py

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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from load_data.convert import MAT_FILE, TrialData  # noqa: E402
from load_data.preprocessing import channel_traces  # noqa: E402
from models.havok import HankelBasis, fit_hankel_basis, modes_for_variance  # noqa: E402
from models.validation import psd_similarity, waveform_correlation  # noqa: E402

from unbias_comparison import CHANNEL, DOWNSAMPLE, load_split  # noqa: E402

DEFAULT_N_DELAYS = 80
DEFAULT_DELAY = 1
DEFAULT_MODE_COUNTS = (1, 2, 3, 5, 8, 12, 16, 20, 24, 32, 48, 80)
OUTPUT_DIR = _PROJECT_ROOT / "outputs/pysindy/havok_mode_probe"


def reconstruction_ceiling(
  basis: HankelBasis, traces: list[np.ndarray], fs: float
) -> dict[str, float]:
  """Score the projection round trip on held-out traces, with no dynamics.

  Args:
    basis: Hankel-SVD coordinates fitted on training trials.
    traces: Held-out preprocessed traces in signal units.
    fs: Processed sampling frequency in hertz.

  Returns:
    Median correlation, PSD similarity, RMSE, and retained variance fraction
    for the reconstructed signal against the measured signal.
  """
  correlations, similarities, errors = [], [], []
  for trace in traces:
    embedded = basis.embed(trace)
    reconstructed = basis.reconstruct(embedded @ basis.modes)
    measured_signal = embedded[:, 0]
    predicted_signal = reconstructed[:, 0]
    correlations.append(waveform_correlation(measured_signal, predicted_signal))
    similarities.append(psd_similarity(measured_signal, predicted_signal, fs=fs))
    errors.append(
      float(np.sqrt(np.mean((measured_signal - predicted_signal) ** 2)))
    )
  return {
    "n_modes": basis.n_modes,
    "explained_variance": basis.explained_variance_ratio,
    "x0_correlation": float(np.median(correlations)),
    "psd_similarity": float(np.median(similarities)),
    "x0_rmse": float(np.median(errors)),
  }


def plot_modes(basis: HankelBasis, dt: float, n_show: int, output_dir: Path) -> Path:
  """Plot the leading mode shapes as waveforms in milliseconds.

  Args:
    basis: Fitted basis holding the modes.
    dt: Processed sample interval in seconds.
    n_show: Number of leading modes to draw.
    output_dir: Directory to write the figure into.

  Returns:
    Path of the written figure.
  """
  n_show = min(n_show, basis.n_modes)
  time_ms = np.arange(basis.n_delays) * dt * 1000.0
  n_cols = 3
  n_rows = int(np.ceil(n_show / n_cols))
  fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.1 * n_rows),
                           squeeze=False, sharex=True)
  for index in range(n_rows * n_cols):
    ax = axes[index // n_cols][index % n_cols]
    if index >= n_show:
      ax.axis("off")
      continue
    ax.plot(time_ms, basis.modes[:, index], lw=1.2)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_title(f"mode {index + 1}", fontsize=9)
    ax.tick_params(labelsize=7)
    if index // n_cols == n_rows - 1:
      ax.set_xlabel("lag (ms)", fontsize=8)
  fig.suptitle(
    f"Leading Hankel-SVD modes -- each is a {time_ms[-1]:.0f} ms waveform shape",
    fontsize=11,
  )
  fig.tight_layout()
  path = output_dir / "hankel_modes.png"
  fig.savefig(path, dpi=150)
  plt.close(fig)
  return path


def plot_ceiling(rows: list[dict], output_dir: Path) -> Path:
  """Plot reconstruction quality against retained mode count.

  Args:
    rows: Ceiling rows produced by :func:`reconstruction_ceiling`.
    output_dir: Directory to write the figure into.

  Returns:
    Path of the written figure.
  """
  counts = [r["n_modes"] for r in rows]
  fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
  for ax, key, label in zip(
    axes,
    ("x0_correlation", "psd_similarity", "x0_rmse"),
    ("correlation with measured", "PSD similarity", "RMSE (uV)"),
  ):
    ax.plot(counts, [r[key] for r in rows], marker="o")
    ax.set_xlabel("modes retained")
    ax.set_ylabel(label)
    ax.set_title(f"reconstruction ceiling: {key}", fontsize=9)
    ax.grid(alpha=0.3)
  fig.suptitle("Projection round trip only -- no dynamics, no fitting", fontsize=11)
  fig.tight_layout()
  path = output_dir / "reconstruction_ceiling.png"
  fig.savefig(path, dpi=150)
  plt.close(fig)
  return path


def main() -> None:
  """Measure and report the truncation ceiling for a range of mode counts."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--n-delays", type=int, default=DEFAULT_N_DELAYS)
  parser.add_argument("--delay", type=int, default=DEFAULT_DELAY)
  parser.add_argument("--lowpass", type=float, default=35.0)
  parser.add_argument(
    "--modes-list", default=",".join(str(m) for m in DEFAULT_MODE_COUNTS)
  )
  parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
  args = parser.parse_args()

  mode_counts = [int(m) for m in args.modes_list.split(",")]
  args.out_dir.mkdir(parents=True, exist_ok=True)

  train_ids, test_ids = load_split()
  print(f"Loading {MAT_FILE} ...", flush=True)
  data = TrialData.load(MAT_FILE)
  dt = DOWNSAMPLE / data.fs
  fs = 1.0 / dt

  train_traces = channel_traces(
    data, channel=CHANNEL, trials=train_ids, downsample=DOWNSAMPLE,
    lowpass_hz=args.lowpass, normalize="none",
  )
  test_traces = channel_traces(
    data, channel=CHANNEL, trials=test_ids, downsample=DOWNSAMPLE,
    lowpass_hz=args.lowpass, normalize="none",
  )

  full_basis = fit_hankel_basis(
    train_traces, n_delays=args.n_delays, delay=args.delay,
    n_modes=max(mode_counts),
  )
  window_ms = args.n_delays * dt * 1000.0
  print(f"\nembedding window: {args.n_delays} samples = {window_ms:.0f} ms "
        f"at {fs:.0f} Hz")
  for target in (0.9, 0.99, 0.999):
    print(f"  modes for {target:.1%} variance: "
          f"{modes_for_variance(full_basis.singular_values, target)}")
  print(f"\nwrote {plot_modes(full_basis, dt, 9, args.out_dir)}")

  print("\nReconstruction ceiling on the 9 held-out trials "
        "(projection only, no dynamics):")
  print(f"{'modes':>7} {'variance':>10} {'correlation':>13} {'psd':>9} {'rmse':>9}")
  rows = []
  for n_modes in mode_counts:
    basis = HankelBasis(
      modes=full_basis.modes[:, :n_modes],
      singular_values=full_basis.singular_values,
      n_delays=args.n_delays,
      delay=args.delay,
    )
    row = reconstruction_ceiling(basis, test_traces, fs)
    rows.append(row)
    print(f"{row['n_modes']:>7} {row['explained_variance']:>9.1%} "
          f"{row['x0_correlation']:>13.4f} {row['psd_similarity']:>9.3f} "
          f"{row['x0_rmse']:>9.2f}")

  path = args.out_dir / "reconstruction_ceiling.csv"
  with open(path, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
  print(f"\nwrote {path}")
  print(f"wrote {plot_ceiling(rows, args.out_dir)}")

  print("\nInterpretation: a fitted model scored in these coordinates cannot")
  print("exceed the ceiling for its mode count. Compare against the fitted")
  print("results in outputs/pysindy/havok_baseline/havok_baseline.csv.")


if __name__ == "__main__":
  main()
