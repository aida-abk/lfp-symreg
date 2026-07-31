"""Score simulated trajectories on waveform SHAPE rather than point-wise RMSE.

Motivation
----------
The saved ``global_analysis`` sweeps record only RMSE and completion status. RMSE
is the wrong criterion for a stochastic signal: a flat line has no spectral
content yet achieves the *lowest possible* RMSE (equal to the signal's own RMS),
while a model that reproduces the spectrum with the wrong phase is penalised
twice and lands near ``sqrt(2)`` times that value. This script therefore records
spectral and distributional agreement, and keeps RMSE only as a secondary column.

Two reference rows are simulated alongside every configuration:

* ``flat_line`` -- constant zero. Minimises RMSE, scores nothing on shape. Shows
  how badly RMSE misranks.
* ``phase_surrogate`` -- the measured trial with its Fourier phases randomised.
  Identical power spectrum, destroyed phase alignment. This is the realistic
  ceiling: a model that captures the spectrum but not the phase should score
  like this, so it is the target to beat on shape metrics.

Configurations are restricted to ``n_delays=2``, the regime where 95% of the
prior sweep's configurations completed simulation, so numerical instability does
not confound the comparison. Fixed settings match the ``global_analysis``
dataset (raw signal, ``normalize_columns=True``, ``alpha=0.05``) so results are
directly comparable with it.

Run with:

    .venv/bin/python scripts/pysindy/shape_analysis.py
"""
from __future__ import annotations

import csv
import json
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
from scipy import signal as scipy_signal  # noqa: E402

from load_data.convert import MAT_FILE, TrialData  # noqa: E402
from load_data.preprocessing import channel_traces  # noqa: E402
from models.sindy import (  # noqa: E402
  SINDyConfig,
  delay_embed_trace,
  delay_embed_trajectories,
  fit_sindy_model,
)
from models.validation import (  # noqa: E402
  SimulationConfig,
  evaluate_simulation,
  psd_similarity,
  simulate_model_detailed_hard_timeout,
  waveform_correlation,
)

csv.field_size_limit(10 * 1024 * 1024)

# --- Fixed backbone (matches the global_analysis sweeps) --------------------
CHANNEL = 0
DOWNSAMPLE = 2
LOWPASS_HZ = 35.0
N_DELAYS = 2          # the 95%-stable regime
DELAY_SAMPLES = 5     # highest completion rate in the prior sweep
SMOOTH_WINDOW = 9
SMOOTHING_POLYORDER = 3
NORMALIZE_COLUMNS = True
ALPHA = 0.05
MAX_ITER = 20

DEGREES = [2, 3, 5, 7, 9]
THRESHOLDS = [1000.0, 10000.0, 20000.0, 50000.0]

SIMULATION_HORIZON_S = 6.0
SIMULATION_WALL_TIMEOUT_S = 20.0
AUTOCORR_MAX_LAG_S = 0.5

SPLIT_METADATA_DIR = (
  _PROJECT_ROOT / "outputs/pysindy/global_analysis/raw_grid_deg2357_t20000/parts"
)
OUTPUT_DIR = _PROJECT_ROOT / "outputs/pysindy/shape_analysis"


@dataclass(frozen=True)
class ShapeMetrics:
  """Shape-oriented agreement between a measured and a candidate waveform.

  Attributes:
    psd_similarity: Correlation of normalized Welch PSD shapes. Unitless, -1..1.
    peak_frequency_error_hz: Absolute difference in dominant Welch peak.
    autocorrelation_similarity: Correlation of autocorrelation functions.
    distribution_ks: Two-sample KS statistic between amplitude distributions.
    collapse_std_ratio: Candidate tail SD divided by measured SD.
    x0_rmse: Point-wise RMSE, retained only as a secondary reference.
    x0_correlation: Point-wise correlation, retained for reference.
  """

  psd_similarity: float
  peak_frequency_error_hz: float
  autocorrelation_similarity: float
  distribution_ks: float
  collapse_std_ratio: float
  x0_rmse: float
  x0_correlation: float


def spectral_peak_hz(x: np.ndarray, fs: float) -> float:
  """Return the frequency carrying the most Welch power.

  Args:
    x: One-dimensional waveform.
    fs: Sampling frequency in hertz.

  Returns:
    Peak frequency in hertz, or NaN when the trace is too short or constant.
  """
  nperseg = min(256, x.size)
  if nperseg < 8 or np.std(x) == 0:
    return float("nan")
  freqs, power = scipy_signal.welch(x, fs=fs, nperseg=nperseg)
  return float(freqs[int(np.argmax(power))])


def autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
  """Return the normalized autocorrelation of a waveform up to ``max_lag``.

  Args:
    x: One-dimensional waveform.
    max_lag: Largest lag in samples.

  Returns:
    Autocorrelation values for lags ``0..max_lag``. All-NaN for a constant
    input, since autocorrelation is undefined without variance.
  """
  values = np.asarray(x, dtype=float)
  values = values - np.mean(values)
  denominator = np.dot(values, values)
  if denominator == 0:
    return np.full(max_lag + 1, np.nan)
  full = np.correlate(values, values, mode="full")[len(values) - 1 :]
  return full[: max_lag + 1] / denominator


def compute_shape_metrics(
  measured: np.ndarray, candidate: np.ndarray, fs: float
) -> ShapeMetrics:
  """Score one candidate waveform against a measured waveform on shape.

  Args:
    measured: Measured state trajectory with shape ``(time, state)``.
    candidate: Candidate trajectory with shape ``(time, state)``.
    fs: Processed sampling frequency in hertz.

  Returns:
    Populated :class:`ShapeMetrics`.
  """
  n = min(measured.shape[0], candidate.shape[0])
  measured, candidate = measured[:n], candidate[:n]
  target, predicted = measured[:, 0], candidate[:, 0]

  standard = evaluate_simulation(
    measured, candidate, fs=fs,
    config=SimulationConfig(simulation_horizon_s=n / fs),
  )
  max_lag = int(round(AUTOCORR_MAX_LAG_S * fs))
  acf_target = autocorrelation(target, max_lag)
  acf_predicted = autocorrelation(predicted, max_lag)
  if np.any(np.isnan(acf_target)) or np.any(np.isnan(acf_predicted)):
    acf_similarity = float("nan")
  else:
    acf_similarity = waveform_correlation(acf_target, acf_predicted)

  peak_target = spectral_peak_hz(target, fs)
  peak_predicted = spectral_peak_hz(predicted, fs)
  peak_error = (
    float("nan")
    if np.isnan(peak_target) or np.isnan(peak_predicted)
    else abs(peak_target - peak_predicted)
  )

  return ShapeMetrics(
    psd_similarity=psd_similarity(target, predicted, fs=fs),
    peak_frequency_error_hz=peak_error,
    autocorrelation_similarity=acf_similarity,
    distribution_ks=float(standard["distribution_ks"]),
    collapse_std_ratio=float(standard["collapse_std_ratio"]),
    x0_rmse=float(standard["x0_rmse"]),
    x0_correlation=float(standard["x0_correlation"]),
  )


def phase_randomized(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
  """Return a surrogate with the same power spectrum but randomized phases.

  Args:
    x: One-dimensional waveform.
    rng: Random generator supplying the replacement phases.

  Returns:
    A real-valued surrogate the same length as ``x``.
  """
  spectrum = np.fft.rfft(x)
  phases = rng.uniform(0, 2 * np.pi, spectrum.shape)
  phases[0] = 0.0
  if x.size % 2 == 0:
    phases[-1] = 0.0
  return np.fft.irfft(np.abs(spectrum) * np.exp(1j * phases), n=x.size)


def load_split(metadata_dir: Path) -> tuple[list[int], list[int]]:
  """Return (train_ids, test_ids) from the first metadata JSON in a directory."""
  candidates = sorted(metadata_dir.glob("*_metadata.json"))
  if not candidates:
    raise FileNotFoundError(f"No *_metadata.json found under {metadata_dir}")
  split = json.loads(candidates[0].read_text())["split"]
  return split["train_trial_ids"], split["test_trial_ids"]


def summarize(values: list[float]) -> float:
  """Return the median of the finite entries, or NaN when none are finite."""
  finite = [v for v in values if np.isfinite(v)]
  return float(np.median(finite)) if finite else float("nan")


def run() -> None:
  """Fit every configuration, simulate all held-out trials, and score shape."""
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  train_ids, test_ids = load_split(SPLIT_METADATA_DIR)

  print(f"Loading {MAT_FILE} ...", flush=True)
  data = TrialData.load(MAT_FILE)
  dt = DOWNSAMPLE / data.fs
  fs = 1.0 / dt
  print(f"dt={dt:g}s  fs={fs:g}Hz  train={len(train_ids)}  test={len(test_ids)}")
  print(f"n_delays={N_DELAYS} delay={DELAY_SAMPLES} smooth={SMOOTH_WINDOW} "
        f"lowpass={LOWPASS_HZ:g}Hz  degrees={DEGREES}  thresholds={THRESHOLDS}\n")

  train_traces = channel_traces(
    data, channel=CHANNEL, trials=train_ids, downsample=DOWNSAMPLE,
    lowpass_hz=LOWPASS_HZ, normalize="none",
  )
  test_traces = channel_traces(
    data, channel=CHANNEL, trials=test_ids, downsample=DOWNSAMPLE,
    lowpass_hz=LOWPASS_HZ, normalize="none",
  )
  embedded_train = delay_embed_trajectories(
    train_traces, n_delays=N_DELAYS, delay=DELAY_SAMPLES
  )
  embedded_test = [
    delay_embed_trace(t, n_delays=N_DELAYS, delay=DELAY_SAMPLES) for t in test_traces
  ]
  n_keep = int(round(SIMULATION_HORIZON_S * fs)) + 1
  embedded_test = [t[:n_keep] for t in embedded_test]

  rows: list[dict] = []

  # --- Reference rows -------------------------------------------------------
  rng = np.random.default_rng(0)
  for label in ("flat_line", "phase_surrogate"):
    per_trial: list[ShapeMetrics] = []
    for measured in embedded_test:
      if label == "flat_line":
        candidate = np.zeros_like(measured)
      else:
        candidate = np.column_stack(
          [phase_randomized(measured[:, c], rng) for c in range(measured.shape[1])]
        )
      per_trial.append(compute_shape_metrics(measured, candidate, fs))
    rows.append(_aggregate_row(label, None, None, None, per_trial, len(embedded_test)))
    print(f"  [reference] {label:<16} "
          f"psd={rows[-1]['psd_similarity']:+.3f} "
          f"acf={rows[-1]['autocorrelation_similarity']:+.3f} "
          f"rmse={rows[-1]['x0_rmse']:.1f}")

  # --- Configurations -------------------------------------------------------
  print()
  for degree in DEGREES:
    for threshold in THRESHOLDS:
      config = SINDyConfig(
        degree=degree, threshold=threshold, alpha=ALPHA,
        normalize_columns=NORMALIZE_COLUMNS, smooth_window=SMOOTH_WINDOW,
        smoothing_polyorder=SMOOTHING_POLYORDER, verbose=False, max_iter=MAX_ITER,
      )
      model = fit_sindy_model(embedded_train, dt=dt, config=config)
      nonzero = int(np.count_nonzero(np.asarray(model.coefficients())))

      per_trial: list[ShapeMetrics] = []
      n_completed = 0
      for measured in embedded_test:
        sim = simulate_model_detailed_hard_timeout(
          model, initial_state=measured[0], dt=dt,
          horizon_s=(len(measured) - 1) * dt,
          wall_timeout_s=SIMULATION_WALL_TIMEOUT_S,
        )
        if sim.completed and sim.trajectory is not None:
          n_completed += 1
          per_trial.append(compute_shape_metrics(measured, sim.trajectory, fs))

      row = _aggregate_row(
        f"deg{degree}_thr{threshold:g}", degree, threshold, nonzero,
        per_trial, len(embedded_test),
      )
      row["n_completed"] = n_completed
      rows.append(row)
      print(f"  deg={degree} thr={threshold:>7g} terms={nonzero:>4} "
            f"completed={n_completed}/{len(embedded_test)}  "
            f"psd={row['psd_similarity']:+.3f} "
            f"acf={row['autocorrelation_similarity']:+.3f} "
            f"peakerr={row['peak_frequency_error_hz']:.2f}Hz "
            f"rmse={row['x0_rmse']:.1f}", flush=True)

  _write_csv(rows)
  _write_plot(rows)
  _print_summary(rows)


def _aggregate_row(
  label: str,
  degree: int | None,
  threshold: float | None,
  nonzero: int | None,
  per_trial: list[ShapeMetrics],
  n_trials: int,
) -> dict:
  """Collapse per-trial metrics into one median row."""
  return {
    "label": label,
    "degree": degree,
    "threshold": threshold,
    "nonzero_terms": nonzero,
    "n_trials_scored": len(per_trial),
    "n_completed": n_trials,
    "psd_similarity": summarize([m.psd_similarity for m in per_trial]),
    "autocorrelation_similarity": summarize(
      [m.autocorrelation_similarity for m in per_trial]
    ),
    "peak_frequency_error_hz": summarize(
      [m.peak_frequency_error_hz for m in per_trial]
    ),
    "distribution_ks": summarize([m.distribution_ks for m in per_trial]),
    "collapse_std_ratio": summarize([m.collapse_std_ratio for m in per_trial]),
    "x0_rmse": summarize([m.x0_rmse for m in per_trial]),
    "x0_correlation": summarize([m.x0_correlation for m in per_trial]),
  }


def _write_csv(rows: list[dict]) -> None:
  """Write one row per configuration and reference to ``shape_metrics.csv``."""
  path = OUTPUT_DIR / "shape_metrics.csv"
  with open(path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
  print(f"\nwrote {path}")


def _write_plot(rows: list[dict]) -> None:
  """Plot shape metrics per configuration against the two reference rows."""
  refs = {r["label"]: r for r in rows if r["degree"] is None}
  configs = [r for r in rows if r["degree"] is not None]
  fig, axes = plt.subplots(1, 2, figsize=(13, 5))

  for ax, key, title in (
    (axes[0], "psd_similarity", "PSD shape similarity (higher is better)"),
    (axes[1], "autocorrelation_similarity", "Autocorrelation similarity (higher is better)"),
  ):
    for threshold in THRESHOLDS:
      subset = [r for r in configs if r["threshold"] == threshold]
      ax.plot(
        [r["degree"] for r in subset], [r[key] for r in subset],
        marker="o", label=f"threshold={threshold:g}",
      )
    if "phase_surrogate" in refs:
      ax.axhline(refs["phase_surrogate"][key], color="tab:green", ls="--",
                 label="phase surrogate (spectrum-matched ceiling)")
    if "flat_line" in refs and np.isfinite(refs["flat_line"][key]):
      ax.axhline(refs["flat_line"][key], color="tab:red", ls=":", label="flat line")
    ax.set_xlabel("polynomial degree")
    ax.set_ylabel(key)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
  axes[0].legend(fontsize=7)
  fig.tight_layout()
  path = OUTPUT_DIR / "shape_metrics.png"
  fig.savefig(path, dpi=150)
  plt.close(fig)
  print(f"wrote {path}")


def _print_summary(rows: list[dict]) -> None:
  """Print a compact table ordered by PSD similarity."""
  print("\n===== SUMMARY (median over held-out trials) =====")
  print(f"{'label':<18} {'terms':>6} {'done':>5} {'psd':>7} {'acf':>7} "
        f"{'peakHz':>7} {'ks':>6} {'collapse':>9} {'rmse':>7} {'corr':>7}")
  ordered = sorted(
    rows, key=lambda r: (-1e9 if not np.isfinite(r["psd_similarity"])
                         else -r["psd_similarity"])
  )
  for r in ordered:
    print(f"{r['label']:<18} {str(r['nonzero_terms'] or '-'):>6} "
          f"{r['n_trials_scored']:>5} "
          f"{r['psd_similarity']:>7.3f} {r['autocorrelation_similarity']:>7.3f} "
          f"{r['peak_frequency_error_hz']:>7.2f} {r['distribution_ks']:>6.3f} "
          f"{r['collapse_std_ratio']:>9.3f} {r['x0_rmse']:>7.1f} "
          f"{r['x0_correlation']:>7.3f}")


if __name__ == "__main__":
  run()
