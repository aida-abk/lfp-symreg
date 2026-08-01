"""Re-score a trained autoencoder from disk, with amplitude as well as shape.

Correlation is scale-invariant. A latent operator with a positive eigenvalue
produces a trajectory whose amplitude grows exponentially, and if the shape is
roughly right that trajectory still scores a high correlation while being
useless as a simulation. The linear latent_dim=12 run has an eigenvalue of
+18.6, i.e. an e-folding time of 54 ms, so over a one-second forecast its
amplitude grows by about eight orders of magnitude -- and it still scored
+0.469 at 50 ms.

Nothing in the correlation-versus-lead curve reveals that. This script adds
the two diagnostics that do, mirroring ``models/validation.py``:

    amplitude_ratio(tau)  spread of the forecasts at lead tau, divided by the
                          spread of the measurements at the same lead. 1.0 is
                          correct; >>1 is exploding, <<1 is collapsing.
    growth factor         median |prediction| at lead tau over |prediction| at
                          the origin, which states the explosion directly.

It also writes the measured-versus-predicted traces, because a plot of the
signal against the forecast settles in one glance what a correlation cannot.

Training is not repeated: the encoder and decoder are restored from the
SavedModel that ``aesindy`` writes into the run directory, and the latent
coefficients from ``aesindy_coefficients.csv``.

Run with:

    python scripts/aesindy/rescore_saved_model.py --run-dir outputs/aesindy_lin_d12

"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(_PROJECT_ROOT))

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
from models.polynomial_library import (  # noqa: E402
  polynomial_exponents,
  polynomial_library_numpy,
)
from models.sindy import delay_embed_trace  # noqa: E402

LOWPASS_HZ = 35.0
PREDICTABILITY_LIMIT_MS = 350.0


def load_saved_parts(run_dir: Path):
  """Restore the encoder, decoder and latent coefficients from a run directory.

  Args:
    run_dir: Directory written by ``scripts/aesindy/run_lfp.py``.

  Returns:
    ``(encoder, decoder, coefficients, config)``.

  Raises:
    SystemExit: If the SavedModel is absent or does not expose the sub-networks.
  """
  import tensorflow as tf

  config = json.loads((run_dir / "aesindy_config.json").read_text())
  coefficients = np.loadtxt(run_dir / "aesindy_coefficients.csv",
                            delimiter=",", ndmin=2)

  candidates = sorted(p for p in run_dir.iterdir()
                      if p.is_dir() and (p / "saved_model.pb").exists())
  if not candidates:
    raise SystemExit(
      f"No SavedModel under {run_dir}. Expected a results_*/ directory "
      f"containing saved_model.pb; without it the encoder and decoder cannot "
      f"be restored and the run must be repeated."
    )
  restored = tf.saved_model.load(str(candidates[-1]))
  encoder = getattr(restored, "encoder", None)
  decoder = getattr(restored, "decoder", None)
  if encoder is None or decoder is None:
    raise SystemExit(
      f"The SavedModel at {candidates[-1]} does not expose .encoder/.decoder. "
      f"Re-run the configuration with the current run_lfp.py, which saves the "
      f"forecast traces directly."
    )
  return encoder, decoder, coefficients, config


def forecast(
  encoder, decoder, coefficients: np.ndarray, exponents: list,
  traces: list[np.ndarray], input_dim: int, dt: float, signal_scale: float,
  max_lead_s: float, origin_stride_s: float,
) -> tuple[np.ndarray, np.ndarray]:
  """Forecast held-out trials from many origins.

  Args:
    encoder: Restored encoder network.
    decoder: Restored decoder network.
    coefficients: Latent coefficient matrix.
    exponents: Polynomial library exponents.
    traces: Held-out traces, already divided by ``signal_scale``.
    input_dim: Hankel embedding dimension.
    dt: Processed sample interval in seconds.
    signal_scale: Amplitude scale to restore microvolts.
    max_lead_s: Longest lead time in seconds.
    origin_stride_s: Spacing between origins in seconds.

  Returns:
    ``(predicted, measured)`` in microvolts, shape ``(n_forecasts, n_leads)``.
  """
  import tensorflow as tf
  from scipy.integrate import solve_ivp

  n_leads = int(round(max_lead_s / dt)) + 1
  stride = max(int(round(origin_stride_s / dt)), 1)
  times = np.arange(n_leads) * dt

  def rhs(_t, z):
    return (polynomial_library_numpy(z[None, :], exponents) @ coefficients)[0]

  predicted_rows, measured_rows = [], []
  for trace in traces:
    embedded = delay_embed_trace(trace, n_delays=input_dim, delay=1)
    latent = np.asarray(
      encoder(tf.convert_to_tensor(embedded, dtype=tf.float32)), dtype=float
    )
    for origin in range(0, max(len(embedded) - n_leads, 0), stride):
      try:
        sol = solve_ivp(rhs, (0.0, times[-1]), latent[origin], t_eval=times,
                        method="LSODA", rtol=1e-6, atol=1e-8)
      except Exception:
        continue
      if not sol.success or sol.y.shape[1] != n_leads:
        continue
      trajectory = sol.y.T
      if not np.all(np.isfinite(trajectory)):
        continue
      decoded = np.asarray(
        decoder(tf.convert_to_tensor(trajectory, dtype=tf.float32)), dtype=float
      )
      predicted_rows.append(decoded[:, 0] * signal_scale)
      measured_rows.append(embedded[origin:origin + n_leads, 0] * signal_scale)

  if not predicted_rows:
    return np.empty((0, n_leads)), np.empty((0, n_leads))
  return np.vstack(predicted_rows), np.vstack(measured_rows)


def amplitude_diagnostics(
  predicted: np.ndarray, measured: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
  """Return per-lead amplitude ratio and growth factor.

  Args:
    predicted: Forecasts, shape ``(n_forecasts, n_leads)``.
    measured: Measurements with the same shape.

  Returns:
    ``(amplitude_ratio, growth_factor)``. The ratio compares forecast spread to
    measured spread at each lead; 1.0 means the amplitude is right. The growth
    factor compares typical forecast magnitude at each lead to its value at the
    origin, so exponential blow-up appears directly.
  """
  measured_spread = np.std(measured, axis=0)
  measured_spread[measured_spread < 1e-12] = np.nan
  amplitude_ratio = np.std(predicted, axis=0) / measured_spread

  origin_magnitude = np.median(np.abs(predicted[:, 0]))
  origin_magnitude = max(origin_magnitude, 1e-12)
  growth_factor = np.median(np.abs(predicted), axis=0) / origin_magnitude
  return amplitude_ratio, growth_factor


def plot_diagnosis(
  predicted, measured, skill, persistence, amplitude_ratio, growth_factor,
  dt: float, run_dir: Path, name: str,
) -> Path:
  """Plot skill, amplitude behaviour and example traces together.

  Args:
    predicted: Forecasts in microvolts.
    measured: Measurements in microvolts.
    skill: Correlation at each lead.
    persistence: Persistence correlation at each lead.
    amplitude_ratio: Forecast spread over measured spread, per lead.
    growth_factor: Forecast magnitude relative to its origin, per lead.
    dt: Processed sample interval in seconds.
    run_dir: Directory to write into.
    name: Run label used in the title.

  Returns:
    Path of the written figure.
  """
  leads_ms = np.arange(skill.size) * dt * 1000.0
  fig = plt.figure(figsize=(15, 7.5))
  grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.1])

  ax = fig.add_subplot(grid[0, 0])
  ax.plot(leads_ms, skill, lw=1.5, label="model")
  ax.plot(leads_ms, persistence, color="black", ls="--", lw=1.8, label="persistence")
  ax.axhline(0, color="gray", lw=0.6)
  ax.axvline(PREDICTABILITY_LIMIT_MS, color="gray", ls=":", lw=0.9)
  ax.set_xlabel("lead (ms)"); ax.set_ylabel("correlation")
  ax.set_title("shape: correlation", fontsize=10); ax.legend(fontsize=8)
  ax.grid(alpha=0.3)

  ax = fig.add_subplot(grid[0, 1])
  ax.semilogy(leads_ms, np.clip(amplitude_ratio, 1e-6, None), lw=1.5)
  ax.axhline(1.0, color="black", ls="--", lw=1.5)
  ax.axvline(PREDICTABILITY_LIMIT_MS, color="gray", ls=":", lw=0.9)
  ax.set_xlabel("lead (ms)"); ax.set_ylabel("forecast sd / measured sd")
  ax.set_title("amplitude: 1.0 is correct", fontsize=10); ax.grid(alpha=0.3)

  ax = fig.add_subplot(grid[0, 2])
  ax.semilogy(leads_ms, np.clip(growth_factor, 1e-6, None), lw=1.5)
  ax.axhline(1.0, color="black", ls="--", lw=1.5)
  ax.set_xlabel("lead (ms)"); ax.set_ylabel("|forecast| / |forecast at origin|")
  ax.set_title("growth relative to start", fontsize=10); ax.grid(alpha=0.3)

  picks = np.linspace(0, predicted.shape[0] - 1, 3).astype(int)
  for column, pick in enumerate(picks):
    ax = fig.add_subplot(grid[1, column])
    ax.plot(leads_ms, measured[pick], color="tab:blue", lw=1.1, label="measured")
    ax.plot(leads_ms, predicted[pick], color="tab:orange", lw=1.1, ls="--",
            label="forecast")
    span = float(np.max(np.abs(measured[pick]))) * 1.8
    ax.set_ylim(-span, span)
    ax.axvline(PREDICTABILITY_LIMIT_MS, color="gray", ls=":", lw=0.9)
    ax.set_xlabel("lead (ms)")
    if column == 0:
      ax.set_ylabel("x0 (uV)")
      ax.legend(fontsize=8)
    ax.set_title(f"forecast {pick} (y-axis fixed to measured)", fontsize=9)
    ax.grid(alpha=0.3)

  fig.suptitle(f"{name}: shape, amplitude, and actual traces", fontsize=12)
  fig.tight_layout()
  path = run_dir / "rescore_diagnosis.png"
  fig.savefig(path, dpi=150)
  plt.close(fig)
  return path


def main() -> None:
  """Re-score one saved run and report shape alongside amplitude."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--run-dir", type=Path, required=True)
  parser.add_argument("--max-lead", type=float, default=1.0)
  parser.add_argument("--origin-stride", type=float, default=0.5)
  args = parser.parse_args()

  encoder, decoder, coefficients, config = load_saved_parts(args.run_dir)
  input_dim = config["input_dim"]
  exponents = polynomial_exponents(config["latent_dim"], config["poly_order"])
  print(f"restored {args.run_dir.name}: latent_dim={config['latent_dim']}, "
        f"poly_order={config['poly_order']}, "
        f"{int(np.count_nonzero(np.abs(coefficients) > 1e-12))} coefficients")

  train_ids, test_ids = load_archived_split()
  data = TrialData.load(MAT_FILE)
  dt = DOWNSAMPLE / data.fs
  train_traces = channel_traces(data, channel=CHANNEL, trials=train_ids,
                                downsample=DOWNSAMPLE, lowpass_hz=LOWPASS_HZ,
                                normalize="none")
  test_traces = channel_traces(data, channel=CHANNEL, trials=test_ids,
                               downsample=DOWNSAMPLE, lowpass_hz=LOWPASS_HZ,
                               normalize="none")
  signal_scale = float(np.std(np.concatenate(train_traces))) or 1.0
  scaled_test = [t / signal_scale for t in test_traces]

  predicted, measured = forecast(
    encoder, decoder, coefficients, exponents, scaled_test, input_dim, dt,
    signal_scale, args.max_lead, args.origin_stride,
  )
  print(f"{predicted.shape[0]} usable forecasts")
  if predicted.shape[0] == 0:
    print("No forecast integrated; nothing to score.")
    return

  skill = skill_by_lead(predicted, measured)
  persistence = persistence_by_lead(measured)
  amplitude_ratio, growth_factor = amplitude_diagnostics(predicted, measured)
  leads = np.arange(skill.size) * dt

  np.savez_compressed(args.run_dir / "rescore_traces.npz",
                      predicted=predicted, measured=measured, dt=dt)
  with open(args.run_dir / "rescore_metrics.csv", "w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["lead_s", "skill", "persistence", "amplitude_ratio",
                     "growth_factor"])
    for row in zip(leads, skill, persistence, amplitude_ratio, growth_factor):
      writer.writerow([f"{v:.6g}" for v in row])

  print(f"\n{'lead':>8} {'corr':>8} {'pers':>8} {'amp ratio':>11} {'growth':>11}")
  for probe in (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 1.0):
    if probe > leads[-1] + 1e-9:
      continue
    i = int(round(probe / dt))
    print(f"{probe*1000:>6.0f}ms {skill[i]:>+8.3f} {persistence[i]:>+8.3f} "
          f"{amplitude_ratio[i]:>11.3g} {growth_factor[i]:>11.3g}")

  print("\namplitude ratio far from 1.0 means the forecast is the wrong size,")
  print("which correlation cannot see. Judge the traces, not the correlation.")
  print(f"\nwrote {plot_diagnosis(predicted, measured, skill, persistence, amplitude_ratio, growth_factor, dt, args.run_dir, args.run_dir.name)}")


if __name__ == "__main__":
  main()
