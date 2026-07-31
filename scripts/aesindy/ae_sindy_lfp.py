"""Deep delay autoencoder on LFP, scored exactly like the PySINDy baselines.

This mirrors ``testcases/lorenzww_basic.py`` from
github.com/josephbakarji/deep-delay-autoencoder -- the reference repository's
only experimental-data example -- with LFP trials in place of the waterwheel
recordings. Their ``RealData.build_solution`` takes a list of trajectories,
one per recording, and Hankel-embeds each separately before stacking; that
maps directly onto trials, so the data layout needs no adaptation.

Held fixed to match the PySINDy runs so the numbers are comparable:
channel, downsample, low-pass, the archived 28/9 trial split, and the
forecast-skill metric with persistence as reference.

Two deliberate departures from the reference, both documented:

* The RK4 ``integral`` loss and the ``x0`` loss are not implemented. See
  ``models/ae_sindy.py``.
* Their train/test split is a temporal cut through the stacked Hankel matrix
  (``train_test_split(..., shuffle=False)``), which puts rows from the same
  recording on both sides. This script splits by *trial* instead, using the
  same archived split as every other result here, so held-out trials are
  genuinely unseen.

Run with:

    .venv/bin/python scripts/pysindy/ae_sindy_lfp.py

"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from load_data.convert import MAT_FILE, TrialData  # noqa: E402
from load_data.preprocessing import channel_traces  # noqa: E402
from models.ae_sindy import (  # noqa: E402
  AESINDy,
  AESINDyConfig,
  polynomial_library_numpy,
  train_ae_sindy,
)
from models.sindy import delay_embed_trace  # noqa: E402

from forecast_skill import persistence_by_lead, skill_by_lead  # noqa: E402
from unbias_comparison import CHANNEL, DOWNSAMPLE, load_split  # noqa: E402

LOWPASS_HZ = 35.0
# Savitzky-Golay window and polynomial order. The reference uses [21, 3] for
# its real-data case, and polyorder 3 is this project's standing assumption.
SG_WINDOW = 21
SG_POLYORDER = 3
OUTPUT_DIR = _PROJECT_ROOT / "outputs/pysindy/ae_sindy"


def build_hankel_dataset(
  traces: list[np.ndarray], input_dim: int, dt: float
) -> tuple[np.ndarray, np.ndarray]:
  """Delay-embed traces and their derivatives, per trial, then stack.

  Embedding each trial separately before stacking is what keeps a delay
  vector from ever spanning a trial boundary.

  Args:
    traces: Preprocessed traces in signal units.
    input_dim: Hankel embedding dimension.
    dt: Processed sample interval in seconds.

  Returns:
    ``(x, xdot)`` with shape ``(n_rows, input_dim)`` each.
  """
  x_blocks, xdot_blocks = [], []
  for trace in traces:
    derivative = savgol_filter(
      trace, window_length=SG_WINDOW, polyorder=SG_POLYORDER, deriv=1, delta=dt
    )
    x_blocks.append(delay_embed_trace(trace, n_delays=input_dim, delay=1))
    xdot_blocks.append(delay_embed_trace(derivative, n_delays=input_dim, delay=1))
  return np.vstack(x_blocks), np.vstack(xdot_blocks)


def simulate_latent(
  model: AESINDy, z0: np.ndarray, dt: float, n_steps: int
) -> np.ndarray | None:
  """Integrate the learned latent ODE from one initial condition.

  Args:
    model: Trained autoencoder holding the latent coefficients.
    z0: Initial latent state with shape ``(latent_dim,)``.
    dt: Processed sample interval in seconds.
    n_steps: Number of samples to produce, including the initial one.

  Returns:
    Latent trajectory with shape ``(n_steps, latent_dim)``, or ``None`` if the
    integration failed or left the region where it stays finite.
  """
  coefficients = (
    (model.coefficients * model.mask).detach().cpu().numpy().astype(float)
  )
  exponents = model.exponents

  def rhs(_t: float, z: np.ndarray) -> np.ndarray:
    return (polynomial_library_numpy(z[None, :], exponents) @ coefficients)[0]

  times = np.arange(n_steps) * dt
  try:
    solution = solve_ivp(
      rhs, (0.0, times[-1]), np.asarray(z0, dtype=float), t_eval=times,
      method="LSODA", rtol=1e-6, atol=1e-8,
    )
  except Exception:
    return None
  if not solution.success or solution.y.shape[1] != n_steps:
    return None
  trajectory = solution.y.T
  if not np.all(np.isfinite(trajectory)):
    return None
  return trajectory


def forecast_from_origins(
  model: AESINDy,
  traces: list[np.ndarray],
  mean: np.ndarray,
  scale: np.ndarray,
  config: AESINDyConfig,
  dt: float,
  max_lead_s: float,
  origin_stride_s: float,
) -> tuple[np.ndarray, np.ndarray]:
  """Forecast from many origins in each held-out trial.

  Args:
    model: Trained autoencoder.
    traces: Held-out preprocessed traces in signal units.
    mean: Per-coordinate standardization mean from the training set.
    scale: Per-coordinate standardization scale from the training set.
    config: Hyperparameters, for the embedding dimension.
    dt: Processed sample interval in seconds.
    max_lead_s: Longest lead time to forecast, in seconds.
    origin_stride_s: Spacing between forecast origins within a trial.

  Returns:
    ``(predicted, measured)`` arrays with shape ``(n_forecasts, n_leads)`` in
    signal units. Failed integrations are omitted.
  """
  import torch

  n_leads = int(round(max_lead_s / dt)) + 1
  stride = max(int(round(origin_stride_s / dt)), 1)
  predicted_rows, measured_rows = [], []

  model.eval()
  for trace in traces:
    embedded = delay_embed_trace(trace, n_delays=config.input_dim, delay=1)
    standardized = (embedded - mean) / scale
    with torch.no_grad():
      latent = model.encoder(
        torch.as_tensor(standardized, dtype=torch.float32)
      ).cpu().numpy()
    usable = len(embedded) - n_leads
    for origin in range(0, max(usable, 0), stride):
      trajectory = simulate_latent(model, latent[origin], dt, n_leads)
      if trajectory is None:
        continue
      with torch.no_grad():
        decoded = model.decoder(
          torch.as_tensor(trajectory, dtype=torch.float32)
        ).cpu().numpy()
      # Undo standardization so predictions are back in microvolts, which is
      # what every PySINDy result in this project is reported in.
      predicted = (decoded * scale + mean)[:, 0]
      truth = embedded[origin : origin + n_leads, 0]
      if len(truth) < n_leads or not np.all(np.isfinite(predicted)):
        continue
      predicted_rows.append(predicted)
      measured_rows.append(truth)

  if not predicted_rows:
    return np.empty((0, n_leads)), np.empty((0, n_leads))
  return np.vstack(predicted_rows), np.vstack(measured_rows)


def plot_results(
  history, leads: np.ndarray, skill: np.ndarray, persistence: np.ndarray,
  output_dir: Path,
) -> Path:
  """Plot training curves and forecast skill side by side.

  Args:
    history: Training history from :func:`train_ae_sindy`.
    leads: Lead times in seconds.
    skill: Autoencoder skill at each lead.
    persistence: Persistence skill at each lead.
    output_dir: Directory to write the figure into.

  Returns:
    Path of the written figure.
  """
  fig, axes = plt.subplots(1, 3, figsize=(15, 4))
  axes[0].plot(history.train_total, label="train")
  axes[0].plot(history.validation_total, label="validation")
  axes[0].set_yscale("log")
  axes[0].set_xlabel("epoch")
  axes[0].set_ylabel("weighted loss")
  axes[0].set_title("training", fontsize=10)
  axes[0].legend(fontsize=8)
  axes[0].grid(alpha=0.3)

  axes[1].plot(history.active_terms)
  axes[1].set_xlabel("epoch")
  axes[1].set_ylabel("active coefficients")
  axes[1].set_title("latent ODE support", fontsize=10)
  axes[1].grid(alpha=0.3)

  axes[2].plot(leads, skill, label="AE-SINDy", lw=1.6)
  axes[2].plot(leads, persistence, label="persistence", color="black",
               ls="--", lw=2.0)
  axes[2].axhline(0, color="gray", lw=0.6)
  axes[2].set_xlabel("lead time (s)")
  axes[2].set_ylabel("correlation with measured signal")
  axes[2].set_title("forecast skill on held-out trials", fontsize=10)
  axes[2].legend(fontsize=8)
  axes[2].grid(alpha=0.3)

  fig.tight_layout()
  path = output_dir / "ae_sindy_results.png"
  fig.savefig(path, dpi=150)
  plt.close(fig)
  return path


def main() -> None:
  """Train one autoencoder and report forecast skill against persistence."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input-dim", type=int, default=80)
  parser.add_argument("--latent-dim", type=int, default=3)
  parser.add_argument("--poly-order", type=int, default=2)
  parser.add_argument("--max-epochs", type=int, default=300)
  parser.add_argument("--batch-size", type=int, default=256)
  parser.add_argument("--learning-rate", type=float, default=1e-3)
  parser.add_argument("--patience", type=int, default=20)
  parser.add_argument("--max-lead", type=float, default=1.0)
  parser.add_argument("--origin-stride", type=float, default=0.5)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
  args = parser.parse_args()

  config = AESINDyConfig(
    input_dim=args.input_dim,
    latent_dim=args.latent_dim,
    poly_order=args.poly_order,
    max_epochs=args.max_epochs,
    batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    patience=args.patience,
    seed=args.seed,
  )
  args.out_dir.mkdir(parents=True, exist_ok=True)

  train_ids, test_ids = load_split()
  print(f"Loading {MAT_FILE} ...", flush=True)
  data = TrialData.load(MAT_FILE)
  dt = DOWNSAMPLE / data.fs

  train_traces = channel_traces(
    data, channel=CHANNEL, trials=train_ids, downsample=DOWNSAMPLE,
    lowpass_hz=LOWPASS_HZ, normalize="none",
  )
  test_traces = channel_traces(
    data, channel=CHANNEL, trials=test_ids, downsample=DOWNSAMPLE,
    lowpass_hz=LOWPASS_HZ, normalize="none",
  )
  print(f"train {len(train_traces)} trials, held-out {len(test_traces)} trials, "
        f"dt={dt*1000:.0f} ms")

  x_train, xdot_train = build_hankel_dataset(train_traces, config.input_dim, dt)
  x_test, xdot_test = build_hankel_dataset(test_traces, config.input_dim, dt)
  print(f"Hankel rows: {x_train.shape[0]:,} train, {x_test.shape[0]:,} held-out "
        f"(input_dim={config.input_dim})")

  # Standardize on training statistics only. Raw microvolts have RMS ~37,
  # which trains poorly; the reference applies StandardScaler for the same
  # reason. Predictions are mapped back before any metric is computed.
  mean = x_train.mean(axis=0, keepdims=True)
  scale = x_train.std(axis=0, keepdims=True)
  scale[scale < 1e-12] = 1.0
  x_train_s = (x_train - mean) / scale
  xdot_train_s = xdot_train / scale
  x_test_s = (x_test - mean) / scale
  xdot_test_s = xdot_test / scale

  model = AESINDy(config)
  n_parameters = sum(p.numel() for p in model.parameters())
  print(f"\nmodel: {config.input_dim} -> latent {config.latent_dim}, "
        f"poly_order {config.poly_order}, {n_parameters:,} parameters")
  print("training ...", flush=True)
  started = time.time()
  history = train_ae_sindy(
    model, x_train_s, xdot_train_s, x_test_s, xdot_test_s, config, verbose=True
  )
  print(f"trained in {time.time() - started:.0f} s, "
        f"{int(model.mask.sum().item())} active coefficients")

  print("\nlearned latent ODE:")
  equations = model.equations()
  for line in equations:
    print(f"  {line}")

  print("\nforecasting from held-out trials ...", flush=True)
  predicted, measured = forecast_from_origins(
    model, test_traces, mean, scale, config, dt, args.max_lead,
    args.origin_stride,
  )
  print(f"{predicted.shape[0]} usable forecasts")
  if predicted.shape[0] == 0:
    print("No forecast completed; the latent ODE did not integrate.")
    return

  skill = skill_by_lead(predicted, measured)
  persistence = persistence_by_lead(measured)
  leads = np.arange(skill.size) * dt

  rows = []
  for lead, model_skill, reference in zip(leads, skill, persistence):
    rows.append({
      "lead_s": float(lead),
      "ae_sindy_skill": float(model_skill),
      "persistence_skill": float(reference),
      "n_forecasts": int(predicted.shape[0]),
    })
  path = args.out_dir / "ae_sindy_forecast_skill.csv"
  with open(path, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

  (args.out_dir / "ae_sindy_equations.txt").write_text("\n".join(equations) + "\n")
  (args.out_dir / "ae_sindy_config.json").write_text(
    json.dumps({**vars(args), "out_dir": str(args.out_dir),
                "active_terms": int(model.mask.sum().item()),
                "n_parameters": n_parameters}, indent=2, default=str) + "\n"
  )

  print("\n===== forecast skill vs lead time =====")
  print(f"{'lead':>10} {'AE-SINDy':>12} {'persistence':>13}")
  for probe in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
    if probe > leads[-1] + 1e-9:
      continue
    index = int(round(probe / dt))
    print(f"{probe*1000:>8.0f}ms {skill[index]:>+12.3f} "
          f"{persistence[index]:>+13.3f}")
  print("\nPySINDy reference at the same leads (earlier runs, 110 forecasts):")
  print("  delay nd4    : 20ms +0.639, 50ms -0.468, 100ms -0.433, 200ms +0.093")
  print("  havok r=12   : 20ms +0.403, 50ms +0.334, 100ms +0.389, 200ms +0.284")

  print(f"\nwrote {path}")
  print(f"wrote {plot_results(history, leads, skill, persistence, args.out_dir)}")


if __name__ == "__main__":
  main()
