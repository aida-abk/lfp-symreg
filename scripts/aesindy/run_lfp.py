"""Run the reference deep delay autoencoder on LFP, scored like the baselines.

This drives the *authors'* implementation -- josephbakarji/deep-delay-autoencoder,
package ``aesindy`` -- rather than a reimplementation, so results can be
reported as "the method from the paper, applied to our data". It follows
``testcases/lorenzww_basic.py``, the repository's only experimental-data
example, substituting LFP trials for waterwheel recordings.

Their ``RealData.build_solution`` accepts a dict holding lists of
trajectories, one entry per recording, and Hankel-embeds each separately
before stacking. Trials map onto that directly, so the data layout needs no
adaptation.

Departures from the reference, all deliberate:

* Only the 28 training trials are handed to ``TrainModel``. The reference
  splits with ``train_test_split(..., shuffle=False)`` *after* stacking the
  Hankel rows, which puts rows from the same recording on both sides of the
  split. Here that internal split is validation only, and the 9 archived
  held-out trials are scored separately by this script -- they never enter
  training.
* ``pdb.set_trace`` is neutralised before ``fit()``. The reference's
  ``save_results`` contains a breakpoint, and ``lorenzww_basic.py`` has
  another before ``trainer.fit()``. Under Slurm either one hangs the job or
  kills it on a closed stdin.
* Scoring uses this project's forecast-skill metric against persistence, so
  the numbers sit in the same table as the PySINDy and HAVOK results.

Run a two-epoch plumbing check first -- it exercises every stage in a few
minutes and is worth doing before committing a full run:

    python scripts/aesindy/run_lfp.py --smoke

Then the real run:

    python scripts/aesindy/run_lfp.py

"""
from __future__ import annotations

import argparse
import csv
import json
import pdb
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
# Imported from the framework-free module, not from models.ae_sindy, which
# requires torch. This environment has TensorFlow instead.
from models.polynomial_library import (  # noqa: E402
  polynomial_exponents,
  polynomial_library_numpy,
)
from models.sindy import delay_embed_trace  # noqa: E402

from load_data.archived_split import (  # noqa: E402
  CHANNEL,
  DOWNSAMPLE,
  load_archived_split,
)
from models.forecast_metrics import persistence_by_lead, skill_by_lead  # noqa: E402

LOWPASS_HZ = 35.0
SG_WINDOW = 21
SG_POLYORDER = 3
OUTPUT_DIR = _PROJECT_ROOT / "outputs/aesindy"


def disable_breakpoints() -> None:
  """Turn ``pdb.set_trace`` into a no-op.

  ``aesindy.training.TrainModel.save_results`` calls it before persisting the
  model. In a batch job that either blocks forever waiting on stdin or raises
  on a closed one, after training has already completed -- the worst possible
  place to lose a run.
  """
  pdb.set_trace = lambda *args, **kwargs: None


def build_reference_params(args) -> dict:
  """Assemble the reference implementation's parameter dict.

  Values follow ``testcases/default_params.py`` with the overrides from
  ``testcases/lorenzww_basic.py``, which is the authors' own configuration for
  experimental data.

  Args:
    args: Parsed command-line arguments.

  Returns:
    The parameter dict expected by ``TrainModel``.
  """
  from testcases.default_params import params  # type: ignore

  params = dict(params)
  params["model"] = "lfp"
  params["case"] = f"fixation_ch{CHANNEL}"
  params["system_coefficients"] = None
  params["noise"] = 0.0

  params["input_dim"] = args.input_dim
  params["latent_dim"] = args.latent_dim
  params["poly_order"] = args.poly_order
  params["include_sine"] = False
  params["exact_features"] = False
  params["fix_coefs"] = False
  params["svd_dim"] = None

  # The LFP is already uniformly sampled after preprocessing, so the
  # reference's cubic re-interpolation (used because the waterwheel data is
  # irregular) is switched off. Derivatives are supplied explicitly instead.
  params["interpolate"] = False

  params["save_checkpoints"] = False
  params["print_progress"] = True
  params["print_frequency"] = 5
  params["max_epochs"] = args.max_epochs
  params["patience"] = args.patience
  params["batch_size"] = args.batch_size
  params["learning_rate"] = args.learning_rate

  # Loss weights exactly as in lorenzww_basic.py.
  params["loss_weight_rec"] = 0.3
  params["loss_weight_sindy_z"] = 0.001
  params["loss_weight_sindy_x"] = 0.001
  params["loss_weight_sindy_regularization"] = 1e-5
  params["loss_weight_integral"] = 0.1
  params["loss_weight_x0"] = 0.01
  params["loss_weight_layer_l2"] = 0.0
  params["loss_weight_layer_l1"] = 0.0

  # Their split is validation only here; the real holdout is by trial.
  params["train_ratio"] = 0.9
  return params


def build_data_dict(traces: list[np.ndarray], dt: float) -> dict:
  """Build the dict consumed by ``RealData.build_solution``.

  Args:
    traces: Preprocessed traces in signal units, one per trial.
    dt: Processed sample interval in seconds.

  Returns:
    A dict with ``time``, ``dt``, ``x`` and ``dx`` in the reference's layout.
  """
  times, values, derivatives = [], [], []
  for trace in traces:
    trace = np.asarray(trace, dtype=float)
    times.append(np.arange(trace.size) * dt)
    values.append(trace)
    derivatives.append(
      savgol_filter(trace, window_length=SG_WINDOW, polyorder=SG_POLYORDER,
                    deriv=1, delta=dt)
    )
  return {"time": times, "dt": dt, "x": values, "dx": derivatives}


def latent_coefficients(model) -> np.ndarray:
  """Extract the masked latent coefficient matrix as numpy.

  Args:
    model: The trained ``Sindy_Autoencoder``.

  Returns:
    Coefficients with shape ``(n_features, latent_dim)``.
  """
  coefficients = np.asarray(model.sindy.coefficients.numpy(), dtype=float)
  mask = np.asarray(model.sindy.coefficients_mask.numpy(), dtype=float)
  return coefficients * mask


def verify_library_ordering(model, latent_dim: int, poly_order: int) -> list:
  """Check this project's polynomial ordering matches the reference's.

  The coefficient matrix is only interpretable alongside the feature ordering
  that produced it. Rather than assume the two libraries agree, this evaluates
  both on random states and compares. Getting this wrong would silently
  scramble the learned equations, so it is checked rather than trusted.

  Args:
    model: The trained ``Sindy_Autoencoder``, whose ``sindy.theta`` is the
      reference library.
    latent_dim: Latent dimension.
    poly_order: Polynomial order.

  Returns:
    Exponent tuples matching the reference ordering.

  Raises:
    RuntimeError: If the two libraries disagree.
  """
  import tensorflow as tf

  exponents = polynomial_exponents(latent_dim, poly_order)
  probe = np.random.default_rng(0).normal(size=(16, latent_dim))
  reference = np.asarray(
    model.sindy.theta(tf.convert_to_tensor(probe, dtype=tf.float32)).numpy(),
    dtype=float,
  )
  local = polynomial_library_numpy(probe, exponents)
  if reference.shape != local.shape:
    raise RuntimeError(
      f"Library size mismatch: reference gives {reference.shape}, this "
      f"project's ordering gives {local.shape}. The latent equations cannot "
      f"be interpreted until these agree."
    )
  if not np.allclose(reference, local, rtol=1e-4, atol=1e-5):
    raise RuntimeError(
      "Reference and local polynomial libraries disagree on ordering. "
      "Simulating with the local ordering would scramble the learned model."
    )
  return exponents


def simulate_latent(
  coefficients: np.ndarray, exponents: list, z0: np.ndarray, dt: float,
  n_steps: int,
) -> np.ndarray | None:
  """Integrate the learned latent ODE from one initial condition.

  Uses LSODA, matching the integrator used for every PySINDy result in this
  project so the comparison is not confounded by the solver.

  Args:
    coefficients: Masked coefficient matrix, shape ``(n_features, latent_dim)``.
    exponents: Library exponents in the verified ordering.
    z0: Initial latent state.
    dt: Processed sample interval in seconds.
    n_steps: Samples to produce, including the initial one.

  Returns:
    Latent trajectory with shape ``(n_steps, latent_dim)``, or ``None`` if the
    integration failed or went non-finite.
  """
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
  return trajectory if np.all(np.isfinite(trajectory)) else None


def forecast_held_out(
  model, coefficients: np.ndarray, exponents: list,
  test_traces: list[np.ndarray], input_dim: int, dt: float,
  max_lead_s: float, origin_stride_s: float,
) -> tuple[np.ndarray, np.ndarray]:
  """Forecast from many origins in each held-out trial.

  Args:
    model: Trained ``Sindy_Autoencoder``.
    coefficients: Masked latent coefficients.
    exponents: Verified library ordering.
    test_traces: Held-out preprocessed traces.
    input_dim: Hankel embedding dimension.
    dt: Processed sample interval in seconds.
    max_lead_s: Longest lead time, in seconds.
    origin_stride_s: Spacing between forecast origins within a trial.

  Returns:
    ``(predicted, measured)`` with shape ``(n_forecasts, n_leads)``, in signal
    units.
  """
  import tensorflow as tf

  n_leads = int(round(max_lead_s / dt)) + 1
  stride = max(int(round(origin_stride_s / dt)), 1)
  predicted_rows, measured_rows = [], []

  for trace in test_traces:
    embedded = delay_embed_trace(trace, n_delays=input_dim, delay=1)
    latent = np.asarray(
      model.encoder(tf.convert_to_tensor(embedded, dtype=tf.float32)).numpy(),
      dtype=float,
    )
    usable = len(embedded) - n_leads
    for origin in range(0, max(usable, 0), stride):
      trajectory = simulate_latent(coefficients, exponents, latent[origin],
                                   dt, n_leads)
      if trajectory is None:
        continue
      decoded = np.asarray(
        model.decoder(
          tf.convert_to_tensor(trajectory, dtype=tf.float32)
        ).numpy(),
        dtype=float,
      )
      predicted = decoded[:, 0]
      truth = embedded[origin : origin + n_leads, 0]
      if len(truth) < n_leads or not np.all(np.isfinite(predicted)):
        continue
      predicted_rows.append(predicted)
      measured_rows.append(truth)

  if not predicted_rows:
    return np.empty((0, n_leads)), np.empty((0, n_leads))
  return np.vstack(predicted_rows), np.vstack(measured_rows)


def plot_results(leads, skill, persistence, output_dir: Path) -> Path:
  """Plot forecast skill against lead time with the persistence reference."""
  fig, ax = plt.subplots(figsize=(7, 4.2))
  ax.plot(leads, skill, label="aesindy (reference impl.)", lw=1.6)
  ax.plot(leads, persistence, label="persistence", color="black", ls="--", lw=2.0)
  ax.axhline(0, color="gray", lw=0.6)
  ax.set_xlabel("lead time (s)")
  ax.set_ylabel("correlation with measured signal")
  ax.set_title("Deep delay autoencoder on LFP, held-out trials", fontsize=11)
  ax.legend(fontsize=8)
  ax.grid(alpha=0.3)
  fig.tight_layout()
  path = output_dir / "aesindy_forecast_skill.png"
  fig.savefig(path, dpi=150)
  plt.close(fig)
  return path


def main() -> None:
  """Train the reference model on LFP and score held-out forecast skill."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input-dim", type=int, default=80)
  parser.add_argument("--latent-dim", type=int, default=3)
  parser.add_argument("--poly-order", type=int, default=2)
  parser.add_argument("--max-epochs", type=int, default=300)
  parser.add_argument("--patience", type=int, default=10)
  parser.add_argument("--batch-size", type=int, default=256)
  parser.add_argument("--learning-rate", type=float, default=1e-3)
  parser.add_argument("--max-lead", type=float, default=1.0)
  parser.add_argument("--origin-stride", type=float, default=0.5)
  parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
  parser.add_argument(
    "--smoke", action="store_true",
    help="Two epochs and a coarse forecast grid, to validate plumbing.",
  )
  args = parser.parse_args()
  if args.smoke:
    args.max_epochs = 2
    args.patience = 2
    args.origin_stride = 4.0

  disable_breakpoints()
  args.out_dir.mkdir(parents=True, exist_ok=True)

  from aesindy.solvers import RealData  # type: ignore
  from aesindy.training import TrainModel  # type: ignore

  train_ids, test_ids = load_archived_split()
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
  print(f"train {len(train_traces)} trials, held-out {len(test_traces)} trials "
        f"(held-out never enters TrainModel), dt={dt*1000:.0f} ms")

  params = build_reference_params(args)
  params["dt"] = dt
  params["tend"] = float(max(t.size for t in train_traces) * dt)
  params["n_ics"] = len(train_traces)
  params["data_path"] = str(args.out_dir) + "/"

  reference_data = RealData(
    input_dim=params["input_dim"],
    interpolate=params["interpolate"],
    interp_dt=params.get("interp_dt", 0.01),
    savgol_interp_coefs=params.get("interp_coefs", [SG_WINDOW, SG_POLYORDER]),
    interp_kind=params.get("interp_kind", "cubic"),
  )
  reference_data.build_solution(build_data_dict(train_traces, dt))
  print(f"Hankel matrix from RealData: x {np.shape(reference_data.x)}, "
        f"dx {np.shape(reference_data.dx)}")

  print("\ntraining (aesindy.TrainModel) ...", flush=True)
  started = time.time()
  trainer = TrainModel(reference_data, params)
  trainer.fit()
  model = trainer.model
  print(f"trained in {time.time() - started:.0f} s")

  exponents = verify_library_ordering(model, args.latent_dim, args.poly_order)
  coefficients = latent_coefficients(model)
  active = int(np.count_nonzero(np.abs(coefficients) > 1e-12))
  print(f"library ordering verified against the reference; "
        f"{active} active coefficients")

  (args.out_dir / "aesindy_coefficients.csv").write_text(
    "\n".join(",".join(f"{v:.8g}" for v in row) for row in coefficients) + "\n"
  )

  print("\nforecasting held-out trials ...", flush=True)
  predicted, measured = forecast_held_out(
    model, coefficients, exponents, test_traces, args.input_dim, dt,
    args.max_lead, args.origin_stride,
  )
  print(f"{predicted.shape[0]} usable forecasts")
  if predicted.shape[0] == 0:
    print("No forecast completed; the latent ODE did not integrate.")
    return

  skill = skill_by_lead(predicted, measured)
  persistence = persistence_by_lead(measured)
  leads = np.arange(skill.size) * dt

  rows = [
    {"lead_s": float(lead), "aesindy_skill": float(a),
     "persistence_skill": float(b), "n_forecasts": int(predicted.shape[0])}
    for lead, a, b in zip(leads, skill, persistence)
  ]
  path = args.out_dir / "aesindy_forecast_skill.csv"
  with open(path, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
  (args.out_dir / "aesindy_config.json").write_text(
    json.dumps({**vars(args), "out_dir": str(args.out_dir),
                "active_coefficients": active}, indent=2, default=str) + "\n"
  )

  print("\n===== forecast skill vs lead time =====")
  print(f"{'lead':>10} {'aesindy':>10} {'persistence':>13}")
  for probe in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
    if probe > leads[-1] + 1e-9:
      continue
    index = int(round(probe / dt))
    print(f"{probe*1000:>8.0f}ms {skill[index]:>+10.3f} "
          f"{persistence[index]:>+13.3f}")
  print("\nPySINDy reference at the same leads (110 forecasts):")
  print("  delay nd4  : 20ms +0.639, 50ms -0.468, 100ms -0.433, 200ms +0.093")
  print("  havok r=12 : 20ms +0.403, 50ms +0.334, 100ms +0.389, 200ms +0.284")

  print(f"\nwrote {path}")
  print(f"wrote {plot_results(leads, skill, persistence, args.out_dir)}")


if __name__ == "__main__":
  main()
