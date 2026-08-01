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
from load_data.convert import load_bhv_trial_table  # noqa: E402
from load_data.trial_selection import select_valid_trials  # noqa: E402
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


def load_default_params() -> dict:
  """Load the reference repository's default parameter dict.

  ``testcases`` is a plain directory in the reference repository, not an
  installed package: it has no ``__init__.py``, and the repository's own
  scripts are run from inside it so they can ``from default_params import
  params`` directly. Installing the project with ``pip install -e .`` exposes
  only the ``aesindy`` package, so the directory is put on the path explicitly
  here, located through the ``ROOTPATH`` that setup writes into
  ``aesindy/config.py``.

  Returns:
    A copy of the reference defaults.
  """
  from aesindy.config import ROOTPATH  # type: ignore

  testcases_dir = Path(ROOTPATH) / "testcases"
  if not (testcases_dir / "default_params.py").exists():
    raise FileNotFoundError(
      f"Expected {testcases_dir / 'default_params.py'}. ROOTPATH in "
      f"aesindy/config.py points at {ROOTPATH!r}; it must be the checkout of "
      f"deep-delay-autoencoder."
    )
  if str(testcases_dir) not in sys.path:
    sys.path.insert(0, str(testcases_dir))
  from default_params import params  # type: ignore

  return dict(params)


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
  params = load_default_params()
  params["model"] = "lfp"
  params["case"] = f"fixation_ch{CHANNEL}"
  params["system_coefficients"] = None
  params["noise"] = 0.0

  params["input_dim"] = args.input_dim
  params["latent_dim"] = args.latent_dim
  params["poly_order"] = args.poly_order
  # A trigonometric library is better matched to a band-limited oscillatory
  # signal than a polynomial one: the Hankel-SVD modes of this data are
  # sinusoids in quadrature pairs. Unlike raising poly_order, adding sin/cos
  # does not introduce terms that diverge in finite time.
  params["include_sine"] = args.include_sine
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

  # Loss weights default to lorenzww_basic.py exactly. The integral weight is
  # exposed because it is the fragile term: it integrates the latent ODE with
  # RK4 during training, and with randomly initialised coefficients of order
  # 10 a quadratic system diverges in finite time. The reference clamps it at
  # +/-500 for this reason. Increasing the number of optimizer steps per epoch
  # -- as pooling in sequence trials does, from 342 batches to 1665 -- raises
  # the chance of hitting that blowup, and a single NaN propagates into the
  # weights permanently.
  params["loss_weight_rec"] = args.loss_weight_rec
  params["loss_weight_sindy_z"] = args.loss_weight_sindy_z
  params["loss_weight_sindy_x"] = args.loss_weight_sindy_x
  params["loss_weight_sindy_regularization"] = 1e-5
  params["loss_weight_integral"] = args.loss_weight_integral
  params["loss_weight_x0"] = args.loss_weight_x0
  params["loss_weight_layer_l2"] = 0.0
  params["loss_weight_layer_l1"] = 0.0

  # Sparsification. Two independent mechanisms exist in the reference, and
  # both are effectively off in its published defaults:
  #
  #   coefficient_threshold / threshold_frequency drive RfeUpdateCallback,
  #     which masks coefficients below the threshold. The default threshold is
  #     1e-6, small enough that nothing is ever masked.
  #   use_sindycall / sindy_threshold drive SindyCall, which periodically
  #     refits the latent coefficients with pysindy's STLSQ. The default is
  #     False, so that refit never happens.
  #
  # Left at their defaults the latent ODE stays fully dense -- 1092 live
  # coefficients at latent_dim=12 -- and a dense quadratic system with
  # coefficients of order 10 diverges in finite time, which is what the
  # latent_dim 6 and 12 runs showed. These flags make that testable rather
  # than assumed.
  params["use_sindycall"] = args.use_sindycall
  params["sindy_threshold"] = args.sindy_threshold
  params["sindycall_freq"] = args.sindycall_freq
  params["coefficient_threshold"] = args.coefficient_threshold
  params["threshold_frequency"] = args.threshold_frequency

  # Their split is validation only here; the real holdout is by trial.
  params["train_ratio"] = 0.9
  return params


def build_trial_sets(
  trial_types: list[str], sequence_holdout: int, seed: int
) -> tuple[list[int], dict[str, list[int]]]:
  """Choose training trials and per-type held-out trials.

  The fixation holdout is always the archived 9-trial set, unchanged, so that
  forecast skill stays directly comparable to every PySINDy, HAVOK and
  single-type autoencoder result in this project. Only the *training* set
  grows when more trial types are requested.

  When ``non_fixation`` is included, a separate random sample of sequence
  trials is also withheld. Training on predominantly sequence trials while
  testing only on fixation would leave a distribution shift invisible;
  scoring both reveals it.

  Args:
    trial_types: Validity-filtered types to train on, e.g.
      ``["fixation", "non_fixation"]``.
    sequence_holdout: Sequence trials to withhold for testing. Ignored when
      ``non_fixation`` is not requested.
    seed: Seed for the sequence holdout draw.

  Returns:
    ``(train_ids, test_sets)`` where ``test_sets`` maps a label to trial ids.

  Raises:
    ValueError: If no valid training trials remain.
  """
  table = load_bhv_trial_table()
  archived_train, fixation_test = load_archived_split()

  test_sets: dict[str, list[int]] = {"fixation": list(fixation_test)}
  withheld = set(fixation_test)
  train_ids: list[int] = []

  for trial_type in trial_types:
    valid = select_valid_trials(table, trial_type)
    if trial_type == "non_fixation" and sequence_holdout > 0:
      available = [i for i in valid if i not in withheld]
      rng = np.random.default_rng(seed)
      chosen = rng.choice(
        len(available), size=min(sequence_holdout, len(available)), replace=False
      )
      sequence_test = sorted(available[i] for i in chosen)
      test_sets["sequence"] = sequence_test
      withheld.update(sequence_test)
    train_ids.extend(valid)

  train_ids = sorted({i for i in train_ids if i not in withheld})
  if not train_ids:
    raise ValueError(
      f"No training trials remain for trial_types={trial_types}."
    )
  # Recorded for transparency: with only fixation requested this must
  # reproduce the archived 28-trial training set exactly.
  if trial_types == ["fixation"] and train_ids != sorted(archived_train):
    print("  note: training set differs from the archived 28-trial split")
  return train_ids, test_sets


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
  max_lead_s: float, origin_stride_s: float, signal_scale: float = 1.0,
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
      # Undo the global scaling so both series are back in microvolts, which
      # is what every PySINDy result in this project reports. Correlation is
      # scale-invariant, so this only affects RMSE-style readings.
      predicted = decoded[:, 0] * signal_scale
      truth = embedded[origin : origin + n_leads, 0] * signal_scale
      if len(truth) < n_leads or not np.all(np.isfinite(predicted)):
        continue
      predicted_rows.append(predicted)
      measured_rows.append(truth)

  if not predicted_rows:
    return np.empty((0, n_leads)), np.empty((0, n_leads))
  return np.vstack(predicted_rows), np.vstack(measured_rows)


def plot_forecast_traces(
  predicted: np.ndarray, measured: np.ndarray, dt: float,
  output_dir: Path, label: str, n_panels: int = 6,
) -> Path:
  """Plot individual forecasts against the measurements they predict.

  The skill curve reports one correlation per lead time, which is the right
  summary but hides what the model is actually producing. These panels show
  single forecasts in microvolts, which is what a reader recognises: whether
  the prediction tracks the signal, flattens out, or drifts out of phase.

  Panels are drawn evenly across the collected forecasts rather than from the
  start, so they span different trials and different points within them.

  Args:
    predicted: Predictions with shape ``(n_forecasts, n_leads)`` in microvolts.
    measured: Matching measurements with the same shape.
    dt: Processed sample interval in seconds.
    output_dir: Directory to write the figure into.
    label: Held-out group name, used in the title and filename.
    n_panels: Number of individual forecasts to draw.

  Returns:
    Path of the written figure.
  """
  n_panels = min(n_panels, predicted.shape[0])
  picks = np.linspace(0, predicted.shape[0] - 1, n_panels).astype(int)
  time_ms = np.arange(predicted.shape[1]) * dt * 1000.0

  n_cols = 2
  n_rows = int(np.ceil(n_panels / n_cols))
  fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.2 * n_cols, 2.4 * n_rows),
                           squeeze=False, sharex=True)
  for index in range(n_rows * n_cols):
    ax = axes[index // n_cols][index % n_cols]
    if index >= n_panels:
      ax.axis("off")
      continue
    row = picks[index]
    ax.plot(time_ms, measured[row], color="tab:blue", lw=1.1, label="measured")
    ax.plot(time_ms, predicted[row], color="tab:orange", lw=1.1, ls="--",
            label="AE-SINDy forecast")
    # The signal's own predictability limit, established from persistence.
    ax.axvline(350, color="gray", ls=":", lw=1.0)
    ax.set_title(f"forecast {row}", fontsize=8)
    ax.tick_params(labelsize=7)
    if index % n_cols == 0:
      ax.set_ylabel("x0 (uV)", fontsize=8)
    if index // n_cols == n_rows - 1:
      ax.set_xlabel("lead time (ms)", fontsize=8)
    if index == 0:
      ax.legend(fontsize=7)
  fig.suptitle(
    f"Individual forecasts, held-out {label} trials "
    f"(dotted line: ~350 ms predictability limit)",
    fontsize=11,
  )
  fig.tight_layout()
  path = output_dir / f"aesindy_forecast_traces_{label}.png"
  fig.savefig(path, dpi=150)
  plt.close(fig)
  return path


def plot_results(leads, skill, persistence, output_dir: Path,
                 label: str = "fixation") -> Path:
  """Plot forecast skill against lead time with the persistence reference."""
  fig, ax = plt.subplots(figsize=(7, 4.2))
  ax.plot(leads, skill, label="aesindy (reference impl.)", lw=1.6)
  ax.plot(leads, persistence, label="persistence", color="black", ls="--", lw=2.0)
  ax.axhline(0, color="gray", lw=0.6)
  ax.set_xlabel("lead time (s)")
  ax.set_ylabel("correlation with measured signal")
  ax.set_title(f"Deep delay autoencoder on LFP, held-out {label} trials",
               fontsize=11)
  ax.legend(fontsize=8)
  ax.grid(alpha=0.3)
  fig.tight_layout()
  path = output_dir / f"aesindy_forecast_skill_{label}.png"
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
  parser.add_argument(
    "--no-scale", action="store_true",
    help="Disable the global amplitude rescaling applied before training.",
  )
  parser.add_argument(
    "--trial-types", default="fixation",
    help="Comma-separated validity-filtered trial types to train on. "
         "'fixation' (37 valid, ~14.2 s each) or "
         "'fixation,non_fixation' (adds 980 valid sequence trials, ~1.9 s "
         "each). The fixation holdout is unchanged either way.",
  )
  parser.add_argument(
    "--sequence-holdout", type=int, default=20,
    help="Sequence trials withheld for a second held-out score, revealing "
         "distribution shift when training is dominated by sequence trials.",
  )
  parser.add_argument(
    "--seed", type=int, default=0,
    help="Seed for the sequence holdout draw. The fixation holdout is the "
         "fixed archived set and does not depend on this.",
  )
  parser.add_argument(
    "--include-sine", action="store_true",
    help="Add sin/cos terms to the latent library. Better matched to an "
         "oscillatory signal than higher polynomial degree, and unlike "
         "quadratic terms these cannot diverge in finite time.",
  )
  parser.add_argument("--loss-weight-rec", type=float, default=0.3)
  parser.add_argument("--loss-weight-sindy-z", type=float, default=0.001)
  parser.add_argument("--loss-weight-sindy-x", type=float, default=0.001)
  parser.add_argument(
    "--loss-weight-integral", type=float, default=0.1,
    help="Weight on the RK4 integral loss. The reference uses 0.1 for real "
         "data. Set to 0 to disable it; it is the term that diverges to NaN "
         "when the latent ODE blows up during training.",
  )
  parser.add_argument("--loss-weight-x0", type=float, default=0.01)
  parser.add_argument(
    "--use-sindycall", action="store_true",
    help="Enable the reference's periodic STLSQ refit of the latent "
         "coefficients. Off in the reference defaults, which leaves the "
         "latent ODE fully dense.",
  )
  parser.add_argument(
    "--sindy-threshold", type=float, default=0.4,
    help="STLSQ threshold used by the refit. The reference default is 0.4.",
  )
  parser.add_argument(
    "--sindycall-freq", type=int, default=1,
    help="Epoch interval between STLSQ refits.",
  )
  parser.add_argument(
    "--coefficient-threshold", type=float, default=1e-6,
    help="Magnitude below which the RFE callback masks a coefficient. The "
         "reference default of 1e-6 never masks anything.",
  )
  parser.add_argument(
    "--threshold-frequency", type=int, default=100,
    help="Epoch interval between RFE masking passes.",
  )
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

  if args.use_sindycall:
    raise SystemExit(
      "--use-sindycall cannot work with real data.\n"
      "\n"
      "aesindy/training.py:get_callbacks guards its SindyCall branch behind\n"
      "params['use_sindycall'], and that branch has three defects: it calls\n"
      "`self.data` from a module-level function where `self` is undefined, it\n"
      "then calls `.run_sim(...)`, which exists only on SynthData and not on\n"
      "RealData, and it references an undefined `data_test`. The authors mark\n"
      "the block `# Change and NOT TESTED`.\n"
      "\n"
      "It is therefore a synthetic-data-only path that has never executed.\n"
      "Use the RFE callback instead, which is always registered and does work;\n"
      "it simply never masks anything at the default 1e-6 threshold:\n"
      "\n"
      "  --coefficient-threshold 0.5 --threshold-frequency 25\n"
    )

  disable_breakpoints()
  args.out_dir.mkdir(parents=True, exist_ok=True)

  from aesindy.solvers import RealData  # type: ignore
  from aesindy.training import TrainModel  # type: ignore

  trial_types = [t.strip() for t in args.trial_types.split(",") if t.strip()]
  train_ids, test_id_sets = build_trial_sets(
    trial_types, args.sequence_holdout, args.seed
  )
  print(f"Loading {MAT_FILE} ...", flush=True)
  data = TrialData.load(MAT_FILE)
  dt = DOWNSAMPLE / data.fs

  train_traces = channel_traces(
    data, channel=CHANNEL, trials=train_ids, downsample=DOWNSAMPLE,
    lowpass_hz=LOWPASS_HZ, normalize="none",
  )
  test_trace_sets = {
    label: channel_traces(
      data, channel=CHANNEL, trials=ids, downsample=DOWNSAMPLE,
      lowpass_hz=LOWPASS_HZ, normalize="none",
    )
    for label, ids in test_id_sets.items()
  }
  test_traces = test_trace_sets["fixation"]
  print(f"trial types: {trial_types}")
  for label, ids in test_id_sets.items():
    print(f"  held-out {label}: {len(ids)} trials")
  print(f"train {len(train_traces)} trials, held-out {len(test_traces)} trials "
        f"(held-out never enters TrainModel), dt={dt*1000:.0f} ms")

  # One global amplitude scale, computed on training trials only.
  #
  # The reference's loss weights (rec 0.3, sindy_z 0.001, sindy_x 0.001) were
  # tuned on waterwheel data of order 1. LFP is in microvolts with derivatives
  # of order 1e3, so the smoke run showed sindy_z at 3.6e9 against a
  # reconstruction loss of 1.5e3 -- six orders of magnitude apart, meaning the
  # network optimises the derivative terms and effectively ignores
  # reconstruction. Dividing by a single constant restores the balance the
  # weights assume.
  #
  # A single global constant is used rather than per-trial z-scoring so that
  # relative amplitude across trials is preserved, and it is inverted before
  # any metric is computed. Correlation is scale-invariant regardless; this
  # matters for the loss balance, not the reported skill.
  signal_scale = 1.0
  if not args.no_scale:
    signal_scale = float(np.std(np.concatenate(train_traces)))
    if signal_scale <= 0:
      signal_scale = 1.0
    train_traces = [t / signal_scale for t in train_traces]
    test_trace_sets = {
      label: [t / signal_scale for t in traces]
      for label, traces in test_trace_sets.items()
    }
    test_traces = test_trace_sets["fixation"]
    print(f"global amplitude scale from training trials: "
          f"{signal_scale:.3f} uV (inverted before scoring)")

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

  if not np.all(np.isfinite(coefficients)):
    print(
      "\nTRAINING DIVERGED: the latent coefficients contain NaN or Inf.\n"
      "\n"
      "Once a NaN reaches the weights it propagates permanently, so every\n"
      "subsequent epoch and every forecast is lost. The usual source is the\n"
      "RK4 integral loss: it integrates the latent ODE during training, and\n"
      "a quadratic system with randomly initialised coefficients of order 10\n"
      "diverges in finite time. More optimizer steps per epoch means more\n"
      "chances to hit it.\n"
      "\n"
      "Try either of these, which separate the two candidate causes:\n"
      "  --loss-weight-integral 0     disable the fragile term outright\n"
      "  --learning-rate 1e-4         keep every reference loss, step smaller\n",
      flush=True,
    )
    return
  print(f"library ordering verified against the reference; "
        f"{active} active coefficients")

  (args.out_dir / "aesindy_coefficients.csv").write_text(
    "\n".join(",".join(f"{v:.8g}" for v in row) for row in coefficients) + "\n"
  )

  (args.out_dir / "aesindy_config.json").write_text(
    json.dumps({**vars(args), "out_dir": str(args.out_dir),
                "active_coefficients": active,
                "train_trials": len(train_ids),
                "held_out": {k: len(v) for k, v in test_id_sets.items()}},
               indent=2, default=str) + "\n"
  )

  # Each held-out group is scored separately. When training is dominated by
  # sequence trials, fixation skill alone cannot distinguish "the model is
  # bad" from "the model learned sequence dynamics and fixation is out of
  # distribution".
  summary: dict[str, tuple] = {}
  for label, traces in test_trace_sets.items():
    print(f"\nforecasting held-out {label} trials ...", flush=True)
    predicted, measured = forecast_held_out(
      model, coefficients, exponents, traces, args.input_dim, dt,
      args.max_lead, args.origin_stride, signal_scale=signal_scale,
    )
    print(f"  {predicted.shape[0]} usable forecasts")
    if predicted.shape[0] == 0:
      print("  No forecast completed; the latent ODE did not integrate.")
      continue
    skill = skill_by_lead(predicted, measured)
    persistence = persistence_by_lead(measured)
    leads = np.arange(skill.size) * dt
    summary[label] = (leads, skill, persistence, predicted.shape[0])

    rows = [
      {"held_out": label, "lead_s": float(lead), "aesindy_skill": float(a),
       "persistence_skill": float(b), "n_forecasts": int(predicted.shape[0])}
      for lead, a, b in zip(leads, skill, persistence)
    ]
    path = args.out_dir / f"aesindy_forecast_skill_{label}.csv"
    with open(path, "w", newline="") as handle:
      writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
      writer.writeheader()
      writer.writerows(rows)
    print(f"  wrote {path}")
    print(f"  wrote {plot_results(leads, skill, persistence, args.out_dir, label)}")
    print(f"  wrote {plot_forecast_traces(predicted, measured, dt, args.out_dir, label)}")
    # The raw forecasts are kept so figures can be redrawn, and compared
    # against other methods, without repeating an hour of training.
    trace_path = args.out_dir / f"aesindy_forecast_traces_{label}.npz"
    np.savez_compressed(trace_path, predicted=predicted, measured=measured, dt=dt)
    print(f"  wrote {trace_path}")

  if not summary:
    print("\nNo held-out group produced a usable forecast.")
    return

  print("\n===== forecast skill vs lead time =====")
  probes = [p for p in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
            if p <= args.max_lead + 1e-9]
  header = f"{'held-out':>10} {'n':>6}" + "".join(f"{p*1000:>10.0f}ms" for p in probes)
  print(header)
  for label, (leads, skill, persistence, n) in summary.items():
    for name, series in (("aesindy", skill), ("persistence", persistence)):
      row = f"{label + '/' + name:>10} {n:>6}"
      for probe in probes:
        row += f"{series[int(round(probe / dt))]:>+12.3f}"
      print(row)

  print("\nPySINDy reference at the same leads (110 forecasts, fixation):")
  print("  delay nd4  : 20ms +0.639, 50ms -0.468, 100ms -0.433, 200ms +0.093")
  print("  havok r=3  : 20ms +0.424, 50ms +0.433, 100ms +0.342, 200ms +0.254")
  print("  havok r=12 : 20ms +0.403, 50ms +0.334, 100ms +0.389, 200ms +0.284")


if __name__ == "__main__":
  main()
