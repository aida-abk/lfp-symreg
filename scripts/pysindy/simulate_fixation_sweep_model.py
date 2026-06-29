from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
PYSINDY_SCRIPTS = SCRIPTS / "pysindy"
for path in (ROOT, SCRIPTS, PYSINDY_SCRIPTS):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

from models.sindy import SINDyConfig, delay_embed_trajectories, fit_sindy_model
from load_data.convert import MAT_FILE, TrialData
from load_data.preprocessing import channel_traces
from pipeline_utils import select_trials, split_trials_random


DEFAULT_MODEL_CSV = (
  ROOT / "outputs" / "pysindy" / "fixation_80hz_linear_stable_top.csv"
)


def load_model_configuration(path: Path, row_index: int) -> dict[str, str]:
  """Load one parameter combination from a sweep CSV."""
  with path.open(newline="") as file:
    rows = list(csv.DictReader(file))
  if not rows:
    raise ValueError(f"No model rows found in {path}")
  if not 0 <= row_index < len(rows):
    raise IndexError(f"row_index must be between 0 and {len(rows) - 1}")
  return rows[row_index]


def extract_linear_system(model) -> tuple[np.ndarray, np.ndarray, list[str]]:

  """Extract the  linear system learned by PySINDy.

    Depending on the number of delays, x can have different dimensions. For example, with six delays:
    x = [x0, x1, x2, x3, x4, x5]

    For degree-1 SINDy models, the learned equation has the form:

      dx/dt = c + A x

    where ``c`` is the constant/intercept vector and ``A`` is the coefficient matrix
    multiplying the delay coordinates.
  """
  feature_names = model.get_feature_names() #get the names of every term in SINDy library
                                          #That will tell us what columns of the coefficient matrix correspond to each delay coordinate.

  coefficients = np.asarray(model.coefficients(), dtype=float) #convert the coefficients to a numpy array
 
  if coefficients.ndim != 2: #make sure the coefficients are a 2D matrix
    raise ValueError(f"Expected 2D coefficient matrix, got {coefficients.shape}")

  n_equations = coefficients.shape[0] #number of equations in the system
  constant = np.zeros(n_equations)
  matrix = np.zeros((n_equations, n_equations)) #initialize the coefficient matrix, each row corresponds to an equation, each column corresponds to a delay coordinate


  #loops through each term in the SINDy library and adds the corresponding coefficient to the coefficient matrix
  for feature_index, feature_name in enumerate(feature_names):
    column = coefficients[:, feature_index]
    if feature_name == "1":
      constant = column #if the term is a constant, add the coefficient to the constant vector
      continue

    if feature_name.startswith("x") and feature_name[1:].isdigit():
      state_index = int(feature_name[1:])

      if state_index >= n_equations: #make sure the state index is within the number of equations
        raise ValueError(
          f"Feature {feature_name} is outside the {n_equations}-state system."
        )
      matrix[:, state_index] = column
      continue

    raise ValueError(
      "Explicit SciPy simulation currently supports only degree-1 models. "
    )

  return constant, matrix, feature_names


def simulate_linear_trajectory_scipy(
  constant: np.ndarray,
  matrix: np.ndarray,
  measured: np.ndarray,
  dt: float,
  horizon_s: float, #how long to simulate the trajectory for
) -> np.ndarray:

  """Simulate a delay-embedded trial with SciPy.

  The initial condition must be one full delay vector.
  If ``n_delays=6``, then ``measured[0]`` contains six values:

      [x(t), x(t-tau), x(t-2tau), ..., x(t-5tau)]

  SciPy then integrates the learned ODE forward:

      dx/dt = c + A x

  The output is sampled at the same ``dt`` as the measured embedded trajectory
  so the two arrays can be compared point-by-point.
  
  """
  # Previous PySINDy simulation version,kept as reference.
  #
  # time = np.arange(n_samples) * dt
  # simulated = np.asarray(
  #   model.simulate(
  #     measured[0], #initial condition is one full delay vector
  #     time, #time points at which to evaluate the ODE
  #     integrator_kws={"method": "LSODA", "rtol": 1e-8, "atol": 1e-10}, #integration settings, LSODA is a method that is good for stiff equations
  #   )
  # )
  #
  # The explicit SciPy version below makes the simulated ODE visible:
  #   dx/dt = c + A x

  n_samples = min(measured.shape[0], int(round(horizon_s / dt)) + 1) #number of samples to simulate

  if n_samples < 2:
    raise ValueError("Simulation horizon must contain at least two samples.")

  time = np.arange(n_samples) * dt #time points at which to evaluate the ODE
  initial_state = measured[0] #initial condition is one full delay vector

  def right_hand_side(_time: float, state: np.ndarray) -> np.ndarray:
    """Evaluate the fitted SINDy equation at one simulated state."""
    return constant + matrix @ state

  solution = solve_ivp(
    right_hand_side,
    t_span=(time[0], time[-1]),
    y0=initial_state,
    t_eval=time,
    method="LSODA",
    rtol=1e-8,
    atol=1e-10,
  )

  if not solution.success:
    raise RuntimeError(f"SciPy integration failed: {solution.message}")

  simulated = solution.y.T
  if simulated.shape != measured[:n_samples].shape:
    raise RuntimeError(
      f"Simulation shape {simulated.shape} does not match "
      f"measured shape {measured[:n_samples].shape}."
    )
  if not np.all(np.isfinite(simulated)):
    raise RuntimeError("Simulation produced NaN or infinite values.")
  return simulated


def correlation(measured: np.ndarray, simulated: np.ndarray) -> float:
  """Calculate measured-vs-simulated waveform correlation.

  This is not autocorrelation. It compares two signals at matching time points:
  the real held-out trajectory and the simulated trajectory.
  """
  if np.std(measured) == 0 or np.std(simulated) == 0:
    return float("nan")
  return float(np.corrcoef(measured, simulated)[0, 1])


def trial_metrics(
  trial: int,
  measured: np.ndarray,
  simulated: np.ndarray,
) -> dict[str, float | int]:
  """Calculate held-out trajectory metrics for one fixation trial."""
  measured = measured[: simulated.shape[0]]
  all_mse = float(mean_squared_error(measured, simulated))
  x0_mse = float(mean_squared_error(measured[:, 0], simulated[:, 0]))
  return {
    "trial": trial,
    "samples": simulated.shape[0],
    "trajectory_rmse_all_coordinates": float(np.sqrt(all_mse)),
    "x0_rmse": float(np.sqrt(x0_mse)),
    "x0_correlation": correlation(measured[:, 0], simulated[:, 0]),
    "measured_x0_max_abs": float(np.max(np.abs(measured[:, 0]))),
    "simulated_x0_max_abs": float(np.max(np.abs(simulated[:, 0]))),
  }


def save_x0_grid(
  path: Path,
  trials: list[int],
  measured_trials: list[np.ndarray],
  simulated_trials: list[np.ndarray],
  dt: float,
  columns: int = 3,
) -> None:
  """Plot measured and simulated x0 for every held-out trial."""
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  rows = math.ceil(len(trials) / columns)
  fig, axes = plt.subplots(
    rows,
    columns,
    figsize=(5 * columns, 2.5 * rows),
    sharex=True,
    sharey=True,
    squeeze=False,
  )
  for axis, trial, measured, simulated in zip(
    axes.ravel(),
    trials,
    measured_trials,
    simulated_trials,
  ):
    time = np.arange(simulated.shape[0]) * dt
    axis.plot(time, measured[: simulated.shape[0], 0], label="Measured", linewidth=1)
    axis.plot(time, simulated[:, 0], label="Simulated", linewidth=1)
    axis.set_title(f"Trial {trial}", fontsize=9)

  for axis in axes.ravel()[len(trials):]:
    axis.set_visible(False)
  axes.ravel()[0].legend(loc="upper right")
  fig.supxlabel("Time from initial state (s)")
  fig.supylabel("Z-scored LFP, x0")
  fig.suptitle("Held-Out Fixation Trials: Measured vs PySINDy Simulation")
  fig.tight_layout(rect=(0.02, 0.02, 1, 0.98))
  path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(path, dpi=180)
  plt.close(fig)


def save_coordinate_plot(
  path: Path,
  trial: int,
  measured: np.ndarray,
  simulated: np.ndarray,
  dt: float,
) -> None:
  """Plot every delay coordinate for one held-out trial."""
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  time = np.arange(simulated.shape[0]) * dt
  n_coordinates = simulated.shape[1]
  fig, axes = plt.subplots(
    n_coordinates,
    1,
    figsize=(10, 2 * n_coordinates),
    sharex=True,
  )
  axes = np.atleast_1d(axes)
  for index, axis in enumerate(axes):
    axis.plot(
      time,
      measured[: simulated.shape[0], index],
      label="Measured",
      linewidth=1,
    )
    axis.plot(time, simulated[:, index], label="Simulated", linewidth=1)
    axis.set_ylabel(f"x{index}")
  axes[0].legend(loc="upper right")
  axes[-1].set_xlabel("Time from initial state (s)")
  fig.suptitle(f"Trial {trial}: All Delay Coordinates")
  fig.tight_layout()
  path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(path, dpi=180)
  plt.close(fig)


def main() -> None:
  """Refit one sweep model and compare SciPy simulations with held-out data."""
  parser = argparse.ArgumentParser(
    description="Simulate a fixation-only model selected from a parameter sweep."
  )
  parser.add_argument("--mat-file", type=Path, default=MAT_FILE)
  parser.add_argument("--model-csv", type=Path, default=DEFAULT_MODEL_CSV)
  parser.add_argument("--row-index", type=int, default=0)
  parser.add_argument("--horizon", type=float, default=1.0)
  parser.add_argument(
    "--out-dir",
    type=Path,
    default=Path("outputs/pysindy/fixation_simulation"),
  )
  args = parser.parse_args()

  configuration = load_model_configuration(args.model_csv, args.row_index)
  if configuration.get("linear_stability_status") == "unstable":
    raise ValueError("Selected sweep row is dynamically unstable.")

  # The sweep CSV stores the modeling choices. The preprocessing choices are
  # fixed here to match the stability sweep: fixation trials, channel 0 unless
  # otherwise stored in the CSV, 80 Hz low-pass filtering, downsample by 2, and
  # per-trial z-score normalization.
  channel = int(configuration["channel"])
  n_delays = int(configuration["n_delays"])
  delay = int(configuration["delay_samples_after_downsample"])
  downsample = 2
  lowpass_hz = 80.0
  normalize = "zscore"
  threshold = float(configuration["threshold"])
  degree = int(configuration["degree"])
  smooth_window = 0
  test_fraction = int(configuration["test_trials"]) / (
    int(configuration["train_trials"]) + int(configuration["test_trials"])
  )

  # Load the original MATLAB data directly. The simulation script does not
  # require a saved .npz cache because it can reconstruct the exact train/test
  # data from the .mat file plus the sweep CSV configuration.
  data = TrialData.load(args.mat_file)

  # Use only fixation trials, then recreate the same random whole-trial split
  # used during the stability sweep. The seed makes the split reproducible.
  fixation = select_trials(data, dataset="fixation")
  train_trials, test_trials = split_trials_random(
    fixation,
    test_fraction=test_fraction,
    seed=int(configuration["random_seed"]),
  )

  # Extract one channel from each selected trial and apply the fixed
  # preprocessing. We keep trials as a list because the trial lengths can differ.
  train_traces = channel_traces(
    data,
    channel=channel,
    trials=train_trials,
    downsample=downsample,
    lowpass_hz=lowpass_hz,
    normalize=normalize,
  )
  test_traces = channel_traces(
    data,
    channel=channel,
    trials=test_trials,
    downsample=downsample,
    lowpass_hz=lowpass_hz,
    normalize=normalize,
  )

  # Convert each one-dimensional LFP trace into a delay-embedded trajectory.
  # For n_delays=6, each row has six state variables:
  # [x(t), x(t-tau), x(t-2tau), x(t-3tau), x(t-4tau), x(t-5tau)].
  train_embedded = delay_embed_trajectories(
    train_traces,
    n_delays=n_delays,
    delay=delay,
  )
  test_embedded = delay_embed_trajectories(
    test_traces,
    n_delays=n_delays,
    delay=delay,
  )

  # After downsampling by 2 from the original 500 Hz sampling rate, dt is
  # 2 / 500 = 0.004 seconds. The delay interval is measured in these samples.
  dt = downsample / data.fs

  # Refit the selected PySINDy model on the training trajectories. We use
  # PySINDy only for fitting the equations; the simulation below is done
  # explicitly with scipy.integrate.solve_ivp.
  model = fit_sindy_model(
    train_embedded,
    dt=dt,
    config=SINDyConfig(
      threshold=threshold,
      degree=degree,
      smooth_window=smooth_window,
    ),
  )

  # For degree=1, PySINDy learns an affine linear system:
  #   dx/dt = c + A x
  # ``constant`` is c, and ``matrix`` is A. Eigenvalues should be computed from
  # A only, but trajectory simulation should include both c and A.
  constant, matrix, feature_names = extract_linear_system(model)

  # Simulate each held-out fixation trial from its first delay vector. This is
  # the important delayed-embedding detail: the initial condition is not one
  # scalar LFP value, it is a vector with n_delays values.
  simulated_trials = [
    simulate_linear_trajectory_scipy(
      constant,
      matrix,
      measured,
      dt=dt,
      horizon_s=args.horizon,
    )
    for measured in test_embedded
  ]

  # Compare simulated trajectories to the measured held-out embedded
  # trajectories over the requested horizon.
  metrics = [
    trial_metrics(trial, measured, simulated)
    for trial, measured, simulated in zip(
      test_trials,
      test_embedded,
      simulated_trials,
    )
  ]

  args.out_dir.mkdir(parents=True, exist_ok=True)

  # Save per-trial metrics so the trajectory comparison can be inspected
  # outside Python.
  metrics_path = args.out_dir / "simulation_metrics.csv"
  with metrics_path.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(metrics[0].keys()))
    writer.writeheader()
    writer.writerows(metrics)

  # Save the x0 coordinate plot. x0 is the first/current LFP value in the delay
  # vector, so this is the easiest view to compare with the original LFP trace.
  x0_plot_path = args.out_dir / "held_out_trials_x0_comparison.png"
  save_x0_grid(
    x0_plot_path,
    trials=test_trials,
    measured_trials=test_embedded,
    simulated_trials=simulated_trials,
    dt=dt,
  )

  # Save one detailed plot with every delay coordinate. This is useful for
  # checking whether the simulated delay coordinates remain mutually coherent.
  coordinate_plot_path = args.out_dir / (
    f"trial_{test_trials[0]}_all_coordinates.png"
  )
  save_coordinate_plot(
    coordinate_plot_path,
    trial=test_trials[0],
    measured=test_embedded[0],
    simulated=simulated_trials[0],
    dt=dt,
  )

  # Save the human-readable equations printed by PySINDy.
  equations_path = args.out_dir / "fitted_equations.txt"
  equations_path.write_text(
    "\n".join(
      f"(x{index})' = {equation}"
      for index, equation in enumerate(model.equations())
    )
    + "\n"
  )

  # Save the explicit SciPy system components. These are the exact numerical
  # objects used for simulation:
  #   constant_vector_c.npy stores c
  #   coefficient_matrix_A.npy stores A
  #   first_initial_condition.npy stores the first held-out delay vector
  constant_path = args.out_dir / "constant_vector_c.npy"
  matrix_path = args.out_dir / "coefficient_matrix_A.npy"
  initial_condition_path = args.out_dir / "first_initial_condition.npy"
  np.save(constant_path, constant)
  np.save(matrix_path, matrix)
  np.save(initial_condition_path, test_embedded[0][0])

  # Save a text version as well, so it can be opened quickly without loading
  # NumPy arrays.
  linear_system_path = args.out_dir / "linear_system_for_scipy.txt"
  linear_system_path.write_text(
    "feature_names:\n"
    + json.dumps(feature_names)
    + "\n\nconstant vector c:\n"
    + np.array2string(constant, precision=8, suppress_small=False)
    + "\n\ncoefficient matrix A:\n"
    + np.array2string(matrix, precision=8, suppress_small=False)
    + "\n\nfirst held-out initial condition:\n"
    + np.array2string(test_embedded[0][0], precision=8, suppress_small=False)
    + "\n"
  )

  summary = {
    "source_model_csv": str(args.model_csv),
    "source_row_index": args.row_index,
    "simulation_horizon_s": args.horizon,
    "configuration": configuration,
    "simulation_method": "scipy.integrate.solve_ivp",
    "ode_form": "dx/dt = c + A x",
    "feature_names": feature_names,
    "dt_seconds": dt,
    "initial_condition_description": (
      "Each simulation starts from measured[0], the first full delay vector "
      "of the held-out trial."
    ),
    "mean_trajectory_rmse_all_coordinates": float(
      np.mean([row["trajectory_rmse_all_coordinates"] for row in metrics])
    ),
    "mean_x0_rmse": float(np.mean([row["x0_rmse"] for row in metrics])),
    "mean_x0_correlation": float(
      np.nanmean([row["x0_correlation"] for row in metrics])
    ),
  }
  summary_path = args.out_dir / "simulation_summary.json"
  summary_path.write_text(json.dumps(summary, indent=2) + "\n")

  print(f"mean trajectory RMSE: {summary['mean_trajectory_rmse_all_coordinates']:.4f}")
  print(f"mean x0 RMSE: {summary['mean_x0_rmse']:.4f}")
  print(f"mean x0 correlation: {summary['mean_x0_correlation']:.4f}")
  print(f"saved: {metrics_path}")
  print(f"saved: {x0_plot_path}")
  print(f"saved: {coordinate_plot_path}")
  print(f"saved: {equations_path}")
  print(f"saved: {constant_path}")
  print(f"saved: {matrix_path}")
  print(f"saved: {initial_condition_path}")
  print(f"saved: {linear_system_path}")
  print(f"saved: {summary_path}")


if __name__ == "__main__":
  main()
