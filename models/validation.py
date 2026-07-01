from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal, stats
from scipy.integrate import solve_ivp
from sklearn.metrics import mean_squared_error


@dataclass(frozen=True)
class SimulationConfig:
  """Explicit simulation and optional divergence settings.

  This configuration contains no scientific rejection thresholds. The active
  pipeline records simulation outcomes and raw metrics for later evaluation.

  Attributes:
    simulation_horizon_s: Maximum integration duration in seconds. Required;
      there is no hidden library default.
    divergence_threshold_std: Optional error threshold in measured-signal
      standard deviations. ``None`` disables time-until-divergence.
    divergence_persistence_s: Required continuous exceedance duration in
      seconds. ``None`` disables time-until-divergence.
  """

  simulation_horizon_s: float
  divergence_threshold_std: float | None = None
  divergence_persistence_s: float | None = None


@dataclass(frozen=True)
class SimulationResult:
  """A complete or partial numerical simulation of a fitted model.

  Attributes:
    trajectory: Simulated array with shape ``(time, state)`` in model units.
    time: Returned simulation times in seconds.
    completed: Whether integration reached the requested horizon.
    failure_reason: Empty on completion; otherwise numerical failure text.
    rhs_evaluations: Number of learned-equation evaluations requested by LSODA.
  """

  trajectory: np.ndarray | None
  time: np.ndarray
  completed: bool
  failure_reason: str
  rhs_evaluations: int

  @property
  def reached_horizon_s(self) -> float:
    """Return the last simulated time in seconds."""
    return float(self.time[-1]) if self.time.size else 0.0


def simulate_model_detailed(
  model,
  initial_state: np.ndarray,
  dt: float,
  horizon_s: float,
) -> SimulationResult:
  """Simulate a model while retaining a trajectory produced before failure.

  Args:
    model: Fitted model whose ``predict`` method returns derivatives.
    initial_state: Initial state vector in the model's signal units.
    dt: Requested output interval in seconds.
    horizon_s: Maximum integration duration in seconds.

  Returns:
    Complete or partial simulation output and an explicit failure reason.

  Notes:
    LSODA is retained as a provisional numerical solver because it can switch
    between non-stiff and stiff integration. The tolerances are numerical
    settings, not model-acceptance criteria.
  """
  n_samples = int(round(horizon_s / dt)) + 1
  if n_samples < 2:
    return SimulationResult(
      trajectory=None,
      time=np.empty(0),
      completed=False,
      failure_reason="simulation horizon contains fewer than two samples",
      rhs_evaluations=0,
    )
  time = np.arange(n_samples, dtype=float) * dt
  rhs_evaluations = 0
  last_rhs_time = 0.0

  def right_hand_side(_time: float, state: np.ndarray) -> np.ndarray:
    """Evaluate the learned right-hand side at one state."""
    nonlocal last_rhs_time, rhs_evaluations
    rhs_evaluations += 1
    last_rhs_time = max(last_rhs_time, float(_time))
    derivative = np.asarray(model.predict(state.reshape(1, -1)), dtype=float)[0]
    if not np.all(np.isfinite(derivative)):
      raise FloatingPointError("model derivative became non-finite")
    return derivative

  try:
    solution = solve_ivp(
      right_hand_side,
      t_span=(time[0], time[-1]),
      y0=np.asarray(initial_state, dtype=float),
      t_eval=time,
      method="LSODA",
      rtol=1e-7,
      atol=1e-9,
      min_step=dt / 100,
    )
  except Exception as exc:
    return SimulationResult(
      trajectory=None,
      time=np.asarray([last_rhs_time]),
      completed=False,
      failure_reason=f"integration error: {exc}",
      rhs_evaluations=rhs_evaluations,
    )

  simulated = solution.y.T
  if not np.all(np.isfinite(simulated)):
    return SimulationResult(
      trajectory=None,
      time=solution.t,
      completed=False,
      failure_reason="simulation produced non-finite values",
      rhs_evaluations=rhs_evaluations,
    )
  if not solution.success:
    return SimulationResult(
      trajectory=simulated,
      time=solution.t,
      completed=False,
      failure_reason=f"integration failed: {solution.message}",
      rhs_evaluations=rhs_evaluations,
    )
  if solution.t.size != time.size:
    return SimulationResult(
      trajectory=simulated,
      time=solution.t,
      completed=False,
      failure_reason="integration ended before the requested horizon",
      rhs_evaluations=rhs_evaluations,
    )
  return SimulationResult(
    trajectory=simulated,
    time=solution.t,
    completed=True,
    failure_reason="",
    rhs_evaluations=rhs_evaluations,
  )


def waveform_correlation(left: np.ndarray, right: np.ndarray) -> float:
  """Return a unitless zero-lag Pearson correlation.

  Args:
    left: First one-dimensional waveform in any amplitude units.
    right: Time-aligned waveform with the same number of samples.

  Returns:
    Unitless correlation from -1 to 1, or NaN for a constant waveform.
  """
  if np.std(left) == 0 or np.std(right) == 0:
    return float("nan")
  return float(np.corrcoef(left, right)[0, 1])


def psd_similarity(measured: np.ndarray, simulated: np.ndarray, fs: float) -> float:
  """Compare normalized Welch PSD shapes using Pearson correlation.

  Args:
    measured: Measured scalar waveform with shape ``(time,)``.
    simulated: Simulated scalar waveform with shape ``(time,)``.
    fs: Sampling frequency in hertz.

  Returns:
    Unitless PSD-shape correlation from -1 to 1, or NaN when too short.
  """
  nperseg = min(256, measured.size, simulated.size)
  if nperseg < 8:
    return float("nan")
  _, measured_psd = signal.welch(measured, fs=fs, nperseg=nperseg)
  _, simulated_psd = signal.welch(simulated, fs=fs, nperseg=nperseg)
  measured_psd = measured_psd / max(np.sum(measured_psd), np.finfo(float).eps)
  simulated_psd = simulated_psd / max(np.sum(simulated_psd), np.finfo(float).eps)
  return waveform_correlation(measured_psd, simulated_psd)


def time_until_divergence(
  measured: np.ndarray,
  simulated: np.ndarray,
  fs: float,
  threshold_std: float,
  persistence_s: float,
  reference_std: float | None = None,
) -> tuple[float, bool]:
  """Return when normalized x0 error first remains large for a set duration.

  The threshold and persistence are deliberately supplied by the caller so the
  project does not silently adopt an unapproved scientific definition.

  Args:
    measured: Measured scalar waveform with shape ``(time,)``.
    simulated: Simulated scalar waveform with shape ``(time,)``.
    fs: Sampling frequency in hertz.
    threshold_std: Absolute-error threshold in measured-signal SD units.
    persistence_s: Required continuous exceedance duration in seconds.
    reference_std: Optional fixed measured-signal SD in signal units.

  Returns:
    Divergence time in seconds and whether divergence occurred.
  """
  if threshold_std <= 0:
    raise ValueError("threshold_std must be positive")
  if persistence_s <= 0:
    raise ValueError("persistence_s must be positive")

  n_samples = min(measured.size, simulated.size)
  target = np.asarray(measured[:n_samples], dtype=float)
  predicted = np.asarray(simulated[:n_samples], dtype=float)
  scale = max(
    float(np.std(target)) if reference_std is None else float(reference_std),
    np.finfo(float).eps,
  )
  exceeds = np.abs(predicted - target) > threshold_std * scale
  persistence_samples = max(1, int(round(persistence_s * fs)))
  if persistence_samples > n_samples:
    return (n_samples - 1) / fs, False

  sustained = np.convolve(
    exceeds.astype(int),
    np.ones(persistence_samples, dtype=int),
    mode="valid",
  )
  matches = np.flatnonzero(sustained == persistence_samples)
  if matches.size:
    return float(matches[0] / fs), True
  return float((n_samples - 1) / fs), False


def evaluate_simulation(
  measured: np.ndarray,
  simulated: np.ndarray,
  fs: float,
  config: SimulationConfig,
  divergence_reference_std: float | None = None,
) -> dict[str, float | bool | None]:
  """Calculate descriptive metrics for one measured/simulated trajectory pair.

  Args:
    measured: Measured state trajectory with shape ``(time, state)``.
    simulated: Simulated state trajectory with shape ``(time, state)``.
    fs: Processed sampling frequency in hertz.
    config: Numerical and optional diagnostic settings.
    divergence_reference_std: Optional fixed x0 scale in signal units.

  Returns:
    Raw descriptive metrics. RMSE values retain the preprocessed signal units;
    correlations, ratios, and KS distance are unitless.
  """
  n_samples = min(measured.shape[0], simulated.shape[0])
  target = measured[:n_samples]
  predicted = simulated[:n_samples]
  target_x0 = target[:, 0]
  predicted_x0 = predicted[:, 0]
  measured_std = max(float(np.std(target_x0)), np.finfo(float).eps)
  simulated_tail_std = float(np.std(predicted_x0[n_samples // 2 :]))
  ks_statistic = float(stats.ks_2samp(target_x0, predicted_x0).statistic)
  metrics: dict[str, float | bool | None] = {
    "trajectory_rmse": float(np.sqrt(mean_squared_error(target, predicted))),
    "x0_rmse": float(np.sqrt(mean_squared_error(target_x0, predicted_x0))),
    "x0_correlation": waveform_correlation(target_x0, predicted_x0),
    "max_amplitude_ratio": float(np.max(np.abs(predicted)))
    / max(float(np.max(np.abs(target))), np.finfo(float).eps),
    "collapse_std_ratio": simulated_tail_std / measured_std,
    "psd_similarity": psd_similarity(target_x0, predicted_x0, fs=fs),
    "distribution_ks": ks_statistic,
    "divergence_time_s": float("nan"),
    "diverged": None,
  }

  if (
    config.divergence_threshold_std is not None
    and config.divergence_persistence_s is not None
  ):
    divergence_time_s, diverged = time_until_divergence(
      target_x0,
      predicted_x0,
      fs=fs,
      threshold_std=config.divergence_threshold_std,
      persistence_s=config.divergence_persistence_s,
      reference_std=divergence_reference_std,
    )
    metrics["divergence_time_s"] = divergence_time_s
    metrics["diverged"] = diverged

  return metrics
