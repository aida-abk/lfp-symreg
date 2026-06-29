from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal, stats
from scipy.integrate import solve_ivp
from sklearn.metrics import mean_squared_error


@dataclass(frozen=True)
class ValidationConfig:
  """Thresholds used to reject unsuitable simulated trajectories."""

  simulation_horizon: float = 2.0
  amplitude_factor: float = 10.0
  collapse_std_fraction: float = 0.05
  min_psd_similarity: float = 0.25
  min_autocorrelation_similarity: float = 0.25
  max_distribution_ks: float = 0.75
  max_rhs_evaluations: int = 20_000


def simulate_model(
  model,
  initial_state: np.ndarray,
  dt: float,
  horizon_s: float,
  amplitude_limit: float,
  max_rhs_evaluations: int,
) -> tuple[np.ndarray | None, str]:
  """Simulate a fitted PySINDy model using SciPy's ODE integrator."""
  n_samples = int(round(horizon_s / dt)) + 1
  if n_samples < 2:
    return None, "simulation horizon contains fewer than two samples"
  time = np.arange(n_samples, dtype=float) * dt
  rhs_evaluations = 0

  def right_hand_side(_time: float, state: np.ndarray) -> np.ndarray:
    """Evaluate the learned right-hand side at one state."""
    nonlocal rhs_evaluations
    rhs_evaluations += 1
    if rhs_evaluations > max_rhs_evaluations:
      raise RuntimeError(f"exceeded {max_rhs_evaluations} RHS evaluations")
    derivative = np.asarray(model.predict(state.reshape(1, -1)), dtype=float)[0]
    if not np.all(np.isfinite(derivative)):
      raise FloatingPointError("model derivative became non-finite")
    return derivative

  def amplitude_event(_time: float, state: np.ndarray) -> float:
    """Stop when the simulated state exceeds the configured amplitude limit."""
    return amplitude_limit - float(np.max(np.abs(state)))

  amplitude_event.terminal = True
  amplitude_event.direction = -1

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
      events=amplitude_event,
    )
  except Exception as exc:
    return None, f"integration error: {exc}"

  if not solution.success:
    return None, f"integration failed: {solution.message}"
  if solution.t.size != time.size:
    return None, f"amplitude exceeded {amplitude_limit:g}"

  simulated = solution.y.T
  if not np.all(np.isfinite(simulated)):
    return None, "simulation produced non-finite values"
  return simulated, ""


def waveform_correlation(left: np.ndarray, right: np.ndarray) -> float:
  """Return Pearson correlation, or NaN if either input is constant."""
  if np.std(left) == 0 or np.std(right) == 0:
    return float("nan")
  return float(np.corrcoef(left, right)[0, 1])


def psd_similarity(measured: np.ndarray, simulated: np.ndarray, fs: float) -> float:
  """Compare normalized Welch PSD curves using Pearson correlation."""
  nperseg = min(256, measured.size, simulated.size)
  if nperseg < 8:
    return float("nan")
  _, measured_psd = signal.welch(measured, fs=fs, nperseg=nperseg)
  _, simulated_psd = signal.welch(simulated, fs=fs, nperseg=nperseg)
  measured_psd = measured_psd / max(np.sum(measured_psd), np.finfo(float).eps)
  simulated_psd = simulated_psd / max(np.sum(simulated_psd), np.finfo(float).eps)
  return waveform_correlation(measured_psd, simulated_psd)


def autocorrelation_similarity(
  measured: np.ndarray,
  simulated: np.ndarray,
  max_lag: int = 100,
) -> float:
  """Compare short-lag autocorrelation curves using Pearson correlation."""
  def autocorrelation(values: np.ndarray) -> np.ndarray:
    centered = values - np.mean(values)
    if np.std(centered) == 0:
      return np.full(max_lag + 1, np.nan)
    corr = np.correlate(centered, centered, mode="full")
    corr = corr[corr.size // 2 :]
    corr = corr[: max_lag + 1]
    return corr / corr[0]

  limit = min(max_lag, measured.size - 1, simulated.size - 1)
  if limit < 2:
    return float("nan")
  return waveform_correlation(
    autocorrelation(measured)[: limit + 1],
    autocorrelation(simulated)[: limit + 1],
  )


def finite_difference_jacobian(model, state: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
  """Estimate the model Jacobian at one state with central differences."""
  state = np.asarray(state, dtype=float)
  jacobian = np.zeros((state.size, state.size), dtype=float)
  for index in range(state.size):
    step = np.zeros_like(state)
    step[index] = epsilon
    plus = np.asarray(model.predict((state + step).reshape(1, -1)), dtype=float)[0]
    minus = np.asarray(model.predict((state - step).reshape(1, -1)), dtype=float)[0]
    jacobian[:, index] = (plus - minus) / (2 * epsilon)
  return jacobian


def evaluate_simulation(
  measured: np.ndarray,
  simulated: np.ndarray,
  fs: float,
  config: ValidationConfig,
) -> dict[str, float | str]:
  """Calculate rejection metrics comparing one measured and simulated trace."""
  n_samples = min(measured.shape[0], simulated.shape[0])
  target = measured[:n_samples]
  predicted = simulated[:n_samples]
  target_x0 = target[:, 0]
  predicted_x0 = predicted[:, 0]
  measured_std = max(float(np.std(target_x0)), np.finfo(float).eps)
  simulated_tail_std = float(np.std(predicted_x0[n_samples // 2 :]))
  ks_statistic = float(stats.ks_2samp(target_x0, predicted_x0).statistic)
  metrics: dict[str, float | str] = {
    "trajectory_rmse": float(np.sqrt(mean_squared_error(target, predicted))),
    "x0_rmse": float(np.sqrt(mean_squared_error(target_x0, predicted_x0))),
    "x0_correlation": waveform_correlation(target_x0, predicted_x0),
    "max_amplitude_ratio": float(np.max(np.abs(predicted))) / max(1.0, float(np.max(np.abs(target)))),
    "collapse_std_ratio": simulated_tail_std / measured_std,
    "psd_similarity": psd_similarity(target_x0, predicted_x0, fs=fs),
    "autocorrelation_similarity": autocorrelation_similarity(target_x0, predicted_x0),
    "distribution_ks": ks_statistic,
    "rejection_reason": "",
  }

  if metrics["collapse_std_ratio"] < config.collapse_std_fraction:
    metrics["rejection_reason"] = "trajectory collapsed toward a constant"
  elif metrics["psd_similarity"] < config.min_psd_similarity:
    metrics["rejection_reason"] = "PSD similarity below threshold"
  elif metrics["autocorrelation_similarity"] < config.min_autocorrelation_similarity:
    metrics["rejection_reason"] = "autocorrelation similarity below threshold"
  elif metrics["distribution_ks"] > config.max_distribution_ks:
    metrics["rejection_reason"] = "amplitude distribution differs too much"

  return metrics
