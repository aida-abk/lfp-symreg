"""PySR utilities: derivative estimation, regression array construction, and metrics."""

from __future__ import annotations

import numpy as np
from scipy import signal
from sklearn.metrics import mean_squared_error, r2_score


def estimate_derivative(
  trajectory: np.ndarray,
  dt: float,
  smooth_window: int,
  polynomial_order: int = 3,
) -> np.ndarray:
  """Estimate derivatives of a delay-embedded trajectory without crossing trial boundaries.

  Args:
    trajectory: State trajectory with shape ``(n_samples, n_states)``.
    dt: Sample interval in seconds.
    smooth_window: Savitzky-Golay window in samples; 0 or 1 uses finite differences.
    polynomial_order: Polynomial order for Savitzky-Golay smoothing.

  Returns:
    Derivative array with the same shape as ``trajectory``.
  """
  if smooth_window <= 2:
    return np.gradient(trajectory, dt, axis=0, edge_order=2)

  window = smooth_window + 1 if smooth_window % 2 == 0 else smooth_window
  if window > trajectory.shape[0]:
    raise ValueError(
      f"smooth_window={window} exceeds trajectory length={trajectory.shape[0]}"
    )
  polyorder = min(polynomial_order, window - 1)
  return signal.savgol_filter(
    trajectory,
    window_length=window,
    polyorder=polyorder,
    deriv=1,
    delta=dt,
    axis=0,
  )


def build_regression_arrays(
  embedded_trials: list[np.ndarray],
  dt: float,
  smooth_window: int,
) -> tuple[np.ndarray, np.ndarray]:
  """Stack delay states and per-trial derivative estimates into regression matrices.

  Args:
    embedded_trials: List of delay-embedded trajectories, each with shape
      ``(n_samples, n_states)``. Trials may have unequal lengths.
    dt: Sample interval in seconds.
    smooth_window: Savitzky-Golay derivative window in samples; 0 uses finite differences.

  Returns:
    ``(X, y)`` where ``X`` is the stacked state matrix and ``y`` is the stacked
    derivative matrix, both with shape ``(total_samples, n_states)``.
  """
  derivatives = [
    estimate_derivative(trial, dt=dt, smooth_window=smooth_window)
    for trial in embedded_trials
  ]
  return np.vstack(embedded_trials), np.vstack(derivatives)


def cap_rows(
  x: np.ndarray,
  y: np.ndarray,
  max_samples: int | None,
) -> tuple[np.ndarray, np.ndarray]:
  """Keep evenly spaced rows when a deterministic training-sample cap is requested.

  Args:
    x: State matrix with shape ``(n_samples, n_features)``.
    y: Derivative matrix with shape ``(n_samples, n_targets)``.
    max_samples: Maximum rows to keep, or ``None`` to keep all.

  Returns:
    Subsampled ``(x, y)`` pair.
  """
  if max_samples is None or x.shape[0] <= max_samples:
    return x, y
  indices = np.linspace(0, x.shape[0] - 1, num=max_samples, dtype=int)
  return x[indices], y[indices]


def selected_equations(model) -> list[str]:
  """Return the selected symbolic equations from a fitted PySRRegressor as strings.

  Args:
    model: Fitted ``PySRRegressor`` instance.

  Returns:
    One string per target dimension.
  """
  equations = model.sympy()
  if isinstance(equations, list):
    return [str(equation) for equation in equations]
  return [str(equations)]


def metric_rows(
  y_true: np.ndarray,
  y_pred: np.ndarray,
) -> list[dict[str, float | int | str]]:
  """Compute per-target and aggregate derivative fit metrics.

  Args:
    y_true: Ground-truth derivatives with shape ``(n_samples, n_targets)``.
    y_pred: Predicted derivatives with the same shape.

  Returns:
    One dict per target plus one aggregate dict (target index ``-1``).
  """
  rows: list[dict[str, float | int | str]] = []
  for target in range(y_true.shape[1]):
    mse = float(mean_squared_error(y_true[:, target], y_pred[:, target]))
    rows.append(
      {
        "target": f"dx{target}_dt",
        "target_index": target,
        "r2": float(r2_score(y_true[:, target], y_pred[:, target])),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
      }
    )
  mse_all = float(mean_squared_error(y_true, y_pred))
  rows.append(
    {
      "target": "all_uniform_average",
      "target_index": -1,
      "r2": float(r2_score(y_true, y_pred, multioutput="uniform_average")),
      "mse": mse_all,
      "rmse": float(np.sqrt(mse_all)),
    }
  )
  return rows


def parse_operators(value: str) -> list[str]:
  """Parse a comma-separated list of PySR operator strings.

  Args:
    value: Comma-separated operator names, e.g. ``"+,-,*"``.

  Returns:
    List of stripped operator strings.
  """
  return [op.strip() for op in value.split(",") if op.strip()]
