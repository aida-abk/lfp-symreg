from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SINDyConfig:
  """PySINDy hyperparameters for one fitted model."""

  threshold: float
  degree: int
  smooth_window: int = 0


def delay_embed_trace(trace: np.ndarray, n_delays: int, delay: int) -> np.ndarray:
  """Build delay coordinates from one scalar time series."""
  if n_delays < 1:
    raise ValueError("n_delays must be >= 1")
  if delay < 1:
    raise ValueError("delay must be >= 1 sample")

  trace = np.asarray(trace, dtype=float).squeeze()
  if trace.ndim != 1:
    raise ValueError(f"Expected a 1D trace, got shape {trace.shape}")

  if n_delays == 1:
    return trace.reshape(-1, 1)

  n_rows = trace.size - (n_delays - 1) * delay
  if n_rows <= 0:
    raise ValueError(
      "Trace is too short for this embedding. "
      f"trace length={trace.size}, n_delays={n_delays}, delay={delay}"
    )
  return np.column_stack(
    [trace[offset : offset + n_rows] for offset in range((n_delays - 1) * delay, -1, -delay)]
  )


def delay_embed_trajectories(
  trajectories: list[np.ndarray],
  n_delays: int,
  delay: int,
) -> list[np.ndarray]:
  """Delay-embed scalar trajectories, leaving multivariate data unchanged."""
  embedded = []
  for trajectory in trajectories:
    values = np.asarray(trajectory, dtype=float)
    if values.ndim == 1:
      embedded.append(delay_embed_trace(values, n_delays=n_delays, delay=delay))
    elif values.ndim == 2:
      embedded.append(values)
    else:
      raise ValueError(f"Expected 1D or 2D trajectory, got shape {values.shape}")
  return embedded


def fit_sindy_model(trajectories: list[np.ndarray], dt: float, config: SINDyConfig):
  """Fit a PySINDy model to one or more trajectories."""
  try:
    import pysindy as ps
  except ImportError as exc:
    raise ImportError("PySINDy is not installed.") from exc

  kwargs = {}
  if config.smooth_window and config.smooth_window > 2:
    window = config.smooth_window
    if window % 2 == 0:
      window += 1
    kwargs["differentiation_method"] = ps.SmoothedFiniteDifference(
      smoother_kws={"window_length": window, "polyorder": 3}
    )

  model = ps.SINDy(
    optimizer=ps.STLSQ(threshold=config.threshold),
    feature_library=ps.PolynomialLibrary(degree=config.degree),
    **kwargs,
  )
  try:
    model.fit(trajectories, t=dt)
  except TypeError:
    model.fit(trajectories, t=dt, multiple_trajectories=True)
  return model


def equation_text(model) -> str:
  """Return fitted equations as a compact single-line string."""
  return " | ".join(
    f"(x{index})' = {equation}"
    for index, equation in enumerate(model.equations())
  )


def count_terms(model) -> int:
  """Count nonzero coefficients in a fitted PySINDy model."""
  return int(np.count_nonzero(np.abs(model.coefficients()) > 1e-12))
