from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StoredPolynomialModel:
  """Evaluate a polynomial SINDy ODE reconstructed from saved coefficients.

  Attributes:
    degree: Maximum polynomial degree. Unitless.
    coefficients: Coefficient matrix with shape ``(state, feature)``. Its
      units are those of the fitted differential equations.
    feature_names: Ordered polynomial-library feature names saved at fitting.
  """

  degree: int
  coefficients: np.ndarray
  feature_names: list[str]

  def __post_init__(self) -> None:
    """Build and validate the polynomial feature library."""
    try:
      import pysindy as ps
    except ImportError as exc:
      raise ImportError("PySINDy is required to reconstruct a stored model.") from exc

    self.coefficients = np.asarray(self.coefficients, dtype=float)
    if self.coefficients.ndim != 2:
      raise ValueError(
        f"Expected a 2D coefficient matrix, got {self.coefficients.shape}."
      )
    n_states, n_features = self.coefficients.shape
    if len(self.feature_names) != n_features:
      raise ValueError(
        f"Stored {len(self.feature_names)} feature names for {n_features} coefficients."
      )

    self._feature_library = ps.PolynomialLibrary(degree=self.degree)
    self._feature_library.fit(np.zeros((2, n_states), dtype=float))
    reconstructed_names = self._feature_library.get_feature_names()
    if reconstructed_names != self.feature_names:
      raise ValueError(
        "Reconstructed polynomial features do not match the stored feature order."
      )

  def predict(self, states: np.ndarray) -> np.ndarray:
    """Evaluate state derivatives for one or more states.

    Args:
      states: State matrix with shape ``(samples, state)`` in the signal units
        used during fitting.

    Returns:
      Derivative matrix with shape ``(samples, state)`` in signal units per
      second.
    """
    values = np.asarray(states, dtype=float)
    if values.ndim != 2 or values.shape[1] != self.coefficients.shape[0]:
      raise ValueError(
        "states must have shape (samples, state) matching the coefficient matrix."
      )
    features = np.asarray(self._feature_library.transform(values), dtype=float)
    return features @ self.coefficients.T


@dataclass(frozen=True)
class SINDyConfig:
  """PySINDy hyperparameters for one fitted model.

  Attributes:
    degree: Maximum polynomial degree in the feature library. Unitless.
    threshold: STLSQ coefficient-removal threshold. Coefficient-scale units;
      fixed at PySINDy's default of 0.1 in the active exploration pipeline.
    alpha: STLSQ ridge regularization strength. Its numerical interpretation
      depends on feature scaling; this records PySINDy's current default of 0.05.
    normalize_columns: Whether STLSQ normalizes library columns internally
      before thresholding and rescales final coefficients to original units.
    smooth_window: Optional Savitzky-Golay smoothing window in samples. A value
      of zero uses PySINDy's default finite-difference derivative.
    smoothing_polyorder: Polynomial order used by Savitzky-Golay smoothing.
  """

  degree: int
  threshold: float = 0.1
  alpha: float = 0.05
  normalize_columns: bool = False
  smooth_window: int = 0
  smoothing_polyorder: int = 3


def delay_embed_trace(trace: np.ndarray, n_delays: int, delay: int) -> np.ndarray:
  """Build delay coordinates from one scalar time series.

  Args:
    trace: Scalar samples with shape ``(n_samples,)``. Units are preserved.
    n_delays: Number of delay coordinates. Unitless count.
    delay: Separation between coordinates in processed samples.

  Returns:
    Array with shape ``(n_embedded_samples, n_delays)``. Column zero is the
    current sample; subsequent columns move backward through the signal.
  """
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
  """Delay-embed scalar trajectories, leaving multivariate data unchanged.

  Args:
    trajectories: Trial trajectories in their original signal units.
    n_delays: Number of delay coordinates. Unitless count.
    delay: Coordinate spacing in processed samples.

  Returns:
    One two-dimensional state trajectory per input trial.
  """
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
  """Fit one autonomous PySINDy model to multiple whole-trial trajectories.

  Args:
    trajectories: State trajectories with shape ``(time, state)``. Signal
      units depend on preprocessing.
    dt: Processed sample interval in seconds.
    config: Polynomial-library and optimizer settings.

  Returns:
    A fitted ``pysindy.SINDy`` model.
  """
  try:
    import pysindy as ps
  except ImportError as exc:
    raise ImportError("PySINDy is not installed.") from exc

  kwargs = {}
  if config.smooth_window and config.smooth_window > 2:
    window = config.smooth_window
    if window % 2 == 0:
      window += 1
    if config.smoothing_polyorder >= window:
      raise ValueError("smoothing_polyorder must be smaller than smooth_window")
    kwargs["differentiation_method"] = ps.SmoothedFiniteDifference(
      smoother_kws={
        "window_length": window,
        "polyorder": config.smoothing_polyorder,
      }
    )

  model = ps.SINDy(
    optimizer=ps.STLSQ(
      threshold=config.threshold,
      alpha=config.alpha,
      normalize_columns=config.normalize_columns,
    ),
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
