from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class SyntheticDataset:
  """Synthetic trajectories and metadata for validating the SINDy pipeline."""

  name: str
  trajectories: list[np.ndarray]
  dt: float
  feature_names: list[str]


def lorenz_rhs(
  _time: float,
  state: np.ndarray,
  sigma: float = 10.0,
  rho: float = 28.0,
  beta: float = 8.0 / 3.0,
) -> np.ndarray:
  """Evaluate the Lorenz system right-hand side."""
  x, y, z = state
  return np.asarray([
    sigma * (y - x),
    x * (rho - z) - y,
    x * y - beta * z,
  ])


def make_lorenz_dataset(
  n_trajectories: int = 8,
  duration: float = 8.0,
  dt: float = 0.01,
  seed: int = 0,
) -> SyntheticDataset:
  """Create Lorenz trajectories through the same multi-trajectory interface."""
  rng = np.random.default_rng(seed)
  time = np.arange(0.0, duration + dt / 2, dt)
  trajectories = []
  for _index in range(n_trajectories):
    initial = np.asarray([-8.0, 8.0, 27.0]) + rng.normal(scale=2.0, size=3)
    solution = solve_ivp(
      lorenz_rhs,
      t_span=(time[0], time[-1]),
      y0=initial,
      t_eval=time,
      method="LSODA",
      rtol=1e-10,
      atol=1e-12,
    )
    if not solution.success:
      raise RuntimeError(f"Lorenz integration failed: {solution.message}")
    trajectories.append(solution.y.T)

  return SyntheticDataset(
    name="lorenz",
    trajectories=trajectories,
    dt=dt,
    feature_names=["x", "y", "z"],
  )
