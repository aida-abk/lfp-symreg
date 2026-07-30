"""Hankel-SVD (HAVOK-style) coordinates for delay-embedded scalar signals.

Takens' theorem guarantees that a delay embedding is diffeomorphic to the
original attractor. It does not guarantee that the dynamics are sparse, or
even polynomial, *in those delay coordinates*. Raw delay coordinates are also
structurally degenerate: consecutive coordinates are shifted copies of one
signal, so ``d/dt`` of coordinate k is very nearly coordinate k-1's
derivative, and the first N-1 equations of a fitted system are near-trivial
shift relations.

Brunton et al. (2017), "Chaos as an intermittently forced linear system"
(Nat. Commun. 8:19), sidesteps this by taking the SVD of the Hankel matrix
first and modelling the dynamics in the leading singular-vector coordinates,
where the dominant behaviour is close to linear.

This module supplies only the coordinate change. Fitting and simulation stay
in :mod:`models.sindy` and :mod:`models.validation`.

The key constraint the API enforces is that the *same* projection is applied
to training and held-out data. Computing a separate SVD per trial would
produce trial-specific coordinates and make held-out simulation meaningless.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.sindy import delay_embed_trace


@dataclass(frozen=True)
class HankelBasis:
  """A shared set of Hankel-SVD coordinates fitted on training trials.

  Attributes:
    modes: Right singular vectors with shape ``(n_delays, n_modes)``. Columns
      are orthonormal and map a delay vector to mode coordinates.
    singular_values: All singular values of the stacked training Hankel
      matrix, in descending order, in the units of the embedded signal.
    n_delays: Embedding dimension the basis was built from. Unitless count.
    delay: Coordinate spacing in processed samples.
  """

  modes: np.ndarray
  singular_values: np.ndarray
  n_delays: int
  delay: int

  @property
  def n_modes(self) -> int:
    """Return the number of retained modes."""
    return int(self.modes.shape[1])

  @property
  def explained_variance_ratio(self) -> float:
    """Return the fraction of squared singular value mass the modes retain."""
    total = float(np.sum(self.singular_values**2))
    kept = float(np.sum(self.singular_values[: self.n_modes] ** 2))
    return kept / total if total > 0 else float("nan")

  def project(self, trace: np.ndarray) -> np.ndarray:
    """Map one scalar trace into mode coordinates.

    Args:
      trace: Scalar samples with shape ``(n_samples,)``.

    Returns:
      Mode-coordinate trajectory with shape ``(n_embedded_samples, n_modes)``.
    """
    embedded = delay_embed_trace(trace, n_delays=self.n_delays, delay=self.delay)
    return embedded @ self.modes

  def embed(self, trace: np.ndarray) -> np.ndarray:
    """Return the raw delay embedding used by :meth:`project`.

    Provided so that a measured trajectory and a reconstructed simulation can
    be compared in the same delay space.

    Args:
      trace: Scalar samples with shape ``(n_samples,)``.

    Returns:
      Delay-coordinate trajectory with shape ``(n_embedded_samples, n_delays)``.
    """
    return delay_embed_trace(trace, n_delays=self.n_delays, delay=self.delay)

  def reconstruct(self, mode_trajectory: np.ndarray) -> np.ndarray:
    """Map a mode-coordinate trajectory back to delay coordinates.

    Because the modes are orthonormal but truncated, this is a projection
    rather than an exact inverse: it returns the closest delay-space
    trajectory expressible in the retained modes.

    Args:
      mode_trajectory: Trajectory with shape ``(time, n_modes)``.

    Returns:
      Delay-coordinate trajectory with shape ``(time, n_delays)``. Column zero
      is the reconstructed current sample of the signal.
    """
    values = np.asarray(mode_trajectory, dtype=float)
    if values.ndim != 2 or values.shape[1] != self.n_modes:
      raise ValueError(
        f"Expected a (time, {self.n_modes}) trajectory, got {values.shape}."
      )
    return values @ self.modes.T


def fit_hankel_basis(
  trajectories: list[np.ndarray],
  n_delays: int,
  delay: int,
  n_modes: int,
) -> HankelBasis:
  """Fit shared Hankel-SVD coordinates from training trajectories.

  Each trajectory is delay-embedded separately and the resulting blocks are
  stacked before the SVD, so no delay vector ever spans a trial boundary.

  Args:
    trajectories: Training traces, each with shape ``(n_samples,)``, in the
      signal units produced by preprocessing.
    n_delays: Embedding dimension. Unitless count.
    delay: Coordinate spacing in processed samples.
    n_modes: Number of leading singular vectors to retain.

  Returns:
    A :class:`HankelBasis` holding the retained modes and the full singular
    value spectrum.

  Raises:
    ValueError: If ``n_modes`` exceeds ``n_delays``.
  """
  if n_modes < 1:
    raise ValueError("n_modes must be at least 1.")
  if n_modes > n_delays:
    raise ValueError(
      f"n_modes={n_modes} cannot exceed the embedding dimension n_delays={n_delays}."
    )

  blocks = [
    delay_embed_trace(np.asarray(t, dtype=float), n_delays=n_delays, delay=delay)
    for t in trajectories
  ]
  stacked = np.vstack(blocks)
  # full_matrices=False keeps the factorization at rank min(rows, n_delays);
  # rows greatly exceeds n_delays here, so Vt is (n_delays, n_delays).
  _, singular_values, vt = np.linalg.svd(stacked, full_matrices=False)
  return HankelBasis(
    modes=np.ascontiguousarray(vt[:n_modes].T),
    singular_values=singular_values,
    n_delays=n_delays,
    delay=delay,
  )


def modes_for_variance(singular_values: np.ndarray, target: float) -> int:
  """Return how many modes are needed to reach a squared-mass target.

  Args:
    singular_values: Descending singular values.
    target: Desired fraction of total squared singular value mass, in ``(0, 1]``.

  Returns:
    The smallest mode count reaching ``target``.
  """
  if not 0 < target <= 1:
    raise ValueError("target must lie in (0, 1].")
  energy = np.asarray(singular_values, dtype=float) ** 2
  total = float(np.sum(energy))
  if total <= 0:
    return len(energy)
  return int(np.searchsorted(np.cumsum(energy) / total, target) + 1)
