"""Polynomial library construction, in numpy only.

These helpers are shared by the PyTorch autoencoder in :mod:`models.ae_sindy`
and by the script that drives the reference TensorFlow implementation. They
live in their own module precisely so that neither framework is required to
use them: importing them must not pull in torch, and must not pull in
TensorFlow either.

The ordering produced here -- constant term first, then ascending total
degree, with exponents within a degree following
``itertools.combinations_with_replacement`` -- is the convention the rest of
this project assumes. Any external library whose coefficients are interpreted
against it should be checked rather than trusted; see
``verify_library_ordering`` in ``scripts/aesindy/run_lfp.py``.
"""
from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np


def polynomial_exponents(latent_dim: int, poly_order: int) -> list[tuple[int, ...]]:
  """Enumerate polynomial library exponents, constant term first.

  Args:
    latent_dim: Number of latent coordinates. Unitless count.
    poly_order: Maximum total degree. Unitless.

  Returns:
    Exponent tuples ordered by increasing total degree, each of length
    ``latent_dim``. The first entry is the all-zero constant term.
  """
  exponents = [tuple([0] * latent_dim)]
  for degree in range(1, poly_order + 1):
    for combination in combinations_with_replacement(range(latent_dim), degree):
      exponent = [0] * latent_dim
      for index in combination:
        exponent[index] += 1
      exponents.append(tuple(exponent))
  return exponents


def feature_names(exponents: list[tuple[int, ...]]) -> list[str]:
  """Return readable names for polynomial library features.

  Args:
    exponents: Exponent tuples from :func:`polynomial_exponents`.

  Returns:
    Names such as ``1``, ``z0``, ``z0 z1``, ``z0^2``.
  """
  names = []
  for exponent in exponents:
    if not any(exponent):
      names.append("1")
      continue
    parts = []
    for index, power in enumerate(exponent):
      if power == 1:
        parts.append(f"z{index}")
      elif power > 1:
        parts.append(f"z{index}^{power}")
    names.append(" ".join(parts))
  return names


def polynomial_library_numpy(
  states: np.ndarray, exponents: list[tuple[int, ...]]
) -> np.ndarray:
  """Evaluate the polynomial library on numpy states.

  Args:
    states: Latent states with shape ``(samples, latent_dim)``.
    exponents: Exponent tuples from :func:`polynomial_exponents`.

  Returns:
    Feature matrix with shape ``(samples, n_features)``.
  """
  values = np.atleast_2d(np.asarray(states, dtype=float))
  columns = []
  for exponent in exponents:
    column = np.ones(values.shape[0], dtype=float)
    for index, power in enumerate(exponent):
      if power:
        column = column * values[:, index] ** power
    columns.append(column)
  return np.stack(columns, axis=1)
