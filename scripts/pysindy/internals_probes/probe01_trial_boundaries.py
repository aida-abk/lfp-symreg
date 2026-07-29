"""Probe: does PySINDy keep separate trajectories isolated during fitting?

Verifies the scientific assumption recorded in CLAUDE.md -- "Trials are
preserved as separate trajectories, never stacked into one matrix" -- against
the actual pysindy 2.1.0 call path, rather than trusting the docstring.

Source claims under test (pysindy 2.1.0):
  1. ``SINDy.fit`` computes derivatives per trajectory. ``_core.py:393`` calls
     ``_process_trajectories``, whose list comprehension (``_core.py:531-538``)
     hands ``calc_trajectory`` exactly one trajectory at a time.
  2. Concatenation happens only afterwards: ``_core.py:405`` (``x_dot``) and
     ``_core.py:407`` (``SampleConcatter`` on the library output), i.e. after
     both differentiation and the library transform.
  3. ``calc_trajectory`` (``feature_library/base.py:69-72``) returns
     ``diff_method.smoothed_x_`` in place of the input ``x``, so with a
     smoothing differentiation method the feature library is built from the
     SMOOTHED signal, not the raw one.

Method: build two trajectories separated by a large offset. If any stage
crossed the boundary, the seam would inject an enormous spurious derivative,
and fitting a list of two trajectories would match fitting their naive
concatenation. Run with:

    .venv/bin/python scripts/pysindy/internals_probes/probe01_trial_boundaries.py
"""
from __future__ import annotations

import numpy as np
import pysindy as ps
from pysindy._core import (
  _adapt_to_multiple_trajectories,
  _check_multiple_trajectories,
  _comprehend_and_validate_inputs,
)


def build_trajectories(dt: float) -> tuple[np.ndarray, np.ndarray]:
  """Return two 2-D trajectories separated by a large constant offset.

  Args:
    dt: Sample interval in seconds.

  Returns:
    Two ``(n_samples, 2)`` arrays. The second is offset by +1000 so that any
    derivative taken across the seam would be about 1000/dt in magnitude.
  """
  t = np.arange(0, 4, dt)
  first = np.column_stack([np.sin(t), np.cos(t)])
  second = np.column_stack([np.sin(t), np.cos(t)]) + 1000.0
  return first, second


def derivatives_via_pysindy(
  trajectories: list[np.ndarray], dt: float, diff_method
) -> list[np.ndarray]:
  """Return per-trajectory derivatives exactly as ``SINDy.fit`` computes them.

  Replicates the input handling at ``_core.py:386-393`` so the result is the
  library's own ``calc_trajectory`` output, not a reimplementation.

  Args:
    trajectories: One or more ``(n_samples, n_coord)`` arrays.
    dt: Sample interval in seconds.
    diff_method: A pysindy differentiation method instance.

  Returns:
    One derivative array per input trajectory.
  """
  library = ps.PolynomialLibrary(degree=1)
  x, t, x_dot, u = trajectories, dt, None, None
  if not _check_multiple_trajectories(x, x_dot, u):
    x, t, x_dot, u = _adapt_to_multiple_trajectories(x, t, x_dot, u)
  x, x_dot, u = _comprehend_and_validate_inputs(x, t, x_dot, u, library)
  return [
    np.asarray(library.calc_trajectory(diff_method, xi, dt)[1]) for xi in x
  ]


def main() -> None:
  """Run every boundary check and print a pass/fail verdict for each."""
  dt = 0.01
  first, second = build_trajectories(dt)
  diff = ps.FiniteDifference()

  # --- Claim 1: derivatives are computed per trajectory --------------------
  joint = derivatives_via_pysindy([first, second], dt, diff)
  alone_first = derivatives_via_pysindy([first], dt, diff)[0]
  alone_second = derivatives_via_pysindy([second], dt, diff)[0]

  isolated = np.allclose(joint[0], alone_first) and np.allclose(
    joint[1], alone_second
  )
  print("=== Claim 1: per-trajectory differentiation ===")
  print(f"  derivative of trial 1 identical whether fit alone or with trial 2: "
        f"{np.allclose(joint[0], alone_first)}")
  print(f"  derivative of trial 2 identical whether fit alone or with trial 1: "
        f"{np.allclose(joint[1], alone_second)}")
  print(f"  max |x_dot| across both trials: {max(np.abs(j).max() for j in joint):.4g}")
  print(f"  (a boundary leak would show ~{1000 / dt:.0f})")
  print(f"  VERDICT: {'PASS - trials isolated' if isolated else 'FAIL - leak detected'}\n")

  # --- Claim 2: naive concatenation is genuinely different -----------------
  # If the list form secretly concatenated, these two fits would agree.
  def coefs(data) -> np.ndarray:
    """Fit a degree-1 model and return its coefficient matrix."""
    model = ps.SINDy(
      optimizer=ps.STLSQ(threshold=0.01, alpha=0.0),
      feature_library=ps.PolynomialLibrary(degree=1),
    )
    model.fit(data, t=dt)
    return model.coefficients()

  as_list = coefs([first, second])
  as_stacked = coefs(np.concatenate([first, second], axis=0))
  differ = not np.allclose(as_list, as_stacked)
  print("=== Claim 2: list-of-trajectories != naive concatenation ===")
  print(f"  max coefficient difference: {np.abs(as_list - as_stacked).max():.6g}")
  print(f"  VERDICT: {'PASS - list form is not a concatenation' if differ else 'FAIL - identical'}\n")

  # --- Claim 3: the library is built from the SMOOTHED signal --------------
  # Uses NOISY data: a clean sine is nearly invariant under Savitzky-Golay, so
  # the substitution is real but invisible (~1e-8) on smooth input. Noise is
  # also the regime that matters for LFP.
  noisy = first + 0.1 * np.random.default_rng(0).standard_normal(first.shape)
  smoother = ps.SmoothedFiniteDifference(
    smoother_kws={"window_length": 9, "polyorder": 3}
  )
  library = ps.PolynomialLibrary(degree=1)
  x, t, x_dot, u = [noisy], dt, None, None
  if not _check_multiple_trajectories(x, x_dot, u):
    x, t, x_dot, u = _adapt_to_multiple_trajectories(x, t, x_dot, u)
  x, x_dot, u = _comprehend_and_validate_inputs(x, t, x_dot, u, library)
  returned_x, _ = library.calc_trajectory(smoother, x[0], dt)
  returned_x = np.asarray(returned_x)

  changed = not np.allclose(returned_x, noisy)
  matches_smoothed = np.allclose(returned_x, np.asarray(smoother.smoothed_x_))
  print("=== Claim 3: calc_trajectory returns the smoothed signal ===")
  print(f"  returned x differs from the raw input: {changed}")
  print(f"  returned x equals diff_method.smoothed_x_: {matches_smoothed}")
  print(f"  max |returned - raw|: {np.abs(returned_x - noisy).max():.6g}")
  print(f"  VERDICT: {'PASS - library sees SMOOTHED x' if changed and matches_smoothed else 'FAIL'}")


if __name__ == "__main__":
  main()
