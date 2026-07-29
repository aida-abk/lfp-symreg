"""Probe: how PySINDy computes derivatives, and what happens at trial edges.

Source claims under test (pysindy 2.1.0):
  1. ``FiniteDifference._differentiate`` (``differentiation/finite_difference.py:247``)
     initializes ``x_dot`` to all-NaN, then fills the interior with centered
     differences and -- only when ``drop_endpoints`` is False (line 281) -- fills
     the boundaries with one-sided forward/backward differences. The class
     docstring (line 38) states the opposite: "If False, endpoints will be set to
     np.nan". The docstring appears to be inverted; this probe settles it.
  2. ``SmoothedFiniteDifference._differentiate``
     (``differentiation/smoothed_finite_difference.py:63-71``) smooths the whole
     trajectory along axis 0, differentiates the SMOOTHED array, and stores
     ``smoothed_x_``. With ``save_smooth=False`` it stores the RAW x instead,
     which decouples the library input from the derivative source.
  3. ``SmoothedFiniteDifference.__init__`` uses a mutable default argument
     (``smoother_kws={}``) and mutates it, adding ``axis=0`` and defaults
     ``window_length=11, polyorder=3``.

Run with:

    .venv/bin/python scripts/pysindy/internals_probes/probe02_differentiation.py
"""
from __future__ import annotations

import numpy as np
import pysindy as ps
from scipy.signal import savgol_filter


def edge_affected_sample_count(n_samples: int, window_length: int) -> int:
  """Return how many samples per trajectory sit inside a Savgol edge window.

  ``savgol_filter`` defaults to ``mode='interp'``, which fits a polynomial to
  the first and last ``window_length`` samples rather than padding. The samples
  whose value comes from that fit rather than a centered window are the first
  and last ``window_length // 2``.

  Args:
    n_samples: Samples in one trajectory.
    window_length: Savitzky-Golay window in samples.

  Returns:
    Total edge-affected samples across both ends of one trajectory.
  """
  half = window_length // 2
  return min(2 * half, n_samples)


def main() -> None:
  """Run each differentiation check and print findings."""
  dt = 0.004  # matches this project: downsample 2 at 500 Hz
  t = np.arange(0, 2, dt)
  clean = np.column_stack([np.sin(2 * np.pi * 3 * t)])
  noisy = clean + 0.1 * np.random.default_rng(0).standard_normal(clean.shape)

  # --- Claim 1: drop_endpoints semantics ----------------------------------
  default_fd = ps.FiniteDifference()
  dropped_fd = ps.FiniteDifference(drop_endpoints=True)
  dot_default = np.asarray(default_fd(clean, t=dt))
  dot_dropped = np.asarray(dropped_fd(clean, t=dt))

  n_nan_default = int(np.isnan(dot_default).sum())
  n_nan_dropped = int(np.isnan(dot_dropped).sum())
  print("=== Claim 1: drop_endpoints semantics ===")
  print(f"  order (default): {default_fd.order}   n_stencil: {default_fd.n_stencil}")
  print(f"  drop_endpoints=False (DEFAULT) -> NaN count: {n_nan_default}")
  print(f"  drop_endpoints=True            -> NaN count: {n_nan_dropped}")
  print(f"  NaN positions when dropped: {np.flatnonzero(np.isnan(dot_dropped[:, 0]))}")
  docstring_says_nan_when_false = n_nan_default > 0
  print(f"  docstring claim ('If False -> NaN') holds: {docstring_says_nan_when_false}")
  print(f"  VERDICT: {'docstring is CORRECT' if docstring_says_nan_when_false else 'docstring is INVERTED — default computes endpoints, no NaN'}\n")

  # Which samples use one-sided (boundary) rather than centered differences?
  half = (default_fd.n_stencil - 1) // 2
  centered = np.full_like(dot_default, np.nan)
  centered[half:-half, 0] = (clean[2 * half:, 0] - clean[:-2 * half, 0]) / (2 * half * dt)
  differs = ~np.isclose(dot_default[:, 0], centered[:, 0], equal_nan=True)
  print("=== Boundary extent ===")
  print(f"  samples differing from a pure centered difference: {np.flatnonzero(differs)}")
  print(f"  -> {int(differs.sum())} one-sided samples per trajectory "
        f"({half} at each end)\n")

  # --- Claim 2: save_smooth controls what the library sees ----------------
  kws = {"window_length": 9, "polyorder": 3}
  saved = ps.SmoothedFiniteDifference(smoother_kws=dict(kws), save_smooth=True)
  unsaved = ps.SmoothedFiniteDifference(smoother_kws=dict(kws), save_smooth=False)
  dot_saved = np.asarray(saved(noisy, t=dt))
  dot_unsaved = np.asarray(unsaved(noisy, t=dt))

  same_derivative = np.allclose(dot_saved, dot_unsaved)
  saved_is_smooth = not np.allclose(np.asarray(saved.smoothed_x_), noisy)
  unsaved_is_raw = np.allclose(np.asarray(unsaved.smoothed_x_), noisy)
  print("=== Claim 2: save_smooth ===")
  print(f"  derivatives identical either way: {same_derivative}")
  print(f"  save_smooth=True  -> smoothed_x_ is smoothed: {saved_is_smooth}")
  print(f"  save_smooth=False -> smoothed_x_ is the RAW input: {unsaved_is_raw}")
  print(f"  VERDICT: {'PASS' if same_derivative and saved_is_smooth and unsaved_is_raw else 'FAIL'}")
  print("  NOTE: with save_smooth=False the library is built from RAW x while")
  print("        x_dot comes from SMOOTHED x -- the two sides disagree.\n")

  # --- Claim 3: mutable default argument ----------------------------------
  first = ps.SmoothedFiniteDifference()
  passed = {"window_length": 7, "polyorder": 2}
  ps.SmoothedFiniteDifference(smoother_kws=passed)
  print("=== Claim 3: mutable default arg + kwargs mutation ===")
  print(f"  defaults injected when none given: {first.smoother_kws}")
  print(f"  caller's own dict after construction: {passed}")
  print(f"  caller's dict was mutated (axis added): {'axis' in passed}\n")

  # --- Practical scale for this project ------------------------------------
  n_trials, n_samples, window = 28, len(t), 9
  per_trial_edges = edge_affected_sample_count(n_samples, window)
  per_trial_onesided = 2 * half
  total = n_trials * n_samples
  print("=== Scale of edge effects for this pipeline ===")
  print(f"  {n_trials} trials x {n_samples} samples = {total} samples")
  print(f"  Savgol edge-window samples: {n_trials * per_trial_edges} "
        f"({100 * n_trials * per_trial_edges / total:.2f}%)")
  print(f"  one-sided derivative samples: {n_trials * per_trial_onesided} "
        f"({100 * n_trials * per_trial_onesided / total:.2f}%)")
  print(f"  savgol_filter mode: {'interp' + ' (scipy default — polynomial fit at edges, no padding)'}")


if __name__ == "__main__":
  main()
