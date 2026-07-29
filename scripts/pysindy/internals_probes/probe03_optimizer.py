"""Probe: what STLSQ actually optimizes, and what alpha does to the result.

Source claims under test (pysindy 2.1.0):
  1. ``BaseOptimizer.fit`` (``optimizers/base.py:250-251``) runs an *unbiasing*
     step by default (``STLSQ.__init__`` sets ``unbias=True``,
     ``optimizers/stlsq.py:113``). ``_unbias`` (``optimizers/base.py:265-274``)
     refits ``LinearRegression(fit_intercept=False)`` -- plain, unregularized
     least squares -- restricted to the support STLSQ selected, and overwrites
     ``coef_``. The reported coefficients are therefore NOT the thresholded
     ridge estimates.
  2. Consequently ``alpha`` influences only WHICH terms are selected. Two alphas
     that select the same support must yield bit-identical final coefficients,
     because both end in the same unregularized refit.
  3. Sub-threshold survivors -- nonzero coefficients whose magnitude is below the
     threshold they were fit at -- are produced by this unbiasing step, not by
     ridge shrinkage. Disabling ``unbias`` should change how many appear.
  4. Thresholding is per-equation and uses ``>=``
     (``optimizers/stlsq.py``, ``_sparse_coefficients``: ``np.abs(c) >= threshold``),
     applied inside a loop over targets in ``_reduce``.

Run with:

    .venv/bin/python scripts/pysindy/internals_probes/probe03_optimizer.py
"""
from __future__ import annotations

import numpy as np
import pysindy as ps


def build_data(n_samples: int = 1500, dt: float = 0.004) -> tuple[np.ndarray, float]:
    """Return a delay-embedded noisy oscillation with collinear coordinates.

    Collinearity is the regime where ridge matters, so it is what makes the
    alpha comparison meaningful.

    Args:
      n_samples: Samples in the trajectory.
      dt: Sample interval in seconds.

    Returns:
      A ``(n_samples, 3)`` state array and the sample interval.
    """
    t = np.arange(n_samples) * dt
    rng = np.random.default_rng(0)
    x = np.sin(2 * np.pi * 6 * t) + 0.4 * rng.standard_normal(t.size)
    return np.column_stack([x[:-10], x[5:-5], x[10:]]), dt


def fit(state, dt, alpha, threshold, unbias, normalize_columns=True):
    """Fit one STLSQ model and return it."""
    model = ps.SINDy(
        optimizer=ps.STLSQ(
            threshold=threshold, alpha=alpha,
            normalize_columns=normalize_columns, unbias=unbias, max_iter=20,
        ),
        feature_library=ps.PolynomialLibrary(degree=2),
    )
    model.fit(state, t=dt)
    return model


def support_of(model) -> np.ndarray:
    """Return the boolean support mask of a fitted model."""
    return np.abs(np.asarray(model.coefficients())) > 1e-14


def subthreshold_count(model, threshold: float) -> int:
    """Count nonzero coefficients whose magnitude is below ``threshold``."""
    c = np.abs(np.asarray(model.coefficients()))
    return int(np.sum((c > 1e-9) & (c < threshold)))


def main() -> None:
    """Run each optimizer check and print findings."""
    state, dt = build_data()
    threshold = 0.5

    print("=== Claim 1: unbias=True is the default, and it changes coefficients ===")
    default_opt = ps.STLSQ()
    print(f"  STLSQ() default unbias = {default_opt.unbias}")
    biased = fit(state, dt, alpha=0.05, threshold=threshold, unbias=False)
    unbiased = fit(state, dt, alpha=0.05, threshold=threshold, unbias=True)
    same_support = np.array_equal(support_of(biased), support_of(unbiased))
    max_delta = float(
        np.max(np.abs(np.asarray(biased.coefficients()) - np.asarray(unbiased.coefficients())))
    )
    print(f"  same support with/without unbias: {same_support}")
    print(f"  max |coefficient difference|:     {max_delta:.6g}")
    print(f"  VERDICT: {'PASS - unbias rewrites coefficient VALUES, keeping the support' if same_support and max_delta > 0 else 'unexpected'}\n")

    print("=== Claim 2: alpha only selects the support; it does not set the values ===")
    results = {}
    for alpha in (0.0, 0.05, 0.5, 5.0):
        m = fit(state, dt, alpha=alpha, threshold=threshold, unbias=True)
        results[alpha] = (support_of(m), np.asarray(m.coefficients()))
        print(f"  alpha={alpha:<5} terms={int(results[alpha][0].sum()):>3}")
    alphas = list(results)
    for i in range(len(alphas)):
        for j in range(i + 1, len(alphas)):
            a, b = alphas[i], alphas[j]
            if np.array_equal(results[a][0], results[b][0]):
                delta = float(np.max(np.abs(results[a][1] - results[b][1])))
                print(f"  alpha={a} and alpha={b} selected the SAME support -> "
                      f"max coefficient difference = {delta:.3g}")
                print(f"    {'PASS - identical values, so alpha did not touch them' if delta < 1e-9 else 'FAIL - values differ'}")
    print()

    print("=== Claim 3: what actually creates sub-threshold survivors ===")
    print("  Two independent mechanisms can leave a nonzero coefficient below the")
    print("  threshold. Separated here by toggling normalize_columns:")
    for nc in (True, False):
        for label, unbias in (("unbias=True ", True), ("unbias=False", False)):
            m = fit(state, dt, alpha=0.05, threshold=threshold, unbias=unbias,
                    normalize_columns=nc)
            nz = int(np.count_nonzero(np.asarray(m.coefficients())))
            print(f"    normalize_columns={str(nc):<5} {label}  nonzero={nz:>3}  "
                  f"below-threshold={subthreshold_count(m, threshold):>3}")
    print("  With normalize_columns=True the threshold is applied in normalized space")
    print("  while coefficients() reports original units, so survivors can sit below it")
    print("  regardless of unbias. With normalize_columns=False the units agree, and")
    print("  only the unregularized unbias refit can push a coefficient under.\n")

    print("=== Claim 4: thresholding is per-equation and uses >= ===")
    print("  (normalize_columns=False so threshold and coefficients share units)")
    m = fit(state, dt, alpha=0.05, threshold=threshold, unbias=False,
            normalize_columns=False)
    coefs = np.asarray(m.coefficients())
    per_eq = [int(np.count_nonzero(row)) for row in coefs]
    print(f"  surviving terms per equation: {per_eq}")
    below = np.abs(coefs[coefs != 0])
    print(f"  smallest surviving |coefficient| = {below.min():.6g} vs threshold {threshold}")
    print(f"  all survivors >= threshold: {bool((below >= threshold).all())}")


if __name__ == "__main__":
    main()
