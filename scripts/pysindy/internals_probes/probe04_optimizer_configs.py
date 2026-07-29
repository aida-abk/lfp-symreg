"""Probe: how each optimizer's configuration determines the fitted result.

Source claims under test (pysindy 2.1.0):
  1. ``threshold`` means different things in different optimizers. STLSQ cuts at
     ``|c| >= threshold`` directly (``optimizers/stlsq.py``,
     ``_sparse_coefficients``). SR3 instead applies a proximal operator with
     weight ``reg_weight_lam * relax_coeff_nu`` (``optimizers/sr3.py``,
     ``_update_sparse_coef``), and for L0 that operator cuts at
     ``sqrt(2 * reg_weight_lam * relax_coeff_nu)`` (``utils/base.py``,
     ``_prox_l0``). So SR3's effective cut is a SQUARE ROOT of its lam, not lam.
  2. Therefore, to match an STLSQ threshold ``T`` with SR3/L0 at ``nu=1`` one must
     set ``reg_weight_lam = T**2 / 2``, not ``reg_weight_lam = T``.
  3. ``regularizer="l2"`` produces NO sparsity: ``_prox_l2`` returns
     ``x / (1 + 2*weight)``, which rescales but never zeroes.
  4. ``unbias`` defaults differ by optimizer -- ``True`` for STLSQ/SSR/FROLS,
     ``False`` for every SR3 variant -- so swapping optimizers silently changes
     whether regularization bias remains in the reported coefficients.

Run with:

    .venv/bin/python scripts/pysindy/internals_probes/probe04_optimizer_configs.py
"""
from __future__ import annotations

import numpy as np
import pysindy as ps


def build_data(n_samples: int = 1200, dt: float = 0.004) -> tuple[np.ndarray, float]:
    """Return a delay-embedded noisy oscillation."""
    t = np.arange(n_samples) * dt
    rng = np.random.default_rng(0)
    x = np.sin(2 * np.pi * 6 * t) + 0.3 * rng.standard_normal(t.size)
    return np.column_stack([x[:-10], x[5:-5], x[10:]]), dt


def n_terms(model) -> int:
    """Return the number of nonzero coefficients in a fitted model."""
    return int(np.count_nonzero(np.asarray(model.coefficients())))


def fit_with(optimizer, state, dt, degree=2):
    """Fit a SINDy model with a given optimizer instance."""
    model = ps.SINDy(
        optimizer=optimizer, feature_library=ps.PolynomialLibrary(degree=degree)
    )
    model.fit(state, t=dt)
    return model


def main() -> None:
    """Run each optimizer-configuration check."""
    state, dt = build_data()

    print("=== Claim 1 & 2: 'threshold' is not a shared unit across optimizers ===")
    print("  STLSQ cuts at |c| >= threshold; SR3/L0 cuts at sqrt(2*lam*nu).\n")
    print(f"  {'T':>8} {'STLSQ terms':>12} | {'SR3 lam=T':>10} {'terms':>6} "
          f"| {'SR3 lam=T^2/2':>14} {'terms':>6}")
    for T in (0.05, 0.2, 0.5, 1.0):
        stlsq = fit_with(ps.STLSQ(threshold=T, alpha=0.05), state, dt)
        sr3_naive = fit_with(
            ps.SR3(reg_weight_lam=T, regularizer="L0", relax_coeff_nu=1.0), state, dt
        )
        sr3_matched = fit_with(
            ps.SR3(reg_weight_lam=T**2 / 2, regularizer="L0", relax_coeff_nu=1.0),
            state, dt,
        )
        print(f"  {T:>8} {n_terms(stlsq):>12} | {T:>10} {n_terms(sr3_naive):>6} "
              f"| {T**2/2:>14.5g} {n_terms(sr3_matched):>6}")
    print("\n  The matched column should track STLSQ; the naive column should not.\n")

    print("=== Effective cut implied by the source formulas ===")
    print(f"  {'lam':>10} {'nu':>6} {'sqrt(2*lam*nu)':>16}")
    for lam, nu in ((0.05, 1.0), (0.05, 0.1), (0.5, 1.0), (20000.0, 1.0)):
        print(f"  {lam:>10g} {nu:>6g} {np.sqrt(2*lam*nu):>16.6g}")
    print()

    print("=== Claim 3: regularizer='l2' yields no sparsity ===")
    for reg in ("L0", "l1", "l2"):
        m = fit_with(
            ps.SR3(reg_weight_lam=0.5, regularizer=reg, relax_coeff_nu=1.0), state, dt
        )
        total = np.asarray(m.coefficients()).size
        print(f"  regularizer={reg:<3} nonzero={n_terms(m):>3} / {total}")
    print()

    print("=== Claim 4: unbias defaults differ by optimizer ===")
    for cls, kwargs in (
        (ps.STLSQ, {}),
        (ps.SR3, {}),
        (ps.SSR, {}),
        (ps.FROLS, {}),
    ):
        inst = cls(**kwargs)
        print(f"  {cls.__name__:<8} unbias default = {inst.unbias}")
    print()

    print("=== Search strategy comparison at matched term counts ===")
    print("  Different optimizers search the same library in fundamentally")
    print("  different ways; term count alone does not make them comparable.")
    candidates = {
        "STLSQ(thr=0.5)": ps.STLSQ(threshold=0.5, alpha=0.05),
        "SR3/L0(lam=0.125)": ps.SR3(reg_weight_lam=0.125, regularizer="L0"),
        "SSR(max_iter=8)": ps.SSR(max_iter=8),
        "FROLS(max_iter=4)": ps.FROLS(max_iter=4),
    }
    for label, opt in candidates.items():
        try:
            m = fit_with(opt, state, dt)
            print(f"  {label:<20} terms={n_terms(m):>3}  "
                  f"R2={m.score(state, t=dt):+.4f}")
        except Exception as exc:
            print(f"  {label:<20} FAILED: {type(exc).__name__}: {exc}")

    print("\n=== Is CVXPY available (required by StableLinearSR3 when lam>0)? ===")
    try:
        import cvxpy  # noqa: F401
        print("  cvxpy is installed -> StableLinearSR3 usable with nonzero lam")
    except ImportError:
        print("  cvxpy NOT installed -> StableLinearSR3 needs it when lam != 0")


if __name__ == "__main__":
    main()
