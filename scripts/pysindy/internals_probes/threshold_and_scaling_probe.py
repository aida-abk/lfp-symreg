"""Why coefficients sit below the threshold, and whether z-scoring helps.

Answers three questions with the real LFP data:

A. **Sub-threshold coefficients with normalize_columns=False.** Stage 4 showed
   ``normalize_columns=True`` breaks the correspondence between the threshold and
   the reported coefficients. But sub-threshold survivors were still observed
   with ``normalize_columns=False`` (e.g. degree 7, alpha 0). The remaining
   candidate is the unbias step (``optimizers/base.py:250-274``), which refits by
   unregularized least squares on the selected support and can move a coefficient
   below the threshold it was selected at. Toggling ``unbias`` isolates it.

B. **How many STLSQ iterations run**, and whether iteration count relates to
   sub-threshold survivors. Verbose output is printed for each fit.

C. **Global z-score versus raw signal.** For a purely linear model the scale
   factor cancels, so z-scoring cannot change the fit. It should matter only
   where conditioning matters -- high polynomial degree. Comparing derivative
   R^2 across degrees under both settings tests that directly.

Run with:

    .venv/bin/python scripts/pysindy/threshold_and_scaling_probe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(_PROJECT_ROOT))

import pysindy as ps  # noqa: E402

from load_data.convert import MAT_FILE, TrialData  # noqa: E402
from load_data.preprocessing import (  # noqa: E402
  apply_global_zscore,
  channel_traces,
  compute_global_zscore_stats,
)
from models.sindy import delay_embed_trajectories  # noqa: E402

CHANNEL = 0
DOWNSAMPLE = 2
LOWPASS_HZ = 35.0
N_DELAYS = 4
DELAY_SAMPLES = 2
SMOOTH_WINDOW = 9
SPLIT_METADATA_DIR = (
  _PROJECT_ROOT / "outputs/pysindy/global_analysis/raw_grid_deg2357_t20000/parts"
)


def load_split() -> list[int]:
  """Return the training trial identifiers from the archived split."""
  candidates = sorted(SPLIT_METADATA_DIR.glob("*_metadata.json"))
  return json.loads(candidates[0].read_text())["split"]["train_trial_ids"]


def fit(trajectories, dt, degree, threshold, alpha, normalize_columns, unbias, verbose):
  """Fit one STLSQ model with explicit control of unbias and verbosity."""
  model = ps.SINDy(
    optimizer=ps.STLSQ(
      threshold=threshold, alpha=alpha, normalize_columns=normalize_columns,
      unbias=unbias, verbose=verbose, max_iter=20,
    ),
    feature_library=ps.PolynomialLibrary(degree=degree),
    differentiation_method=ps.SmoothedFiniteDifference(
      smoother_kws={"window_length": SMOOTH_WINDOW, "polyorder": 3}
    ),
  )
  model.fit(trajectories, t=dt)
  return model


def counts(model, threshold):
  """Return (nonzero, below-threshold) coefficient counts."""
  c = np.abs(np.asarray(model.coefficients()))
  return int(np.sum(c > 1e-9)), int(np.sum((c > 1e-9) & (c < threshold)))


def main() -> None:
  """Run all three investigations."""
  train_ids = load_split()
  print(f"Loading {MAT_FILE} ...", flush=True)
  data = TrialData.load(MAT_FILE)
  dt = DOWNSAMPLE / data.fs

  raw = channel_traces(data, channel=CHANNEL, trials=train_ids,
                       downsample=DOWNSAMPLE, lowpass_hz=LOWPASS_HZ, normalize="none")
  stats = compute_global_zscore_stats(raw, channel=CHANNEL)
  zscored = apply_global_zscore(raw, stats)
  print(f"global z-score: mean={stats.mean:.4g}  std={stats.std:.4g}\n")

  emb_raw = delay_embed_trajectories(raw, n_delays=N_DELAYS, delay=DELAY_SAMPLES)
  emb_z = delay_embed_trajectories(zscored, n_delays=N_DELAYS, delay=DELAY_SAMPLES)

  # ---- A + B: sub-threshold survivors with normalize_columns=False ----------
  threshold = 33.61  # calibrated earlier for z-scored signal, nc=False
  print("=" * 76)
  print("A/B. Sub-threshold coefficients with normalize_columns=FALSE (z-scored)")
  print(f"     threshold={threshold}  (units match coefficients when nc=False)")
  print("=" * 76)
  print(f"\n{'degree':>7} {'alpha':>6} {'unbias':>7} {'nonzero':>8} {'below_thr':>10}")
  results = {}
  for degree in (3, 7):
    for alpha in (0.05, 0.0):
      for unbias in (True, False):
        m = fit(emb_z, dt, degree, threshold, alpha,
                normalize_columns=False, unbias=unbias, verbose=False)
        nz, sub = counts(m, threshold)
        results[(degree, alpha, unbias)] = (nz, sub)
        print(f"{degree:>7} {alpha:>6} {str(unbias):>7} {nz:>8} {sub:>10}")

  print("\n  Reading: if below_thr > 0 only when unbias=True, the unregularized")
  print("  refit is the cause. If it is nonzero for both, something else is.\n")

  # ---- verbose logs for the interesting cases ------------------------------
  print("=" * 76)
  print("B. STLSQ verbose logs (iteration counts) for the degree-7 cases")
  print("=" * 76)
  for alpha in (0.05, 0.0):
    for unbias in (True, False):
      print(f"\n--- degree=7 alpha={alpha} unbias={unbias} nc=False thr={threshold} ---")
      m = fit(emb_z, dt, 7, threshold, alpha,
              normalize_columns=False, unbias=unbias, verbose=True)
      nz, sub = counts(m, threshold)
      hist = getattr(m.optimizer, "history_", None)
      print(f"  -> history_ length={len(hist) if hist is not None else 'n/a'} "
            f"(= printed rows + 1 seed)   nonzero={nz}  below_thr={sub}")

  # ---- C: z-score versus raw ----------------------------------------------
  print("\n" + "=" * 76)
  print("C. Global z-score versus raw signal: derivative R^2 by degree")
  print("   (threshold=0 -> ceiling R^2, so this isolates conditioning)")
  print("=" * 76)
  print(f"\n{'degree':>7} {'R2 raw':>12} {'R2 zscored':>12} {'difference':>12} {'terms raw':>10} {'terms z':>9}")
  for degree in (2, 3, 5, 7, 9):
    out = {}
    for label, emb in (("raw", emb_raw), ("z", emb_z)):
      try:
        m = fit(emb, dt, degree, 0.0, 0.05,
                normalize_columns=True, unbias=True, verbose=False)
        out[label] = (float(m.score(emb, t=dt)),
                      int(np.count_nonzero(np.asarray(m.coefficients()))))
      except Exception as exc:
        out[label] = (float("nan"), -1)
        print(f"    (degree {degree} {label} failed: {type(exc).__name__})")
    dr, tr = out["raw"]
    dz, tz = out["z"]
    print(f"{degree:>7} {dr:>+12.6f} {dz:>+12.6f} {dz - dr:>+12.3e} {tr:>10} {tz:>9}")

  print("\n  A linear rescaling cannot change a scale-invariant R^2, so any")
  print("  difference here is numerical conditioning, not modelling content.")


if __name__ == "__main__":
  main()
