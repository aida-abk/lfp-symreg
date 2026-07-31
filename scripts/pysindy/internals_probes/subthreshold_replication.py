"""Replicate the two sub-threshold mechanisms across conditions and trial subsets.

Earlier evidence was thin:

* Mechanism 1 (``normalize_columns=True`` -> threshold applied in normalized
  space while coefficients are reported in original units) was tested only on
  synthetic data.
* Mechanism 2 (``unbias=True`` -> unregularized refit relocates coefficients
  after selection) rested on a SINGLE configuration cell (degree 7, alpha 0, one
  threshold, one trial split). Every other cell showed zero survivors in both
  states and was therefore uninformative.

This script varies degree, alpha, threshold, ``normalize_columns``, ``unbias``,
and -- critically -- the subset of training trials, so each mechanism is either
reproduced or refuted across independent data samples.

Run with:

    .venv/bin/python scripts/pysindy/subthreshold_replication.py
"""
from __future__ import annotations

import csv
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

DEGREES = [3, 5, 7]
ALPHAS = [0.0, 0.05]
N_SUBSETS = 5
SUBSET_SIZE = 20

SPLIT_METADATA_DIR = (
  _PROJECT_ROOT / "outputs/pysindy/global_analysis/raw_grid_deg2357_t20000/parts"
)
OUTPUT_DIR = _PROJECT_ROOT / "outputs/pysindy/shape_analysis"


def load_train_ids() -> list[int]:
  """Return the archived training trial identifiers."""
  candidates = sorted(SPLIT_METADATA_DIR.glob("*_metadata.json"))
  return json.loads(candidates[0].read_text())["split"]["train_trial_ids"]


def fit(trajectories, dt, degree, threshold, alpha, normalize_columns, unbias):
  """Fit one STLSQ model with explicit unbias/normalize control."""
  model = ps.SINDy(
    optimizer=ps.STLSQ(
      threshold=threshold, alpha=alpha, normalize_columns=normalize_columns,
      unbias=unbias, verbose=False, max_iter=20,
    ),
    feature_library=ps.PolynomialLibrary(degree=degree),
    differentiation_method=ps.SmoothedFiniteDifference(
      smoother_kws={"window_length": SMOOTH_WINDOW, "polyorder": 3}
    ),
  )
  model.fit(trajectories, t=dt)
  return model


def counts(model, threshold: float) -> tuple[int, int]:
  """Return (nonzero, below-threshold) coefficient counts."""
  c = np.abs(np.asarray(model.coefficients()))
  return int(np.sum(c > 1e-9)), int(np.sum((c > 1e-9) & (c < threshold)))


def calibrate(trajectories, dt, degree, normalize_columns, target=20) -> float:
  """Return a threshold giving roughly ``target`` terms, by sweeping."""
  base = fit(trajectories, dt, degree, 0.0, 0.05, normalize_columns, unbias=True)
  scale = float(np.max(np.abs(np.asarray(base.coefficients()))))
  if scale <= 0:
    return 0.0
  best, best_gap = 0.0, 10**9
  for thr in np.geomspace(scale * 1e-4, scale * 1e2, 22):
    n = int(np.count_nonzero(np.asarray(
      fit(trajectories, dt, degree, float(thr), 0.05, normalize_columns, True).coefficients()
    )))
    gap = abs(n - target)
    if gap < best_gap or (gap == best_gap and thr > best):
      best, best_gap = float(thr), gap
  return best


def main() -> None:
  """Run the replication across subsets and conditions."""
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  train_ids = load_train_ids()
  print(f"Loading {MAT_FILE} ...", flush=True)
  data = TrialData.load(MAT_FILE)
  dt = DOWNSAMPLE / data.fs

  raw_all = channel_traces(data, channel=CHANNEL, trials=train_ids,
                           downsample=DOWNSAMPLE, lowpass_hz=LOWPASS_HZ,
                           normalize="none")
  rng = np.random.default_rng(0)
  subsets = [
    sorted(rng.choice(len(raw_all), size=SUBSET_SIZE, replace=False).tolist())
    for _ in range(N_SUBSETS)
  ]

  rows: list[dict] = []
  for s_i, subset in enumerate(subsets):
    traces = [raw_all[i] for i in subset]
    stats = compute_global_zscore_stats(traces, channel=CHANNEL)
    z = apply_global_zscore(traces, stats)
    emb = delay_embed_trajectories(z, n_delays=N_DELAYS, delay=DELAY_SAMPLES)

    for nc in (False, True):
      for degree in DEGREES:
        thr = calibrate(emb, dt, degree, nc)
        for alpha in ALPHAS:
          for unbias in (True, False):
            m = fit(emb, dt, degree, thr, alpha, nc, unbias)
            nz, sub = counts(m, thr)
            rows.append(dict(subset=s_i, normalize_columns=nc, degree=degree,
                             alpha=alpha, unbias=unbias, threshold=thr,
                             nonzero=nz, below_threshold=sub))
        print(f"  subset {s_i} nc={str(nc):<5} deg={degree} thr={thr:.4g} done", flush=True)

  path = OUTPUT_DIR / "subthreshold_replication.csv"
  with open(path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
  print(f"\nwrote {path}\n")
  _summarize(rows)


def _summarize(rows: list[dict]) -> None:
  """Print replication tables for both mechanisms."""
  print("=" * 78)
  print("MECHANISM 1: does normalize_columns=True produce sub-threshold survivors?")
  print("=" * 78)
  print(f"\n{'nc':>6} {'degree':>7} {'subsets with >0':>17} {'median below_thr':>18}")
  for nc in (True, False):
    for degree in DEGREES:
      sel = [r for r in rows if r["normalize_columns"] == nc and r["degree"] == degree]
      pos = sum(1 for r in sel if r["below_threshold"] > 0)
      med = np.median([r["below_threshold"] for r in sel])
      print(f"{str(nc):>6} {degree:>7} {pos:>10}/{len(sel):<6} {med:>18.0f}")

  print("\n" + "=" * 78)
  print("MECHANISM 2: with nc=False, does unbias=True cause them?")
  print("=" * 78)
  print(f"\n{'degree':>7} {'alpha':>6} {'unbias':>7} {'subsets >0':>12} {'median below_thr':>18}")
  for degree in DEGREES:
    for alpha in ALPHAS:
      for unbias in (True, False):
        sel = [r for r in rows if not r["normalize_columns"] and r["degree"] == degree
               and r["alpha"] == alpha and r["unbias"] == unbias]
        if not sel:
          continue
        pos = sum(1 for r in sel if r["below_threshold"] > 0)
        med = np.median([r["below_threshold"] for r in sel])
        print(f"{degree:>7} {alpha:>6} {str(unbias):>7} {pos:>7}/{len(sel):<4} {med:>18.0f}")


if __name__ == "__main__":
  main()
