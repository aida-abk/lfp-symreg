"""Four-scenario normalization validation on raw-grid config 4.

Config 4 (from ``raw_grid_deg2357_t20000``) is the simplest degree-2 model with
both linear and quadratic surviving terms that also integrates stably:

    lowpass = 35 Hz, degree = 2, n_delays = 2, delay = 2 samples,
    smooth_window = 0, alpha = 0.05, threshold = 20000, normalize_columns = True.

This script refits that exact configuration under the four normalization schemes

    1. no z-score, normalize_columns = True   (reproduces the saved config-4 fit)
    2. no z-score, normalize_columns = False
    3. global z-score, normalize_columns = True
    4. global z-score, normalize_columns = False

and prints two comparison tables:

  * A "theory-isolation" pass (threshold = 0, alpha = 0 -> plain least squares)
    that keeps every candidate term so the coefficient transform can be verified
    directly: linear coefficients must be identical across all four schemes, and
    an order-k coefficient must scale by sigma^(k-1) under z-scoring.

  * A "selection-behavior" pass at the real config-4 threshold that shows how the
    surviving support (not just the values) differs across schemes.

Run:
    .venv/bin/python scripts/pysindy/normalization_validation/validate_config4_normalization.py
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

# Project root is three levels up: scripts/pysindy/normalization_validation/<file>
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
import sys

if str(_PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(_PROJECT_ROOT))

from load_data.convert import MAT_FILE, TrialData
from load_data.preprocessing import (
  apply_global_zscore,
  channel_traces,
  compute_global_zscore_stats,
)
from models.sindy import SINDyConfig, delay_embed_trajectories, fit_sindy_model

# ---------------------------------------------------------------------------
# Config 4 settings and the exact training split, copied from
# outputs/pysindy/raw_grid_deg2357_t20000/parts/part_lp35_degree2_metadata.json
# (random whole-trial split, test_fraction = 0.25, seed = 0). Hardcoding the IDs
# makes the check reproducible without reloading the slow behavioral table.
# ---------------------------------------------------------------------------
TRAIN_TRIAL_IDS = [
  1398, 1, 0, 776, 934, 775, 311, 1086, 310, 156, 930, 621, 777, 158,
  467, 931, 313, 932, 314, 157, 155, 469, 779, 465, 933, 1087, 466, 1085,
]
CHANNEL = 0
DOWNSAMPLE = 2
LOWPASS_HZ = 35.0
DEGREE = 2
N_DELAYS = 2
DELAY = 2
SMOOTH_WINDOW = 0
CONFIG4_THRESHOLD = 20000.0
CONFIG4_ALPHA = 0.05

SCENARIOS = [
  ("1. no-z,  NC=True", False, True),
  ("2. no-z,  NC=False", False, False),
  ("3. zscore, NC=True", True, True),
  ("4. zscore, NC=False", True, False),
]


def feature_order(name: str) -> int:
  """Return the total polynomial order of a PySINDy feature name.

  Args:
    name: Feature label such as ``"1"``, ``"x0"``, ``"x0^2"``, or ``"x0 x1"``.

  Returns:
    The summed polynomial order (0 for the constant, 1 for linear, and so on).
  """
  if name.strip() in ("1", "1.0"):
    return 0
  order = 0
  for token in name.split():
    match = re.match(r"x\d+(?:\^(\d+))?$", token)
    order += int(match.group(1)) if match and match.group(1) else 1
  return order


def fit_scenario(
  train_raw_none: list[np.ndarray],
  zscore: bool,
  normalize_columns: bool,
  threshold: float,
  alpha: float,
) -> tuple[list[str], np.ndarray, float]:
  """Fit config 4 under one normalization scheme.

  Args:
    train_raw_none: Lowpass-filtered, detrended training traces in microvolts
      (normalize="none"); the same list is reused for every scenario.
    zscore: Whether to apply the global z-score to the state before embedding.
    normalize_columns: STLSQ ``normalize_columns`` flag.
    threshold: STLSQ coefficient threshold.
    alpha: STLSQ ridge strength.

  Returns:
    Feature names, the coefficient matrix (n_targets, n_features), and the
    global standard deviation used (1.0 when no z-score is applied).
  """
  sigma = 1.0
  traces = train_raw_none
  if zscore:
    stats = compute_global_zscore_stats(train_raw_none, channel=CHANNEL)
    sigma = stats.std
    traces = apply_global_zscore(train_raw_none, stats)

  embedded = delay_embed_trajectories(traces, n_delays=N_DELAYS, delay=DELAY)
  dt = DOWNSAMPLE / FS
  model = fit_sindy_model(
    embedded,
    dt=dt,
    config=SINDyConfig(
      degree=DEGREE,
      threshold=threshold,
      alpha=alpha,
      normalize_columns=normalize_columns,
      smooth_window=SMOOTH_WINDOW,
      smoothing_polyorder=3,
      optimizer="stlsq",
    ),
  )
  return model.get_feature_names(), np.asarray(model.coefficients()), sigma


def print_table(
  title: str,
  names: list[str],
  coefs_by_scenario: dict[str, np.ndarray],
  target_index: int,
) -> None:
  """Print an aligned coefficient table for one target equation.

  Args:
    title: Table heading.
    names: Feature names shared across scenarios.
    coefs_by_scenario: Scenario label -> coefficient matrix.
    target_index: Which state-equation row to display.
  """
  labels = list(coefs_by_scenario)
  print(f"\n{title}  —  equation (x{target_index})'")
  header = f"{'term':10s} {'ord':>3s} " + " ".join(f"{lab:>20s}" for lab in labels)
  print(header)
  print("-" * len(header))
  for j, name in enumerate(names):
    row_vals = [coefs_by_scenario[lab][target_index, j] for lab in labels]
    if all(abs(v) < 1e-9 for v in row_vals):
      continue
    cells = " ".join(f"{v:>20.6g}" for v in row_vals)
    print(f"{name:10s} {feature_order(name):>3d} {cells}")


def verify_transform(
  names: list[str],
  coefs: dict[str, np.ndarray],
  sigma: float,
  target_index: int,
) -> None:
  """Check linear invariance and sigma^(k-1) scaling against scenario 1.

  Args:
    names: Feature names.
    coefs: Scenario label -> coefficient matrix (theory-isolation pass).
    sigma: Global standard deviation from the z-scored scenarios.
    target_index: State-equation row to verify.
  """
  base = coefs["1. no-z,  NC=True"][target_index]
  zsc = coefs["4. zscore, NC=False"][target_index]
  print(f"\nTransform check (x{target_index})' — predicted zscore coef = raw * sigma^(k-1),"
        f" sigma={sigma:.4f}")
  print(f"{'term':10s} {'ord':>3s} {'raw (sc1)':>14s} {'zscore (sc4)':>14s}"
        f" {'predicted':>14s} {'match':>7s}")
  for j, name in enumerate(names):
    if abs(base[j]) < 1e-9 and abs(zsc[j]) < 1e-9:
      continue
    k = feature_order(name)
    predicted = base[j] * sigma ** (k - 1)
    ok = "OK" if abs(predicted - zsc[j]) <= 1e-6 * (1 + abs(zsc[j])) else "DIFF"
    print(f"{name:10s} {k:>3d} {base[j]:>14.6g} {zsc[j]:>14.6g}"
          f" {predicted:>14.6g} {ok:>7s}")


def main() -> None:
  """Load config-4 training data and run both validation passes."""
  global FS
  print(f"Loading {MAT_FILE} ...", flush=True)
  data = TrialData.load(MAT_FILE)
  FS = data.fs
  print(f"fs={FS} Hz, dt={DOWNSAMPLE / FS:.4f} s, train trials={len(TRAIN_TRIAL_IDS)}")

  # Shared preprocessed training traces (detrended, 35 Hz lowpass, no z-score).
  train_raw_none = channel_traces(
    data,
    channel=CHANNEL,
    trials=TRAIN_TRIAL_IDS,
    downsample=DOWNSAMPLE,
    lowpass_hz=LOWPASS_HZ,
    normalize="none",
  )

  # --- Pass A: theory isolation (keep all terms) ---------------------------
  theory_coefs: dict[str, np.ndarray] = {}
  names_ref: list[str] | None = None
  sigma_used = 1.0
  for label, zscore, nc in SCENARIOS:
    names, coefs, sigma = fit_scenario(
      train_raw_none, zscore=zscore, normalize_columns=nc,
      threshold=0.0, alpha=0.0,
    )
    theory_coefs[label] = coefs
    names_ref = names
    if zscore:
      sigma_used = sigma

  print("\n" + "=" * 78)
  print("PASS A — theory isolation (threshold=0, alpha=0; every term retained)")
  print("=" * 78)
  for target in range(theory_coefs[SCENARIOS[0][0]].shape[0]):
    print_table("PASS A", names_ref, theory_coefs, target)
  for target in range(theory_coefs[SCENARIOS[0][0]].shape[0]):
    verify_transform(names_ref, theory_coefs, sigma_used, target)

  # --- Pass B: real config-4 selection behavior ----------------------------
  select_coefs: dict[str, np.ndarray] = {}
  for label, zscore, nc in SCENARIOS:
    names, coefs, _ = fit_scenario(
      train_raw_none, zscore=zscore, normalize_columns=nc,
      threshold=CONFIG4_THRESHOLD, alpha=CONFIG4_ALPHA,
    )
    select_coefs[label] = coefs
    names_ref = names

  print("\n" + "=" * 78)
  print(f"PASS B — config-4 selection (threshold={CONFIG4_THRESHOLD}, alpha={CONFIG4_ALPHA})")
  print("=" * 78)
  for target in range(select_coefs[SCENARIOS[0][0]].shape[0]):
    print_table("PASS B", names_ref, select_coefs, target)

  print("\nSupport size (nonzero terms) per scenario, Pass B:")
  for label, _, _ in SCENARIOS:
    nnz = int(np.count_nonzero(np.abs(select_coefs[label]) > 1e-9))
    print(f"  {label:22s} nonzero_terms={nnz}")


if __name__ == "__main__":
  FS = 500.0  # overwritten in main() from the loaded data
  main()
