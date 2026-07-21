"""Test whether alpha=0 removes sub-threshold surviving coefficients.

Background
----------
The run ``raw_grid_nc_false_gsz_t1`` (global z-score, normalize_columns=False,
threshold=1.0, alpha=0.05) contains equations with surviving coefficients whose
absolute value is below the threshold of 1.0. That is possible because STLSQ
selects support with a *ridge* regression (alpha=0.05) and thresholds that ridge
coefficient, but the reported value comes from the unbiased *plain OLS* refit on
the surviving support. Ridge and OLS differ, so a term that passed selection at
>=1 can be printed below 1.

"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(_PROJECT_ROOT))

from load_data.convert import MAT_FILE, TrialData
from load_data.preprocessing import (
  apply_global_zscore,
  channel_traces,
  compute_global_zscore_stats,
)
from models.sindy import SINDyConfig, delay_embed_trajectories, fit_sindy_model

csv.field_size_limit(10 * 1024 * 1024)

SOURCE_RUN = _PROJECT_ROOT / "outputs/pysindy/raw_grid_nc_false_gsz_t1"
SOURCE_CSV = SOURCE_RUN / "raw_grid_merged.csv"
SOURCE_META = sorted((SOURCE_RUN / "parts").glob("*_metadata.json"))[0]
OUTPUT_DIR = _PROJECT_ROOT / "outputs/pysindy/nc_false_gsz_t1_alpha0_probe"

THRESHOLD = 1.0
CHANNEL = 0
DOWNSAMPLE = 2
TEST_FRACTION = 0.25
SEED = 0
N_CONFIGS = 12  # how many affected configs to probe


def count_subthreshold(coefs: np.ndarray, threshold: float) -> int:
  """Return the number of surviving coefficients with 0 < |c| < threshold.

  Args:
    coefs: Coefficient matrix (n_targets, n_features).
    threshold: STLSQ threshold the fit was run at.

  Returns:
    Count of nonzero coefficients whose magnitude is below the threshold.
  """
  a = np.abs(np.asarray(coefs, dtype=float))
  return int(np.sum((a > 1e-9) & (a < threshold)))


def format_equations(names: list[str], coefs: np.ndarray) -> str:
  """Build a human-readable equation string per target state.

  Args:
    names: Feature names.
    coefs: Coefficient matrix (n_targets, n_features).

  Returns:
    One string with each equation as "(xi)' = c name + ...", targets joined by
    " || ".
  """
  coefs = np.asarray(coefs, dtype=float)
  eqs = []
  for i, row in enumerate(coefs):
    terms = [f"{c:+.4f} {name}" for name, c in zip(names, row) if abs(c) > 1e-9]
    eqs.append(f"(x{i})' = " + (" ".join(terms) if terms else "0"))
  return " || ".join(eqs)


def subthreshold_terms(names: list[str], coefs: np.ndarray, threshold: float) -> str:
  """List only the surviving terms with 0 < |c| < threshold.

  Args:
    names: Feature names.
    coefs: Coefficient matrix (n_targets, n_features).
    threshold: STLSQ threshold.

  Returns:
    A "; "-joined string of "(xi)': c name" entries, empty if none.
  """
  coefs = np.asarray(coefs, dtype=float)
  out = []
  for i, row in enumerate(coefs):
    for name, c in zip(names, row):
      if 1e-9 < abs(c) < threshold:
        out.append(f"(x{i})': {c:+.4f} {name}")
  return "; ".join(out)


def load_source_rows() -> list[dict]:
  """Load the existing alpha=0.05 run rows keyed for lookup."""
  with open(SOURCE_CSV) as f:
    return list(csv.DictReader(f))


def main() -> None:
  """Select affected configs, refit at alpha=0, and compare survivor counts."""
  rows = load_source_rows()

  # Rank configs by number of sub-threshold survivors in the original run.
  affected = []
  for r in rows:
    if r["fit_status"] != "success" or not r["coefficients_json"]:
      continue
    coefs = np.asarray(json.loads(r["coefficients_json"]), dtype=float)
    n_sub = count_subthreshold(coefs, THRESHOLD)
    if n_sub > 0:
      affected.append((n_sub, r))
  affected.sort(key=lambda t: -t[0])

  # Pick a diverse dozen: spread across degrees rather than only the worst ones.
  by_degree: dict[str, list] = {}
  for n_sub, r in affected:
    by_degree.setdefault(r["degree"], []).append((n_sub, r))
  selected: list[tuple[int, dict]] = []
  degrees = sorted(by_degree)
  i = 0
  while len(selected) < N_CONFIGS and any(by_degree.values()):
    deg = degrees[i % len(degrees)]
    if by_degree[deg]:
      selected.append(by_degree[deg].pop(0))
    i += 1
  print(f"{len(affected)} of {len(rows)} configs have sub-threshold survivors; "
        f"probing {len(selected)}.")

  # Data + split (same seed/fraction as the source run).
  meta = json.loads(SOURCE_META.read_text())
  train_ids = meta["split"]["train_trial_ids"]
  print(f"Loading {MAT_FILE} ... (train trials={len(train_ids)})", flush=True)
  data = TrialData.load(MAT_FILE)
  fs = data.fs
  dt = DOWNSAMPLE / fs

  # Precompute z-scored training traces per lowpass value (train-only stats).
  train_z: dict[float, list[np.ndarray]] = {}
  for lp in sorted({float(r["lowpass_hz"]) for _, r in selected}):
    raw = channel_traces(data, channel=CHANNEL, trials=train_ids,
                         downsample=DOWNSAMPLE, lowpass_hz=lp, normalize="none")
    stats = compute_global_zscore_stats(raw, channel=CHANNEL)
    train_z[lp] = apply_global_zscore(raw, stats)

  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  out_rows = []
  print()
  header = (f"{'cfg':>4} {'deg':>3} {'lp':>4} {'nd':>3} {'dly':>3} {'sm':>3} "
            f"{'sub@a=0.05':>10} {'sub@a=0.0':>10} {'min|c|@0.05':>12} {'min|c|@0.0':>12}")
  print(header)
  print("-" * len(header))
  for n_sub_orig, r in selected:
    lp = float(r["lowpass_hz"])
    n_delays, delay, smooth = int(r["n_delays"]), int(r["delay_samples"]), int(r["smooth_window_samples"])
    degree = int(r["degree"])
    embedded = delay_embed_trajectories(train_z[lp], n_delays=n_delays, delay=delay)
    model = fit_sindy_model(
      embedded, dt=dt,
      config=SINDyConfig(
        degree=degree, threshold=THRESHOLD, alpha=0.0,
        normalize_columns=False, smooth_window=smooth, smoothing_polyorder=3,
      ),
    )
    coefs_a0 = np.asarray(model.coefficients(), dtype=float)
    names_a0 = model.get_feature_names()
    coefs_orig = np.asarray(json.loads(r["coefficients_json"]), dtype=float)
    names_orig = json.loads(r["feature_names_json"])
    n_sub_a0 = count_subthreshold(coefs_a0, THRESHOLD)

    def min_nonzero_abs(c: np.ndarray) -> float:
      a = np.abs(c)
      a = a[a > 1e-9]
      return float(a.min()) if a.size else float("nan")

    print(f"{r['configuration_index']:>4} {degree:>3} {lp:>4.0f} {n_delays:>3} "
          f"{delay:>3} {smooth:>3} {n_sub_orig:>10} {n_sub_a0:>10} "
          f"{min_nonzero_abs(coefs_orig):>12.4f} {min_nonzero_abs(coefs_a0):>12.4f}")
    out_rows.append({
      "configuration_index": r["configuration_index"], "degree": degree,
      "lowpass_hz": lp, "n_delays": n_delays, "delay_samples": delay,
      "smooth_window_samples": smooth,
      "threshold": THRESHOLD,
      "nonzero_terms_alpha_0p05": int(np.sum(np.abs(coefs_orig) > 1e-9)),
      "subthreshold_count_alpha_0p05": n_sub_orig,
      "subthreshold_terms_alpha_0p05": subthreshold_terms(names_orig, coefs_orig, THRESHOLD),
      "equations_alpha_0p05": format_equations(names_orig, coefs_orig),
      "nonzero_terms_alpha_0p0": int(np.sum(np.abs(coefs_a0) > 1e-9)),
      "subthreshold_count_alpha_0p0": n_sub_a0,
      "subthreshold_terms_alpha_0p0": subthreshold_terms(names_a0, coefs_a0, THRESHOLD),
      "equations_alpha_0p0": format_equations(names_a0, coefs_a0),
    })

  out_csv = OUTPUT_DIR / "alpha0_subthreshold_comparison.csv"
  with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
    w.writeheader()
    w.writerows(out_rows)

  total_orig = sum(r["subthreshold_count_alpha_0p05"] for r in out_rows)
  total_a0 = sum(r["subthreshold_count_alpha_0p0"] for r in out_rows)
  print("\nSummary across probed configs:")
  print(f"  sub-threshold survivors at alpha=0.05: {total_orig}")
  print(f"  sub-threshold survivors at alpha=0.0 : {total_a0}")
  print(f"  wrote {out_csv.relative_to(_PROJECT_ROOT)}")


if __name__ == "__main__":
  main()
