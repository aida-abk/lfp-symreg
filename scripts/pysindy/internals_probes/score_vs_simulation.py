"""Compare what SINDy optimizes against how the model is being judged.

STLSQ minimizes ``||Xdot - Theta(x) Xi||^2 + alpha ||Xi||^2`` -- a ONE-STEP
derivative regression evaluated at measured states (see
``docs/pysindy_internals.md``, Stage 4). ``SINDy.score`` reports R^2 on exactly
that quantity. Free-running simulation, by contrast, integrates the model from a
single initial condition and is never referenced by the loss.

This script measures both for the same fitted models. The hypothesis under test:
SINDy succeeds at the objective it was given (high derivative R^2) while failing
the criterion it was never given (trajectory/spectral agreement). If confirmed,
the limitation is the objective, not the hyperparameters.

Configurations mirror ``scripts/pysindy/shape_analysis.py`` so the score column
can be read directly against the shape metrics already computed there.

Run with:

    .venv/bin/python scripts/pysindy/score_vs_simulation.py
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

from load_data.convert import MAT_FILE, TrialData  # noqa: E402
from load_data.preprocessing import channel_traces  # noqa: E402
from models.sindy import (  # noqa: E402
  SINDyConfig,
  delay_embed_trace,
  delay_embed_trajectories,
  fit_sindy_model,
)

csv.field_size_limit(10 * 1024 * 1024)

CHANNEL = 0
DOWNSAMPLE = 2
LOWPASS_HZ = 35.0
N_DELAYS = 2
DELAY_SAMPLES = 5
SMOOTH_WINDOW = 9
SMOOTHING_POLYORDER = 3
NORMALIZE_COLUMNS = True
ALPHA = 0.05
MAX_ITER = 20

DEGREES = [2, 3, 5, 7, 9]
THRESHOLDS = [1000.0, 10000.0, 20000.0, 50000.0]

SPLIT_METADATA_DIR = (
  _PROJECT_ROOT / "outputs/pysindy/global_analysis/raw_grid_deg2357_t20000/parts"
)
OUTPUT_DIR = _PROJECT_ROOT / "outputs/pysindy/shape_analysis"


def load_split(metadata_dir: Path) -> tuple[list[int], list[int]]:
  """Return (train_ids, test_ids) from the first metadata JSON in a directory."""
  candidates = sorted(metadata_dir.glob("*_metadata.json"))
  if not candidates:
    raise FileNotFoundError(f"No *_metadata.json found under {metadata_dir}")
  split = json.loads(candidates[0].read_text())["split"]
  return split["train_trial_ids"], split["test_trial_ids"]


def median_finite(values: list[float]) -> float:
  """Return the median of finite entries, or NaN when none are finite."""
  finite = [v for v in values if np.isfinite(v)]
  return float(np.median(finite)) if finite else float("nan")


def run() -> None:
  """Fit each configuration and report derivative R^2 on train and held-out data."""
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  train_ids, test_ids = load_split(SPLIT_METADATA_DIR)

  print(f"Loading {MAT_FILE} ...", flush=True)
  data = TrialData.load(MAT_FILE)
  dt = DOWNSAMPLE / data.fs
  print(f"dt={dt:g}s  train={len(train_ids)}  test={len(test_ids)}")
  print("score = R^2 of the DERIVATIVE fit -- the quantity STLSQ actually minimizes\n")

  train_traces = channel_traces(
    data, channel=CHANNEL, trials=train_ids, downsample=DOWNSAMPLE,
    lowpass_hz=LOWPASS_HZ, normalize="none",
  )
  test_traces = channel_traces(
    data, channel=CHANNEL, trials=test_ids, downsample=DOWNSAMPLE,
    lowpass_hz=LOWPASS_HZ, normalize="none",
  )
  embedded_train = delay_embed_trajectories(
    train_traces, n_delays=N_DELAYS, delay=DELAY_SAMPLES
  )
  embedded_test = [
    delay_embed_trace(t, n_delays=N_DELAYS, delay=DELAY_SAMPLES) for t in test_traces
  ]

  rows: list[dict] = []
  for degree in DEGREES:
    for threshold in THRESHOLDS:
      config = SINDyConfig(
        degree=degree, threshold=threshold, alpha=ALPHA,
        normalize_columns=NORMALIZE_COLUMNS, smooth_window=SMOOTH_WINDOW,
        smoothing_polyorder=SMOOTHING_POLYORDER, verbose=False, max_iter=MAX_ITER,
      )
      model = fit_sindy_model(embedded_train, dt=dt, config=config)
      nonzero = int(np.count_nonzero(np.asarray(model.coefficients())))

      train_score = float(model.score(embedded_train, t=dt))
      test_scores = [float(model.score([trial], t=dt)) for trial in embedded_test]

      row = {
        "label": f"deg{degree}_thr{threshold:g}",
        "degree": degree,
        "threshold": threshold,
        "nonzero_terms": nonzero,
        "derivative_r2_train": train_score,
        "derivative_r2_test_median": median_finite(test_scores),
        "derivative_r2_test_min": min(test_scores),
        "derivative_r2_test_max": max(test_scores),
      }
      rows.append(row)
      print(f"  deg={degree} thr={threshold:>7g} terms={nonzero:>4}  "
            f"R2_train={train_score:+.4f}  "
            f"R2_test_median={row['derivative_r2_test_median']:+.4f}  "
            f"[{row['derivative_r2_test_min']:+.3f}, {row['derivative_r2_test_max']:+.3f}]",
            flush=True)

  path = OUTPUT_DIR / "derivative_scores.csv"
  with open(path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
  print(f"\nwrote {path}")

  _print_comparison(rows)


def _print_comparison(rows: list[dict]) -> None:
  """Print derivative R^2 beside the shape metrics, when those exist."""
  shape_path = OUTPUT_DIR / "shape_metrics.csv"
  if not shape_path.exists():
    return
  shape = {
    r["label"]: r for r in csv.DictReader(open(shape_path)) if r["degree"]
  }
  print("\n===== OBJECTIVE vs EVALUATION =====")
  print("derivative R^2 = what STLSQ minimizes | PSD/ACF = free-running simulation")
  print(f"{'label':<16} {'terms':>6} {'R2_train':>9} {'R2_test':>9} {'psd':>7} {'acf':>7}")
  for r in sorted(rows, key=lambda r: -r["derivative_r2_test_median"]):
    s = shape.get(r["label"])
    psd = f"{float(s['psd_similarity']):+.3f}" if s else "   --  "
    acf = f"{float(s['autocorrelation_similarity']):+.3f}" if s else "   --  "
    print(f"{r['label']:<16} {r['nonzero_terms']:>6} "
          f"{r['derivative_r2_train']:>+9.4f} {r['derivative_r2_test_median']:>+9.4f} "
          f"{psd:>7} {acf:>7}")


if __name__ == "__main__":
  run()
