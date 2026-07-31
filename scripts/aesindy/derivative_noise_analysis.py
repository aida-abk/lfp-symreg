"""Test whether the derivative-fit ceiling (R^2 ~ 0.35) is noise or model class.

Background
----------
Fitting 20 configurations at ``n_delays=2`` produced derivative R^2 between 0.347
and 0.358 regardless of degree or threshold, while a 25x increase in term count
bought only 0.011 (see ``docs/pysindy_internals.md``, Stage 5). Two explanations
remain, and they imply opposite next steps:

* **Noise-limited.** The estimated ``xdot`` is dominated by high-frequency noise,
  so R^2 is capped no matter what library is used. The fix would be a formulation
  that avoids pointwise derivatives (e.g. the weak/integral form).
* **Model-class-limited.** A 2-coordinate delay embedding is too small to carry
  the dynamics. The fix would be a richer state, not a different objective.

Three measurements separate them:

A. **R^2 versus smoothing window.** Climbing steeply implies derivative noise was
   being removed. Interpret with care: heavier smoothing also makes the target
   intrinsically more predictable, so some rise is expected. The informative part
   is whether it saturates.
B. **R^2 versus number of delay coordinates.** A larger embedding gives a linear
   model strictly more information. Climbing implies the embedding was the limit;
   flat implies a noise floor.
C. **Agreement between differentiation methods.** If independent estimators of the
   same derivative disagree, the estimate is noise-dominated.

All fits use ``threshold=0`` so the result is the *ceiling* R^2 for that library
(unrestricted least squares after the unbias step), isolating the data question
from the sparsity question.

Run with:

    .venv/bin/python scripts/pysindy/derivative_noise_analysis.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pysindy as ps

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
ALPHA = 0.05
NORMALIZE_COLUMNS = True
CEILING_THRESHOLD = 0.0

# A: smoothing sweep (n_delays and degree fixed)
SMOOTH_WINDOWS = [0, 5, 9, 15, 25, 51]
SMOOTH_SWEEP_N_DELAYS = 2
SMOOTH_SWEEP_DEGREE = 2

# B: embedding sweep (smoothing and degree fixed)
N_DELAYS_LIST = [2, 4, 6, 8, 12, 16]
EMBED_SWEEP_SMOOTH = 9
EMBED_SWEEP_DEGREE = 2

DELAY_SAMPLES = 5
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


def ceiling_scores(
  train_traces: list[np.ndarray],
  test_traces: list[np.ndarray],
  dt: float,
  n_delays: int,
  degree: int,
  smooth_window: int,
) -> tuple[float, float, int]:
  """Fit an unrestricted model and return (train R^2, median test R^2, terms).

  Args:
    train_traces: Preprocessed training traces.
    test_traces: Preprocessed held-out traces.
    dt: Sample interval in seconds.
    n_delays: Delay coordinates in the embedding.
    degree: Polynomial degree.
    smooth_window: Savitzky-Golay window in samples; 0 uses plain finite
      differences.

  Returns:
    Training R^2, median held-out R^2, and the surviving term count.
  """
  embedded_train = delay_embed_trajectories(
    train_traces, n_delays=n_delays, delay=DELAY_SAMPLES
  )
  embedded_test = [
    delay_embed_trace(t, n_delays=n_delays, delay=DELAY_SAMPLES) for t in test_traces
  ]
  model = fit_sindy_model(
    embedded_train, dt=dt,
    config=SINDyConfig(
      degree=degree, threshold=CEILING_THRESHOLD, alpha=ALPHA,
      normalize_columns=NORMALIZE_COLUMNS, smooth_window=smooth_window,
      smoothing_polyorder=3, verbose=False, max_iter=20,
    ),
  )
  train_r2 = float(model.score(embedded_train, t=dt))
  test_r2 = float(np.median([model.score([t], t=dt) for t in embedded_test]))
  terms = int(np.count_nonzero(np.asarray(model.coefficients())))
  return train_r2, test_r2, terms


def derivative_agreement(trace: np.ndarray, dt: float) -> None:
  """Print pairwise agreement between differentiation methods on one trace."""
  column = np.asarray(trace, dtype=float).reshape(-1, 1)
  methods: dict[str, np.ndarray] = {
    "finite_diff": np.asarray(ps.FiniteDifference()(column, t=dt)).ravel(),
  }
  for window in (5, 9, 25, 51):
    methods[f"savgol_{window}"] = np.asarray(
      ps.SmoothedFiniteDifference(
        smoother_kws={"window_length": window, "polyorder": 3}
      )(column, t=dt)
    ).ravel()
  try:
    methods["spectral"] = np.asarray(
      ps.SpectralDerivative()(column, t=dt)
    ).ravel()
  except Exception as exc:  # pragma: no cover - informational only
    print(f"  (spectral derivative unavailable: {exc})")

  names = list(methods)
  print(f"  {'':>12} " + "".join(f"{n:>12}" for n in names))
  for a in names:
    cells = []
    for b in names:
      left, right = methods[a], methods[b]
      n = min(left.size, right.size)
      mask = np.isfinite(left[:n]) & np.isfinite(right[:n])
      corr = float(np.corrcoef(left[:n][mask], right[:n][mask])[0, 1])
      cells.append(f"{corr:>12.4f}")
    print(f"  {a:>12} " + "".join(cells))

  fd = methods["finite_diff"]
  finite = fd[np.isfinite(fd)]
  centered = finite - finite.mean()
  lag1 = float(
    np.dot(centered[:-1], centered[1:]) / np.dot(centered, centered)
  )
  print(f"\n  lag-1 autocorrelation of the plain finite-difference derivative: {lag1:+.4f}")
  print("  (a white-noise-dominated derivative from differencing tends strongly")
  print("   negative near -0.5; a smooth, well-resolved derivative stays near +1)")


def run() -> None:
  """Execute all three measurements and write a summary CSV."""
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  train_ids, test_ids = load_split(SPLIT_METADATA_DIR)

  print(f"Loading {MAT_FILE} ...", flush=True)
  data = TrialData.load(MAT_FILE)
  dt = DOWNSAMPLE / data.fs
  train_traces = channel_traces(
    data, channel=CHANNEL, trials=train_ids, downsample=DOWNSAMPLE,
    lowpass_hz=LOWPASS_HZ, normalize="none",
  )
  test_traces = channel_traces(
    data, channel=CHANNEL, trials=test_ids, downsample=DOWNSAMPLE,
    lowpass_hz=LOWPASS_HZ, normalize="none",
  )
  print(f"dt={dt:g}s  train={len(train_traces)}  test={len(test_traces)}\n")

  rows: list[dict] = []

  print("=== C. Do differentiation methods agree? ===")
  print("(pairwise correlation of derivative estimates, one training trial)")
  derivative_agreement(train_traces[0], dt)

  print("\n=== A. Ceiling R^2 versus smoothing window ===")
  print(f"(n_delays={SMOOTH_SWEEP_N_DELAYS}, degree={SMOOTH_SWEEP_DEGREE}, threshold=0)")
  print(f"  {'window':>7} {'terms':>6} {'R2_train':>10} {'R2_test':>10}")
  for window in SMOOTH_WINDOWS:
    tr, te, terms = ceiling_scores(
      train_traces, test_traces, dt,
      n_delays=SMOOTH_SWEEP_N_DELAYS, degree=SMOOTH_SWEEP_DEGREE,
      smooth_window=window,
    )
    print(f"  {window:>7} {terms:>6} {tr:>+10.4f} {te:>+10.4f}", flush=True)
    rows.append({"sweep": "smoothing", "value": window, "terms": terms,
                 "r2_train": tr, "r2_test": te})

  print(f"\n=== B. Ceiling R^2 versus embedding dimension ===")
  print(f"(smooth={EMBED_SWEEP_SMOOTH}, degree={EMBED_SWEEP_DEGREE}, threshold=0)")
  print(f"  {'n_delays':>9} {'terms':>6} {'R2_train':>10} {'R2_test':>10}")
  for n_delays in N_DELAYS_LIST:
    tr, te, terms = ceiling_scores(
      train_traces, test_traces, dt,
      n_delays=n_delays, degree=EMBED_SWEEP_DEGREE,
      smooth_window=EMBED_SWEEP_SMOOTH,
    )
    print(f"  {n_delays:>9} {terms:>6} {tr:>+10.4f} {te:>+10.4f}", flush=True)
    rows.append({"sweep": "n_delays", "value": n_delays, "terms": terms,
                 "r2_train": tr, "r2_test": te})

  path = OUTPUT_DIR / "derivative_noise_analysis.csv"
  with open(path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
  print(f"\nwrote {path}")


if __name__ == "__main__":
  run()
