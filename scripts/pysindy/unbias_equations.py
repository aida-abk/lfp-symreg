"""Print the fitted equations with unbias on and off, term by term.

``unbias`` keeps the support STLSQ selected and recomputes the coefficient
values by unregularized least squares (``optimizers/base.py:265-274``). This
script shows exactly which coefficients move and by how much, for the same four
archived configurations compared in ``unbias_comparison.py``.

Output goes to the console and to
``outputs/pysindy/unbias_comparison/unbias_equations.txt``.

Run with:

    .venv/bin/python scripts/pysindy/unbias_equations.py
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
from load_data.preprocessing import channel_traces  # noqa: E402
from models.sindy import delay_embed_trajectories  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unbias_comparison import (  # noqa: E402
  ALPHA,
  CASES,
  CHANNEL,
  DOWNSAMPLE,
  NORMALIZE_COLUMNS,
  OUTPUT_DIR,
  SPLIT_METADATA_DIR,
  fit_case,
)

MAX_TERMS_SHOWN = 10**9   # show every surviving term


def load_train_ids() -> list[int]:
  """Return the archived training trial identifiers."""
  candidates = sorted(SPLIT_METADATA_DIR.glob("*_metadata.json"))
  return json.loads(candidates[0].read_text())["split"]["train_trial_ids"]


def format_equation(names: list[str], coefs: np.ndarray, precision: int = 4) -> str:
  """Render one equation row as a readable sum of terms."""
  parts = []
  for name, value in zip(names, coefs):
    if abs(value) <= 1e-12:
      continue
    parts.append(f"{value:+.{precision}g} {name}" if name != "1" else f"{value:+.{precision}g}")
  return " ".join(parts) if parts else "0"


def main() -> None:
  """Fit each case both ways and report coefficient-level differences."""
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  train_ids = load_train_ids()
  print(f"Loading {MAT_FILE} ...", flush=True)
  data = TrialData.load(MAT_FILE)
  dt = DOWNSAMPLE / data.fs

  lines: list[str] = []

  def emit(text: str = "") -> None:
    """Print and record one output line."""
    print(text)
    lines.append(text)

  emit("Coefficients with unbias=True (archive default) vs unbias=False.")
  emit("Same support in every case; only the values differ.")
  emit(f"alpha={ALPHA}, normalize_columns={NORMALIZE_COLUMNS}, raw signal (no z-score).")

  for case in CASES:
    traces = channel_traces(data, channel=CHANNEL, trials=train_ids,
                            downsample=DOWNSAMPLE, lowpass_hz=case.lowpass,
                            normalize="none")
    embedded = delay_embed_trajectories(traces, n_delays=case.n_delays, delay=case.delay)

    models = {u: fit_case(embedded, dt, case, u) for u in (True, False)}
    names = models[True].get_feature_names()
    a = np.asarray(models[True].coefficients())   # unbias=True
    b = np.asarray(models[False].coefficients())  # unbias=False

    emit("\n" + "=" * 78)
    emit(f"{case.label}: {case.sweep} cfg {case.index}  "
         f"(degree={case.degree}, n_delays={case.n_delays}, delay={case.delay}, "
         f"smooth={case.smooth}, threshold={case.threshold:g}, lowpass={case.lowpass:g} Hz)")
    emit("=" * 78)

    support = np.abs(a) > 1e-12
    emit(f"terms: {int(support.sum())}   support identical: "
         f"{np.array_equal(support, np.abs(b) > 1e-12)}")

    # Full equations when small enough to read.
    if int(support.sum()) <= MAX_TERMS_SHOWN:
      emit("\n-- unbias=True (what the archive contains) --")
      for i in range(a.shape[0]):
        emit(f"  (x{i})' = {format_equation(names, a[i])}")
      emit("\n-- unbias=False --")
      for i in range(b.shape[0]):
        emit(f"  (x{i})' = {format_equation(names, b[i])}")

    # Term-by-term table, largest surviving coefficients first.
    idx = np.dstack(np.unravel_index(
      np.argsort(np.abs(a).ravel())[::-1], a.shape))[0]
    shown = [(i, j) for i, j in idx if support[i, j]][:MAX_TERMS_SHOWN]
    emit(f"\n{'equation':>9} {'term':<12} {'unbias=True':>14} {'unbias=False':>14} "
         f"{'False/True':>11} {'change':>9}")
    for i, j in shown:
      ratio = b[i, j] / a[i, j] if abs(a[i, j]) > 1e-12 else float("nan")
      emit(f"{'(x%d)' % i:>9} {names[j]:<12} {a[i, j]:>14.5g} {b[i, j]:>14.5g} "
           f"{ratio:>11.4f} {100 * (ratio - 1):>+8.1f}%")

    ratios = (b[support] / a[support])
    emit(f"\n  ratio False/True over all {support.sum()} surviving terms: "
         f"median={np.median(ratios):.4f}  "
         f"min={ratios.min():.4f}  max={ratios.max():.4f}")
    emit(f"  |coefficient| totals: unbias=True {np.abs(a).sum():.4g}   "
         f"unbias=False {np.abs(b).sum():.4g}   "
         f"ratio {np.abs(b).sum() / np.abs(a).sum():.4f}")
    emit(f"  fraction of terms SHRUNK by unbias=False: "
         f"{100 * np.mean(np.abs(b[support]) < np.abs(a[support])):.1f}%")

  path = OUTPUT_DIR / "unbias_equations.txt"
  path.write_text("\n".join(lines))
  print(f"\nwrote {path}")


if __name__ == "__main__":
  main()
