"""Render the sub-threshold replication results as a self-contained text report.

Reads ``outputs/pysindy/shape_analysis/subthreshold_replication.csv`` (written by
``subthreshold_replication.py``) and produces a report explaining what a
sub-threshold coefficient is, which two mechanisms produce them, and what the
replication across five independent training-trial subsets showed.

No fitting is performed here -- this only formats existing results.

Run with:

    .venv/bin/python scripts/pysindy/subthreshold_report.py
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(_PROJECT_ROOT))

CSV_PATH = _PROJECT_ROOT / "outputs/pysindy/shape_analysis/subthreshold_replication.csv"
OUT_PATH = _PROJECT_ROOT / "outputs/pysindy/shape_analysis/subthreshold_report.txt"

HEADER = """\
================================================================================
SUB-THRESHOLD COEFFICIENTS IN PySINDy: WHAT CAUSES THEM
================================================================================

A "sub-threshold coefficient" is a coefficient the optimizer KEPT (it is nonzero
in the fitted model) whose reported magnitude is nevertheless BELOW the threshold
the model was fit at. On its face this looks like a bug: the threshold is
supposed to remove exactly those terms.

It is not a bug. Two separate parts of the pipeline can produce it, and they have
very different scope. This report separates them.

--------------------------------------------------------------------------------
WHERE THE THRESHOLD IS APPLIED  (pysindy 2.1.0)
--------------------------------------------------------------------------------

BaseOptimizer.fit  (pysindy/optimizers/base.py)

  line 229-231   if normalize_columns: divide each library column by its L2 norm
  line 234       ordinary least-squares initial guess
  line 247       _reduce()  -> the STLSQ loop: ridge solve, then hard-threshold
                              |c| >= threshold, repeated until the support settles
  line 248       ind_ = |coef_| > 1e-14        (record WHICH terms survived)
  line 250-251   if unbias:  _unbias()          (default True for STLSQ)
  line 265-274   _unbias: refit by UNREGULARIZED least squares, restricted to the
                          surviving support, and overwrite the coefficient values
  line 254-259   if normalize_columns: divide coefficients by the column norms to
                          return them to original units

Two consequences follow directly from that ordering:

  MECHANISM 1 -- unit mismatch.
    With normalize_columns=True the threshold is applied to coefficients in
    NORMALIZED-column space (line 247), but coefficients() reports values in
    ORIGINAL units (line 254-259). The two numbers are not on the same scale, so
    comparing a reported coefficient against the threshold is meaningless.

  MECHANISM 2 -- the unbias refit.
    The threshold was applied to the values produced INSIDE the loop. The unbias
    step (line 250-251) then replaces those values with an unregularized refit.
    The reported coefficients are therefore not the ones that were thresholded,
    and the refit can move them below it.

--------------------------------------------------------------------------------
HOW THIS WAS TESTED
--------------------------------------------------------------------------------

Real LFP data, channel 0, fixation trials. 5 RESAMPLED training-trial subsets
(20 of the 28 training trials each, drawn at random without replacement, seed 0),
crossed with
degrees {3,5,7} x alpha {0, 0.05} x normalize_columns {True,False} x
unbias {True,False}. 120 fits in total.

The threshold was recalibrated separately for every subset and every regime (by
sweeping until the fit kept about 20 terms), so no result depends on one lucky
threshold value.

Preprocessing: 35 Hz low-pass, downsample 2 (dt = 0.004 s), global z-score,
n_delays = 4, delay = 2 samples, Savitzky-Golay smoothing window 9.
"""

FOOTER = """\
--------------------------------------------------------------------------------
CONCLUSIONS
--------------------------------------------------------------------------------

1. MECHANISM 1 (normalize_columns=True) is GENERAL and DETERMINISTIC.
   It fired in every subset at every degree tested. If normalize_columns is True,
   reported coefficients simply cannot be compared against the threshold. This is
   the mechanism that applies to this project's archived sweeps, which all used
   normalize_columns=True.

2. MECHANISM 2 (unbias=True) is REAL but NARROW.
   It reproduced perfectly -- 5/5 subsets versus 0/5 -- but ONLY at degree 7 with
   alpha=0. It did not appear at degree 3 or 5, and did not appear at alpha=0.05
   even at degree 7. It is a symptom of severe ill-conditioning, not a general
   property of the unbias step.

3. alpha=0.05 ELIMINATES mechanism 2.
   This is a concrete justification for a nonzero ridge that has nothing to do
   with shrinkage. Separately measured: alpha does not change the fitted
   coefficient VALUES at all, because the unbias refit discards the regularized
   estimates (alpha in {0, 0.05, 0.5} gave bit-identical coefficients when the
   selected support matched). Alpha's role is to keep the selection
   well-conditioned enough that the refit does not relocate coefficients across
   the threshold.

--------------------------------------------------------------------------------
PRACTICAL GUIDANCE
--------------------------------------------------------------------------------

* To reason about coefficient magnitudes against the threshold you set, use
  normalize_columns=False. Then the units agree.
* Threshold values are NOT comparable across normalize_columns settings, across
  signal normalizations, or across optimizers. Always record the regime with the
  value.
* Keep a nonzero alpha at high polynomial degree.
* unbias=True (the default) is generally desirable -- it removes ridge bias from
  the final estimates -- but it means the reported coefficients are not the ones
  that were thresholded.

--------------------------------------------------------------------------------
SCOPE AND LIMITS
--------------------------------------------------------------------------------

All five subsets are drawn from the same 28 training trials of a single session
and a single channel, and they OVERLAP: any two share about 14 of their 20 trials
(71%), which is what random 20-of-28 draws give. They are correlated resamples,
not independent samples. The correct reading is that the effect is not driven by
any one trial or small group of trials -- NOT that it has been validated on
independent data. Establishing that would need separate sessions or subjects. Degrees above 7 and n_delays other than 4 were
not tested here.

Source: pysindy 2.1.0, verified byte-identical to upstream tag v2.1.0.
Data:   outputs/pysindy/shape_analysis/subthreshold_replication.csv
Script: scripts/pysindy/subthreshold_replication.py
"""


def load_rows() -> list[dict]:
  """Read and type-convert the replication CSV."""
  if not CSV_PATH.exists():
    raise FileNotFoundError(
      f"{CSV_PATH} not found. Run scripts/pysindy/subthreshold_replication.py first."
    )
  rows = []
  for r in csv.DictReader(open(CSV_PATH)):
    rows.append(dict(
      subset=int(r["subset"]),
      normalize_columns=r["normalize_columns"] == "True",
      degree=int(r["degree"]),
      alpha=float(r["alpha"]),
      unbias=r["unbias"] == "True",
      threshold=float(r["threshold"]),
      nonzero=int(r["nonzero"]),
      below_threshold=int(r["below_threshold"]),
    ))
  return rows


def section_mechanism1(rows: list[dict]) -> str:
  """Build the normalize_columns table."""
  out = ["-" * 80,
         "MECHANISM 1: normalize_columns  (all degrees, both alphas, both unbias)",
         "-" * 80, "",
         f"{'normalize_columns':>18} {'degree':>7} {'fits with survivors':>21} "
         f"{'median count':>14} {'max count':>11}", ""]
  for nc in (True, False):
    for degree in (3, 5, 7):
      sel = [r for r in rows if r["normalize_columns"] == nc and r["degree"] == degree]
      if not sel:
        continue
      pos = sum(1 for r in sel if r["below_threshold"] > 0)
      med = np.median([r["below_threshold"] for r in sel])
      mx = max(r["below_threshold"] for r in sel)
      out.append(f"{str(nc):>18} {degree:>7} {pos:>12}/{len(sel):<8} "
                 f"{med:>14.0f} {mx:>11}")
  out += ["",
          "Reading: with normalize_columns=True every single fit produced",
          "sub-threshold survivors, at every degree. With False they are absent",
          "except in the degree-7 cells explained by mechanism 2 below.", ""]
  return "\n".join(out)


def section_mechanism2(rows: list[dict]) -> str:
  """Build the unbias table, restricted to normalize_columns=False."""
  out = ["-" * 80,
         "MECHANISM 2: unbias   (normalize_columns=False, so units agree)",
         "-" * 80, "",
         f"{'degree':>7} {'alpha':>7} {'unbias':>8} {'subsets with survivors':>24} "
         f"{'median count':>14}", ""]
  for degree in (3, 5, 7):
    for alpha in (0.0, 0.05):
      for unbias in (True, False):
        sel = [r for r in rows if not r["normalize_columns"] and r["degree"] == degree
               and r["alpha"] == alpha and r["unbias"] == unbias]
        if not sel:
          continue
        pos = sum(1 for r in sel if r["below_threshold"] > 0)
        med = np.median([r["below_threshold"] for r in sel])
        flag = "   <<<" if pos > 0 else ""
        out.append(f"{degree:>7} {alpha:>7} {str(unbias):>8} "
                   f"{pos:>15}/{len(sel):<8} {med:>14.0f}{flag}")
  out += ["",
          "Reading: the ONLY cell that produces survivors is degree 7 with",
          "alpha=0 and unbias=True. Turning unbias off in that same cell removes",
          "them completely (0/5), which isolates the refit as the cause.", ""]
  return "\n".join(out)


def section_per_subset(rows: list[dict]) -> str:
  """Show the decisive cell subset by subset, to demonstrate replication."""
  out = ["-" * 80,
         "PER-SUBSET DETAIL: the decisive cell (degree 7, alpha 0, nc=False)",
         "-" * 80, "",
         f"{'subset':>7} {'threshold':>12} {'unbias':>8} {'nonzero':>9} "
         f"{'below threshold':>17}", ""]
  sel = [r for r in rows if not r["normalize_columns"] and r["degree"] == 7
         and r["alpha"] == 0.0]
  for r in sorted(sel, key=lambda r: (r["subset"], not r["unbias"])):
    out.append(f"{r['subset']:>7} {r['threshold']:>12.4g} {str(r['unbias']):>8} "
               f"{r['nonzero']:>9} {r['below_threshold']:>17}")
  out += ["",
          "Every subset shows the same pattern: many survivors with unbias=True,",
          "exactly zero with unbias=False, on the same data and threshold.", ""]
  return "\n".join(out)


def section_contrast(rows: list[dict]) -> str:
  """Show that alpha=0.05 removes the effect at degree 7."""
  out = ["-" * 80,
         "WHY alpha MATTERS: degree 7, nc=False, unbias=True, alpha 0 vs 0.05",
         "-" * 80, "",
         f"{'subset':>7} {'alpha':>7} {'nonzero':>9} {'below threshold':>17}", ""]
  sel = [r for r in rows if not r["normalize_columns"] and r["degree"] == 7
         and r["unbias"]]
  for r in sorted(sel, key=lambda r: (r["subset"], r["alpha"])):
    out.append(f"{r['subset']:>7} {r['alpha']:>7} {r['nonzero']:>9} "
               f"{r['below_threshold']:>17}")
  out += ["",
          "alpha=0 leaves the regression ill-conditioned: the loop and the unbias",
          "refit then solve a near-singular system differently and disagree",
          "wildly. alpha=0.05 makes it well-conditioned and the disagreement",
          "vanishes.", ""]
  return "\n".join(out)


def main() -> None:
  """Write the report to disk and echo it."""
  rows = load_rows()
  report = "\n".join([
    HEADER,
    section_mechanism1(rows),
    section_mechanism2(rows),
    section_per_subset(rows),
    section_contrast(rows),
    FOOTER,
  ])
  OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
  OUT_PATH.write_text(report)
  print(report)
  print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
  main()
