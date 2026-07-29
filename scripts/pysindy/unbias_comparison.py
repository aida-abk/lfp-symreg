"""Refit archived configurations with unbias on and off, and compare simulations.

``unbias=True`` (the STLSQ default) discards the regularized coefficient values
after selection and recomputes them by unregularized least squares on the chosen
support (see ``docs/pysindy_internals.md``, Stage 4). Every archived sweep
therefore reports unbiased coefficients.

This script refits four archived configurations both ways, holding everything
else identical, and asks whether the choice changes:

* the coefficient values (expected -- that is what unbias does),
* the eigenvalues of the linear operator, which set oscillation frequency and
  decay time constant,
* the simulated trajectory on held-out trials.

Configurations are the four used in the slide deck, spanning ``n_delays`` 2-8.

Run with:

    .venv/bin/python scripts/pysindy/unbias_comparison.py
"""
from __future__ import annotations

import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pysindy as ps  # noqa: E402

from load_data.convert import MAT_FILE, TrialData  # noqa: E402
from load_data.preprocessing import channel_traces  # noqa: E402
from models.validation import (  # noqa: E402
  SimulationConfig,
  evaluate_simulation,
  simulate_model_detailed,
)
from models.sindy import delay_embed_trace, delay_embed_trajectories  # noqa: E402

csv.field_size_limit(10 * 1024 * 1024)

CHANNEL = 0
DOWNSAMPLE = 2
ALPHA = 0.05
NORMALIZE_COLUMNS = True
SIM_WALL_TIMEOUT_S = 240.0
# Cap the horizon: 6 s is ample to compare trajectories, and the 152-term
# n_delays=6 models are slow per right-hand-side evaluation at full trial length.
MAX_HORIZON_S = 6.0

GLOBAL = _PROJECT_ROOT / "outputs/pysindy/global_analysis"
SPLIT_METADATA_DIR = GLOBAL / "raw_grid_deg2357_t20000/parts"
OUTPUT_DIR = _PROJECT_ROOT / "outputs/pysindy/unbias_comparison"


@dataclass(frozen=True)
class Case:
  """One archived configuration to refit both ways."""

  label: str
  sweep: str
  index: str
  lowpass: float
  n_delays: int
  delay: int
  smooth: int
  degree: int
  threshold: float


CASES = [
  Case("nd2", "raw_grid_deg2357_t50000", "45", 35.0, 2, 5, 9, 3, 50000.0),
  Case("nd4", "raw_grid_deg2357_t20000", "49", 35.0, 4, 2, 0, 3, 20000.0),
  Case("nd6", "raw_grid_deg2357_t20000", "55", 35.0, 6, 1, 0, 3, 20000.0),
  Case("nd8", "raw_grid_deg2357_t50000", "67", 35.0, 8, 2, 0, 3, 50000.0),
]


def load_split() -> tuple[list[int], list[int]]:
  """Return (train_ids, test_ids) from the archived split metadata."""
  candidates = sorted(SPLIT_METADATA_DIR.glob("*_metadata.json"))
  split = json.loads(candidates[0].read_text())["split"]
  return split["train_trial_ids"], split["test_trial_ids"]


def archived_coefficients(case: Case) -> np.ndarray | None:
  """Return the coefficients stored in the archive for this configuration."""
  path = GLOBAL / case.sweep / "raw_grid_merged.csv"
  for row in csv.DictReader(open(path)):
    if row["configuration_index"] == case.index:
      return np.array(json.loads(row["coefficients_json"]))
  return None


def fit_case(trajectories, dt: float, case: Case, unbias: bool):
  """Fit one archived configuration with an explicit unbias setting."""
  kwargs = {}
  if case.smooth > 2:
    kwargs["differentiation_method"] = ps.SmoothedFiniteDifference(
      smoother_kws={"window_length": case.smooth, "polyorder": 3}
    )
  model = ps.SINDy(
    optimizer=ps.STLSQ(
      threshold=case.threshold, alpha=ALPHA,
      normalize_columns=NORMALIZE_COLUMNS, unbias=unbias,
      verbose=False, max_iter=20,
    ),
    feature_library=ps.PolynomialLibrary(degree=case.degree),
    **kwargs,
  )
  model.fit(trajectories, t=dt)
  return model


def linear_spectrum(model) -> tuple[list[float], float]:
  """Return (oscillation frequencies in Hz, max real part) of the linear part."""
  names = model.get_feature_names()
  coefs = np.asarray(model.coefficients())
  linear = [i for i, n in enumerate(names) if re.fullmatch(r"x\d+", n.strip())]
  A = coefs[:, linear]
  if A.shape[0] != A.shape[1]:
    return [], float("nan")
  eigenvalues = np.linalg.eigvals(A)
  freqs = sorted({round(abs(v.imag) / (2 * np.pi), 2) for v in eigenvalues if abs(v.imag) > 1e-9})
  return freqs, float(max(v.real for v in eigenvalues))


def main() -> None:
  """Refit every case both ways, simulate, and write comparison outputs."""
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  train_ids, test_ids = load_split()
  print(f"Loading {MAT_FILE} ...", flush=True)
  data = TrialData.load(MAT_FILE)
  dt = DOWNSAMPLE / data.fs
  fs = 1.0 / dt

  rows: list[dict] = []
  panels: dict[tuple[str, bool], tuple[np.ndarray, np.ndarray | None]] = {}

  for case in CASES:
    train = channel_traces(data, channel=CHANNEL, trials=train_ids,
                           downsample=DOWNSAMPLE, lowpass_hz=case.lowpass,
                           normalize="none")
    test = channel_traces(data, channel=CHANNEL, trials=test_ids,
                          downsample=DOWNSAMPLE, lowpass_hz=case.lowpass,
                          normalize="none")
    emb_train = delay_embed_trajectories(train, n_delays=case.n_delays, delay=case.delay)
    emb_test = [delay_embed_trace(t, n_delays=case.n_delays, delay=case.delay) for t in test]

    stored = archived_coefficients(case)
    print(f"\n=== {case.label}: {case.sweep} cfg {case.index} "
          f"(deg={case.degree}, nd={case.n_delays}, thr={case.threshold:g}) ===")

    fitted = {}
    for unbias in (True, False):
      model = fit_case(emb_train, dt, case, unbias)
      fitted[unbias] = model
      coefs = np.asarray(model.coefficients())
      nz = int(np.count_nonzero(coefs))
      freqs, max_re = linear_spectrum(model)
      match = ("n/a" if stored is None or stored.shape != coefs.shape
               else f"{np.max(np.abs(stored - coefs)):.4g}")
      print(f"  unbias={str(unbias):<5} terms={nz:>4}  maxRe={max_re:+7.3f}  "
            f"freqs={freqs[:4]}  |archive-refit|={match}")

      # simulate every held-out trial
      metrics_list = []
      failures: list[str] = []
      for trial_i, measured in enumerate(emb_test):
        horizon = min((len(measured) - 1) * dt, MAX_HORIZON_S)
        # In-process integration with a SIGALRM cap. The subprocess variant
        # (simulate_model_detailed_hard_timeout) was measured to exceed a 60 s
        # wall clock on these configurations even though the same integration
        # completes in seconds in-process, so it is not used here.
        sim = simulate_model_detailed(
          model, initial_state=measured[0], dt=dt, horizon_s=horizon,
          wall_timeout_s=SIM_WALL_TIMEOUT_S,
        )
        if not sim.completed and sim.failure_reason:
          failures.append(sim.failure_reason)
        if sim.completed and sim.trajectory is not None:
          m = evaluate_simulation(measured, sim.trajectory, fs=fs,
                                  config=SimulationConfig(simulation_horizon_s=horizon))
          metrics_list.append(m)
          if trial_i == 0:
            panels[(case.label, unbias)] = (measured[:, 0], sim.trajectory[:, 0])
        elif trial_i == 0:
          panels[(case.label, unbias)] = (measured[:, 0], None)

      def med(key):
        vals = [float(m[key]) for m in metrics_list if np.isfinite(float(m[key]))]
        return float(np.median(vals)) if vals else float("nan")

      rows.append(dict(
        case=case.label, unbias=unbias, degree=case.degree, n_delays=case.n_delays,
        threshold=case.threshold, nonzero_terms=nz, max_real_eigenvalue=max_re,
        frequencies=";".join(str(f) for f in freqs),
        n_completed=len(metrics_list), n_trials=len(emb_test),
        psd_similarity=med("psd_similarity"),
        x0_correlation=med("x0_correlation"),
        collapse_std_ratio=med("collapse_std_ratio"),
        x0_rmse=med("x0_rmse"),
      ))
      print(f"    completed {len(metrics_list)}/{len(emb_test)}  "
            f"psd={rows[-1]['psd_similarity']:+.3f} "
            f"collapse={rows[-1]['collapse_std_ratio']:.3f} "
            f"rmse={rows[-1]['x0_rmse']:.1f}", flush=True)
      if failures:
        print(f"      first failure: {failures[0][:110]}")

    a = np.asarray(fitted[True].coefficients())
    b = np.asarray(fitted[False].coefficients())
    same_support = np.array_equal(np.abs(a) > 1e-12, np.abs(b) > 1e-12)
    print(f"  --> same support: {same_support}   max|coef difference|: {np.max(np.abs(a - b)):.4g}")

  path = OUTPUT_DIR / "unbias_comparison.csv"
  with open(path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
  print(f"\nwrote {path}")
  _plot(panels, dt)
  _summary(rows)


def _plot(panels, dt: float) -> None:
  """Write a case × unbias grid of measured versus simulated trajectories."""
  labels = [c.label for c in CASES]
  fig, axes = plt.subplots(len(labels), 2, figsize=(13, 2.4 * len(labels)),
                           squeeze=False, sharex=True)
  for r, label in enumerate(labels):
    for c, unbias in enumerate((True, False)):
      ax = axes[r][c]
      entry = panels.get((label, unbias))
      if entry is None:
        ax.set_title(f"{label}  unbias={unbias}  [no simulation]", fontsize=8)
        continue
      measured, simulated = entry
      t = np.arange(measured.size) * dt
      ax.plot(t, measured, color="tab:blue", lw=0.6, label="measured")
      if simulated is not None:
        ts = np.arange(simulated.size) * dt
        ax.plot(ts, simulated, color="tab:orange", lw=0.7, ls="--", label="simulated")
      ax.set_title(f"{label}   unbias={unbias}", fontsize=9)
      ax.tick_params(labelsize=7)
      if c == 0:
        ax.set_ylabel("x0 (µV)", fontsize=8)
      if r == len(labels) - 1:
        ax.set_xlabel("time (s)", fontsize=8)
      if r == 0 and c == 0:
        ax.legend(fontsize=7)
  fig.tight_layout()
  out = OUTPUT_DIR / "unbias_trajectories.png"
  fig.savefig(out, dpi=150)
  plt.close(fig)
  print(f"wrote {out}")


def _summary(rows: list[dict]) -> None:
  """Print a compact side-by-side table."""
  print("\n===== unbias=True vs unbias=False =====")
  print(f"{'case':>5} {'unbias':>7} {'terms':>6} {'maxRe':>8} {'done':>6} "
        f"{'psd':>7} {'collapse':>9} {'rmse':>7}  frequencies")
  for r in rows:
    print(f"{r['case']:>5} {str(r['unbias']):>7} {r['nonzero_terms']:>6} "
          f"{r['max_real_eigenvalue']:>+8.3f} {r['n_completed']:>3}/{r['n_trials']:<2} "
          f"{r['psd_similarity']:>+7.3f} {r['collapse_std_ratio']:>9.3f} "
          f"{r['x0_rmse']:>7.1f}  {r['frequencies']}")


if __name__ == "__main__":
  main()
