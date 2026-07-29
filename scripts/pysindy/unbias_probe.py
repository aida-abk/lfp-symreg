"""Measure the effect of STLSQ ``unbias`` on simulated LFP trajectories.

PySINDy's ``unbias=True`` (its default) appends an unregularized least-squares
refit on the support STLSQ selected, undoing the shrinkage the ridge ``alpha``
imposed. Because that refit runs *after* thresholding and only reads the
support (pysindy ``optimizers/base.py:_unbias``), it cannot change which terms
survive -- only their magnitudes. This probe therefore isolates exactly one
question: does ridge shrinkage of the retained coefficients change the
simulated dynamics of a held-out trial?

Each configuration is fit twice (``unbias=True`` and ``unbias=False``) with
everything else held identical, then both models are simulated from the same
initial state. ``alpha=0.0`` is included as a control: STLSQ's ridge solve
reduces to ordinary least squares there, so the two arms must agree to
numerical precision. If they do not, the experiment is misconfigured.

The backbone (channel, filtering, embedding, smoothing, train/test split) is
reused from ``scenario_config.py`` so results are comparable with the existing
scenario probe. Model code in ``models/sindy.py`` is used as-is via the
``SINDyConfig.unbias`` field.

Example:
    .venv/bin/python scripts/pysindy/unbias_probe.py
"""
from __future__ import annotations

import csv
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

from load_data.convert import MAT_FILE, TrialData  # noqa: E402
from models.sindy import (  # noqa: E402
  SINDyConfig,
  count_subthreshold_terms,
  equation_text,
  fit_with_iteration_count,
)
from models.validation import (  # noqa: E402
  SimulationConfig,
  evaluate_simulation,
  simulate_model_detailed_hard_timeout,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scenario_config as cfg  # noqa: E402
from scenario_probe import (  # noqa: E402
  calibrate_threshold,
  load_split,
  prepare_group,
  stability_verdict,
)

# --- Probe grid -------------------------------------------------------------
# Signal normalization is fixed; the varying axes are degree, the STLSQ
# normalize_columns regime (which interacts with unbiasing, since the refit
# happens in normalized-column space before rescaling), and alpha.
SIGNAL_NORMALIZATION = "global_zscore"
DEGREES = cfg.DEGREES
NORMALIZE_COLUMNS = [True, False]
ALPHAS = [0.05, 0.0]
UNBIAS_ARMS = [True, False]

OUTPUT_DIR = _PROJECT_ROOT / "outputs/pysindy/unbias_probe"


@dataclass(frozen=True)
class ArmResult:
  """Fit, simulation, and diagnostics for one ``unbias`` arm.

  Attributes:
    unbias: Whether the unregularized refit was applied.
    coefficients: Fitted coefficient matrix, shape ``(n_targets, n_features)``.
    n_iterations: STLSQ outer iterations, or ``None`` for non-STLSQ optimizers.
    converged: Whether STLSQ's support stabilized before ``max_iter``.
    nonzero_terms: Count of surviving coefficients.
    subthreshold_count: Survivors whose reported magnitude is below threshold.
    equations: Human-readable fitted equations.
    sim_x0: Simulated first delay coordinate, or ``None`` if integration failed.
    verdict: Compact stability verdict from ``stability_verdict``.
    completed: Whether integration reached the requested horizon.
    failure_reason: Integrator failure description, empty when successful.
    metrics: Descriptive simulation metrics, or ``None`` when unavailable.
  """

  unbias: bool
  coefficients: np.ndarray
  n_iterations: int | None
  converged: bool | None
  nonzero_terms: int
  subthreshold_count: int
  equations: str
  sim_x0: np.ndarray | None
  verdict: str
  completed: bool
  failure_reason: str
  metrics: dict[str, float | bool | None] | None


def support_of(coefficients: np.ndarray, tol: float = 1e-9) -> np.ndarray:
  """Return the boolean support mask of a coefficient matrix.

  Args:
    coefficients: Coefficient matrix in the model's reported units.
    tol: Magnitude below which a coefficient counts as zero.

  Returns:
    Boolean array with the same shape as ``coefficients``.
  """
  return np.abs(np.asarray(coefficients, dtype=float)) > tol


def coefficient_shift(
  unbiased: np.ndarray,
  biased: np.ndarray,
) -> tuple[bool, float, float]:
  """Quantify how far ridge shrinkage moved coefficients from the OLS refit.

  The unbiased (OLS) fit is the reference, since it is the estimator the
  original STLSQ formulation reports. Relative change is measured only on the
  shared support, where both arms have a value to compare.

  Args:
    unbiased: Coefficients from the ``unbias=True`` arm.
    biased: Coefficients from the ``unbias=False`` arm.

  Returns:
    ``(supports_match, max_relative_shift, coefficient_norm_ratio)``. The
    relative shift is ``max |biased - unbiased| / |unbiased|`` over the shared
    support and is ``0.0`` when that support is empty. The norm ratio is
    ``||biased|| / ||unbiased||``; ridge shrinkage drives it below one.
  """
  unbiased = np.asarray(unbiased, dtype=float)
  biased = np.asarray(biased, dtype=float)
  mask_u, mask_b = support_of(unbiased), support_of(biased)
  supports_match = bool(np.array_equal(mask_u, mask_b))

  shared = mask_u & mask_b
  if not np.any(shared):
    max_relative = 0.0
  else:
    max_relative = float(
      np.max(np.abs(biased[shared] - unbiased[shared]) / np.abs(unbiased[shared]))
    )

  norm_u = float(np.linalg.norm(unbiased))
  norm_ratio = float(np.linalg.norm(biased) / norm_u) if norm_u > 0 else float("nan")
  return supports_match, max_relative, norm_ratio


def run_arm(
  embedded_train: list[np.ndarray],
  embedded_test: np.ndarray,
  dt: float,
  fs_processed: float,
  degree: int,
  threshold: float,
  normalize_columns: bool,
  alpha: float,
  unbias: bool,
) -> ArmResult:
  """Fit one configuration at one ``unbias`` setting and simulate the test trial.

  Args:
    embedded_train: Delay-embedded training trajectories.
    embedded_test: Delay-embedded held-out trajectory to simulate against.
    dt: Processed sample interval in seconds.
    fs_processed: Processed sampling rate in Hz.
    degree: Polynomial library degree.
    threshold: STLSQ coefficient-removal threshold.
    normalize_columns: STLSQ ``normalize_columns`` setting.
    alpha: STLSQ ridge regularization strength.
    unbias: Whether to apply the unregularized refit on the selected support.

  Returns:
    Fit diagnostics, simulated trajectory, and evaluation metrics for this arm.
  """
  config = SINDyConfig(
    degree=degree,
    threshold=threshold,
    alpha=alpha,
    normalize_columns=normalize_columns,
    smooth_window=cfg.SMOOTH_WINDOW,
    smoothing_polyorder=cfg.SMOOTHING_POLYORDER,
    verbose=False,
    max_iter=cfg.MAX_ITER,
    unbias=unbias,
  )
  model, n_iterations, converged = fit_with_iteration_count(
    embedded_train, dt=dt, config=config
  )
  nonzero_terms, subthreshold_count = count_subthreshold_terms(model, threshold)

  horizon_s = min((len(embedded_test) - 1) * dt, cfg.MAX_SIMULATION_HORIZON_S)
  sim = simulate_model_detailed_hard_timeout(
    model,
    initial_state=embedded_test[0],
    dt=dt,
    horizon_s=horizon_s,
    wall_timeout_s=cfg.SIMULATION_WALL_TIMEOUT_S,
  )

  metrics = None
  if sim.trajectory is not None and sim.completed:
    metrics = evaluate_simulation(
      embedded_test,
      sim.trajectory,
      fs=fs_processed,
      config=SimulationConfig(
        simulation_horizon_s=horizon_s,
        divergence_threshold_std=cfg.DIVERGENCE_THRESHOLD_STD,
        divergence_persistence_s=cfg.DIVERGENCE_PERSISTENCE_S,
      ),
    )

  return ArmResult(
    unbias=unbias,
    coefficients=np.asarray(model.coefficients(), dtype=float),
    n_iterations=n_iterations,
    converged=converged,
    nonzero_terms=nonzero_terms,
    subthreshold_count=subthreshold_count,
    equations=equation_text(model),
    sim_x0=sim.trajectory[:, 0] if sim.trajectory is not None else None,
    verdict=stability_verdict(sim.completed, metrics),
    completed=sim.completed,
    failure_reason=sim.failure_reason,
    metrics=metrics,
  )


def run() -> None:
  """Execute the unbias probe and write CSV, equations, and plots."""
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  train_ids, test_ids = load_split(cfg.SPLIT_METADATA_GLOB)
  if cfg.SIMULATION_TEST_TRIAL not in test_ids:
    raise ValueError(
      f"SIMULATION_TEST_TRIAL={cfg.SIMULATION_TEST_TRIAL} not in test set {test_ids}"
    )

  print(f"Loading {MAT_FILE} ...", flush=True)
  data = TrialData.load(MAT_FILE)
  dt = cfg.DOWNSAMPLE / data.fs
  fs_processed = 1.0 / dt
  print(
    f"dt={dt:g}s  fs_processed={fs_processed:g}Hz  "
    f"train={len(train_ids)} test_trial={cfg.SIMULATION_TEST_TRIAL}"
  )

  group = prepare_group(
    data, train_ids, cfg.SIMULATION_TEST_TRIAL, SIGNAL_NORMALIZATION
  )

  # One threshold per normalize_columns regime, held fixed across both unbias
  # arms. Safe because unbiasing provably cannot alter the selected support.
  thresholds: dict[bool, float] = {}
  print(
    "\nCalibrating thresholds (target ~"
    f"{cfg.CALIBRATION_TARGET_TERMS} terms at degree {cfg.CALIBRATION_DEGREE}):"
  )
  for nc in NORMALIZE_COLUMNS:
    thr, _table = calibrate_threshold(
      group.embedded_train,
      dt=dt,
      degree=cfg.CALIBRATION_DEGREE,
      normalize_columns=nc,
      target_terms=cfg.CALIBRATION_TARGET_TERMS,
      n_candidates=cfg.CALIBRATION_N_CANDIDATES,
    )
    thresholds[nc] = thr
    print(f"  normalize_columns={str(nc):>5}: threshold = {thr:.6g}")

  rows: list[dict] = []
  # panels[(degree, nc, alpha)] -> (true_x0, {unbias: (sim_x0, verdict)})
  panels: dict[tuple[int, bool, float], tuple[np.ndarray, dict[bool, tuple]]] = {}
  test = group.embedded_test

  print("\nFitting and simulating:")
  for degree in DEGREES:
    for nc in NORMALIZE_COLUMNS:
      for alpha in ALPHAS:
        threshold = thresholds[nc]
        arms = {
          unbias: run_arm(
            group.embedded_train, test, dt, fs_processed,
            degree=degree, threshold=threshold, normalize_columns=nc,
            alpha=alpha, unbias=unbias,
          )
          for unbias in UNBIAS_ARMS
        }
        supports_match, max_rel_shift, norm_ratio = coefficient_shift(
          arms[True].coefficients, arms[False].coefficients
        )
        panels[(degree, nc, alpha)] = (
          test[:, 0],
          {u: (arms[u].sim_x0, arms[u].verdict) for u in UNBIAS_ARMS},
        )

        for unbias in UNBIAS_ARMS:
          arm = arms[unbias]
          metrics = arm.metrics
          rows.append({
            "degree": degree,
            "normalize_columns": nc,
            "alpha": alpha,
            "unbias": unbias,
            "threshold": threshold,
            "n_iterations": arm.n_iterations,
            "converged": arm.converged,
            "nonzero_terms": arm.nonzero_terms,
            "subthreshold_count": arm.subthreshold_count,
            "supports_match_across_arms": supports_match,
            "max_relative_coef_shift": max_rel_shift,
            "coef_norm_ratio_biased_over_unbiased": norm_ratio,
            "sim_verdict": arm.verdict,
            "sim_completed": arm.completed,
            "failure_reason": arm.failure_reason,
            "x0_correlation": None if metrics is None else metrics["x0_correlation"],
            "psd_similarity": None if metrics is None else metrics["psd_similarity"],
            "max_amplitude_ratio": (
              None if metrics is None else metrics["max_amplitude_ratio"]
            ),
            "collapse_std_ratio": (
              None if metrics is None else metrics["collapse_std_ratio"]
            ),
            "equations": arm.equations,
          })

        label = f"deg{degree} nc={str(nc):<5} a={alpha:<5}"
        print(
          f"  {label} support_match={supports_match} "
          f"max_rel_shift={max_rel_shift:.3g} norm_ratio={norm_ratio:.3g} "
          f"| unbias=True -> {arms[True].verdict:<9} "
          f"unbias=False -> {arms[False].verdict}"
        )

  _write_csv(rows)
  _write_equations(rows)
  _write_plot(panels, dt)
  _print_summary(rows)


def _write_csv(rows: list[dict]) -> None:
  """Write one row per (degree x normalize_columns x alpha x unbias) run."""
  path = OUTPUT_DIR / "unbias_probe.csv"
  with open(path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
  print(f"\nwrote {path}")


def _write_equations(rows: list[dict]) -> None:
  """Write a readable per-run equation dump for side-by-side comparison."""
  path = OUTPUT_DIR / "equations.txt"
  lines = []
  for row in rows:
    lines.append(
      f"[deg{row['degree']} nc={row['normalize_columns']} "
      f"alpha={row['alpha']} unbias={row['unbias']}] "
      f"thr={row['threshold']:.4g} nz={row['nonzero_terms']} "
      f"verdict={row['sim_verdict']}"
    )
    lines.append(f"  {row['equations']}")
    lines.append("")
  path.write_text("\n".join(lines))
  print(f"wrote {path}")


def _write_plot(
  panels: dict[tuple[int, bool, float], tuple[np.ndarray, dict[bool, tuple]]],
  dt: float,
) -> None:
  """Write a degree x (normalize_columns, alpha) grid overlaying both arms."""
  columns = [(nc, alpha) for nc in NORMALIZE_COLUMNS for alpha in ALPHAS]
  n_rows, n_cols = len(DEGREES), len(columns)
  fig, axes = plt.subplots(
    n_rows, n_cols, figsize=(3.4 * n_cols, 2.4 * n_rows),
    squeeze=False, sharex=True,
  )
  for r, degree in enumerate(DEGREES):
    for c, (nc, alpha) in enumerate(columns):
      ax = axes[r][c]
      true_x0, arms = panels[(degree, nc, alpha)]
      t_true = np.arange(true_x0.size) * dt
      ax.plot(t_true, true_x0, color="black", lw=0.9, label="true", zorder=3)

      # Clip divergent traces so the measured signal stays legible.
      cap = 6 * (np.std(true_x0) if np.std(true_x0) > 0 else 1.0)
      cap += float(np.abs(true_x0).max())
      styles = {
        True: dict(color="tab:blue", lw=1.0, ls="-", label="unbias=True"),
        False: dict(color="tab:red", lw=1.0, ls="--", label="unbias=False"),
      }
      verdicts = []
      for unbias in UNBIAS_ARMS:
        sim_x0, verdict = arms[unbias]
        verdicts.append(f"{'T' if unbias else 'F'}:{verdict}")
        if sim_x0 is None:
          continue
        finite = sim_x0[np.isfinite(sim_x0)]
        plotted = (
          np.clip(sim_x0, -cap, cap)
          if finite.size and np.abs(finite).max() > cap
          else sim_x0
        )
        ax.plot(np.arange(sim_x0.size) * dt, plotted, alpha=0.85, **styles[unbias])

      ax.set_title(
        f"deg{degree}  nc={nc}  alpha={alpha}\n[{'  '.join(verdicts)}]", fontsize=7
      )
      ax.tick_params(labelsize=6)
      if c == 0:
        ax.set_ylabel("x0 (z-scored)", fontsize=7)
      if r == n_rows - 1:
        ax.set_xlabel("time (s)", fontsize=7)
      if r == 0 and c == 0:
        ax.legend(fontsize=6, loc="upper right")
  fig.tight_layout()
  path = OUTPUT_DIR / "unbias_trajectories.png"
  fig.savefig(path, dpi=150)
  plt.close(fig)
  print(f"wrote {path}")


def _print_summary(rows: list[dict]) -> None:
  """Print the control check and a verdict-change tally."""
  print("\n--- Summary ---")

  mismatched = [r for r in rows if not r["supports_match_across_arms"]]
  if mismatched:
    print(f"WARNING: {len(mismatched)} configurations had differing supports "
          "across arms (unbiasing should never change support).")
  else:
    print("Support identical across arms in every configuration (as predicted).")

  control = [r for r in rows if r["alpha"] == 0.0]
  worst_control = max((r["max_relative_coef_shift"] for r in control), default=0.0)
  print(f"alpha=0 control: max relative coefficient shift = {worst_control:.3g} "
        "(should be ~0)")

  treated = [r for r in rows if r["alpha"] != 0.0 and r["unbias"]]
  if treated:
    worst = max(treated, key=lambda r: r["max_relative_coef_shift"])
    print(
      f"alpha>0: largest coefficient shift = {worst['max_relative_coef_shift']:.3g} "
      f"at deg{worst['degree']} nc={worst['normalize_columns']}"
    )

  changed = []
  by_key: dict[tuple, dict[bool, str]] = {}
  for row in rows:
    key = (row["degree"], row["normalize_columns"], row["alpha"])
    by_key.setdefault(key, {})[row["unbias"]] = row["sim_verdict"]
  for key, verdicts in sorted(by_key.items()):
    if verdicts.get(True) != verdicts.get(False):
      changed.append((key, verdicts))
  print(f"\nVerdict changed in {len(changed)}/{len(by_key)} configurations:")
  for (degree, nc, alpha), verdicts in changed:
    print(f"  deg{degree} nc={nc} alpha={alpha}: "
          f"unbias=True -> {verdicts[True]}, unbias=False -> {verdicts[False]}")


if __name__ == "__main__":
  run()
