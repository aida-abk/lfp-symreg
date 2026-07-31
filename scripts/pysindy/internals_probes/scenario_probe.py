"""Run the z-score / normalize-columns / alpha scenario probe.

For every (degree x scenario) combination this fits one autonomous SINDy model,
records how STLSQ behaved (iteration count, convergence warning, coefficients
kept below the threshold), simulates a single held-out trial, and reports a
stability verdict. Two thresholds are calibrated first -- one per signal
normalization group -- because z-scoring shifts coefficient magnitudes by
orders of magnitude.

All experiment settings live in ``scenario_config.py``; the canonical model
code in ``models/sindy.py`` is not modified.

Example:
    .venv/bin/python scripts/pysindy/scenario_probe.py
"""
from __future__ import annotations

import argparse
import csv
import json
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
from load_data.preprocessing import (  # noqa: E402
  apply_global_zscore,
  channel_traces,
  compute_global_zscore_stats,
)
from models.sindy import (  # noqa: E402
  SINDyConfig,
  count_subthreshold_terms,
  delay_embed_trace,
  delay_embed_trajectories,
  equation_text,
  fit_sindy_model,
  fit_with_iteration_count,
  maximum_polynomial_terms,
)
from models.validation import (  # noqa: E402
  SimulationConfig,
  evaluate_simulation,
  simulate_model_detailed_hard_timeout,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scenario_config as cfg  # noqa: E402

csv.field_size_limit(10 * 1024 * 1024)


@dataclass
class PreparedSignals:
  """Delay-embedded train trajectories and one test trajectory for a group.

  Attributes:
    embedded_train: One ``(time, n_delays)`` array per training trial.
    embedded_test: One ``(time, n_delays)`` array for the held-out trial.
  """

  embedded_train: list[np.ndarray]
  embedded_test: np.ndarray


def load_split(metadata_dir: Path) -> tuple[list[int], list[int]]:
  """Return (train_ids, test_ids) from the first metadata JSON in a directory.

  Args:
    metadata_dir: Directory holding ``*_metadata.json`` split files.

  Returns:
    Training and test trial identifier lists.
  """
  candidates = sorted(metadata_dir.glob("*_metadata.json"))
  if not candidates:
    raise FileNotFoundError(f"No *_metadata.json found under {metadata_dir}")
  split = json.loads(candidates[0].read_text())["split"]
  return split["train_trial_ids"], split["test_trial_ids"]


def prepare_group(
  data: TrialData,
  train_ids: list[int],
  test_id: int,
  signal_normalization: str,
) -> PreparedSignals:
  """Preprocess, optionally z-score, and delay-embed one signal group.

  Args:
    data: Loaded trial data.
    train_ids: Training trial identifiers.
    test_id: Single held-out trial identifier to simulate.
    signal_normalization: ``"none"`` or ``"global_zscore"``.

  Returns:
    Delay-embedded train and test trajectories for the requested group.
  """
  raw_train = channel_traces(
    data, channel=cfg.CHANNEL, trials=train_ids,
    downsample=cfg.DOWNSAMPLE, lowpass_hz=cfg.LOWPASS_HZ, normalize="none",
  )
  raw_test = channel_traces(
    data, channel=cfg.CHANNEL, trials=[test_id],
    downsample=cfg.DOWNSAMPLE, lowpass_hz=cfg.LOWPASS_HZ, normalize="none",
  )[0]

  if signal_normalization == "global_zscore":
    stats = compute_global_zscore_stats(raw_train, channel=cfg.CHANNEL)
    train_traces = apply_global_zscore(raw_train, stats)
    test_trace = apply_global_zscore([raw_test], stats)[0]
  elif signal_normalization == "none":
    train_traces = raw_train
    test_trace = raw_test
  else:
    raise ValueError(f"Unknown signal_normalization: {signal_normalization}")

  embedded_train = delay_embed_trajectories(
    train_traces, n_delays=cfg.N_DELAYS, delay=cfg.DELAY_SAMPLES
  )
  embedded_test = delay_embed_trace(
    test_trace, n_delays=cfg.N_DELAYS, delay=cfg.DELAY_SAMPLES
  )
  return PreparedSignals(embedded_train=embedded_train, embedded_test=embedded_test)


def _fit_nz(
  embedded_train: list[np.ndarray],
  dt: float,
  degree: int,
  threshold: float,
  normalize_columns: bool,
) -> tuple[object, int]:
  """Fit the baseline (alpha=0.05) at one threshold; return (model, term count)."""
  config = SINDyConfig(
    degree=degree, threshold=threshold, alpha=0.05,
    normalize_columns=normalize_columns,
    smooth_window=cfg.SMOOTH_WINDOW, smoothing_polyorder=cfg.SMOOTHING_POLYORDER,
    verbose=False, max_iter=cfg.MAX_ITER,
  )
  model = fit_sindy_model(embedded_train, dt=dt, config=config)
  return model, int(np.count_nonzero(np.asarray(model.coefficients())))


def calibrate_threshold(
  embedded_train: list[np.ndarray],
  dt: float,
  degree: int,
  normalize_columns: bool,
  target_terms: int,
  n_candidates: int,
) -> tuple[float, list[tuple[float, int]]]:
  """Sweep thresholds and pick the one whose refit keeps ~``target_terms`` terms.

  Chosen by the *actual* refit term count rather than derived from reported
  coefficient magnitudes, because ``normalize_columns=True`` makes STLSQ
  threshold in a space that is unit-incompatible with the rescaled coefficients
  it reports. Ties in closeness to the target prefer the larger (sparser,
  more numerically stable) threshold.

  Args:
    embedded_train: Delay-embedded training trajectories for the regime.
    dt: Processed sample interval in seconds.
    degree: Polynomial degree used for calibration.
    normalize_columns: STLSQ ``normalize_columns`` for this regime.
    target_terms: Desired number of surviving coefficients.
    n_candidates: Number of geometrically spaced thresholds to try.

  Returns:
    The chosen threshold and the full ``(threshold, term_count)`` sweep table.
  """
  base_model, _ = _fit_nz(embedded_train, dt, degree, 0.0, normalize_columns)
  scale = float(np.max(np.abs(np.asarray(base_model.coefficients()))))
  if scale <= 0:
    return 0.0, []
  candidates = np.geomspace(scale * 1e-3, scale * 1e3, n_candidates)
  table = [
    (float(thr), _fit_nz(embedded_train, dt, degree, float(thr), normalize_columns)[1])
    for thr in candidates
  ]
  chosen = min(table, key=lambda pair: (abs(pair[1] - target_terms), -pair[0]))
  return chosen[0], table


def stability_verdict(
  completed: bool,
  metrics: dict[str, float | bool | None] | None,
) -> str:
  """Summarize simulation outcome as a compact verdict string.

  Args:
    completed: Whether integration reached the requested horizon.
    metrics: Descriptive metrics from ``evaluate_simulation``, or ``None`` when
      no usable trajectory was produced.

  Returns:
    One of ``"failed"``, ``"diverged"``, ``"collapsed"``, or ``"ok"``.
  """
  if not completed or metrics is None:
    return "failed"
  amp = metrics.get("max_amplitude_ratio")
  collapse = metrics.get("collapse_std_ratio")
  if metrics.get("diverged") or (amp is not None and amp > 100):
    return "diverged"
  if collapse is not None and collapse < 0.02:
    return "collapsed"
  return "ok"


def run() -> None:
  """Execute the full scenario probe and write CSV, equations, and plots."""
  cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  train_ids, test_ids = load_split(cfg.SPLIT_METADATA_GLOB)
  if cfg.SIMULATION_TEST_TRIAL not in test_ids:
    raise ValueError(
      f"SIMULATION_TEST_TRIAL={cfg.SIMULATION_TEST_TRIAL} not in test set {test_ids}"
    )

  print(f"Loading {MAT_FILE} ...", flush=True)
  data = TrialData.load(MAT_FILE)
  dt = cfg.DOWNSAMPLE / data.fs
  fs_processed = 1.0 / dt
  print(f"dt={dt:g}s  fs_processed={fs_processed:g}Hz  "
        f"train={len(train_ids)} test_trial={cfg.SIMULATION_TEST_TRIAL}")

  # Prepare each signal group once and reuse across scenarios/degrees.
  groups = {
    name: prepare_group(data, train_ids, cfg.SIMULATION_TEST_TRIAL, name)
    for name in {s.signal_normalization for s in cfg.SCENARIOS}
  }

  # Calibrate one threshold per (signal_normalization, normalize_columns) regime.
  regimes = sorted({(s.signal_normalization, s.normalize_columns) for s in cfg.SCENARIOS})
  thresholds: dict[tuple[str, bool], float] = {}
  print("\nCalibrating thresholds (target ~"
        f"{cfg.CALIBRATION_TARGET_TERMS} terms at degree {cfg.CALIBRATION_DEGREE}):")
  for signal_norm, nc in regimes:
    thr, table = calibrate_threshold(
      groups[signal_norm].embedded_train, dt=dt, degree=cfg.CALIBRATION_DEGREE,
      normalize_columns=nc, target_terms=cfg.CALIBRATION_TARGET_TERMS,
      n_candidates=cfg.CALIBRATION_N_CANDIDATES,
    )
    thresholds[(signal_norm, nc)] = thr
    sweep = "  ".join(f"{t:.3g}:{n}" for t, n in table[::4])
    print(f"  regime signal={signal_norm:>13} nc={str(nc):>5}: threshold = {thr:.6g}")
    print(f"      sweep (threshold:nz) {sweep}")

  rows: list[dict] = []
  # panels[(degree, scenario_name)] -> (true_x0, sim_x0_or_None, verdict)
  panels: dict[tuple[int, str], tuple[np.ndarray, np.ndarray | None, str]] = {}

  for degree in cfg.DEGREES:
    for scenario in cfg.SCENARIOS:
      group = groups[scenario.signal_normalization]
      threshold = thresholds[(scenario.signal_normalization, scenario.normalize_columns)]
      label = f"deg{degree}/{scenario.name}"

      config = SINDyConfig(
        degree=degree, threshold=threshold, alpha=scenario.alpha,
        normalize_columns=scenario.normalize_columns,
        smooth_window=cfg.SMOOTH_WINDOW, smoothing_polyorder=cfg.SMOOTHING_POLYORDER,
        verbose=False, max_iter=cfg.MAX_ITER,
      )

      model, n_iterations, converged = fit_with_iteration_count(
        group.embedded_train, dt=dt, config=config
      )
      nonzero_terms, subthreshold_count = count_subthreshold_terms(model, threshold)
      possible_terms = maximum_polynomial_terms(cfg.N_DELAYS, degree)
      equations = equation_text(model)

      test = group.embedded_test
      horizon_s = min((len(test) - 1) * dt, cfg.MAX_SIMULATION_HORIZON_S)
      sim = simulate_model_detailed_hard_timeout(
        model, initial_state=test[0], dt=dt, horizon_s=horizon_s,
        wall_timeout_s=cfg.SIMULATION_WALL_TIMEOUT_S,
      )

      metrics = None
      if sim.trajectory is not None and sim.completed:
        metrics = evaluate_simulation(
          test, sim.trajectory, fs=fs_processed,
          config=SimulationConfig(
            simulation_horizon_s=horizon_s,
            divergence_threshold_std=cfg.DIVERGENCE_THRESHOLD_STD,
            divergence_persistence_s=cfg.DIVERGENCE_PERSISTENCE_S,
          ),
        )
      verdict = stability_verdict(sim.completed, metrics)
      sim_x0 = sim.trajectory[:, 0] if sim.trajectory is not None else None
      panels[(degree, scenario.name)] = (test[:, 0], sim_x0, verdict)

      rows.append({
        "degree": degree,
        "scenario": scenario.name,
        "signal_normalization": scenario.signal_normalization,
        "normalize_columns": scenario.normalize_columns,
        "alpha": scenario.alpha,
        "threshold": threshold,
        "n_iterations": n_iterations,
        "converged": converged,
        "convergence_warning": (converged is False),
        "nonzero_terms": nonzero_terms,
        "subthreshold_count": subthreshold_count,
        "possible_terms": possible_terms,
        "sim_verdict": verdict,
        "sim_completed": sim.completed,
        "failure_reason": sim.failure_reason,
        "x0_correlation": None if metrics is None else metrics["x0_correlation"],
        "psd_similarity": None if metrics is None else metrics["psd_similarity"],
        "max_amplitude_ratio": None if metrics is None else metrics["max_amplitude_ratio"],
        "collapse_std_ratio": None if metrics is None else metrics["collapse_std_ratio"],
        "equations": equations,
      })
      print(f"  {label:<28} iters={n_iterations} conv={converged} "
            f"nz={nonzero_terms} sub={subthreshold_count} -> {verdict}")

  _write_csv(rows)
  _write_equations(rows)
  _write_plot(panels, dt)
  _print_summary(rows, thresholds)


def _write_csv(rows: list[dict]) -> None:
  """Write one row per (degree x scenario) to ``scenario_probe.csv``."""
  path = cfg.OUTPUT_DIR / "scenario_probe.csv"
  with open(path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
  print(f"\nwrote {path}")


def _write_equations(rows: list[dict]) -> None:
  """Write a readable per-run equation dump to ``equations.txt``."""
  path = cfg.OUTPUT_DIR / "equations.txt"
  lines = []
  for row in rows:
    lines.append(
      f"[deg{row['degree']} {row['scenario']}] "
      f"thr={row['threshold']:.4g} nz={row['nonzero_terms']} "
      f"sub={row['subthreshold_count']} verdict={row['sim_verdict']}"
    )
    lines.append(f"  {row['equations']}")
    lines.append("")
  path.write_text("\n".join(lines))
  print(f"wrote {path}")


def _write_plot(
  panels: dict[tuple[int, str], tuple[np.ndarray, np.ndarray | None, str]],
  dt: float,
) -> None:
  """Write a degree x scenario grid of simulated-vs-true x0 trajectories."""
  scenario_names = [s.name for s in cfg.SCENARIOS]
  degrees = cfg.DEGREES
  n_rows, n_cols = len(degrees), len(scenario_names)
  fig, axes = plt.subplots(
    n_rows, n_cols, figsize=(3.0 * n_cols, 2.2 * n_rows),
    squeeze=False, sharex=True,
  )
  for r, degree in enumerate(degrees):
    for c, name in enumerate(scenario_names):
      ax = axes[r][c]
      true_x0, sim_x0, verdict = panels[(degree, name)]
      t_true = np.arange(true_x0.size) * dt
      ax.plot(t_true, true_x0, color="black", lw=0.8, label="true")
      if sim_x0 is not None:
        t_sim = np.arange(sim_x0.size) * dt
        color = {"ok": "tab:green", "collapsed": "tab:orange",
                 "diverged": "tab:red", "failed": "tab:red"}.get(verdict, "tab:blue")
        # Clip wildly divergent traces so the true signal stays visible.
        finite = sim_x0[np.isfinite(sim_x0)]
        cap = 6 * (np.std(true_x0) if np.std(true_x0) > 0 else 1.0) + np.abs(true_x0).max()
        plotted = np.clip(sim_x0, -cap, cap) if finite.size and np.abs(finite).max() > cap else sim_x0
        ax.plot(t_sim, plotted, color=color, lw=0.8, alpha=0.9, label="sim")
      ax.set_title(f"deg{degree} {name}\n[{verdict}]", fontsize=6)
      ax.tick_params(labelsize=5)
      if c == 0:
        ax.set_ylabel("x0", fontsize=6)
      if r == n_rows - 1:
        ax.set_xlabel("time (s)", fontsize=6)
  fig.tight_layout()
  path = cfg.OUTPUT_DIR / "scenario_trajectories.png"
  fig.savefig(path, dpi=150)
  plt.close(fig)
  print(f"wrote {path}")


def _print_summary(rows: list[dict], thresholds: dict[str, float]) -> None:
  """Print a compact console summary grouped by degree."""
  print("\n===== SUMMARY =====")
  print("thresholds:", {k: round(v, 6) for k, v in thresholds.items()})
  header = (f"{'degree':>6} {'scenario':<18} {'iters':>5} {'conv':>5} "
            f"{'nz':>4} {'sub':>4} {'verdict':>9} {'x0corr':>7} {'psd':>6}")
  print(header)
  for row in rows:
    x0c = row["x0_correlation"]
    psd = row["psd_similarity"]
    print(f"{row['degree']:>6} {row['scenario']:<18} "
          f"{str(row['n_iterations']):>5} {str(row['converged']):>5} "
          f"{row['nonzero_terms']:>4} {row['subthreshold_count']:>4} "
          f"{row['sim_verdict']:>9} "
          f"{'--' if x0c is None else f'{x0c:>7.3f}'} "
          f"{'--' if psd is None else f'{psd:>6.3f}'}")


def load_thresholds_from_csv(path: Path) -> dict[tuple[str, bool], float]:
  """Read per-regime thresholds recorded by a previous full run.

  Args:
    path: Path to a ``scenario_probe.csv`` written by :func:`run`.

  Returns:
    Mapping of ``(signal_normalization, normalize_columns)`` to threshold.
  """
  with open(path) as f:
    rows = list(csv.DictReader(f))
  if not rows:
    raise ValueError(f"{path} contains no rows; run the full probe first.")
  return {
    (row["signal_normalization"], row["normalize_columns"] == "True"): float(row["threshold"])
    for row in rows
  }


def run_verbose_fits() -> None:
  """Refit every configuration with STLSQ verbose=True and print its table.

  Reuses the thresholds calibrated by a previous :func:`run` so the printed
  iteration tables correspond exactly to the runs in ``scenario_probe.csv``.
  Simulation and calibration are skipped.
  """
  csv_path = cfg.OUTPUT_DIR / "scenario_probe.csv"
  thresholds = load_thresholds_from_csv(csv_path)
  print(f"Reusing thresholds from {csv_path}:")
  for (signal_norm, nc), thr in sorted(thresholds.items()):
    print(f"  signal={signal_norm:>13} nc={str(nc):>5} -> threshold={thr:.6g}")

  train_ids, _ = load_split(cfg.SPLIT_METADATA_GLOB)
  print(f"\nLoading {MAT_FILE} ...", flush=True)
  data = TrialData.load(MAT_FILE)
  dt = cfg.DOWNSAMPLE / data.fs
  groups = {
    name: prepare_group(data, train_ids, cfg.SIMULATION_TEST_TRIAL, name)
    for name in {s.signal_normalization for s in cfg.SCENARIOS}
  }

  print("\nSTLSQ verbose columns: Iteration | |y-Xw|^2 (data misfit) | "
        "a*|w|_2 (ridge penalty) | |w|_0 (surviving terms) | Total error\n")

  for degree in cfg.DEGREES:
    for scenario in cfg.SCENARIOS:
      threshold = thresholds[(scenario.signal_normalization, scenario.normalize_columns)]
      print("=" * 78)
      print(f"deg{degree} / {scenario.name}")
      print(f"  signal={scenario.signal_normalization}  "
            f"normalize_columns={scenario.normalize_columns}  "
            f"alpha={scenario.alpha}  threshold={threshold:.6g}  degree={degree}")
      print("=" * 78, flush=True)

      config = SINDyConfig(
        degree=degree, threshold=threshold, alpha=scenario.alpha,
        normalize_columns=scenario.normalize_columns,
        smooth_window=cfg.SMOOTH_WINDOW, smoothing_polyorder=cfg.SMOOTHING_POLYORDER,
        verbose=True, max_iter=cfg.MAX_ITER,
      )
      model, n_iterations, converged = fit_with_iteration_count(
        groups[scenario.signal_normalization].embedded_train, dt=dt, config=config
      )
      nonzero_terms, subthreshold_count = count_subthreshold_terms(model, threshold)
      print(f"  -> iterations={n_iterations}  converged={converged}  "
            f"nonzero={nonzero_terms}  subthreshold={subthreshold_count}\n", flush=True)


def parse_args() -> argparse.Namespace:
  """Parse CLI arguments."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--verbose-fits", action="store_true",
    help="Refit every configuration with STLSQ verbose=True and print its "
         "per-iteration table. Reuses thresholds from the existing "
         "scenario_probe.csv; skips calibration and simulation.",
  )
  return parser.parse_args()


if __name__ == "__main__":
  args = parse_args()
  if args.verbose_fits:
    run_verbose_fits()
  else:
    run()
