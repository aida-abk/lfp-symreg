"""Simulate and compare hand-picked sweep configurations under STLSQ overrides.

Background
----------
General-purpose replacement for the one-off alpha=0.05-vs-alpha=0 probe
(``normalization_validation/probe_alpha0_subthreshold.py`` /
``simulate_alpha0_subthreshold.py``). Configuration selection now happens by
eye in the ``outputs/pysindy/global_analysis`` dashboard, not by an automated
"which configs are good" script (``scripts/pysindy/sweep_analysis/`` and
``run_analysis.py`` were archived for exactly this reason). This script takes
whatever ``configuration_index`` values you picked there, refits each one
under one or more (alpha, threshold, max_iter) variants, simulates every
selected held-out trial, and plots measured-vs-simulated for direct visual
comparison.

Example: does alpha=0 actually remove alpha=0.05's sub-threshold surviving
coefficients, and does that change simulated behavior?

    .venv/bin/python scripts/pysindy/simulate_selected_configurations.py \\
      --configuration-indices 29,64,173 \\
      --alpha-overrides 0.05,0.0
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(_PROJECT_ROOT))

from load_data.convert import LFP_AMPLITUDE_UNIT, MAT_FILE, TrialData
from load_data.preprocessing import (
  apply_global_zscore,
  channel_traces,
  compute_global_zscore_stats,
)
from models.sindy import SINDyConfig, delay_embed_trajectories, equation_text, fit_sindy_model
from models.validation import SimulationConfig, evaluate_simulation, simulate_model_detailed

csv.field_size_limit(10 * 1024 * 1024)

CHANNEL = 0
DOWNSAMPLE = 2
YLIM_MEASURED_MULTIPLE = 4.0
YLIM_FLOOR = 0.5

METRIC_KEYS = [
  "trajectory_rmse",
  "x0_correlation",
  "max_amplitude_ratio",
  "collapse_std_ratio",
  "psd_similarity",
  "distribution_ks",
]

DEFAULT_GRID_CSV = _PROJECT_ROOT / "outputs/pysindy/raw_grid_nc_false_gsz_t1/raw_grid_merged.csv"


@dataclass(frozen=True)
class Variant:
  """One (alpha, threshold, max_iter) refit to simulate for a configuration."""

  alpha: float
  threshold: float
  max_iter: int

  @property
  def label(self) -> str:
    """Human-readable legend label."""
    return f"alpha={self.alpha:g}, threshold={self.threshold:g}, max_iter={self.max_iter}"

  @property
  def short_label(self) -> str:
    """Compact label for space-constrained panel titles."""
    return f"a={self.alpha:g},t={self.threshold:g},mi={self.max_iter}"


def parse_float_list(text: str | None) -> list[float] | None:
  """Parse a comma-separated float list, or ``None`` if ``text`` is ``None``."""
  if text is None:
    return None
  return [float(x) for x in text.split(",") if x.strip()]


def parse_int_list(text: str | None) -> list[int] | None:
  """Parse a comma-separated int list, or ``None`` if ``text`` is ``None``."""
  if text is None:
    return None
  return [int(x) for x in text.split(",") if x.strip()]


def build_variants(
  row: dict[str, str],
  alpha_overrides: list[float] | None,
  threshold_overrides: list[float] | None,
  max_iter_overrides: list[int] | None,
) -> list[Variant]:
  """Build the cross product of alpha/threshold/max_iter overrides for one row.

  Falls back to the row's own stored alpha/threshold when no override is
  given, and to pysindy's own STLSQ default (20) when no max_iter override is
  given. Everything is refit freshly (never reconstructed from stored
  coefficients), since that's the only way to also vary max_iter.
  """
  alphas = alpha_overrides if alpha_overrides is not None else [float(row["alpha"])]
  thresholds = (
    threshold_overrides if threshold_overrides is not None
    else [float(row["stlsq_threshold"])]
  )
  max_iters = max_iter_overrides if max_iter_overrides is not None else [20]
  return [
    Variant(alpha=a, threshold=t, max_iter=mi)
    for a in alphas for t in thresholds for mi in max_iters
  ]


def empty_metrics() -> dict[str, float]:
  """Return NaN-filled metrics for a simulation that produced no trajectory."""
  return {key: float("nan") for key in METRIC_KEYS}


def mean_finite(values: list[float]) -> float:
  """Average the finite entries of ``values``, or NaN if none are finite."""
  finite = [v for v in values if np.isfinite(v)]
  return float(np.mean(finite)) if finite else float("nan")


def display_path(path: Path) -> str:
  """Show ``path`` relative to the project root, or absolute if it's elsewhere."""
  try:
    return str(path.relative_to(_PROJECT_ROOT))
  except ValueError:
    return str(path)


def count_terms(model, threshold: float) -> tuple[int, int]:
  """Count nonzero and sub-threshold-surviving coefficients in a fitted model.

  A "sub-threshold survivor" is a coefficient STLSQ kept (nonzero) whose final
  reported magnitude is nonetheless below the threshold it fit at -- possible
  because STLSQ selects support with the (possibly ridge) regression during
  fitting but reports the last refit's coefficients, which can differ. See
  ``normalization_validation/probe_alpha0_subthreshold.py`` for the original
  discovery.

  Returns:
    ``(nonzero_terms, subthreshold_count)``.
  """
  coefs = np.abs(np.asarray(model.coefficients(), dtype=float))
  nonzero_terms = int(np.sum(coefs > 1e-9))
  subthreshold_count = int(np.sum((coefs > 1e-9) & (coefs < threshold)))
  return nonzero_terms, subthreshold_count


def fit_with_iteration_count(trajectories, dt: float, config: SINDyConfig):
  """Fit a SINDy model and report how many STLSQ outer iterations it took.

  STLSQ appends one entry to its optimizer's ``history_`` per fit/threshold/
  refit cycle (fit/threshold/refit until the surviving support stops
  changing, or ``max_iter`` is reached -- see
  ``.venv/.../pysindy/optimizers/stlsq.py:_reduce``). ``history_`` is absent
  for non-STLSQ optimizers (e.g. SR3), in which case iteration count is
  ``None``. PySINDy raises ``ConvergenceWarning`` exactly when the support
  never stabilized within ``max_iter`` iterations; that warning is captured
  here rather than re-derived, since "reached max_iter" and "the support
  happened to stabilize on the very last allowed iteration" are otherwise
  indistinguishable from the history length alone.

  Returns:
    ``(model, n_iterations, converged)``. ``n_iterations`` and ``converged``
    are ``None`` when the optimizer exposes no ``history_``.
  """
  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    model = fit_sindy_model(trajectories, dt=dt, config=config)
  history = getattr(getattr(model, "optimizer", None), "history_", None)
  if history is None:
    return model, None, None
  converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
  return model, len(history), converged


def load_grid_rows(grid_csv: Path, configuration_indices: list[int]) -> list[dict]:
  """Load the requested configuration rows from a raw-grid CSV, in request order."""
  with open(grid_csv) as f:
    by_index = {row["configuration_index"]: row for row in csv.DictReader(f)}
  rows = []
  for idx in configuration_indices:
    key = str(idx)
    if key not in by_index:
      raise KeyError(f"configuration_index {idx} not found in {grid_csv}")
    rows.append(by_index[key])
  return rows


def _draw_variant_panel(
  axis,
  measured: np.ndarray,
  variant: Variant,
  sim,
  dt: float,
  signal_units: str,
) -> None:
  """Draw measured vs. exactly one variant's simulation onto ``axis``.

  One variant per panel -- no overlapping traces to disentangle. The y-axis
  is capped at a multiple of the measured signal's own amplitude so a
  diverging variant does not squash the measured-vs-simulated comparison into
  a flat line; if the trace exceeds the cap it is still drawn (matplotlib
  clips it at the axis edge) and its true peak is reported in the title.
  """
  measured_time = np.arange(measured.shape[0]) * dt
  measured_peak = float(np.max(np.abs(measured[:, 0]))) if measured.size else 0.0
  margin = max(YLIM_MEASURED_MULTIPLE * measured_peak, YLIM_FLOOR)

  axis.plot(
    measured_time, measured[:, 0],
    color="steelblue", linewidth=1.1, label=f"measured ({signal_units})",
  )

  status = "ok" if sim.completed else "failed"
  title = f"{variant.short_label}: {status}@{sim.reached_horizon_s:.2f}s"
  if sim.trajectory is not None and sim.trajectory.size:
    trace = sim.trajectory[:, 0]
    peak = float(np.max(np.abs(trace)))
    if peak > margin:
      title += f"  [peak {peak:.1f}, off-scale]"
    axis.plot(sim.time, trace, color="darkorange", linestyle="--", linewidth=1.1, label="simulated")

  axis.set_ylim(-margin, margin)
  axis.set_title(title, fontsize=9)
  axis.set_xlabel("Time (s)", fontsize=9)
  axis.set_ylabel(f"x0 ({signal_units})", fontsize=9)
  axis.grid(alpha=0.2)
  axis.legend(loc="upper right", fontsize=8)


def plot_trial_comparison(
  path: Path,
  row: dict[str, str],
  trial_id: int,
  measured: np.ndarray,
  sims: list[tuple[Variant, object]],
  dt: float,
  signal_units: str,
) -> None:
  """Plot one trial; one figure per (configuration, trial), one sub-panel per variant.

  Small multiples instead of overlaid traces -- every variant gets its own
  axis against the same measured trace, so variants that draw identically
  (e.g. max_iter values that converged to the same fit) are still visibly
  separate panels rather than indistinguishable stacked lines.
  """
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  columns = min(3, len(sims)) or 1
  n_rows = math.ceil(len(sims) / columns)
  figure, axes = plt.subplots(
    n_rows, columns,
    figsize=(4.5 * columns, 3.2 * n_rows),
    sharex=False, sharey=False, squeeze=False,
  )
  for axis, (variant, sim) in zip(axes.ravel(), sims):
    _draw_variant_panel(axis, measured, variant, sim, dt, signal_units)
  for axis in axes.ravel()[len(sims):]:
    axis.set_visible(False)

  figure.suptitle(
    f"Configuration {row['configuration_index']}, Trial {trial_id}: "
    f"LP={row['lowpass_hz']} Hz, degree={row['degree']}, delays={row['n_delays']}, "
    f"spacing={row['delay_samples']} samples, "
    f"smoothing={row['smooth_window_samples']} samples",
    fontsize=10,
  )
  figure.tight_layout(rect=(0, 0, 1, 0.93))
  path.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(path, dpi=160)
  plt.close(figure)


def process_configuration(
  row: dict[str, str],
  variants: list[Variant],
  train_z: dict[float, list[np.ndarray]],
  test_z: dict[float, list[np.ndarray]],
  test_ids: list[int],
  selected_trial_ids: list[int],
  dt: float,
  fs_processed: float,
  horizon_s: float,
  trial_timeout_s: float,
  output_dir: Path,
  figure_format: str,
  normalize_columns: bool,
  signal_units: str,
) -> tuple[list[dict], list[dict]]:
  """Fit every variant for one configuration, simulate selected trials, and plot.

  Returns:
    ``(trial_rows, summary_rows)`` for this one configuration.
  """
  lp = float(row["lowpass_hz"])
  n_delays, delay = int(row["n_delays"]), int(row["delay_samples"])
  smooth = int(row["smooth_window_samples"])
  degree = int(row["degree"])

  embedded_train = delay_embed_trajectories(train_z[lp], n_delays=n_delays, delay=delay)
  embedded_test_by_trial = dict(zip(
    test_ids, delay_embed_trajectories(test_z[lp], n_delays=n_delays, delay=delay)
  ))
  trial_ids = [t for t in selected_trial_ids if t in embedded_test_by_trial]

  models: dict[Variant, object] = {}
  fit_iterations: dict[Variant, int | None] = {}
  fit_converged: dict[Variant, bool | None] = {}
  for variant in variants:
    model, n_iterations, converged = fit_with_iteration_count(
      embedded_train, dt=dt,
      config=SINDyConfig(degree=degree, threshold=variant.threshold, alpha=variant.alpha,
                         normalize_columns=normalize_columns, smooth_window=smooth, smoothing_polyorder=3,
                         max_iter=variant.max_iter),
    )
    models[variant] = model
    fit_iterations[variant] = n_iterations
    fit_converged[variant] = converged

  sim_config = SimulationConfig(simulation_horizon_s=horizon_s)
  trial_rows = []
  metrics_by_variant: dict[Variant, list[dict]] = {v: [] for v in variants}
  completed_by_variant: dict[Variant, int] = {v: 0 for v in variants}
  plot_paths: list[Path] = []

  for trial_id in trial_ids:
    measured = embedded_test_by_trial[trial_id]
    if measured.shape[0] < 2:
      continue
    x0 = measured[0]
    trial_sims = []
    for variant in variants:
      sim = simulate_model_detailed(models[variant], initial_state=x0, dt=dt,
                                     horizon_s=horizon_s, wall_timeout_s=trial_timeout_s)
      metric = (
        empty_metrics() if sim.trajectory is None
        else evaluate_simulation(measured, sim.trajectory, fs=fs_processed, config=sim_config)
      )
      trial_rows.append({
        "configuration_index": row["configuration_index"], "degree": degree, "lowpass_hz": lp,
        "n_delays": n_delays, "delay_samples": delay, "smooth_window_samples": smooth,
        "alpha": variant.alpha, "threshold": variant.threshold, "max_iter": variant.max_iter,
        "trial_id": trial_id, "completed": sim.completed, "failure_reason": sim.failure_reason,
        "reached_horizon_s": sim.reached_horizon_s,
        **{key: metric[key] for key in METRIC_KEYS},
      })
      metrics_by_variant[variant].append(metric)
      if sim.completed:
        completed_by_variant[variant] += 1
      trial_sims.append((variant, sim))

    plot_path = (
      output_dir / "plots"
      / f"cfg{row['configuration_index']}_trial_{trial_id:04d}.{figure_format}"
    )
    plot_trial_comparison(plot_path, row, trial_id, measured, trial_sims, dt, signal_units)
    plot_paths.append(plot_path)

  summary_rows = []
  for variant in variants:
    metrics = metrics_by_variant[variant]
    n_completed = completed_by_variant[variant]
    nonzero_terms, subthreshold_count = count_terms(models[variant], variant.threshold)
    summary = {
      "configuration_index": row["configuration_index"], "degree": degree, "lowpass_hz": lp,
      "n_delays": n_delays, "delay_samples": delay, "smooth_window_samples": smooth,
      "alpha": variant.alpha, "threshold": variant.threshold, "max_iter": variant.max_iter,
      "nonzero_terms": nonzero_terms, "subthreshold_count": subthreshold_count,
      "fit_n_iterations": fit_iterations[variant], "fit_converged": fit_converged[variant],
      "equations": equation_text(models[variant]),
      "n_trials": len(metrics), "n_completed": n_completed, "n_failed": len(metrics) - n_completed,
    }
    for key in METRIC_KEYS:
      summary[f"mean_{key}"] = mean_finite([m[key] for m in metrics])
    summary_rows.append(summary)

  non_converged = [v.short_label for v in variants if fit_converged[v] is False]
  iteration_note = (
    f"  NOT CONVERGED: {', '.join(non_converged)}" if non_converged else ""
  )
  print(f"  cfg {row['configuration_index']:>4} deg={degree} lp={lp:>4.0f}  "
        f"variants={len(variants)}  {len(plot_paths)} plots in "
        f"{display_path(plot_paths[0].parent) if plot_paths else '(none)'}"
        f"{iteration_note}")
  return trial_rows, summary_rows


def parse_args() -> argparse.Namespace:
  """Parse CLI arguments for local or Slurm-array invocation."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--configuration-indices", type=str, required=True,
    help="Comma-separated configuration_index values to simulate, e.g. "
         "hand-picked from the global_analysis dashboard.",
  )
  parser.add_argument(
    "--configuration-index", type=int, default=None,
    help="1-based index into --configuration-indices, for one Slurm array task. "
         "Omit to run every listed configuration locally.",
  )
  parser.add_argument("--grid-csv", type=Path, default=DEFAULT_GRID_CSV)
  parser.add_argument(
    "--metadata-json", type=Path, default=None,
    help="Defaults to the first parts/*_metadata.json next to --grid-csv.",
  )
  parser.add_argument(
    "--alpha-overrides", type=str, default=None,
    help="Comma-separated alpha values to refit, e.g. '0.05,0.0'. "
         "Default: each configuration's own stored alpha (still refit fresh).",
  )
  parser.add_argument(
    "--threshold-overrides", type=str, default=None,
    help="Comma-separated threshold values to refit. "
         "Default: each configuration's own stored threshold.",
  )
  parser.add_argument(
    "--max-iter-overrides", type=str, default=None,
    help="Comma-separated STLSQ max_iter values to refit, e.g. '20,150'. "
         "Default: pysindy's own default of 20.",
  )
  parser.add_argument(
    "--trial-ids", type=str, default=None,
    help="Comma-separated held-out trial IDs to simulate/plot. "
         "Default: every held-out trial in the metadata split.",
  )
  parser.add_argument(
    "--max-trials", type=int, default=None,
    help="Cap the number of trials simulated/plotted (applied after --trial-ids).",
  )
  parser.add_argument("--trial-timeout-s", type=float, default=60.0)
  parser.add_argument("--horizon-s", type=float, default=2.0)
  parser.add_argument(
    "--output-dir", type=Path, default=None,
    help="Default: <grid-csv's parent>/configuration_probes/",
  )
  parser.add_argument("--figure-format", type=str, default="png")
  parser.add_argument(
    "--normalize-columns", action=argparse.BooleanOptionalAction, default=None,
    help="STLSQ normalize_columns for every refit. Default: auto-detected from "
         "--metadata-json's fixed_model_settings.normalize_columns (falls back "
         "to False if absent). Override to test a different scheme than the "
         "sweep was originally fit under.",
  )
  parser.add_argument(
    "--signal-normalization", type=str, default=None, choices=["none", "global_zscore"],
    help="Signal preprocessing before delay embedding. Default: auto-detected "
         "from --metadata-json's preprocessing.normalization. Override to test "
         "a different scheme than the sweep was originally preprocessed under.",
  )
  return parser.parse_args()


def main() -> None:
  """Refit selected configurations under override variants, simulate, and plot."""
  args = parse_args()
  args.grid_csv = args.grid_csv.resolve()
  if args.metadata_json is not None:
    args.metadata_json = args.metadata_json.resolve()
  if args.output_dir is not None:
    args.output_dir = args.output_dir.resolve()
  configuration_indices = parse_int_list(args.configuration_indices)
  if not configuration_indices:
    raise ValueError("--configuration-indices must list at least one configuration_index")

  if args.configuration_index is not None:
    if not 1 <= args.configuration_index <= len(configuration_indices):
      raise ValueError(f"--configuration-index must be in 1..{len(configuration_indices)}")
    configuration_indices = [configuration_indices[args.configuration_index - 1]]

  alpha_overrides = parse_float_list(args.alpha_overrides)
  threshold_overrides = parse_float_list(args.threshold_overrides)
  max_iter_overrides = parse_int_list(args.max_iter_overrides)
  requested_trial_ids = parse_int_list(args.trial_ids)

  metadata_json = args.metadata_json
  if metadata_json is None:
    candidates = sorted((args.grid_csv.parent / "parts").glob("*_metadata.json"))
    if not candidates:
      raise FileNotFoundError(f"No *_metadata.json found under {args.grid_csv.parent / 'parts'}")
    metadata_json = candidates[0]

  output_dir = args.output_dir or (args.grid_csv.parent / "configuration_probes")

  rows = load_grid_rows(args.grid_csv, configuration_indices)
  print(f"Simulating {len(rows)} configuration(s) from "
        f"{display_path(args.grid_csv)} "
        f"(horizon={args.horizon_s:g}s, trial timeout={args.trial_timeout_s:g}s) ...")

  meta = json.loads(metadata_json.read_text())
  train_ids = meta["split"]["train_trial_ids"]
  test_ids = meta["split"]["test_trial_ids"]

  normalize_columns = args.normalize_columns
  if normalize_columns is None:
    normalize_columns = bool(meta.get("fixed_model_settings", {}).get("normalize_columns", False))
  print(f"normalize_columns={normalize_columns} "
        f"({'explicit' if args.normalize_columns is not None else 'auto-detected from metadata'})")

  signal_normalization = args.signal_normalization
  if signal_normalization is None:
    signal_normalization = meta.get("preprocessing", {}).get("normalization", "none")
  if signal_normalization not in ("none", "global_zscore"):
    raise ValueError(
      f"Unsupported normalization '{signal_normalization}'; expected 'none' or 'global_zscore'."
    )
  signal_units = "z-score" if signal_normalization == "global_zscore" else LFP_AMPLITUDE_UNIT
  print(f"signal normalization={signal_normalization!r} "
        f"({'explicit' if args.signal_normalization is not None else 'auto-detected from metadata'})")

  if requested_trial_ids is not None:
    missing = sorted(set(requested_trial_ids) - set(test_ids))
    if missing:
      raise ValueError(f"--trial-ids {missing} are not in the held-out test split")
  selected_trial_ids = requested_trial_ids if requested_trial_ids is not None else list(test_ids)
  if args.max_trials is not None:
    selected_trial_ids = selected_trial_ids[: args.max_trials]

  print(f"Loading {MAT_FILE} ...", flush=True)
  data = TrialData.load(MAT_FILE)
  dt = DOWNSAMPLE / data.fs
  fs_processed = 1.0 / dt

  lowpass_values = sorted({float(r["lowpass_hz"]) for r in rows})
  train_z: dict[float, list[np.ndarray]] = {}
  test_z: dict[float, list[np.ndarray]] = {}
  for lp in lowpass_values:
    raw_train = channel_traces(data, channel=CHANNEL, trials=train_ids,
                                downsample=DOWNSAMPLE, lowpass_hz=lp, normalize="none")
    raw_test = channel_traces(data, channel=CHANNEL, trials=test_ids,
                               downsample=DOWNSAMPLE, lowpass_hz=lp, normalize="none")
    if signal_normalization == "global_zscore":
      stats = compute_global_zscore_stats(raw_train, channel=CHANNEL)
      train_z[lp] = apply_global_zscore(raw_train, stats)
      test_z[lp] = apply_global_zscore(raw_test, stats)
    else:
      train_z[lp] = raw_train
      test_z[lp] = raw_test

  all_trial_rows: list[dict] = []
  all_summary_rows: list[dict] = []
  for row in rows:
    variants = build_variants(row, alpha_overrides, threshold_overrides, max_iter_overrides)
    trial_rows, summary_rows = process_configuration(
      row, variants, train_z, test_z, test_ids, selected_trial_ids,
      dt, fs_processed, args.horizon_s, args.trial_timeout_s, output_dir, args.figure_format,
      normalize_columns, signal_units,
    )
    all_trial_rows.extend(trial_rows)
    all_summary_rows.extend(summary_rows)

  if args.configuration_index is not None:
    parts_dir = output_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    tag = f"cfg{args.configuration_index:02d}"
    trial_csv = parts_dir / f"{tag}_trials.csv"
    summary_csv = parts_dir / f"{tag}_summary.csv"
  else:
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_csv = output_dir / "simulation_trials.csv"
    summary_csv = output_dir / "simulation_summary.csv"

  with open(trial_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(all_trial_rows[0].keys()))
    w.writeheader()
    w.writerows(all_trial_rows)
  with open(summary_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(all_summary_rows[0].keys()))
    w.writeheader()
    w.writerows(all_summary_rows)

  print(f"\nwrote {display_path(trial_csv)}")
  print(f"wrote {display_path(summary_csv)}")


if __name__ == "__main__":
  main()
