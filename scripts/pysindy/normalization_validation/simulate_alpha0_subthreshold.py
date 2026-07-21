"""Simulate the alpha=0.05 vs alpha=0 equations from the subthreshold probe.

Background
----------
``probe_alpha0_subthreshold.py`` showed that STLSQ's alpha=0.05 ridge fit can
leave surviving coefficients below the nominal threshold once the reported
value is refit by plain OLS on the selected support, while alpha=0 removes
that artifact. This script checks whether the resulting equations are only
numerically different or also *qualitatively* different in behavior: does
forward simulation on held-out trials diverge, collapse to a constant, or
otherwise change character between the two alpha settings.

For each of the 12 probed configurations, the alpha=0.05 model is
reconstructed from the coefficients already stored in
``raw_grid_nc_false_gsz_t1/raw_grid_merged.csv`` (no refit needed); the
alpha=0 model is refit the same way ``probe_alpha0_subthreshold.py`` does.
Both are simulated from every held-out test trial's own initial condition and
scored with the project's standard descriptive metrics (no pass/fail
rejection).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
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
from models.sindy import SINDyConfig, StoredPolynomialModel, delay_embed_trajectories, fit_sindy_model
from models.validation import SimulationConfig, evaluate_simulation, simulate_model_detailed

csv.field_size_limit(10 * 1024 * 1024)

SOURCE_RUN = _PROJECT_ROOT / "outputs/pysindy/raw_grid_nc_false_gsz_t1"
SOURCE_CSV = SOURCE_RUN / "raw_grid_merged.csv"
SOURCE_META = sorted((SOURCE_RUN / "parts").glob("*_metadata.json"))[0]
PROBE_CSV = (
  _PROJECT_ROOT / "outputs/pysindy/nc_false_gsz_t1_alpha0_probe/alpha0_subthreshold_comparison.csv"
)
OUTPUT_DIR = _PROJECT_ROOT / "outputs/pysindy/nc_false_gsz_t1_alpha0_probe"

THRESHOLD = 1.0
CHANNEL = 0
DOWNSAMPLE = 2
HORIZON_S = 2.0
SIGNAL_UNITS = "z-score"
FIGURE_FORMAT = "svg"

METRIC_KEYS = [
  "trajectory_rmse",
  "x0_correlation",
  "max_amplitude_ratio",
  "collapse_std_ratio",
  "psd_similarity",
  "distribution_ks",
]


def load_probed_rows() -> list[dict]:
  """Load the 12 configs already selected by the subthreshold probe.

  Returns:
    Rows from ``raw_grid_merged.csv`` matching the probe's configuration
    indices, in the probe's original order.
  """
  with open(PROBE_CSV) as f:
    probed_indices = [row["configuration_index"] for row in csv.DictReader(f)]

  with open(SOURCE_CSV) as f:
    by_index = {row["configuration_index"]: row for row in csv.DictReader(f)}

  return [by_index[idx] for idx in probed_indices]


def empty_metrics() -> dict[str, float]:
  """Return NaN-filled metrics for a simulation that produced no trajectory."""
  return {key: float("nan") for key in METRIC_KEYS}


def mean_finite(values: list[float]) -> float:
  """Average the finite entries of ``values``, or NaN if none are finite."""
  finite = [v for v in values if np.isfinite(v)]
  return float(np.mean(finite)) if finite else float("nan")


def _draw_comparison_panel(
  axis,
  trial_id: int,
  measured: np.ndarray,
  sim_a05,
  sim_a0,
  dt: float,
) -> None:
  """Draw one trial's measured trace plus both alpha simulations onto ``axis``."""
  measured_time = np.arange(measured.shape[0]) * dt
  axis.plot(
    measured_time, measured[:, 0],
    color="steelblue", linewidth=1.1, label=f"measured ({SIGNAL_UNITS})",
  )
  if sim_a05.trajectory is not None and sim_a05.trajectory.size:
    axis.plot(
      sim_a05.time, sim_a05.trajectory[:, 0],
      color="darkorange", linestyle="--", linewidth=1.1, label="simulated (alpha=0.05)",
    )
  if sim_a0.trajectory is not None and sim_a0.trajectory.size:
    axis.plot(
      sim_a0.time, sim_a0.trajectory[:, 0],
      color="seagreen", linestyle="-.", linewidth=1.1, label="simulated (alpha=0.0)",
    )
  status_a05 = "complete" if sim_a05.completed else "failed"
  status_a0 = "complete" if sim_a0.completed else "failed"
  axis.set_title(
    f"Trial {trial_id}: alpha=0.05 {status_a05} @{sim_a05.reached_horizon_s:.2f}s, "
    f"alpha=0.0 {status_a0} @{sim_a0.reached_horizon_s:.2f}s",
    fontsize=9,
  )
  axis.set_xlabel("Time from embedded initial state (s)", fontsize=8)
  axis.set_ylabel(f"x0 ({SIGNAL_UNITS})", fontsize=8)
  axis.grid(alpha=0.2)
  axis.legend(loc="upper right", fontsize=7)


def plot_configuration_comparison(
  path: Path,
  row: dict[str, str],
  trial_ids: list[int],
  measured_trials: list[np.ndarray],
  results_a05: list,
  results_a0: list,
  dt: float,
) -> None:
  """Plot every held-out trial in a compact grid; one figure per configuration.

  Each panel overlays the measured trace with both the alpha=0.05 and the
  alpha=0 simulations so a qualitative difference (divergence, collapse to a
  constant, phase drift) is visible directly, not just in the metrics CSV.
  """
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  columns = 3
  n_rows = math.ceil(len(trial_ids) / columns)
  figure, axes = plt.subplots(
    n_rows, columns,
    figsize=(5 * columns, 2.7 * n_rows),
    sharex=False, sharey=False, squeeze=False,
  )
  for axis, trial_id, measured, sim_a05, sim_a0 in zip(
    axes.ravel(), trial_ids, measured_trials, results_a05, results_a0
  ):
    _draw_comparison_panel(axis, trial_id, measured, sim_a05, sim_a0, dt)

  for axis in axes.ravel()[len(trial_ids):]:
    axis.set_visible(False)

  threshold_text = f", threshold={THRESHOLD:g}"
  figure.suptitle(
    f"Configuration {row['configuration_index']}: LP={row['lowpass_hz']} Hz, "
    f"degree={row['degree']}, delays={row['n_delays']}, "
    f"spacing={row['delay_samples']} samples, "
    f"smoothing={row['smooth_window_samples']} samples{threshold_text}"
  )
  figure.tight_layout(rect=(0, 0, 1, 0.95))
  path.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(path, dpi=160)
  plt.close(figure)


def parse_args() -> argparse.Namespace:
  """Parse CLI arguments for local or Slurm-array invocation."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--configuration-index", type=int, default=None,
    help="1-based index into the 12 probed configs, for one Slurm array task. "
         "Omit to run all configs locally and write merged CSVs.",
  )
  parser.add_argument(
    "--trial-timeout-s", type=float, default=60.0,
    help="Wall-clock cap per simulate_model_detailed call. Protects against "
         "an unstable model hanging LSODA's adaptive step size.",
  )
  return parser.parse_args()


def main() -> None:
  """Simulate alpha=0.05 and alpha=0 models on held-out trials and compare."""
  args = parse_args()
  rows = load_probed_rows()
  if args.configuration_index is not None:
    if not 1 <= args.configuration_index <= len(rows):
      raise ValueError(f"--configuration-index must be in 1..{len(rows)}")
    rows = [rows[args.configuration_index - 1]]
  print(f"Simulating {len(rows)} config(s) at alpha=0.05 and alpha=0.0 "
        f"(horizon={HORIZON_S:g}s, trial timeout={args.trial_timeout_s:g}s) ...")

  meta = json.loads(SOURCE_META.read_text())
  train_ids = meta["split"]["train_trial_ids"]
  test_ids = meta["split"]["test_trial_ids"]

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
    stats = compute_global_zscore_stats(raw_train, channel=CHANNEL)
    train_z[lp] = apply_global_zscore(raw_train, stats)
    raw_test = channel_traces(data, channel=CHANNEL, trials=test_ids,
                               downsample=DOWNSAMPLE, lowpass_hz=lp, normalize="none")
    test_z[lp] = apply_global_zscore(raw_test, stats)

  sim_config = SimulationConfig(simulation_horizon_s=HORIZON_S)

  trial_rows = []
  summary_rows = []

  for r in rows:
    lp = float(r["lowpass_hz"])
    n_delays, delay, smooth = int(r["n_delays"]), int(r["delay_samples"]), int(r["smooth_window_samples"])
    degree = int(r["degree"])

    coefs_a05 = np.asarray(json.loads(r["coefficients_json"]), dtype=float)
    names_a05 = json.loads(r["feature_names_json"])
    model_a05 = StoredPolynomialModel(degree=degree, coefficients=coefs_a05, feature_names=names_a05)

    embedded_train = delay_embed_trajectories(train_z[lp], n_delays=n_delays, delay=delay)
    model_a0 = fit_sindy_model(
      embedded_train, dt=dt,
      config=SINDyConfig(degree=degree, threshold=THRESHOLD, alpha=0.0,
                         normalize_columns=False, smooth_window=smooth, smoothing_polyorder=3),
    )

    embedded_test = delay_embed_trajectories(test_z[lp], n_delays=n_delays, delay=delay)

    metrics_by_alpha: dict[str, list[dict]] = {"0.05": [], "0.0": []}
    plotted_trial_ids: list[int] = []
    plotted_measured: list[np.ndarray] = []
    plotted_results_a05: list = []
    plotted_results_a0: list = []
    for trial_id, measured in zip(test_ids, embedded_test):
      if measured.shape[0] < 2:
        continue
      x0 = measured[0]
      sims: dict[str, object] = {}
      for alpha_label, model in (("0.05", model_a05), ("0.0", model_a0)):
        sim = simulate_model_detailed(model, initial_state=x0, dt=dt, horizon_s=HORIZON_S,
                                       wall_timeout_s=args.trial_timeout_s)
        sims[alpha_label] = sim
        metric = (
          empty_metrics() if sim.trajectory is None
          else evaluate_simulation(measured, sim.trajectory, fs=fs_processed, config=sim_config)
        )
        trial_rows.append({
          "configuration_index": r["configuration_index"], "degree": degree, "lowpass_hz": lp,
          "n_delays": n_delays, "delay_samples": delay, "smooth_window_samples": smooth,
          "alpha": alpha_label, "trial_id": trial_id,
          "completed": sim.completed, "failure_reason": sim.failure_reason,
          "reached_horizon_s": sim.reached_horizon_s,
          **{key: metric[key] for key in METRIC_KEYS},
        })
        metrics_by_alpha[alpha_label].append(metric)
      plotted_trial_ids.append(trial_id)
      plotted_measured.append(measured)
      plotted_results_a05.append(sims["0.05"])
      plotted_results_a0.append(sims["0.0"])

    plot_path = OUTPUT_DIR / "simulation_plots" / f"cfg{r['configuration_index']}.{FIGURE_FORMAT}"
    plot_configuration_comparison(
      plot_path, r, plotted_trial_ids, plotted_measured,
      plotted_results_a05, plotted_results_a0, dt,
    )

    summary = {
      "configuration_index": r["configuration_index"], "degree": degree, "lowpass_hz": lp,
      "n_delays": n_delays, "delay_samples": delay, "smooth_window_samples": smooth,
      "n_test_trials": len(metrics_by_alpha["0.05"]),
    }
    for key in METRIC_KEYS:
      m05 = mean_finite([m[key] for m in metrics_by_alpha["0.05"]])
      m0 = mean_finite([m[key] for m in metrics_by_alpha["0.0"]])
      summary[f"{key}_alpha_0p05"] = m05
      summary[f"{key}_alpha_0p0"] = m0
      summary[f"{key}_delta"] = m0 - m05
    summary_rows.append(summary)
    print(f"  cfg {r['configuration_index']:>4} deg={degree} lp={lp:>4.0f}  "
          f"collapse_std_ratio delta={summary['collapse_std_ratio_delta']:+.3f}  "
          f"psd_similarity delta={summary['psd_similarity_delta']:+.3f}  "
          f"plot={plot_path.relative_to(_PROJECT_ROOT)}")

  if args.configuration_index is not None:
    parts_dir = OUTPUT_DIR / "simulation_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    tag = f"cfg{args.configuration_index:02d}"
    trial_csv = parts_dir / f"{tag}_trials.csv"
    summary_csv = parts_dir / f"{tag}_summary.csv"
  else:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trial_csv = OUTPUT_DIR / "alpha0_subthreshold_simulation_trials.csv"
    summary_csv = OUTPUT_DIR / "alpha0_subthreshold_simulation_summary.csv"

  with open(trial_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(trial_rows[0].keys()))
    w.writeheader()
    w.writerows(trial_rows)

  with open(summary_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
    w.writeheader()
    w.writerows(summary_rows)

  print(f"\nwrote {trial_csv.relative_to(_PROJECT_ROOT)}")
  print(f"wrote {summary_csv.relative_to(_PROJECT_ROOT)}")


if __name__ == "__main__":
  main()
