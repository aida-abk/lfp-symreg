from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time

csv.field_size_limit(10 * 1024 * 1024)  # degree=7 coefficient JSON exceeds 128KB default
from pathlib import Path

import numpy as np

# Project imports
ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
PYSINDY_SCRIPTS = SCRIPTS / "pysindy"
for path in (ROOT, SCRIPTS, PYSINDY_SCRIPTS):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

from load_data.convert import LFP_AMPLITUDE_UNIT, MAT_FILE, TrialData
from load_data.preprocessing import channel_traces, pooled_trace_rms
from models.sindy import StoredFourierModel, StoredPolynomialModel, delay_embed_trajectories
from models.validation import SimulationResult, simulate_model_detailed
from raw_grid_io import write_csv_checkpoint

# Default raw-grid artifacts
DEFAULT_GRID = ROOT / "outputs" / "pysindy" / "raw_grid" / "raw_grid_merged.csv"
DEFAULT_METADATA = (
  ROOT
  / "outputs"
  / "pysindy"
  / "raw_grid"
  / "parts"
  / "part_lp35_degree1_metadata.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "pysindy" / "raw_grid" / "simulations"

STATUS_FIELDS = [
  "configuration_index",
  "test_trial_id",
  "simulation_status",
  "failure_reason",
  "requested_duration_s",
  "reached_duration_s",
  "simulation_runtime_s",
  "rhs_evaluations",
  "simulated_samples",
  "simulated_x0_rms_uv",
  "compared_samples",
  "x0_rmse_uv",
  "trajectory_rmse_uv",
]

# One row per (configuration, trial, window) in windowed re-anchored simulation mode.
WINDOW_STATUS_FIELDS = [
  "configuration_index",
  "test_trial_id",
  "window_index",
  "window_start_s",
  "window_duration_s",
  "simulation_status",
  "failure_reason",
  "reached_duration_s",
  "simulation_runtime_s",
  "rhs_evaluations",
  "simulated_samples",
  "simulated_x0_rms_uv",
  "compared_samples",
  "x0_rmse_uv",
  "trajectory_rmse_uv",
]


def load_grid(path: Path) -> list[dict[str, str]]:
  """Load successful stored-equation rows from the merged raw-grid CSV."""
  with path.open(newline="") as file:
    rows = list(csv.DictReader(file))
  if not rows:
    raise ValueError(f"No configurations found in {path}.")
  failed = [row for row in rows if row["fit_status"] != "success"]
  if failed:
    raise ValueError(f"The grid contains {len(failed)} unsuccessful fits.")
  return rows


def select_benchmark_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
  """Select one fixed middle-sized configuration per degree/filter pair.

  The benchmark fixes four delay coordinates, two-sample spacing, and a
  five-sample smoothing window. This selection estimates runtime only and does
  not rank the scientific quality of those settings.
  """
  selected = []
  pairs = sorted({(float(row["lowpass_hz"]), int(row["degree"])) for row in rows})
  for lowpass_hz, degree in pairs:
    matches = [
      row
      for row in rows
      if float(row["lowpass_hz"]) == lowpass_hz
      and int(row["degree"]) == degree
      and int(row["n_delays"]) == 4
      and int(row["delay_samples"]) == 2
      and int(row["smooth_window_samples"]) == 5
    ]
    if len(matches) != 1:
      raise ValueError(
        "Expected one benchmark row for "
        f"lowpass={lowpass_hz}, degree={degree}; found {len(matches)}."
      )
    selected.append(matches[0])
  return selected


def select_rows(
  rows: list[dict[str, str]],
  configuration_index: int | None,
  benchmark: bool,
) -> list[dict[str, str]]:
  """Select all rows, one global configuration, or six benchmark rows."""
  if configuration_index is not None and benchmark:
    raise ValueError("Use either --configuration-index or --benchmark, not both.")
  if benchmark:
    return select_benchmark_rows(rows)
  if configuration_index is None:
    return rows
  matches = [
    row for row in rows if int(row["configuration_index"]) == configuration_index
  ]
  if len(matches) != 1:
    raise ValueError(
      f"Expected one row for configuration {configuration_index}; found {len(matches)}."
    )
  return matches


def _is_fourier_row(row: dict[str, str]) -> bool:
  """Return True when the CSV row comes from a Fourier-library sweep."""
  return "n_frequencies" in row and bool(row.get("n_frequencies"))


def _model_order_label(row: dict[str, str]) -> str:
  """Return a short label describing model order for figure titles."""
  if _is_fourier_row(row):
    return f"n_freq={row['n_frequencies']}"
  return f"degree={row.get('degree', '?')}"


def reconstruct_model(row: dict[str, str]) -> StoredPolynomialModel | StoredFourierModel:
  """Reconstruct one fitted ODE from stored coefficients.

  Automatically selects ``StoredFourierModel`` when the row contains an
  ``n_frequencies`` column, and ``StoredPolynomialModel`` otherwise.
  """
  coefs = np.asarray(json.loads(row["coefficients_json"]), dtype=float)
  names = list(json.loads(row["feature_names_json"]))
  if _is_fourier_row(row):
    return StoredFourierModel(
      n_frequencies=int(row["n_frequencies"]),
      coefficients=coefs,
      feature_names=names,
    )
  return StoredPolynomialModel(
    degree=int(row["degree"]),
    coefficients=coefs,
    feature_names=names,
  )


def _config_suptitle(row: dict[str, str]) -> str:
  """Return a figure suptitle string from a raw-grid configuration row."""
  threshold_text = (
    f", threshold={row['stlsq_threshold']}" if row.get("stlsq_threshold") else ""
  )
  return (
    f"Configuration {row['configuration_index']}: "
    f"LP={row['lowpass_hz']} Hz, {_model_order_label(row)}, "
    f"delays={row['n_delays']}, spacing={row['delay_samples']} samples, "
    f"smoothing={row['smooth_window_samples']} samples{threshold_text}"
  )


def _draw_trial_panel(
  axis,
  trial_id: int,
  measured: np.ndarray,
  result: SimulationResult,
  dt: float,
  raw_unfiltered: np.ndarray | None,
  signal_units: str,
  title_fontsize: int = 10,
  label_fontsize: int = 9,
) -> None:
  """Draw one trial's lowpass, simulated, and optional raw traces onto ``axis``."""
  measured_time = np.arange(measured.shape[0]) * dt

  ax2 = None
  if raw_unfiltered is not None:
    ax2 = axis.twinx()
    raw = raw_unfiltered[: measured.shape[0]]
    ax2.plot(
      measured_time[: len(raw)], raw,
      color="grey", linewidth=0.8, alpha=0.4, label="raw (µV)",
    )
    ax2.set_ylabel("µV (raw)", fontsize=label_fontsize - 1, color="grey")
    ax2.tick_params(axis="y", labelcolor="grey", labelsize=label_fontsize - 2)
    ax2.spines["right"].set_color("grey")
    ax2.spines["right"].set_alpha(0.5)
    axis.set_zorder(ax2.get_zorder() + 1)
    axis.patch.set_visible(False)

  axis.plot(
    measured_time, measured[:, 0],
    color="steelblue", linewidth=1.1, label=f"lowpass ({signal_units})",
  )
  if result.trajectory is not None and result.trajectory.size:
    axis.plot(
      result.time, result.trajectory[:, 0],
      color="darkorange", linestyle="--", linewidth=1.1, label="simulated",
    )
  status = "complete" if result.completed else "failed"
  axis.set_title(
    f"Trial {trial_id}: {status}, reached {result.reached_horizon_s:.2f} s",
    fontsize=title_fontsize,
  )
  axis.set_xlabel("Time from embedded initial state (s)", fontsize=label_fontsize)
  axis.set_ylabel(f"x0 ({signal_units})", fontsize=label_fontsize)
  axis.grid(alpha=0.2)

  lines, labels = axis.get_legend_handles_labels()
  if ax2 is not None:
    lines2, labels2 = ax2.get_legend_handles_labels()
    axis.legend(lines2 + lines, labels2 + labels, loc="upper right",
                fontsize=label_fontsize - 1)
  else:
    axis.legend(loc="upper right", fontsize=label_fontsize - 1)


def plot_configuration(
  path: Path,
  row: dict[str, str],
  trial_ids: list[int],
  measured_trials: list[np.ndarray],
  results: list[SimulationResult],
  dt: float,
  title: str | None = None,
  raw_unfiltered_trials: list[np.ndarray] | None = None,
  signal_units: str = LFP_AMPLITUDE_UNIT,
) -> None:
  """Plot all held-out trials in a compact grid; one PNG per configuration."""
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
  for i, (axis, trial_id, measured, result) in enumerate(
    zip(axes.ravel(), trial_ids, measured_trials, results)
  ):
    raw = (raw_unfiltered_trials[i] if raw_unfiltered_trials is not None
           and i < len(raw_unfiltered_trials) else None)
    _draw_trial_panel(axis, trial_id, measured, result, dt, raw, signal_units,
                      title_fontsize=9, label_fontsize=8)

  for axis in axes.ravel()[len(trial_ids):]:
    axis.set_visible(False)
  figure.suptitle(title or _config_suptitle(row))
  figure.tight_layout(rect=(0, 0, 1, 0.97))
  path.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(path, dpi=160)
  plt.close(figure)


def plot_trial(
  path: Path,
  row: dict[str, str],
  trial_id: int,
  measured: np.ndarray,
  result: SimulationResult,
  dt: float,
  raw_unfiltered: np.ndarray | None = None,
  signal_units: str = LFP_AMPLITUDE_UNIT,
) -> None:
  """Plot a single held-out trial at full figure size; one PNG per trial."""
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  figure, axis = plt.subplots(figsize=(11, 4.5))
  _draw_trial_panel(axis, trial_id, measured, result, dt, raw_unfiltered,
                    signal_units, title_fontsize=12, label_fontsize=10)
  figure.suptitle(_config_suptitle(row), fontsize=10)
  figure.tight_layout(rect=(0, 0, 1, 0.94))
  path.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(path, dpi=160)
  plt.close(figure)


def simulate_configuration(
  row: dict[str, str],
  test_raw: list[np.ndarray],
  test_trial_ids: list[int],
  dt: float,
  output_dir: Path,
  trial_timeout_s: float | None,
  test_raw_unfiltered: list[np.ndarray] | None = None,
  signal_units: str = LFP_AMPLITUDE_UNIT,
  per_trial_figures: bool = False,
  figure_format: str = "png",
) -> dict[str, object]:
  """Simulate one stored equation from every held-out initial condition.

  Args:
    row: Stored raw-grid configuration.
    test_raw: Lowpass-filtered held-out traces in model input units (µV or z-score).
    test_trial_ids: Original zero-based held-out trial identifiers.
    dt: Processed sample interval in seconds.
    output_dir: Destination for status checkpoints and figures.
    trial_timeout_s: Optional operational wall-time limit per simulation.
    test_raw_unfiltered: Optional unfiltered (detrended, downsampled) µV traces for
      the grey raw-signal overlay.
    signal_units: Units label for the primary axis in simulation plots.
    per_trial_figures: When True, save one full-size PNG per trial instead of a
      compact multi-trial grid.

  Returns:
    Configuration-level simulation summary.
  """
  configuration_index = int(row["configuration_index"])
  measured_trials = delay_embed_trajectories(
    test_raw,
    n_delays=int(row["n_delays"]),
    delay=int(row["delay_samples"]),
  )
  model = reconstruct_model(row)
  results = []
  status_rows = []
  configuration_started = time.perf_counter()
  stem = f"config_{configuration_index:04d}"
  status_path = output_dir / "status" / f"{stem}.csv"

  for trial_id, measured in zip(test_trial_ids, measured_trials):
    requested_duration_s = (measured.shape[0] - 1) * dt
    started = time.perf_counter()
    result = simulate_model_detailed(
      model,
      initial_state=measured[0],
      dt=dt,
      horizon_s=requested_duration_s,
      wall_timeout_s=trial_timeout_s,
    )
    runtime_s = time.perf_counter() - started
    results.append(result)
    simulated_x0 = (
      result.trajectory[:, 0]
      if result.trajectory is not None and result.trajectory.size
      else None
    )
    compared_samples = (
      0
      if result.trajectory is None
      else min(measured.shape[0], result.trajectory.shape[0])
    )
    if compared_samples:
      measured_segment = measured[:compared_samples]
      simulated_segment = result.trajectory[:compared_samples]
      with np.errstate(over="ignore", invalid="ignore"):
        x0_rmse_uv = float(
          np.sqrt(
            np.mean((simulated_segment[:, 0] - measured_segment[:, 0]) ** 2)
          )
        )
        trajectory_rmse_uv = float(
          np.sqrt(np.mean((simulated_segment - measured_segment) ** 2))
        )
    else:
      x0_rmse_uv = ""
      trajectory_rmse_uv = ""
    status_rows.append(
      {
        "configuration_index": configuration_index,
        "test_trial_id": trial_id,
        "simulation_status": "success" if result.completed else "failed",
        "failure_reason": result.failure_reason,
        "requested_duration_s": requested_duration_s,
        "reached_duration_s": result.reached_horizon_s,
        "simulation_runtime_s": runtime_s,
        "rhs_evaluations": result.rhs_evaluations,
        "simulated_samples": 0 if simulated_x0 is None else simulated_x0.size,
        "simulated_x0_rms_uv": (
          "" if simulated_x0 is None else pooled_trace_rms([simulated_x0])
        ),
        "compared_samples": compared_samples,
        "x0_rmse_uv": x0_rmse_uv,
        "trajectory_rmse_uv": trajectory_rmse_uv,
      }
    )
    write_csv_checkpoint(status_path, STATUS_FIELDS, status_rows)
    print(
      f"config={configuration_index} trial={trial_id} "
      f"status={status_rows[-1]['simulation_status']} "
      f"reached={result.reached_horizon_s:.2f}/{requested_duration_s:.2f}s "
      f"x0_rmse={x0_rmse_uv} uV "
      f"runtime={runtime_s:.1f}s",
      flush=True,
    )

  # Crop unfiltered traces to align with delay-embedded x0 (same offset applied in
  # delay_embed_trace: first valid row starts at (n_delays - 1) * delay samples).
  raw_unfiltered_x0: list[np.ndarray] | None = None
  if test_raw_unfiltered is not None:
    offset = (int(row["n_delays"]) - 1) * int(row["delay_samples"])
    raw_unfiltered_x0 = [tr[offset:] for tr in test_raw_unfiltered]

  figures_dir = output_dir / "figures"
  if per_trial_figures:
    for trial_id, measured, result, raw in zip(
      test_trial_ids, measured_trials, results,
      raw_unfiltered_x0 if raw_unfiltered_x0 is not None else [None] * len(results),
    ):
      plot_trial(
        figures_dir / f"{stem}_trial_{trial_id:04d}.{figure_format}",
        row=row,
        trial_id=trial_id,
        measured=measured,
        result=result,
        dt=dt,
        raw_unfiltered=raw,
        signal_units=signal_units,
      )
    # Per-trial mode writes one figure per trial; record the directory holding them.
    figure_path = figures_dir
  else:
    figure_path = figures_dir / f"{stem}.{figure_format}"
    plot_configuration(
      figure_path,
      row=row,
      trial_ids=test_trial_ids,
      measured_trials=measured_trials,
      results=results,
      dt=dt,
      raw_unfiltered_trials=raw_unfiltered_x0,
      signal_units=signal_units,
    )
  model_order_key = "n_frequencies" if _is_fourier_row(row) else "degree"
  return {
    "configuration_index": configuration_index,
    "lowpass_hz": float(row["lowpass_hz"]),
    model_order_key: int(row[model_order_key]),
    "n_delays": int(row["n_delays"]),
    "delay_samples": int(row["delay_samples"]),
    "smooth_window_samples": int(row["smooth_window_samples"]),
    "test_trials": len(test_trial_ids),
    "successful_simulations": sum(result.completed for result in results),
    "configuration_runtime_s": time.perf_counter() - configuration_started,
    "figure": str(figure_path),
    "status_csv": str(status_path),
  }


def simulate_trial_windows(
  model,
  measured: np.ndarray,
  dt: float,
  window_samples: int,
  step_samples: int,
  trial_timeout_s: float | None,
) -> list[dict[str, object]]:
  """Simulate one trial as a sequence of re-anchored fixed-length windows.

  Each window is an independent free-run: its initial condition is the *measured*
  delay-embedded state at the window's start sample, and it is integrated forward
  for the window duration. This isolates short-horizon predictive accuracy across
  the whole trial instead of letting a single long free-run diverge.

  Args:
    model: Reconstructed ODE whose ``predict`` returns state derivatives.
    measured: Delay-embedded held-out trajectory with shape ``(time, state)``.
    dt: Processed sample interval in seconds.
    window_samples: Nominal window length in processed samples.
    step_samples: Spacing between successive window starts in processed samples.
      Equal to ``window_samples`` for non-overlapping tiling.
    trial_timeout_s: Optional wall-time limit per window simulation.

  Returns:
    One dict per window holding status metrics plus the raw ``SimulationResult``
    and the window's start/length in samples (for plotting). A trailing partial
    window shorter than the nominal length is kept and integrated to trial end.
  """
  total_samples = measured.shape[0]
  windows: list[dict[str, object]] = []
  window_index = 0
  for start in range(0, total_samples - 1, step_samples):
    this_window_samples = min(window_samples, total_samples - start)
    if this_window_samples < 2:
      break
    window_duration_s = (this_window_samples - 1) * dt
    started = time.perf_counter()
    result = simulate_model_detailed(
      model,
      initial_state=measured[start],
      dt=dt,
      horizon_s=window_duration_s,
      wall_timeout_s=trial_timeout_s,
    )
    runtime_s = time.perf_counter() - started

    measured_window = measured[start : start + this_window_samples]
    simulated_x0 = (
      result.trajectory[:, 0]
      if result.trajectory is not None and result.trajectory.size
      else None
    )
    compared_samples = (
      0
      if result.trajectory is None
      else min(measured_window.shape[0], result.trajectory.shape[0])
    )
    if compared_samples:
      measured_segment = measured_window[:compared_samples]
      simulated_segment = result.trajectory[:compared_samples]
      with np.errstate(over="ignore", invalid="ignore"):
        x0_rmse_uv = float(
          np.sqrt(np.mean((simulated_segment[:, 0] - measured_segment[:, 0]) ** 2))
        )
        trajectory_rmse_uv = float(
          np.sqrt(np.mean((simulated_segment - measured_segment) ** 2))
        )
    else:
      x0_rmse_uv = ""
      trajectory_rmse_uv = ""

    windows.append(
      {
        "window_index": window_index,
        "window_start_s": start * dt,
        "window_duration_s": window_duration_s,
        "simulation_status": "success" if result.completed else "failed",
        "failure_reason": result.failure_reason,
        "reached_duration_s": result.reached_horizon_s,
        "simulation_runtime_s": runtime_s,
        "rhs_evaluations": result.rhs_evaluations,
        "simulated_samples": 0 if simulated_x0 is None else simulated_x0.size,
        "simulated_x0_rms_uv": (
          "" if simulated_x0 is None else pooled_trace_rms([simulated_x0])
        ),
        "compared_samples": compared_samples,
        "x0_rmse_uv": x0_rmse_uv,
        "trajectory_rmse_uv": trajectory_rmse_uv,
        "_result": result,
        "_start_sample": start,
      }
    )
    window_index += 1
  return windows


def _draw_windowed_trial_panel(
  axis,
  trial_id: int,
  measured: np.ndarray,
  windows: list[dict[str, object]],
  dt: float,
  raw_unfiltered: np.ndarray | None,
  signal_units: str,
  title_fontsize: int = 10,
  label_fontsize: int = 9,
) -> None:
  """Draw one trial: measured x0 plus each window's re-anchored simulated x0."""
  measured_time = np.arange(measured.shape[0]) * dt

  ax2 = None
  if raw_unfiltered is not None:
    ax2 = axis.twinx()
    raw = raw_unfiltered[: measured.shape[0]]
    ax2.plot(
      measured_time[: len(raw)], raw,
      color="grey", linewidth=0.8, alpha=0.4, label="raw (µV)",
    )
    ax2.set_ylabel("µV (raw)", fontsize=label_fontsize - 1, color="grey")
    ax2.tick_params(axis="y", labelcolor="grey", labelsize=label_fontsize - 2)
    ax2.spines["right"].set_color("grey")
    ax2.spines["right"].set_alpha(0.5)
    axis.set_zorder(ax2.get_zorder() + 1)
    axis.patch.set_visible(False)

  axis.plot(
    measured_time, measured[:, 0],
    color="steelblue", linewidth=1.1, label=f"lowpass ({signal_units})",
  )

  finite_rmse = []
  for i, window in enumerate(windows):
    result: SimulationResult = window["_result"]  # type: ignore[assignment]
    start = int(window["_start_sample"])
    axis.axvline(start * dt, color="grey", linestyle=":", linewidth=0.6, alpha=0.5)
    if result.trajectory is not None and result.trajectory.size:
      segment_time = start * dt + np.arange(result.trajectory.shape[0]) * dt
      axis.plot(
        segment_time, result.trajectory[:, 0],
        color="darkorange", linestyle="--", linewidth=1.1,
        label="simulated (windows)" if i == 0 else None,
      )
    if isinstance(window["x0_rmse_uv"], float) and math.isfinite(window["x0_rmse_uv"]):
      finite_rmse.append(window["x0_rmse_uv"])

  successful = sum(w["simulation_status"] == "success" for w in windows)
  mean_rmse_text = (
    f"mean x0 RMSE {np.mean(finite_rmse):.1f}" if finite_rmse else "no finite RMSE"
  )
  axis.set_title(
    f"Trial {trial_id}: {successful}/{len(windows)} windows complete, {mean_rmse_text}",
    fontsize=title_fontsize,
  )
  axis.set_xlabel("Time from embedded initial state (s)", fontsize=label_fontsize)
  axis.set_ylabel(f"x0 ({signal_units})", fontsize=label_fontsize)
  axis.grid(alpha=0.2)

  lines, labels = axis.get_legend_handles_labels()
  if ax2 is not None:
    lines2, labels2 = ax2.get_legend_handles_labels()
    axis.legend(lines2 + lines, labels2 + labels, loc="upper right",
                fontsize=label_fontsize - 1)
  else:
    axis.legend(loc="upper right", fontsize=label_fontsize - 1)


def plot_windowed_trial(
  path: Path,
  row: dict[str, str],
  trial_id: int,
  measured: np.ndarray,
  windows: list[dict[str, object]],
  dt: float,
  raw_unfiltered: np.ndarray | None = None,
  signal_units: str = LFP_AMPLITUDE_UNIT,
) -> None:
  """Plot a single held-out trial's windowed simulation at full figure size."""
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  figure, axis = plt.subplots(figsize=(11, 4.5))
  _draw_windowed_trial_panel(axis, trial_id, measured, windows, dt, raw_unfiltered,
                             signal_units, title_fontsize=12, label_fontsize=10)
  figure.suptitle(_config_suptitle(row), fontsize=10)
  figure.tight_layout(rect=(0, 0, 1, 0.94))
  path.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(path, dpi=160)
  plt.close(figure)


def plot_windowed_trial_stacked(
  path: Path,
  row: dict[str, str],
  trial_id: int,
  measured: np.ndarray,
  windows: list[dict[str, object]],
  dt: float,
  raw_unfiltered: np.ndarray | None = None,
  signal_units: str = LFP_AMPLITUDE_UNIT,
) -> None:
  """Plot one trial as separate zoomed panels, one per re-anchored window.

  Each window gets its own vertically-stacked subplot on a shared "time from
  window start" x-axis (so recurring features line up across windows). Every
  panel overlays the measured lowpass x0 with that window's simulated forecast,
  each free-run from the measured state at the window's start.

  Args:
    path: Destination PNG path.
    row: Stored raw-grid configuration.
    trial_id: Held-out trial identifier.
    measured: Delay-embedded measured trajectory, shape ``(time, state)``.
    windows: Per-window dicts from ``simulate_trial_windows``.
    dt: Processed sample interval in seconds.
    raw_unfiltered: Optional offset-aligned unfiltered µV trace for a grey overlay.
    signal_units: Units label for the primary axis.
  """
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  n = len(windows)
  if n == 0:
    return
  figure, axes = plt.subplots(
    n, 1, figsize=(9, max(2.2, 1.5 * n)), sharex=True, squeeze=False,
  )
  axes = axes.ravel()
  for axis, window in zip(axes, windows):
    result: SimulationResult = window["_result"]  # type: ignore[assignment]
    start = int(window["_start_sample"])
    window_samples = int(round(float(window["window_duration_s"]) / dt)) + 1
    measured_segment = measured[start : start + window_samples, 0]
    measured_time = np.arange(measured_segment.shape[0]) * dt

    if raw_unfiltered is not None:
      ax2 = axis.twinx()
      raw_segment = raw_unfiltered[start : start + window_samples]
      ax2.plot(
        measured_time[: len(raw_segment)], raw_segment,
        color="grey", linewidth=0.7, alpha=0.35,
      )
      ax2.tick_params(axis="y", labelcolor="grey", labelsize=6)
      axis.set_zorder(ax2.get_zorder() + 1)
      axis.patch.set_visible(False)

    axis.plot(
      measured_time, measured_segment,
      color="steelblue", linewidth=1.1, label=f"lowpass ({signal_units})",
    )
    if result.trajectory is not None and result.trajectory.size:
      simulated = result.trajectory[:, 0]
      axis.plot(
        np.arange(simulated.shape[0]) * dt, simulated,
        color="darkorange", linestyle="--", linewidth=1.1, label="simulated",
      )

    rmse = window["x0_rmse_uv"]
    rmse_text = (
      f"RMSE {rmse:.1f}" if isinstance(rmse, float) and math.isfinite(rmse)
      else "RMSE n/a"
    )
    start_s = float(window["window_start_s"])
    end_s = start_s + float(window["window_duration_s"])
    axis.set_title(
      f"window {window['window_index']}: {start_s:.1f}–{end_s:.1f}s  |  "
      f"{window['simulation_status']}  |  {rmse_text}",
      fontsize=8, loc="left",
    )
    axis.set_ylabel(f"x0 ({signal_units})", fontsize=8)
    axis.tick_params(labelsize=7)
    axis.grid(alpha=0.2)

  axes[-1].set_xlabel("Time from window start (s)", fontsize=9)
  axes[0].legend(loc="upper right", fontsize=7)
  figure.suptitle(f"Trial {trial_id} — {_config_suptitle(row)}", fontsize=9)
  figure.tight_layout(rect=(0, 0, 1, 0.98))
  path.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(path, dpi=160)
  plt.close(figure)


def plot_windowed_configuration(
  path: Path,
  row: dict[str, str],
  trial_ids: list[int],
  measured_trials: list[np.ndarray],
  windows_per_trial: list[list[dict[str, object]]],
  dt: float,
  raw_unfiltered_trials: list[np.ndarray] | None = None,
  signal_units: str = LFP_AMPLITUDE_UNIT,
) -> None:
  """Plot all held-out trials' windowed simulations in a compact grid."""
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
  for i, (axis, trial_id, measured, windows) in enumerate(
    zip(axes.ravel(), trial_ids, measured_trials, windows_per_trial)
  ):
    raw = (raw_unfiltered_trials[i] if raw_unfiltered_trials is not None
           and i < len(raw_unfiltered_trials) else None)
    _draw_windowed_trial_panel(axis, trial_id, measured, windows, dt, raw,
                               signal_units, title_fontsize=9, label_fontsize=8)
  for axis in axes.ravel()[len(trial_ids):]:
    axis.set_visible(False)
  figure.suptitle(_config_suptitle(row))
  figure.tight_layout(rect=(0, 0, 1, 0.97))
  path.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(path, dpi=160)
  plt.close(figure)


def simulate_windowed_configuration(
  row: dict[str, str],
  test_raw: list[np.ndarray],
  test_trial_ids: list[int],
  dt: float,
  output_dir: Path,
  window_s: float,
  window_step_s: float,
  trial_timeout_s: float | None,
  test_raw_unfiltered: list[np.ndarray] | None = None,
  signal_units: str = LFP_AMPLITUDE_UNIT,
  per_trial_figures: bool = False,
  stacked: bool = False,
  figure_format: str = "png",
) -> dict[str, object]:
  """Simulate one stored equation as re-anchored windows over every held-out trial.

  Mirrors ``simulate_configuration`` but replaces the single full-trial free-run
  with a sequence of fixed-length windows re-initialized from measured data. Writes
  one status row per (trial, window) and one figure per trial (or a compact grid).

  Args:
    row: Stored raw-grid configuration.
    test_raw: Lowpass-filtered held-out traces in model input units.
    test_trial_ids: Original zero-based held-out trial identifiers.
    dt: Processed sample interval in seconds.
    output_dir: Destination for per-window status checkpoints and figures.
    window_s: Window length in seconds.
    window_step_s: Spacing between window starts in seconds (== window_s to tile).
    trial_timeout_s: Optional wall-time limit per window simulation.
    test_raw_unfiltered: Optional unfiltered µV traces for the grey overlay.
    signal_units: Units label for the primary axis.
    per_trial_figures: When True, one PNG per trial; otherwise a compact grid.

  Returns:
    Configuration-level windowed-simulation summary.
  """
  configuration_index = int(row["configuration_index"])
  measured_trials = delay_embed_trajectories(
    test_raw,
    n_delays=int(row["n_delays"]),
    delay=int(row["delay_samples"]),
  )
  model = reconstruct_model(row)
  window_samples = int(round(window_s / dt))
  step_samples = max(1, int(round(window_step_s / dt)))
  if window_samples < 2:
    raise ValueError(
      f"window_s={window_s} yields fewer than two samples at dt={dt}."
    )

  configuration_started = time.perf_counter()
  stem = f"config_{configuration_index:04d}"
  status_path = output_dir / "status" / f"{stem}.csv"
  status_rows: list[dict[str, object]] = []
  windows_per_trial: list[list[dict[str, object]]] = []
  total_windows = 0
  successful_windows = 0

  for trial_id, measured in zip(test_trial_ids, measured_trials):
    windows = simulate_trial_windows(
      model,
      measured=measured,
      dt=dt,
      window_samples=window_samples,
      step_samples=step_samples,
      trial_timeout_s=trial_timeout_s,
    )
    windows_per_trial.append(windows)
    for window in windows:
      total_windows += 1
      successful_windows += window["simulation_status"] == "success"
      status_rows.append(
        {
          "configuration_index": configuration_index,
          "test_trial_id": trial_id,
          **{key: value for key, value in window.items() if not key.startswith("_")},
        }
      )
    write_csv_checkpoint(status_path, WINDOW_STATUS_FIELDS, status_rows)
    print(
      f"config={configuration_index} trial={trial_id} "
      f"windows={len(windows)} "
      f"success={sum(w['simulation_status'] == 'success' for w in windows)} "
      f"elapsed={time.perf_counter() - configuration_started:.1f}s",
      flush=True,
    )

  raw_unfiltered_x0: list[np.ndarray] | None = None
  if test_raw_unfiltered is not None:
    offset = (int(row["n_delays"]) - 1) * int(row["delay_samples"])
    raw_unfiltered_x0 = [tr[offset:] for tr in test_raw_unfiltered]

  figures_dir = output_dir / "figures"
  if stacked or per_trial_figures:
    plot_fn = plot_windowed_trial_stacked if stacked else plot_windowed_trial
    for trial_id, measured, windows, raw in zip(
      test_trial_ids, measured_trials, windows_per_trial,
      raw_unfiltered_x0 if raw_unfiltered_x0 is not None
      else [None] * len(measured_trials),
    ):
      plot_fn(
        figures_dir / f"{stem}_trial_{trial_id:04d}.{figure_format}",
        row=row,
        trial_id=trial_id,
        measured=measured,
        windows=windows,
        dt=dt,
        raw_unfiltered=raw,
        signal_units=signal_units,
      )
    figure_path = figures_dir
  else:
    figure_path = figures_dir / f"{stem}.{figure_format}"
    plot_windowed_configuration(
      figure_path,
      row=row,
      trial_ids=test_trial_ids,
      measured_trials=measured_trials,
      windows_per_trial=windows_per_trial,
      dt=dt,
      raw_unfiltered_trials=raw_unfiltered_x0,
      signal_units=signal_units,
    )

  model_order_key = "n_frequencies" if _is_fourier_row(row) else "degree"
  return {
    "configuration_index": configuration_index,
    "lowpass_hz": float(row["lowpass_hz"]),
    model_order_key: int(row[model_order_key]),
    "n_delays": int(row["n_delays"]),
    "delay_samples": int(row["delay_samples"]),
    "smooth_window_samples": int(row["smooth_window_samples"]),
    "window_s": window_s,
    "window_step_s": window_step_s,
    "test_trials": len(test_trial_ids),
    "total_windows": total_windows,
    "successful_windows": successful_windows,
    "configuration_runtime_s": time.perf_counter() - configuration_started,
    "figure": str(figure_path),
    "status_csv": str(status_path),
  }


def run(args: argparse.Namespace) -> list[dict[str, object]]:
  """Run the selected stored-equation simulations and visualizations."""
  if args.figure_format == "svg":
    # Keep text as editable text objects (not outlined paths) for vector editing.
    import matplotlib
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["pdf.fonttype"] = 42
  rows = select_rows(
    load_grid(args.grid_csv),
    configuration_index=args.configuration_index,
    benchmark=args.benchmark,
  )
  metadata = json.loads(args.metadata_json.read_text())
  normalization = metadata["preprocessing"]["normalization"]
  if normalization not in ("none", "global_zscore"):
    raise ValueError(
      f"Unsupported normalization '{normalization}'; expected 'none' or 'global_zscore'."
    )
  signal_units = "z-score" if normalization == "global_zscore" else LFP_AMPLITUDE_UNIT
  # Load per-lowpass (μ, σ) computed from training data; keyed by float lowpass_hz.
  global_zscore_stats: dict[float, dict] = {}
  if normalization == "global_zscore" and "global_zscore" in metadata:
    for lp_key, stats in metadata["global_zscore"].items():
      global_zscore_stats[float(lp_key)] = stats

  data = TrialData.load(args.mat_file)
  if float(metadata["raw_sampling_hz"]) != float(data.fs):
    raise ValueError("Local data sampling frequency does not match the sweep metadata.")
  test_trial_ids = [int(value) for value in metadata["split"]["test_trial_ids"]]
  if args.max_test_trials is not None:
    test_trial_ids = test_trial_ids[: args.max_test_trials]
  downsample = int(metadata["downsample_factor"])
  channel = int(metadata["channel"])
  dt = downsample / data.fs

  # Cache lowpass-filtered traces (model target) and unfiltered traces (grey overlay).
  # Keyed by lowpass_hz; unfiltered uses lowpass_hz=None (detrend + downsample only).
  traces_by_lowpass: dict[float, list[np.ndarray]] = {}
  traces_unfiltered: list[np.ndarray] = channel_traces(
    data,
    channel=channel,
    trials=test_trial_ids,
    downsample=downsample,
    lowpass_hz=None,
    normalize="none",
  )
  summaries = []
  for index, row in enumerate(rows, start=1):
    lowpass_hz = float(row["lowpass_hz"])
    if lowpass_hz not in traces_by_lowpass:
      raw_traces = channel_traces(
        data,
        channel=channel,
        trials=test_trial_ids,
        downsample=downsample,
        lowpass_hz=lowpass_hz,
        normalize="none",
      )
      if lowpass_hz in global_zscore_stats:
        stats = global_zscore_stats[lowpass_hz]
        raw_traces = [(t - stats["mean"]) / stats["std"] for t in raw_traces]
      traces_by_lowpass[lowpass_hz] = raw_traces
    print(
      f"[{index}/{len(rows)}] configuration={row['configuration_index']} "
      f"{_model_order_label(row)} lowpass={row['lowpass_hz']}",
      flush=True,
    )
    if args.window_s is not None:
      summaries.append(
        simulate_windowed_configuration(
          row,
          test_raw=traces_by_lowpass[lowpass_hz],
          test_trial_ids=test_trial_ids,
          dt=dt,
          output_dir=args.output_dir,
          window_s=args.window_s,
          window_step_s=(
            args.window_step_s if args.window_step_s is not None else args.window_s
          ),
          trial_timeout_s=args.trial_timeout_s,
          test_raw_unfiltered=traces_unfiltered,
          signal_units=signal_units,
          per_trial_figures=args.per_trial_figures,
          stacked=args.stacked_windows,
          figure_format=args.figure_format,
        )
      )
    else:
      summaries.append(
        simulate_configuration(
          row,
          test_raw=traces_by_lowpass[lowpass_hz],
          test_trial_ids=test_trial_ids,
          dt=dt,
          output_dir=args.output_dir,
          trial_timeout_s=args.trial_timeout_s,
          test_raw_unfiltered=traces_unfiltered,
          signal_units=signal_units,
          per_trial_figures=args.per_trial_figures,
          figure_format=args.figure_format,
        )
      )

  summary_name = (
    "benchmark_summary.json"
    if args.benchmark
    else f"config_{args.configuration_index:04d}_summary.json"
    if args.configuration_index is not None
    else "all_configurations_summary.json"
  )
  summary_path = args.output_dir / summary_name
  summary_path.parent.mkdir(parents=True, exist_ok=True)
  summary_path.write_text(json.dumps(summaries, indent=2) + "\n")
  print(f"saved: {summary_path}")
  return summaries


def main() -> None:
  """Parse CLI arguments and visualize raw-grid simulations."""
  parser = argparse.ArgumentParser(
    description=(
      "Reconstruct stored raw-grid equations and compare full held-out "
      "simulations with measured x0 in microvolts."
    )
  )
  parser.add_argument("--mat-file", type=Path, default=MAT_FILE)
  parser.add_argument("--grid-csv", type=Path, default=DEFAULT_GRID)
  parser.add_argument("--metadata-json", type=Path, default=DEFAULT_METADATA)
  parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
  parser.add_argument("--configuration-index", type=int, default=None)
  parser.add_argument("--benchmark", action="store_true")
  parser.add_argument(
    "--max-test-trials",
    type=int,
    default=None,
    help="Optional computational smoke-test limit; default uses every held-out trial.",
  )
  parser.add_argument(
    "--per-trial-figures",
    action="store_true",
    default=False,
    help=(
      "Save one full-size PNG per trial instead of a compact multi-trial grid. "
      "Files are named config_NNNN_trial_TTTT.png."
    ),
  )
  parser.add_argument(
    "--trial-timeout-s",
    type=float,
    default=None,
    help=(
      "Optional operational wall-time limit per held-out simulation in seconds; "
      "timeouts are saved as failed simulations."
    ),
  )
  parser.add_argument(
    "--window-s",
    type=float,
    default=None,
    help=(
      "Enable re-anchored windowed simulation: tile each trial into windows of "
      "this length (seconds), each free-run from the measured state at its start. "
      "Omit to keep the default single full-trial free-run."
    ),
  )
  parser.add_argument(
    "--window-step-s",
    type=float,
    default=None,
    help=(
      "Spacing between window starts in seconds; defaults to --window-s "
      "(non-overlapping tiling). Smaller values give overlapping windows."
    ),
  )
  parser.add_argument(
    "--figure-format",
    choices=("png", "svg"),
    default="png",
    help=(
      "Output image format for saved figures. 'svg' produces vector files with "
      "editable text (svg.fonttype=none) for Illustrator; 'png' is the default raster."
    ),
  )
  parser.add_argument(
    "--stacked-windows",
    action="store_true",
    default=False,
    help=(
      "With --window-s, save one PNG per trial containing each window as its own "
      "vertically-stacked zoomed panel (shared time-from-window-start x-axis) "
      "instead of overlaying all windows on a single full-trial axis."
    ),
  )
  args = parser.parse_args()
  if args.max_test_trials is not None and args.max_test_trials < 1:
    parser.error("--max-test-trials must be at least 1.")
  if args.trial_timeout_s is not None and args.trial_timeout_s <= 0:
    parser.error("--trial-timeout-s must be positive.")
  if args.window_s is not None and args.window_s <= 0:
    parser.error("--window-s must be positive.")
  if args.window_step_s is not None:
    if args.window_s is None:
      parser.error("--window-step-s requires --window-s.")
    if args.window_step_s <= 0:
      parser.error("--window-step-s must be positive.")
  if args.stacked_windows and args.window_s is None:
    parser.error("--stacked-windows requires --window-s.")
  run(args)


if __name__ == "__main__":
  main()
