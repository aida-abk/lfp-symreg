from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Project imports
ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
PYSINDY_SCRIPTS = SCRIPTS / "pysindy"
for path in (ROOT, SCRIPTS, PYSINDY_SCRIPTS):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

from exploration_sweep import (
  parse_optional_float_list,
  prepare_lfp_trials,
)
from load_data.convert import LFP_AMPLITUDE_UNIT, MAT_FILE
from load_data.preprocessing import channel_traces
from models.sindy import (
  SINDyConfig,
  count_terms,
  delay_embed_trajectories,
  equation_text,
  fit_sindy_model,
)
from pipeline_utils import parse_int_list

# Raw-grid outputs
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "pysindy" / "raw_grid"
DEFAULT_RESULTS = DEFAULT_OUTPUT_DIR / "raw_grid.csv"
DEFAULT_EQUATIONS = DEFAULT_OUTPUT_DIR / "raw_grid_equations.txt"
DEFAULT_METADATA = DEFAULT_OUTPUT_DIR / "run_metadata.json"

FIELDNAMES = [
  "configuration_index",
  "lowpass_hz",
  "degree",
  "n_delays",
  "delay_samples",
  "delay_ms",
  "embedding_span_ms",
  "smooth_window_samples",
  "smooth_window_ms",
  "derivative_method",
  "fit_status",
  "fit_failure_reason",
  "nonzero_terms",
  "fit_runtime_s",
  "feature_names_json",
  "coefficients_json",
  "equations",
]


def initialize_csv(path: Path) -> None:
  """Create an empty raw-grid CSV with its header."""
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="") as file:
    csv.DictWriter(file, fieldnames=FIELDNAMES).writeheader()


def append_result(path: Path, row: dict[str, object]) -> None:
  """Append one fitted configuration so interrupted jobs retain completed work."""
  with path.open("a", newline="") as file:
    csv.DictWriter(file, fieldnames=FIELDNAMES).writerow(row)


def write_equations(path: Path, rows: list[dict[str, object]]) -> None:
  """Write readable equations for every successful raw-grid fit."""
  sections = []
  for row in rows:
    if row["fit_status"] != "success":
      continue
    sections.append(
      f"Configuration {row['configuration_index']}: "
      f"lowpass={row['lowpass_hz']} Hz, degree={row['degree']}, "
      f"n_delays={row['n_delays']}, delay={row['delay_samples']} samples, "
      f"smooth={row['smooth_window_samples']} samples\n{row['equations']}"
    )
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text("\n\n".join(sections) + ("\n" if sections else ""))


def derivative_method_label(smooth_window: int) -> str:
  """Describe the derivative method represented by a smoothing-window value."""
  if smooth_window == 0:
    return "finite_difference"
  return "smoothed_finite_difference_savgol_order_3"


def build_metadata(
  args: argparse.Namespace,
  data,
  train_ids: list[int],
  test_ids: list[int],
  lowpass_values: list[float | None],
  degree_values: list[int],
  n_delay_values: list[int],
  delay_values: list[int],
  smooth_values: list[int],
) -> dict[str, object]:
  """Build one manifest containing all fixed settings and trial assignments."""
  return {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "command": sys.argv,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    "source": "lfp",
    "mat_file": str(args.mat_file),
    "session": str(data.sessname),
    "trial_type": args.trial_type,
    "validity_rule": (
      "is_fixation_trial & goodFix"
      if args.trial_type == "fixation"
      else "is_sequence_trial & goodFix_wholeseq"
    ),
    "channel": args.channel,
    "signal_units": LFP_AMPLITUDE_UNIT,
    "raw_sampling_hz": data.fs,
    "downsample_factor": args.downsample,
    "processed_sampling_hz": data.fs / args.downsample,
    "preprocessing": {
      "demean_each_trial": True,
      "lowpass_filter": "fourth-order Butterworth, zero-phase sosfiltfilt",
      "normalization": "none",
      "software_highpass": "none",
      "full_stored_trial": True,
    },
    "split": {
      "method": "random whole-trial split",
      "test_fraction": args.test_fraction,
      "seed": args.seed,
      "train_trial_ids": train_ids,
      "test_trial_ids": test_ids,
    },
    "fixed_model_settings": {
      "optimizer": "STLSQ",
      "threshold": 0.1,
      "alpha": 0.05,
      "normalize_columns": True,
      "savgol_polyorder": 3,
      "simulation_performed": False,
      "diagnostics_calculated": False,
    },
    "grid": {
      "lowpass_hz": lowpass_values,
      "degree": degree_values,
      "n_delays": n_delay_values,
      "delay_samples": delay_values,
      "smooth_window_samples": smooth_values,
    },
    "expected_configurations": (
      len(lowpass_values)
      * len(degree_values)
      * len(n_delay_values)
      * len(delay_values)
      * len(smooth_values)
    ),
  }


def run_raw_grid(args: argparse.Namespace) -> list[dict[str, object]]:
  """Fit the requested raw PySINDy grid without simulation or diagnostics."""
  lowpass_values = parse_optional_float_list(args.lowpass_list)
  degree_values = parse_int_list(args.degree_list)
  n_delay_values = parse_int_list(args.n_delays_list)
  delay_values = parse_int_list(args.delay_list)
  smooth_values = parse_int_list(args.smooth_window_list)
  if any(window not in {0, 5, 9} for window in smooth_values):
    raise ValueError("Approved smoothing windows are 0, 5, and 9 samples.")

  data, train_ids, test_ids = prepare_lfp_trials(args)
  metadata = build_metadata(
    args,
    data,
    train_ids,
    test_ids,
    lowpass_values,
    degree_values,
    n_delay_values,
    delay_values,
    smooth_values,
  )
  args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
  args.metadata_out.write_text(json.dumps(metadata, indent=2) + "\n")
  initialize_csv(args.out_csv)

  rows = []
  total = int(metadata["expected_configurations"])
  configuration_index = 0
  for lowpass_hz in lowpass_values:
    train_raw = channel_traces(
      data,
      channel=args.channel,
      trials=train_ids,
      downsample=args.downsample,
      lowpass_hz=lowpass_hz,
      normalize="none",
    )
    dt = args.downsample / data.fs
    for degree, n_delays, delay, smooth_window in itertools.product(
      degree_values,
      n_delay_values,
      delay_values,
      smooth_values,
    ):
      configuration_index += 1
      started = time.perf_counter()
      row: dict[str, object] = {
        "configuration_index": configuration_index,
        "lowpass_hz": lowpass_hz if lowpass_hz is not None else "none",
        "degree": degree,
        "n_delays": n_delays,
        "delay_samples": delay,
        "delay_ms": 1000 * delay * dt,
        "embedding_span_ms": 1000 * (n_delays - 1) * delay * dt,
        "smooth_window_samples": smooth_window,
        "smooth_window_ms": 1000 * smooth_window * dt,
        "derivative_method": derivative_method_label(smooth_window),
        "fit_status": "failed",
        "fit_failure_reason": "",
        "nonzero_terms": 0,
        "fit_runtime_s": float("nan"),
        "feature_names_json": "",
        "coefficients_json": "",
        "equations": "",
      }
      try:
        train = delay_embed_trajectories(
          train_raw,
          n_delays=n_delays,
          delay=delay,
        )
        model = fit_sindy_model(
          train,
          dt=dt,
          config=SINDyConfig(
            degree=degree,
            threshold=0.1,
            alpha=0.05,
            normalize_columns=True,
            smooth_window=smooth_window,
            smoothing_polyorder=3,
          ),
        )
        row["fit_status"] = "success"
        row["nonzero_terms"] = count_terms(model)
        row["feature_names_json"] = json.dumps(model.get_feature_names())
        row["coefficients_json"] = json.dumps(model.coefficients().tolist())
        row["equations"] = equation_text(model)
      except Exception as exc:
        row["fit_failure_reason"] = str(exc)
      row["fit_runtime_s"] = time.perf_counter() - started
      rows.append(row)
      append_result(args.out_csv, row)
      print(
        f"[{configuration_index}/{total}] status={row['fit_status']} "
        f"lowpass={row['lowpass_hz']} degree={degree} delays={n_delays} "
        f"delay={delay} smooth={smooth_window} terms={row['nonzero_terms']} "
        f"runtime={float(row['fit_runtime_s']):.1f}s",
        flush=True,
      )

  write_equations(args.equations_out, rows)
  return rows


def main() -> None:
  """Run the fitting-only raw-grid CLI."""
  parser = argparse.ArgumentParser(
    description="Fit a raw PySINDy parameter grid without simulation diagnostics."
  )
  parser.add_argument("--mat-file", type=Path, default=MAT_FILE)
  parser.add_argument("--trial-type", choices=("fixation", "non_fixation"), default="fixation")
  parser.add_argument("--channel", type=int, default=0)
  parser.add_argument("--max-trials", type=int, default=None)
  parser.add_argument("--test-fraction", type=float, default=0.25)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--downsample", type=int, default=2)
  parser.add_argument("--lowpass-list", default="35,80", help="Cutoffs in Hz.")
  parser.add_argument("--degree-list", default="1,2,3")
  parser.add_argument("--n-delays-list", default="2,4,6,8")
  parser.add_argument("--delay-list", default="1,2,5", help="Processed samples.")
  parser.add_argument("--smooth-window-list", default="0,5,9", help="Processed samples.")
  parser.add_argument("--out-csv", type=Path, default=DEFAULT_RESULTS)
  parser.add_argument("--equations-out", type=Path, default=DEFAULT_EQUATIONS)
  parser.add_argument("--metadata-out", type=Path, default=DEFAULT_METADATA)
  args = parser.parse_args()

  rows = run_raw_grid(args)
  successful = sum(row["fit_status"] == "success" for row in rows)
  print(f"saved: {args.out_csv}")
  print(f"saved: {args.equations_out}")
  print(f"saved: {args.metadata_out}")
  print(f"successful fits: {successful}/{len(rows)}")


if __name__ == "__main__":
  main()
