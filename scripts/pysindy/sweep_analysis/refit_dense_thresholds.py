from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

# Project imports
ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
PYSINDY_SCRIPTS = SCRIPTS / "pysindy"
for path in (ROOT, SCRIPTS, PYSINDY_SCRIPTS):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

from load_data.convert import MAT_FILE, TrialData
from load_data.preprocessing import channel_traces
from models.sindy import (
  SINDyConfig,
  count_terms,
  delay_embed_trajectories,
  equation_text,
  fit_sindy_model,
)
from pipeline_utils import parse_float_list
from raw_grid_io import load_successful_grid

# Targeted threshold experiment outputs
DEFAULT_GRID = ROOT / "outputs" / "pysindy" / "raw_grid" / "raw_grid_merged.csv"
DEFAULT_METADATA = (
  ROOT
  / "outputs"
  / "pysindy"
  / "raw_grid"
  / "parts"
  / "part_lp35_degree1_metadata.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "pysindy" / "dense_threshold_refits"

FIELDNAMES = [
  "baseline_configuration_index",
  "baseline_threshold",
  "baseline_nonzero_terms",
  "stlsq_threshold",
  "refit_nonzero_terms",
  "terms_removed",
  "term_reduction_fraction",
  "lowpass_hz",
  "degree",
  "n_delays",
  "delay_samples",
  "smooth_window_samples",
  "alpha",
  "normalize_columns",
  "fit_status",
  "fit_failure_reason",
  "fit_runtime_s",
  "feature_names_json",
  "coefficients_json",
  "equations",
]


def dense_rows(
  rows: list[dict[str, str]],
  minimum_terms: int,
) -> list[dict[str, str]]:
  """Select baseline rows strictly above a nonzero-term count."""
  selected = [row for row in rows if int(row["nonzero_terms"]) > minimum_terms]
  return sorted(selected, key=lambda row: int(row["configuration_index"]))


def select_dense_rank(
  rows: list[dict[str, str]],
  dense_rank: int | None,
) -> list[dict[str, str]]:
  """Select every dense row or one 1-based dense-row rank for an array task."""
  if dense_rank is None:
    return rows
  if not 1 <= dense_rank <= len(rows):
    raise ValueError(f"dense_rank must be between 1 and {len(rows)}.")
  return [rows[dense_rank - 1]]


def output_path(
  output_dir: Path,
  selected_rows: list[dict[str, str]],
  dense_rank: int | None,
) -> Path:
  """Return a unique part path or the single-process combined path."""
  if dense_rank is None:
    return output_dir / "dense_threshold_refits.csv"
  configuration = int(selected_rows[0]["configuration_index"])
  return output_dir / "parts" / f"config_{configuration:04d}.csv"


def initialize_csv(path: Path) -> None:
  """Create an empty targeted-refit CSV with its header."""
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="") as file:
    csv.DictWriter(file, fieldnames=FIELDNAMES).writeheader()


def append_row(path: Path, row: dict[str, object]) -> None:
  """Append one threshold refit result immediately."""
  with path.open("a", newline="") as file:
    csv.DictWriter(file, fieldnames=FIELDNAMES).writerow(row)


def run(args: argparse.Namespace) -> list[dict[str, object]]:
  """Refit dense baseline configurations at higher STLSQ thresholds."""
  thresholds = parse_float_list(args.threshold_list)
  if not thresholds or any(value <= 0 for value in thresholds):
    raise ValueError("STLSQ thresholds must be positive.")
  if any(value == 0.1 for value in thresholds):
    raise ValueError("Threshold 0.1 is already represented by the baseline grid.")

  baseline = dense_rows(
    load_successful_grid(args.grid_csv),
    minimum_terms=args.minimum_terms,
  )
  selected = select_dense_rank(baseline, args.dense_rank)
  path = output_path(args.output_dir, selected, args.dense_rank)
  initialize_csv(path)

  metadata = json.loads(args.metadata_json.read_text())
  if metadata["preprocessing"]["normalization"] != "none":
    raise ValueError("Baseline preprocessing must use normalization='none'.")
  data = TrialData.load(args.mat_file)
  if float(metadata["raw_sampling_hz"]) != float(data.fs):
    raise ValueError("Local sampling frequency does not match baseline metadata.")
  train_ids = [int(value) for value in metadata["split"]["train_trial_ids"]]
  downsample = int(metadata["downsample_factor"])
  channel = int(metadata["channel"])
  dt = downsample / data.fs

  # Reuse preprocessed training traces for rows with the same cutoff.
  traces_by_lowpass: dict[float, list[np.ndarray]] = {}
  results = []
  total = len(selected) * len(thresholds)
  completed = 0
  for baseline_row in selected:
    lowpass_hz = float(baseline_row["lowpass_hz"])
    if lowpass_hz not in traces_by_lowpass:
      traces_by_lowpass[lowpass_hz] = channel_traces(
        data,
        channel=channel,
        trials=train_ids,
        downsample=downsample,
        lowpass_hz=lowpass_hz,
        normalize="none",
      )
    embedded = delay_embed_trajectories(
      traces_by_lowpass[lowpass_hz],
      n_delays=int(baseline_row["n_delays"]),
      delay=int(baseline_row["delay_samples"]),
    )
    baseline_terms = int(baseline_row["nonzero_terms"])

    for threshold in thresholds:
      completed += 1
      started = time.perf_counter()
      row: dict[str, object] = {
        "baseline_configuration_index": int(baseline_row["configuration_index"]),
        "baseline_threshold": 0.1,
        "baseline_nonzero_terms": baseline_terms,
        "stlsq_threshold": threshold,
        "refit_nonzero_terms": 0,
        "terms_removed": "",
        "term_reduction_fraction": "",
        "lowpass_hz": lowpass_hz,
        "degree": int(baseline_row["degree"]),
        "n_delays": int(baseline_row["n_delays"]),
        "delay_samples": int(baseline_row["delay_samples"]),
        "smooth_window_samples": int(baseline_row["smooth_window_samples"]),
        "alpha": 0.05,
        "normalize_columns": True,
        "fit_status": "failed",
        "fit_failure_reason": "",
        "fit_runtime_s": float("nan"),
        "feature_names_json": "",
        "coefficients_json": "",
        "equations": "",
      }
      try:
        model = fit_sindy_model(
          embedded,
          dt=dt,
          config=SINDyConfig(
            degree=int(baseline_row["degree"]),
            threshold=threshold,
            alpha=0.05,
            normalize_columns=True,
            smooth_window=int(baseline_row["smooth_window_samples"]),
            smoothing_polyorder=3,
          ),
        )
        refit_terms = count_terms(model)
        row["fit_status"] = "success"
        row["refit_nonzero_terms"] = refit_terms
        row["terms_removed"] = baseline_terms - refit_terms
        row["term_reduction_fraction"] = (
          (baseline_terms - refit_terms) / baseline_terms
        )
        row["feature_names_json"] = json.dumps(model.get_feature_names())
        row["coefficients_json"] = json.dumps(model.coefficients().tolist())
        row["equations"] = equation_text(model)
      except Exception as exc:
        row["fit_failure_reason"] = str(exc)
      row["fit_runtime_s"] = time.perf_counter() - started
      results.append(row)
      append_row(path, row)
      print(
        f"[{completed}/{total}] baseline={row['baseline_configuration_index']} "
        f"threshold={threshold} status={row['fit_status']} "
        f"terms={baseline_terms}->{row['refit_nonzero_terms']} "
        f"runtime={float(row['fit_runtime_s']):.1f}s",
        flush=True,
      )

  print(f"dense baseline configurations: {len(baseline)}")
  print(f"saved: {path}")
  return results


def main() -> None:
  """Parse CLI arguments and run targeted dense-model refits."""
  parser = argparse.ArgumentParser(
    description=(
      "Refit only baseline models above a term-count limit while changing "
      "the STLSQ threshold."
    )
  )
  parser.add_argument("--mat-file", type=Path, default=MAT_FILE)
  parser.add_argument("--grid-csv", type=Path, default=DEFAULT_GRID)
  parser.add_argument("--metadata-json", type=Path, default=DEFAULT_METADATA)
  parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
  parser.add_argument("--minimum-terms", type=int, default=100)
  parser.add_argument("--threshold-list", default="0.2,0.3")
  parser.add_argument(
    "--dense-rank",
    type=int,
    default=None,
    help="Optional 1-based rank among dense baseline rows for a Slurm task.",
  )
  args = parser.parse_args()
  if args.minimum_terms < 0:
    parser.error("--minimum-terms must be nonnegative.")
  run(args)


if __name__ == "__main__":
  main()
