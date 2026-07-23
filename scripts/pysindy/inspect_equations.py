"""Fit hand-picked configurations under STLSQ overrides and report equations.

Background
----------
The core question this answers: how does the alpha/threshold interaction
change a fitted equation? No simulation, no plotting, no Slurm -- just fit
each requested (alpha, threshold, max_iter) variant on the training data and
report the equation text, term counts, and STLSQ convergence behavior.
Configuration selection happens by eye in the ``outputs/pysindy/global_analysis``
dashboard; pass whatever ``configuration_index`` values you picked there.

Example: does alpha=0 actually remove alpha=0.05's sub-threshold surviving
coefficients?

    .venv/bin/python scripts/pysindy/inspect_equations.py \\
      --configuration-indices 46,47 \\
      --alpha-overrides 0.05,0.0
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(_PROJECT_ROOT))

from load_data.convert import MAT_FILE, TrialData
from load_data.preprocessing import (
  apply_global_zscore,
  channel_traces,
  compute_global_zscore_stats,
)
from models.sindy import (
  SINDyConfig,
  count_subthreshold_terms,
  delay_embed_trajectories,
  equation_text,
  fit_with_iteration_count,
)

csv.field_size_limit(10 * 1024 * 1024)

CHANNEL = 0
DOWNSAMPLE = 2
DEFAULT_GRID_CSV = _PROJECT_ROOT / "outputs/pysindy/global_analysis/raw_grid_deg2357_t20000/raw_grid_merged.csv"


@dataclass(frozen=True)
class Variant:
  """One (alpha, threshold, max_iter) refit to inspect for a configuration."""

  alpha: float
  threshold: float
  max_iter: int

  @property
  def label(self) -> str:
    """Human-readable label for console output and CSV rows."""
    return f"alpha={self.alpha:g}, threshold={self.threshold:g}, max_iter={self.max_iter}"


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
  given.
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


def parse_args() -> argparse.Namespace:
  """Parse CLI arguments."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--configuration-indices", type=str, required=True,
    help="Comma-separated configuration_index values to inspect.",
  )
  parser.add_argument("--grid-csv", type=Path, default=DEFAULT_GRID_CSV)
  parser.add_argument(
    "--metadata-json", type=Path, default=None,
    help="Defaults to the first parts/*_metadata.json next to --grid-csv.",
  )
  parser.add_argument(
    "--alpha-overrides", type=str, default=None,
    help="Comma-separated alpha values to refit, e.g. '0.05,0.0'. "
         "Default: each configuration's own stored alpha.",
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
    "--normalize-columns", action=argparse.BooleanOptionalAction, default=None,
    help="STLSQ normalize_columns for every refit. Default: auto-detected "
         "from --metadata-json's fixed_model_settings.normalize_columns.",
  )
  parser.add_argument(
    "--signal-normalization", type=str, default=None, choices=["none", "global_zscore"],
    help="Signal preprocessing before delay embedding. Default: auto-detected "
         "from --metadata-json's preprocessing.normalization.",
  )
  parser.add_argument(
    "--output-csv", type=Path, default=None,
    help="Default: <grid-csv's parent>/equation_inspection.csv",
  )
  return parser.parse_args()


def main() -> None:
  """Fit every requested variant per configuration and report equations."""
  args = parse_args()
  configuration_indices = parse_int_list(args.configuration_indices)
  if not configuration_indices:
    raise ValueError("--configuration-indices must list at least one configuration_index")

  alpha_overrides = parse_float_list(args.alpha_overrides)
  threshold_overrides = parse_float_list(args.threshold_overrides)
  max_iter_overrides = parse_int_list(args.max_iter_overrides)

  metadata_json = args.metadata_json
  if metadata_json is None:
    candidates = sorted((args.grid_csv.parent / "parts").glob("*_metadata.json"))
    if not candidates:
      raise FileNotFoundError(f"No *_metadata.json found under {args.grid_csv.parent / 'parts'}")
    metadata_json = candidates[0]

  rows = load_grid_rows(args.grid_csv, configuration_indices)
  meta = json.loads(metadata_json.read_text())
  train_ids = meta["split"]["train_trial_ids"]

  normalize_columns = args.normalize_columns
  if normalize_columns is None:
    normalize_columns = bool(meta.get("fixed_model_settings", {}).get("normalize_columns", False))
  signal_normalization = args.signal_normalization
  if signal_normalization is None:
    signal_normalization = meta.get("preprocessing", {}).get("normalization", "none")
  print(f"normalize_columns={normalize_columns}  signal_normalization={signal_normalization!r}")

  print(f"Loading {MAT_FILE} ...", flush=True)
  data = TrialData.load(MAT_FILE)
  dt = DOWNSAMPLE / data.fs

  lowpass_values = sorted({float(r["lowpass_hz"]) for r in rows})
  train_by_lowpass: dict[float, list] = {}
  for lp in lowpass_values:
    raw_train = channel_traces(data, channel=CHANNEL, trials=train_ids,
                                downsample=DOWNSAMPLE, lowpass_hz=lp, normalize="none")
    if signal_normalization == "global_zscore":
      stats = compute_global_zscore_stats(raw_train, channel=CHANNEL)
      train_by_lowpass[lp] = apply_global_zscore(raw_train, stats)
    else:
      train_by_lowpass[lp] = raw_train

  output_rows = []
  for row in rows:
    lp = float(row["lowpass_hz"])
    n_delays, delay = int(row["n_delays"]), int(row["delay_samples"])
    smooth = int(row["smooth_window_samples"])
    degree = int(row["degree"])
    embedded_train = delay_embed_trajectories(train_by_lowpass[lp], n_delays=n_delays, delay=delay)

    variants = build_variants(row, alpha_overrides, threshold_overrides, max_iter_overrides)
    print(f"\n=== Configuration {row['configuration_index']}: degree={degree}, "
          f"lowpass={lp:g} Hz, n_delays={n_delays}, delay={delay}, smoothing={smooth} ===")

    for variant in variants:
      model, n_iterations, converged = fit_with_iteration_count(
        embedded_train, dt=dt,
        config=SINDyConfig(degree=degree, threshold=variant.threshold, alpha=variant.alpha,
                           normalize_columns=normalize_columns, smooth_window=smooth,
                           smoothing_polyorder=3, max_iter=variant.max_iter),
      )
      nonzero_terms, subthreshold_count = count_subthreshold_terms(model, variant.threshold)
      equations = equation_text(model)

      print(f"\n  {variant.label}")
      print(f"    nonzero_terms={nonzero_terms}  subthreshold_count={subthreshold_count}  "
            f"fit_n_iterations={n_iterations}  fit_converged={converged}")
      print(f"    equations: {equations}")

      output_rows.append({
        "configuration_index": row["configuration_index"], "degree": degree, "lowpass_hz": lp,
        "n_delays": n_delays, "delay_samples": delay, "smooth_window_samples": smooth,
        "alpha": variant.alpha, "threshold": variant.threshold, "max_iter": variant.max_iter,
        "nonzero_terms": nonzero_terms, "subthreshold_count": subthreshold_count,
        "fit_n_iterations": n_iterations, "fit_converged": converged,
        "equations": equations,
      })

  output_csv = args.output_csv or (args.grid_csv.parent / "equation_inspection.csv")
  with open(output_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(output_rows[0].keys()))
    w.writeheader()
    w.writerows(output_rows)
  print(f"\nwrote {output_csv}")


if __name__ == "__main__":
  main()
