from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

# Project paths
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GRID = ROOT / "outputs" / "pysindy" / "raw_grid" / "raw_grid_merged.csv"
DEFAULT_REFIT_CSVS = [
  ROOT
  / "outputs"
  / "pysindy"
  / "dense_threshold_refits"
  / "dense_threshold_refits.csv",
  ROOT
  / "outputs"
  / "pysindy"
  / "dense_threshold_refits"
  / "range_pilot_rank90"
  / "parts"
  / "config_0216.csv",
]
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "pysindy" / "raw_grid" / "analysis"

SUMMARY_FIELDS = [
  "configuration_index",
  "stlsq_threshold",
  "nonzero_terms",
  "possible_terms",
  "term_utilization_percent",
  "terms_removed_from_baseline",
]


def read_csv(path: Path) -> list[dict[str, str]]:
  """Read one nonempty CSV into string-valued dictionaries."""
  with path.open(newline="") as file:
    rows = list(csv.DictReader(file))
  if not rows:
    raise ValueError(f"No rows found in {path}.")
  return rows


def collect_threshold_models(
  grid_csv: Path,
  refit_csvs: list[Path],
  configuration_index: int,
) -> list[dict[str, object]]:
  """Collect one baseline equation and all matching threshold refits."""
  baseline_matches = [
    row
    for row in read_csv(grid_csv)
    if int(row["configuration_index"]) == configuration_index
  ]
  if len(baseline_matches) != 1:
    raise ValueError(
      f"Expected one baseline configuration {configuration_index}; "
      f"found {len(baseline_matches)}."
    )
  baseline = baseline_matches[0]
  possible_terms = int(baseline["possible_terms"])
  models: dict[float, dict[str, object]] = {
    0.1: {
      "threshold": 0.1,
      "nonzero_terms": int(baseline["nonzero_terms"]),
      "feature_names": list(json.loads(baseline["feature_names_json"])),
      "coefficients": np.asarray(
        json.loads(baseline["coefficients_json"]), dtype=float
      ),
      "configuration": baseline,
    }
  }

  for path in refit_csvs:
    for row in read_csv(path):
      if int(row["baseline_configuration_index"]) != configuration_index:
        continue
      if row["fit_status"] != "success":
        continue
      threshold = float(row["stlsq_threshold"])
      models[threshold] = {
        "threshold": threshold,
        "nonzero_terms": int(row["refit_nonzero_terms"]),
        "feature_names": list(json.loads(row["feature_names_json"])),
        "coefficients": np.asarray(json.loads(row["coefficients_json"]), dtype=float),
        "configuration": baseline,
      }

  ordered = [models[threshold] for threshold in sorted(models)]
  expected_shape = ordered[0]["coefficients"].shape
  expected_features = ordered[0]["feature_names"]
  for model in ordered:
    if model["coefficients"].shape != expected_shape:
      raise ValueError("Coefficient matrix shape changed across thresholds.")
    if model["feature_names"] != expected_features:
      raise ValueError("Polynomial feature order changed across thresholds.")
    if model["coefficients"].size != possible_terms:
      raise ValueError("Coefficient count does not match possible_terms.")
  return ordered


def write_summary(
  path: Path,
  models: list[dict[str, object]],
  configuration_index: int,
) -> None:
  """Write the term count and utilization at each threshold."""
  possible_terms = int(models[0]["coefficients"].size)
  baseline_terms = int(models[0]["nonzero_terms"])
  rows = []
  for model in models:
    nonzero_terms = int(model["nonzero_terms"])
    rows.append(
      {
        "configuration_index": configuration_index,
        "stlsq_threshold": float(model["threshold"]),
        "nonzero_terms": nonzero_terms,
        "possible_terms": possible_terms,
        "term_utilization_percent": 100 * nonzero_terms / possible_terms,
        "terms_removed_from_baseline": baseline_terms - nonzero_terms,
      }
    )
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
    writer.writeheader()
    writer.writerows(rows)


def plot_threshold_path(
  path: Path,
  models: list[dict[str, object]],
  configuration_index: int,
) -> None:
  """Plot term utilization and coefficient support across thresholds."""
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  from matplotlib.colors import ListedColormap

  thresholds = np.asarray([float(model["threshold"]) for model in models])
  term_counts = np.asarray([int(model["nonzero_terms"]) for model in models])
  coefficient_support = np.vstack(
    [
      np.abs(np.asarray(model["coefficients"], dtype=float)).ravel() > 1e-12
      for model in models
    ]
  )
  possible_terms = coefficient_support.shape[1]
  n_equations, n_features = models[0]["coefficients"].shape
  configuration = models[0]["configuration"]

  figure, (count_axis, support_axis) = plt.subplots(
    2,
    1,
    figsize=(13, 8),
    gridspec_kw={"height_ratios": [1, 2]},
  )
  count_axis.plot(thresholds, term_counts, marker="o", color="#176b57")
  count_axis.set_xscale("log")
  count_axis.set_ylabel("Nonzero terms")
  count_axis.set_xlabel("STLSQ threshold")
  count_axis.set_ylim(0, possible_terms * 1.05)
  count_axis.grid(alpha=0.25)
  utilization_axis = count_axis.twinx()
  utilization_axis.set_ylabel("Term utilization (%)")
  utilization_axis.set_ylim(0, 105)

  support_axis.imshow(
    coefficient_support,
    aspect="auto",
    interpolation="nearest",
    cmap=ListedColormap(["#f1f3f4", "#202124"]),
    vmin=0,
    vmax=1,
  )
  support_axis.set_yticks(np.arange(len(thresholds)))
  support_axis.set_yticklabels([f"{threshold:g}" for threshold in thresholds])
  support_axis.set_ylabel("STLSQ threshold")
  support_axis.set_xlabel(
    "Coefficient index grouped by state equation (dark = retained)"
  )
  for equation_index in range(1, n_equations):
    support_axis.axvline(
      equation_index * n_features - 0.5,
      color="#d93025",
      linewidth=0.6,
      alpha=0.7,
    )
  equation_centers = [index * n_features + (n_features - 1) / 2 for index in range(n_equations)]
  support_axis.set_xticks(equation_centers)
  support_axis.set_xticklabels([f"x{index}'" for index in range(n_equations)])

  figure.suptitle(
    f"Configuration {configuration_index}: threshold sparsification path\n"
    f"degree={configuration['degree']}, delays={configuration['n_delays']}, "
    f"spacing={configuration['delay_samples']} samples, "
    f"LP={configuration['lowpass_hz']} Hz, "
    f"smoothing={configuration['smooth_window_samples']} samples"
  )
  figure.tight_layout(rect=(0, 0, 1, 0.94))
  path.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(path, dpi=180)
  plt.close(figure)


def main() -> None:
  """Parse CLI arguments and plot one configuration's threshold path."""
  parser = argparse.ArgumentParser(
    description="Plot how one stored SINDy equation changes with STLSQ threshold."
  )
  parser.add_argument("--configuration-index", type=int, default=216)
  parser.add_argument("--grid-csv", type=Path, default=DEFAULT_GRID)
  parser.add_argument(
    "--refit-csv",
    type=Path,
    action="append",
    default=None,
    help="Repeat for each threshold-refit CSV; defaults to the saved pilots.",
  )
  parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
  args = parser.parse_args()

  models = collect_threshold_models(
    args.grid_csv,
    refit_csvs=args.refit_csv or DEFAULT_REFIT_CSVS,
    configuration_index=args.configuration_index,
  )
  stem = f"configuration_{args.configuration_index:04d}_threshold_path"
  summary_path = args.output_dir / f"{stem}.csv"
  figure_path = args.output_dir / f"{stem}.png"
  write_summary(summary_path, models, args.configuration_index)
  plot_threshold_path(figure_path, models, args.configuration_index)
  print(f"thresholds plotted: {len(models)}")
  print(f"saved: {summary_path}")
  print(f"saved: {figure_path}")


if __name__ == "__main__":
  main()
