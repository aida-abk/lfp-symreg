from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from analyze_simulation_grid import DEFAULT_GRID, DEFAULT_OUTPUT_DIR, read_csv

# Output schema
FIELDNAMES = [
  "lowpass_hz",
  "degree",
  "n_delays",
  "delay_samples",
  "configuration_smooth_0",
  "configuration_smooth_5",
  "configuration_smooth_9",
  "coefficient_count",
  "max_variance",
  "max_standard_deviation",
  "max_variance_equation_index",
  "max_variance_feature_index",
  "max_variance_feature_name",
  "mean_coefficient_at_max_variance",
  "mean_elementwise_variance",
  "relative_rms_variability",
  "mean_pairwise_cosine_similarity",
  "minimum_pairwise_cosine_similarity",
]


def smoothing_group_key(row: dict[str, str]) -> tuple[float, int, int, int]:
  """Return parameters held constant within one smoothing comparison."""
  return (
    float(row["lowpass_hz"]),
    int(row["degree"]),
    int(row["n_delays"]),
    int(row["delay_samples"]),
  )


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
  """Return cosine similarity between flattened coefficient matrices."""
  left_flat = left.ravel()
  right_flat = right.ravel()
  denominator = max(
    float(np.linalg.norm(left_flat) * np.linalg.norm(right_flat)),
    np.finfo(float).eps,
  )
  return float(np.dot(left_flat, right_flat) / denominator)


def coefficient_matrix_variability(matrices: np.ndarray) -> dict[str, object]:
  """Calculate whole-matrix variability across matched fitted equations.

  Args:
    matrices: Coefficient matrices with shape
      ``(models, equations, library_terms)``. All models must use the same
      state dimension and feature ordering.

  Returns:
    Mean and variance matrices, dimensionless relative RMS variability, and
    pairwise cosine similarities.
  """
  if matrices.ndim != 3 or matrices.shape[0] < 2:
    raise ValueError(
      "Expected at least two matched 2D coefficient matrices, "
      f"got shape {matrices.shape}."
    )
  mean_matrix = np.mean(matrices, axis=0)
  squared_differences = (matrices - mean_matrix) ** 2
  variance_matrix = np.mean(squared_differences, axis=0)
  rms_variability = float(np.sqrt(np.mean(squared_differences)))
  mean_matrix_rms = max(
    float(np.sqrt(np.mean(mean_matrix**2))),
    np.finfo(float).eps,
  )
  pairwise_cosines = [
    cosine_similarity(matrices[left], matrices[right])
    for left, right in combinations(range(matrices.shape[0]), 2)
  ]
  return {
    "mean_matrix": mean_matrix,
    "variance_matrix": variance_matrix,
    "relative_rms_variability": rms_variability / mean_matrix_rms,
    "mean_pairwise_cosine_similarity": float(np.mean(pairwise_cosines)),
    "minimum_pairwise_cosine_similarity": float(np.min(pairwise_cosines)),
  }


def analyze_smoothing_group(
  key: tuple[float, int, int, int],
  rows_by_smoothing: dict[int, dict[str, str]],
) -> dict[str, object]:
  """Calculate the requested coefficient-variance matrix for one triple."""
  expected_windows = {0, 5, 9}
  if set(rows_by_smoothing) != expected_windows:
    raise ValueError(
      f"Smoothing group {key} has windows {sorted(rows_by_smoothing)}; "
      f"expected {sorted(expected_windows)}."
    )

  ordered_rows = [rows_by_smoothing[window] for window in (0, 5, 9)]
  matrices = np.stack(
    [
      np.asarray(json.loads(row["coefficients_json"]), dtype=float)
      for row in ordered_rows
    ]
  )
  feature_names = list(json.loads(ordered_rows[0]["feature_names_json"]))
  if any(
    list(json.loads(row["feature_names_json"])) != feature_names
    for row in ordered_rows[1:]
  ):
    raise ValueError(f"Feature order changed within smoothing group {key}.")

  # User-requested population variance across smoothing windows.
  variability = coefficient_matrix_variability(matrices)
  mean_matrix = variability["mean_matrix"]
  variance_matrix = variability["variance_matrix"]
  max_location = np.unravel_index(np.argmax(variance_matrix), variance_matrix.shape)
  max_variance = float(variance_matrix[max_location])

  return {
    "lowpass_hz": key[0],
    "degree": key[1],
    "n_delays": key[2],
    "delay_samples": key[3],
    "configuration_smooth_0": int(ordered_rows[0]["configuration_index"]),
    "configuration_smooth_5": int(ordered_rows[1]["configuration_index"]),
    "configuration_smooth_9": int(ordered_rows[2]["configuration_index"]),
    "coefficient_count": int(matrices[0].size),
    "max_variance": max_variance,
    "max_standard_deviation": float(np.sqrt(max_variance)),
    "max_variance_equation_index": int(max_location[0]),
    "max_variance_feature_index": int(max_location[1]),
    "max_variance_feature_name": feature_names[max_location[1]],
    "mean_coefficient_at_max_variance": float(mean_matrix[max_location]),
    "mean_elementwise_variance": float(np.mean(variance_matrix)),
    "relative_rms_variability": variability["relative_rms_variability"],
    "mean_pairwise_cosine_similarity": variability[
      "mean_pairwise_cosine_similarity"
    ],
    "minimum_pairwise_cosine_similarity": variability[
      "minimum_pairwise_cosine_similarity"
    ],
  }


def analyze_grid(grid_rows: list[dict[str, str]]) -> list[dict[str, object]]:
  """Analyze every complete smoothing triple in the raw parameter grid."""
  grouped: dict[
    tuple[float, int, int, int],
    dict[int, dict[str, str]],
  ] = defaultdict(dict)
  for row in grid_rows:
    key = smoothing_group_key(row)
    smoothing = int(row["smooth_window_samples"])
    if smoothing in grouped[key]:
      raise ValueError(f"Duplicate smoothing window {smoothing} in group {key}.")
    grouped[key][smoothing] = row
  return [
    analyze_smoothing_group(key, rows_by_smoothing)
    for key, rows_by_smoothing in sorted(grouped.items())
  ]


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
  """Write smoothing-variability rows with a stable schema."""
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
  """Summarize dimensionless smoothing variability by cutoff and degree."""
  output: dict[str, object] = {}
  for cutoff in sorted({float(row["lowpass_hz"]) for row in rows}):
    cutoff_rows = [row for row in rows if float(row["lowpass_hz"]) == cutoff]
    cutoff_summary = {}
    for degree in sorted({int(row["degree"]) for row in cutoff_rows}):
      degree_rows = [row for row in cutoff_rows if int(row["degree"]) == degree]
      relative = np.asarray(
        [float(row["relative_rms_variability"]) for row in degree_rows]
      )
      cosine = np.asarray(
        [float(row["minimum_pairwise_cosine_similarity"]) for row in degree_rows]
      )
      cutoff_summary[f"degree_{degree}"] = {
        "groups": len(degree_rows),
        "median_relative_rms_variability": float(np.median(relative)),
        "maximum_relative_rms_variability": float(np.max(relative)),
        "median_minimum_pairwise_cosine_similarity": float(np.median(cosine)),
        "minimum_pairwise_cosine_similarity": float(np.min(cosine)),
      }
    output[f"lowpass_{cutoff:g}_hz"] = cutoff_summary
  return output


def plot_degree_panel(axis, rows: list[dict[str, object]], degree: int) -> None:
  """Plot smoothing variability by delay count and spacing on one axis."""
  colors = {1: "#176b57", 2: "#d97706", 5: "#4f46e5"}
  degree_rows = [row for row in rows if int(row["degree"]) == degree]
  for delay in (1, 2, 5):
    delay_rows = sorted(
      [row for row in degree_rows if int(row["delay_samples"]) == delay],
      key=lambda row: int(row["n_delays"]),
    )
    axis.plot(
      [int(row["n_delays"]) for row in delay_rows],
      [100 * float(row["relative_rms_variability"]) for row in delay_rows],
      marker="o",
      label=f"Spacing {delay}",
      color=colors[delay],
    )
  axis.set_xticks([2, 4, 6, 8])
  axis.grid(alpha=0.25)


def plot_cutoff(
  path: Path,
  rows: list[dict[str, object]],
  lowpass_hz: float,
) -> None:
  """Plot dimensionless smoothing variability for one low-pass branch."""
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  cutoff_rows = [
    row for row in rows if float(row["lowpass_hz"]) == lowpass_hz
  ]
  figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
  for axis, degree in zip(axes, (1, 2, 3)):
    plot_degree_panel(axis, cutoff_rows, degree)
    axis.set_title(f"Polynomial degree {degree}")
    axis.set_xlabel("Delay coordinates")
  axes[0].set_ylabel("Relative coefficient RMS variability (%)")
  axes[-1].legend()
  figure.suptitle(
    f"{lowpass_hz:g} Hz models: coefficient variability across smoothing 0, 5, 9"
  )
  figure.tight_layout(rect=(0, 0, 1, 0.93))
  path.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(path, dpi=180)
  plt.close(figure)


def plot_cutoff_comparison(
  path: Path,
  rows: list[dict[str, object]],
) -> None:
  """Plot 35 and 80 Hz smoothing variability on one shared y-axis."""
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  figure, axes = plt.subplots(
    2,
    3,
    figsize=(14, 8),
    sharex=True,
    sharey=True,
  )
  for cutoff_index, cutoff in enumerate((35.0, 80.0)):
    cutoff_rows = [
      row for row in rows if float(row["lowpass_hz"]) == cutoff
    ]
    for degree_index, degree in enumerate((1, 2, 3)):
      axis = axes[cutoff_index, degree_index]
      plot_degree_panel(axis, cutoff_rows, degree)
      if cutoff_index == 0:
        axis.set_title(f"Polynomial degree {degree}")
      if cutoff_index == 1:
        axis.set_xlabel("Delay coordinates")
      if degree_index == 0:
        axis.set_ylabel(
          f"{cutoff:g} Hz\nRelative coefficient RMS variability (%)"
        )
  axes[0, -1].legend()
  figure.suptitle(
    "Coefficient variability across smoothing windows 0, 5, and 9\n"
    "Shared y-axis for 35 Hz and 80 Hz models"
  )
  figure.tight_layout(rect=(0, 0, 1, 0.94))
  path.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(path, dpi=180)
  plt.close(figure)


def main() -> None:
  """Parse CLI arguments and analyze smoothing-window variability."""
  parser = argparse.ArgumentParser(
    description="Measure coefficient variability across smoothing windows 0, 5, and 9."
  )
  parser.add_argument("--grid-csv", type=Path, default=DEFAULT_GRID)
  parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
  args = parser.parse_args()

  rows = analyze_grid(read_csv(args.grid_csv))
  rows_35hz = [row for row in rows if float(row["lowpass_hz"]) == 35.0]
  rows_80hz = [row for row in rows if float(row["lowpass_hz"]) == 80.0]
  write_rows(args.output_dir / "smoothing_variability_all.csv", rows)
  write_rows(args.output_dir / "smoothing_variability_35hz.csv", rows_35hz)
  write_rows(args.output_dir / "smoothing_variability_80hz.csv", rows_80hz)
  summary = summarize(rows)
  (args.output_dir / "smoothing_variability_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n"
  )
  plot_cutoff(
    args.output_dir / "smoothing_variability_35hz.png",
    rows,
    lowpass_hz=35.0,
  )
  plot_cutoff(
    args.output_dir / "smoothing_variability_80hz.png",
    rows,
    lowpass_hz=80.0,
  )
  plot_cutoff_comparison(
    args.output_dir / "smoothing_variability_35hz_vs_80hz.png",
    rows,
  )
  print(f"smoothing triples analyzed: {len(rows)}")
  print(f"35 Hz triples analyzed: {len(rows_35hz)}")
  print(f"80 Hz triples analyzed: {len(rows_80hz)}")
  print(f"saved analyses: {args.output_dir}")


if __name__ == "__main__":
  main()
