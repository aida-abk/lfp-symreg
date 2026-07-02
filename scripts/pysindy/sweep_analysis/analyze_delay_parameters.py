from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_simulation_grid import (
  DEFAULT_GRID,
  DEFAULT_OUTPUT_DIR,
  DEFAULT_SIMULATION_DIR,
  read_csv,
)
from analyze_smoothing_variability import coefficient_matrix_variability

# Output schemas
SPACING_FIELDS = [
  "lowpass_hz",
  "degree",
  "n_delays",
  "smooth_window_samples",
  "configuration_spacing_1",
  "configuration_spacing_2",
  "configuration_spacing_5",
  "coefficient_count",
  "relative_deviation_spacing_1",
  "relative_deviation_spacing_2",
  "relative_deviation_spacing_5",
  "relative_rms_variability",
  "mean_pairwise_cosine_similarity",
  "minimum_pairwise_cosine_similarity",
]

COMPLETION_FIELDS = [
  "lowpass_hz",
  "degree",
  "n_delays",
  "delay_samples",
  "configurations",
  "trial_simulations",
  "completed_trial_simulations",
  "completion_fraction",
]


def spacing_group_key(row: dict[str, str]) -> tuple[float, int, int, int]:
  """Return parameters held constant across a delay-spacing comparison."""
  return (
    float(row["lowpass_hz"]),
    int(row["degree"]),
    int(row["n_delays"]),
    int(row["smooth_window_samples"]),
  )


def analyze_spacing_variability(
  grid_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
  """Compare coefficient matrices across spacing 1, 2, and 5 samples.

  Args:
    grid_rows: Raw-grid rows containing stored coefficient matrices.

  Returns:
    One coefficient-variability row for every matched spacing triple.
  """
  grouped: dict[
    tuple[float, int, int, int], dict[int, dict[str, str]]
  ] = defaultdict(dict)
  for row in grid_rows:
    key = spacing_group_key(row)
    spacing = int(row["delay_samples"])
    if spacing in grouped[key]:
      raise ValueError(f"Duplicate spacing {spacing} in group {key}.")
    grouped[key][spacing] = row

  output = []
  expected_spacings = {1, 2, 5}
  for key, rows_by_spacing in sorted(grouped.items()):
    if set(rows_by_spacing) != expected_spacings:
      raise ValueError(
        f"Spacing group {key} has {sorted(rows_by_spacing)}; "
        f"expected {sorted(expected_spacings)}."
      )
    ordered_rows = [rows_by_spacing[value] for value in (1, 2, 5)]
    feature_names = [
      list(json.loads(row["feature_names_json"])) for row in ordered_rows
    ]
    if any(names != feature_names[0] for names in feature_names[1:]):
      raise ValueError(f"Feature order changed within spacing group {key}.")
    matrices = np.stack(
      [
        np.asarray(json.loads(row["coefficients_json"]), dtype=float)
        for row in ordered_rows
      ]
    )
    variability = coefficient_matrix_variability(matrices)
    mean_matrix = variability["mean_matrix"]
    mean_norm = max(float(np.linalg.norm(mean_matrix)), np.finfo(float).eps)
    relative_deviations = [
      float(np.linalg.norm(matrix - mean_matrix) / mean_norm)
      for matrix in matrices
    ]
    output.append(
      {
        "lowpass_hz": key[0],
        "degree": key[1],
        "n_delays": key[2],
        "smooth_window_samples": key[3],
        "configuration_spacing_1": int(ordered_rows[0]["configuration_index"]),
        "configuration_spacing_2": int(ordered_rows[1]["configuration_index"]),
        "configuration_spacing_5": int(ordered_rows[2]["configuration_index"]),
        "coefficient_count": int(matrices[0].size),
        "relative_deviation_spacing_1": relative_deviations[0],
        "relative_deviation_spacing_2": relative_deviations[1],
        "relative_deviation_spacing_5": relative_deviations[2],
        "relative_rms_variability": variability["relative_rms_variability"],
        "mean_pairwise_cosine_similarity": variability[
          "mean_pairwise_cosine_similarity"
        ],
        "minimum_pairwise_cosine_similarity": variability[
          "minimum_pairwise_cosine_similarity"
        ],
      }
    )
  return output


def simulation_completion(
  grid_rows: list[dict[str, str]],
  status_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
  """Aggregate trial-level completion by delay count and spacing.

  Args:
    grid_rows: Raw-grid configuration rows.
    status_rows: Trial-level numerical simulation statuses.

  Returns:
    Completion summaries split by cutoff, degree, delay count, and spacing.
  """
  configuration_parameters = {
    int(row["configuration_index"]): (
      float(row["lowpass_hz"]),
      int(row["degree"]),
      int(row["n_delays"]),
      int(row["delay_samples"]),
    )
    for row in grid_rows
  }
  grouped: dict[tuple[float, int, int, int], list[dict[str, str]]] = defaultdict(list)
  configurations: dict[tuple[float, int, int, int], set[int]] = defaultdict(set)
  for row in status_rows:
    configuration_id = int(row["configuration_index"])
    if configuration_id not in configuration_parameters:
      raise ValueError(f"Unknown configuration {configuration_id} in statuses.")
    key = configuration_parameters[configuration_id]
    grouped[key].append(row)
    configurations[key].add(configuration_id)

  output = []
  for key, trials in sorted(grouped.items()):
    completed = sum(row["simulation_status"] == "success" for row in trials)
    output.append(
      {
        "lowpass_hz": key[0],
        "degree": key[1],
        "n_delays": key[2],
        "delay_samples": key[3],
        "configurations": len(configurations[key]),
        "trial_simulations": len(trials),
        "completed_trial_simulations": completed,
        "completion_fraction": completed / len(trials),
      }
    )
  return output


def write_csv(
  path: Path,
  rows: list[dict[str, object]],
  fieldnames: list[str],
) -> None:
  """Write analysis rows with an explicit column order."""
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def plot_delay_geometry(path: Path, grid_rows: list[dict[str, str]]) -> None:
  """Plot embedding duration produced by delay count and spacing."""
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  unique = {
    (int(row["n_delays"]), int(row["delay_samples"])): float(
      row["embedding_span_ms"]
    )
    for row in grid_rows
  }
  colors = {1: "#176b57", 2: "#d97706", 5: "#4f46e5"}
  figure, axis = plt.subplots(figsize=(8, 5))
  for spacing in (1, 2, 5):
    counts = (2, 4, 6, 8)
    spans = [unique[(count, spacing)] for count in counts]
    axis.plot(
      counts,
      spans,
      marker="o",
      color=colors[spacing],
      label=f"Spacing {spacing} sample{'s' if spacing > 1 else ''}",
    )
    for count, span in zip(counts, spans):
      axis.annotate(f"{span:g} ms", (count, span), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=8)
  axis.set_xticks([2, 4, 6, 8])
  axis.set_xlabel("Number of delay coordinates")
  axis.set_ylabel("Embedding span (ms)")
  axis.set_title("Time history represented by the delay embedding")
  axis.grid(alpha=0.25)
  axis.legend()
  figure.tight_layout()
  path.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(path, dpi=180)
  plt.close(figure)


def plot_library_growth(path: Path, grid_rows: list[dict[str, str]]) -> None:
  """Plot maximum fitted coefficients as delay count increases."""
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  unique = {
    (int(row["degree"]), int(row["n_delays"])): int(row["possible_terms"])
    for row in grid_rows
  }
  colors = {1: "#176b57", 2: "#d97706", 3: "#4f46e5"}
  figure, axis = plt.subplots(figsize=(8, 5))
  for degree in (1, 2, 3):
    counts = (2, 4, 6, 8)
    terms = [unique[(degree, count)] for count in counts]
    axis.plot(
      counts,
      terms,
      marker="o",
      color=colors[degree],
      label=f"Degree {degree}",
    )
    for count, term_count in zip(counts, terms):
      axis.annotate(str(term_count), (count, term_count), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=8)
  axis.set_yscale("log")
  axis.set_xticks([2, 4, 6, 8])
  axis.set_xlabel("Number of delay coordinates")
  axis.set_ylabel("Possible equation coefficients (log scale)")
  axis.set_title("Polynomial-library growth with additional delays")
  axis.grid(alpha=0.25)
  axis.legend()
  figure.tight_layout()
  path.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(path, dpi=180)
  plt.close(figure)


def plot_spacing_variability(
  path: Path,
  rows: list[dict[str, object]],
) -> None:
  """Plot coefficient sensitivity to spacing on a shared y-axis."""
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  colors = {0: "#176b57", 5: "#d97706", 9: "#4f46e5"}
  figure, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
  for cutoff_index, cutoff in enumerate((35.0, 80.0)):
    for degree_index, degree in enumerate((1, 2, 3)):
      axis = axes[cutoff_index, degree_index]
      for smoothing in (0, 5, 9):
        selected = sorted(
          [
            row
            for row in rows
            if float(row["lowpass_hz"]) == cutoff
            and int(row["degree"]) == degree
            and int(row["smooth_window_samples"]) == smoothing
          ],
          key=lambda row: int(row["n_delays"]),
        )
        axis.plot(
          [int(row["n_delays"]) for row in selected],
          [100 * float(row["relative_rms_variability"]) for row in selected],
          marker="o",
          color=colors[smoothing],
          label=f"Smoothing {smoothing}",
        )
      axis.set_xticks([2, 4, 6, 8])
      axis.grid(alpha=0.25)
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
    "Coefficient variability across delay spacings 1, 2, and 5 samples\n"
    "Each line fixes the derivative-smoothing window"
  )
  figure.tight_layout(rect=(0, 0, 1, 0.94))
  path.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(path, dpi=180)
  plt.close(figure)


def plot_spacing_deviations(
  path: Path,
  rows: list[dict[str, object]],
  smoothing: int,
) -> None:
  """Plot each spacing's coefficient distance from its spacing-group mean.

  Args:
    path: Destination PNG path.
    rows: Matched spacing-group analysis rows.
    smoothing: Fixed derivative-smoothing window in samples.
  """
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  colors = {2: "#176b57", 4: "#d97706", 6: "#4f46e5", 8: "#b91c1c"}
  figure, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
  for cutoff_index, cutoff in enumerate((35.0, 80.0)):
    for degree_index, degree in enumerate((1, 2, 3)):
      axis = axes[cutoff_index, degree_index]
      for n_delays in (2, 4, 6, 8):
        matches = [
          row
          for row in rows
          if float(row["lowpass_hz"]) == cutoff
          and int(row["degree"]) == degree
          and int(row["smooth_window_samples"]) == smoothing
          and int(row["n_delays"]) == n_delays
        ]
        if len(matches) != 1:
          raise ValueError(
            "Expected one spacing group for "
            f"cutoff={cutoff}, degree={degree}, smoothing={smoothing}, "
            f"n_delays={n_delays}; found {len(matches)}."
          )
        row = matches[0]
        axis.plot(
          [1, 2, 5],
          [
            100 * float(row["relative_deviation_spacing_1"]),
            100 * float(row["relative_deviation_spacing_2"]),
            100 * float(row["relative_deviation_spacing_5"]),
          ],
          marker="o",
          color=colors[n_delays],
          label=f"{n_delays} delays",
        )
      axis.set_xticks([1, 2, 5])
      axis.grid(alpha=0.25)
      if cutoff_index == 0:
        axis.set_title(f"Polynomial degree {degree}")
      if cutoff_index == 1:
        axis.set_xlabel("Delay spacing (samples)")
      if degree_index == 0:
        axis.set_ylabel(
          f"{cutoff:g} Hz\nCoefficient distance from mean (%)"
        )
  axes[0, -1].legend()
  figure.suptitle(
    "Spacing-specific coefficient sensitivity\n"
    f"Derivative smoothing fixed at {smoothing} samples"
  )
  figure.tight_layout(rect=(0, 0, 1, 0.94))
  path.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(path, dpi=180)
  plt.close(figure)


def plot_completion_heatmaps(
  path: Path,
  rows: list[dict[str, object]],
) -> None:
  """Plot full-trial numerical simulation completion by delay settings."""
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  figure, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
  image = None
  for cutoff_index, cutoff in enumerate((35.0, 80.0)):
    for degree_index, degree in enumerate((1, 2, 3)):
      axis = axes[cutoff_index, degree_index]
      matrix = np.full((3, 4), np.nan)
      for row in rows:
        if (
          float(row["lowpass_hz"]) == cutoff
          and int(row["degree"]) == degree
        ):
          spacing_index = (1, 2, 5).index(int(row["delay_samples"]))
          delay_index = (2, 4, 6, 8).index(int(row["n_delays"]))
          matrix[spacing_index, delay_index] = float(row["completion_fraction"])
      image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
      for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
          value = matrix[row_index, column_index]
          axis.text(
            column_index,
            row_index,
            f"{100 * value:.0f}%",
            ha="center",
            va="center",
            color="white" if value < 0.65 else "black",
            fontsize=9,
          )
      axis.set_xticks(range(4), labels=[2, 4, 6, 8])
      axis.set_yticks(range(3), labels=[1, 2, 5])
      if cutoff_index == 0:
        axis.set_title(f"Polynomial degree {degree}")
      if cutoff_index == 1:
        axis.set_xlabel("Delay coordinates")
      if degree_index == 0:
        axis.set_ylabel(f"{cutoff:g} Hz\nDelay spacing (samples)")
  colorbar_axis = figure.add_axes((0.91, 0.15, 0.018, 0.68))
  colorbar = figure.colorbar(image, cax=colorbar_axis)
  colorbar.set_label("Completed full-trial simulations (fraction)")
  figure.suptitle(
    "Numerical simulation completion by delay count and spacing\n"
    "Each cell pools smoothing windows 0, 5, and 9"
  )
  figure.subplots_adjust(left=0.08, right=0.88, bottom=0.08, top=0.88,
                         wspace=0.18, hspace=0.22)
  path.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(path, dpi=180)
  plt.close(figure)


def main() -> None:
  """Parse CLI arguments and analyze delay-count and spacing effects."""
  parser = argparse.ArgumentParser(
    description="Analyze timing, complexity, coefficients, and simulation outcomes."
  )
  parser.add_argument("--grid-csv", type=Path, default=DEFAULT_GRID)
  parser.add_argument(
    "--status-csv",
    type=Path,
    default=DEFAULT_SIMULATION_DIR / "simulation_status_merged.csv",
  )
  parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
  args = parser.parse_args()

  grid_rows = read_csv(args.grid_csv)
  status_rows = read_csv(args.status_csv)
  spacing_rows = analyze_spacing_variability(grid_rows)
  completion_rows = simulation_completion(grid_rows, status_rows)

  write_csv(
    args.output_dir / "delay_spacing_coefficient_variability.csv",
    spacing_rows,
    SPACING_FIELDS,
  )
  write_csv(
    args.output_dir / "delay_parameter_simulation_completion.csv",
    completion_rows,
    COMPLETION_FIELDS,
  )
  plot_delay_geometry(args.output_dir / "delay_embedding_geometry.png", grid_rows)
  plot_library_growth(args.output_dir / "delay_library_growth.png", grid_rows)
  plot_spacing_variability(
    args.output_dir / "delay_spacing_coefficient_variability.png",
    spacing_rows,
  )
  for smoothing in (0, 5, 9):
    plot_spacing_deviations(
      args.output_dir
      / f"delay_spacing_sensitivity_smoothing_{smoothing}.png",
      spacing_rows,
      smoothing=smoothing,
    )
  plot_completion_heatmaps(
    args.output_dir / "delay_parameter_simulation_completion.png",
    completion_rows,
  )
  print(f"spacing triples analyzed: {len(spacing_rows)}")
  print(f"simulation parameter groups: {len(completion_rows)}")
  print(f"saved analyses: {args.output_dir}")


if __name__ == "__main__":
  main()
