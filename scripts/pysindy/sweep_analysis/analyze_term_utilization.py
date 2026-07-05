from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_simulation_grid import DEFAULT_GRID, DEFAULT_OUTPUT_DIR, read_csv

# Output schema
SUMMARY_FIELDS = [
  "parameter",
  "value",
  "configurations",
  "mean_utilization_percent",
  "median_utilization_percent",
  "minimum_utilization_percent",
  "maximum_utilization_percent",
  "fully_utilized_configurations",
]

GROUP_PARAMETERS = [
  "degree",
  "lowpass_hz",
  "n_delays",
  "delay_samples",
  "smooth_window_samples",
]


def grouped_utilization(
  rows: list[dict[str, str]],
) -> list[dict[str, object]]:
  """Summarize term utilization by every swept parameter.

  Args:
    rows: Raw-grid rows containing ``term_utilization_percent``.

  Returns:
    Parameter-level utilization summary rows.
  """
  output = []
  for parameter in GROUP_PARAMETERS:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
      groups[row[parameter]].append(float(row["term_utilization_percent"]))
    for value, values in sorted(groups.items(), key=lambda item: float(item[0])):
      array = np.asarray(values, dtype=float)
      output.append(
        {
          "parameter": parameter,
          "value": value,
          "configurations": array.size,
          "mean_utilization_percent": float(np.mean(array)),
          "median_utilization_percent": float(np.median(array)),
          "minimum_utilization_percent": float(np.min(array)),
          "maximum_utilization_percent": float(np.max(array)),
          "fully_utilized_configurations": int(np.sum(np.isclose(array, 100.0))),
        }
      )
  return output


def overall_utilization(rows: list[dict[str, str]]) -> dict[str, object]:
  """Calculate sweep-wide term-utilization statistics."""
  values = np.asarray(
    [float(row["term_utilization_percent"]) for row in rows],
    dtype=float,
  )
  nonzero_terms = sum(int(row["nonzero_terms"]) for row in rows)
  possible_terms = sum(int(row["possible_terms"]) for row in rows)
  return {
    "configurations": len(rows),
    "stlsq_thresholds": sorted(
      {float(row["stlsq_threshold"]) for row in rows}
    ),
    "total_nonzero_terms": nonzero_terms,
    "total_possible_terms": possible_terms,
    "pooled_term_utilization_percent": 100 * nonzero_terms / possible_terms,
    "mean_configuration_utilization_percent": float(np.mean(values)),
    "median_configuration_utilization_percent": float(np.median(values)),
    "minimum_configuration_utilization_percent": float(np.min(values)),
    "maximum_configuration_utilization_percent": float(np.max(values)),
    "fully_utilized_configurations": int(np.sum(np.isclose(values, 100.0))),
  }


def write_summary(
  path: Path,
  rows: list[dict[str, object]],
) -> None:
  """Write grouped term-utilization rows to CSV."""
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
    writer.writeheader()
    writer.writerows(rows)


def plot_utilization(path: Path, rows: list[dict[str, str]]) -> None:
  """Plot utilization against delay count with smoothing ranges visible."""
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  colors = {1: "#176b57", 2: "#d97706", 5: "#4f46e5"}
  figure, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
  for cutoff_index, cutoff in enumerate((35.0, 80.0)):
    for degree_index, degree in enumerate((1, 2, 3)):
      axis = axes[cutoff_index, degree_index]
      for spacing in (1, 2, 5):
        means = []
        minima = []
        maxima = []
        for n_delays in (2, 4, 6, 8):
          values = np.asarray(
            [
              float(row["term_utilization_percent"])
              for row in rows
              if float(row["lowpass_hz"]) == cutoff
              and int(row["degree"]) == degree
              and int(row["delay_samples"]) == spacing
              and int(row["n_delays"]) == n_delays
            ],
            dtype=float,
          )
          if values.size != 3:
            raise ValueError(
              "Expected smoothing windows 0, 5, and 9 for "
              f"cutoff={cutoff}, degree={degree}, spacing={spacing}, "
              f"n_delays={n_delays}; found {values.size}."
            )
          means.append(float(np.mean(values)))
          minima.append(float(np.min(values)))
          maxima.append(float(np.max(values)))
        axis.plot(
          [2, 4, 6, 8],
          means,
          marker="o",
          color=colors[spacing],
          label=f"Spacing {spacing}",
        )
        axis.fill_between(
          [2, 4, 6, 8],
          minima,
          maxima,
          color=colors[spacing],
          alpha=0.12,
        )
      axis.set_ylim(0, 102)
      axis.set_xticks([2, 4, 6, 8])
      axis.grid(alpha=0.25)
      if cutoff_index == 0:
        axis.set_title(f"Polynomial degree {degree}")
      if cutoff_index == 1:
        axis.set_xlabel("Delay coordinates")
      if degree_index == 0:
        axis.set_ylabel(f"{cutoff:g} Hz\nTerm utilization (%)")
  axes[0, -1].legend()
  figure.suptitle(
    "STLSQ term utilization at threshold 100\n"
    "Lines average smoothing 0, 5, and 9; bands show their range"
  )
  figure.tight_layout(rect=(0, 0, 1, 0.94))
  path.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(path, dpi=180)
  plt.close(figure)


def main() -> None:
  """Parse CLI arguments and analyze STLSQ term utilization."""
  parser = argparse.ArgumentParser(
    description="Summarize and plot fitted-library term utilization."
  )
  parser.add_argument("--grid-csv", type=Path, default=DEFAULT_GRID)
  parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
  args = parser.parse_args()

  rows = read_csv(args.grid_csv)
  grouped = grouped_utilization(rows)
  overall = overall_utilization(rows)
  write_summary(args.output_dir / "term_utilization_by_parameter.csv", grouped)
  args.output_dir.mkdir(parents=True, exist_ok=True)
  (args.output_dir / "term_utilization_summary.json").write_text(
    json.dumps(overall, indent=2) + "\n"
  )
  plot_utilization(args.output_dir / "term_utilization_by_delay.png", rows)
  print(
    "pooled utilization: "
    f"{overall['pooled_term_utilization_percent']:.2f}%"
  )
  print(
    "mean configuration utilization: "
    f"{overall['mean_configuration_utilization_percent']:.2f}%"
  )
  print(f"saved analyses: {args.output_dir}")


if __name__ == "__main__":
  main()
