from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from analyze_simulation_grid import (
  DEFAULT_GRID,
  DEFAULT_OUTPUT_DIR,
  DEFAULT_SIMULATION_DIR,
  configuration_summary,
  read_csv,
  validate_inputs,
  write_csv,
)

# Paired comparison schemas
PAIR_FIELDS = [
  "degree",
  "n_delays",
  "delay_samples",
  "smooth_window_samples",
  "configuration_35hz",
  "configuration_80hz",
  "completed_trials_35hz",
  "completed_trials_80hz",
  "completed_trials_difference_80_minus_35",
  "all_9_completed_35hz",
  "all_9_completed_80hz",
  "coefficient_cosine_similarity",
  "coefficient_relative_l2_change_from_35hz",
]

TRIAL_PAIR_FIELDS = [
  "degree",
  "n_delays",
  "delay_samples",
  "smooth_window_samples",
  "test_trial_id",
  "completed_35hz",
  "completed_80hz",
  "reached_fraction_35hz",
  "reached_fraction_80hz",
  "reached_fraction_difference_80_minus_35",
]

GROUP_FIELDS = [
  "scope",
  "degree",
  "lowpass_hz",
  "configurations",
  "all_9_completed_count",
  "all_9_completed_fraction",
  "simulation_attempts",
  "completed_simulations",
  "simulation_completion_fraction",
  "mean_reached_fraction",
]

PAIRED_SCOPE_FIELDS = [
  "scope",
  "degree",
  "matched_configuration_pairs",
  "configuration_wins_35hz",
  "configuration_wins_80hz",
  "configuration_ties",
  "mean_completed_trial_difference_80_minus_35",
  "all9_35hz_only",
  "all9_80hz_only",
  "all9_discordance_exact_p",
  "matched_trial_pairs",
  "trial_completion_35hz_only",
  "trial_completion_80hz_only",
  "trial_completion_discordance_exact_p",
  "mean_reached_fraction_difference_80_minus_35",
]

COEFFICIENT_GROUP_FIELDS = [
  "parameter",
  "value",
  "matched_equation_pairs",
  "mean_coefficient_cosine_similarity",
  "median_coefficient_cosine_similarity",
  "minimum_coefficient_cosine_similarity",
  "mean_relative_l2_change_from_35hz",
  "median_relative_l2_change_from_35hz",
  "maximum_relative_l2_change_from_35hz",
]

COEFFICIENT_GROUP_PARAMETERS = [
  "degree",
  "n_delays",
  "delay_samples",
  "smooth_window_samples",
]


def pairing_key(row: dict[str, object]) -> tuple[int, int, int, int]:
  """Return parameters that are identical across the two filter cutoffs."""
  return (
    int(row["degree"]),
    int(row["n_delays"]),
    int(row["delay_samples"]),
    int(row["smooth_window_samples"]),
  )


def coefficient_comparison(
  row_35hz: dict[str, str],
  row_80hz: dict[str, str],
) -> tuple[float, float]:
  """Return cosine similarity and relative L2 coefficient change."""
  left = np.asarray(json.loads(row_35hz["coefficients_json"]), dtype=float).ravel()
  right = np.asarray(json.loads(row_80hz["coefficients_json"]), dtype=float).ravel()
  if left.shape != right.shape:
    raise ValueError("Paired coefficient matrices have different shapes.")
  left_norm = float(np.linalg.norm(left))
  right_norm = float(np.linalg.norm(right))
  denominator = max(left_norm * right_norm, np.finfo(float).eps)
  cosine = float(np.dot(left, right) / denominator)
  relative_l2 = float(np.linalg.norm(right - left)) / max(
    left_norm, np.finfo(float).eps
  )
  return cosine, relative_l2


def build_configuration_pairs(
  configuration_rows: list[dict[str, object]],
  grid_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
  """Create one paired model row per shared non-filter configuration."""
  grid_by_id = {int(row["configuration_index"]): row for row in grid_rows}
  grouped: dict[tuple[int, int, int, int], dict[float, dict[str, object]]] = (
    defaultdict(dict)
  )
  for row in configuration_rows:
    grouped[pairing_key(row)][float(row["lowpass_hz"])] = row

  output = []
  for key, cutoff_rows in sorted(grouped.items()):
    if set(cutoff_rows) != {35.0, 80.0}:
      raise ValueError(f"Incomplete cutoff pair for parameters {key}.")
    left = cutoff_rows[35.0]
    right = cutoff_rows[80.0]
    cosine, relative_l2 = coefficient_comparison(
      grid_by_id[int(left["configuration_index"])],
      grid_by_id[int(right["configuration_index"])],
    )
    output.append(
      {
        "degree": key[0],
        "n_delays": key[1],
        "delay_samples": key[2],
        "smooth_window_samples": key[3],
        "configuration_35hz": int(left["configuration_index"]),
        "configuration_80hz": int(right["configuration_index"]),
        "completed_trials_35hz": int(left["completed_simulations"]),
        "completed_trials_80hz": int(right["completed_simulations"]),
        "completed_trials_difference_80_minus_35": (
          int(right["completed_simulations"]) - int(left["completed_simulations"])
        ),
        "all_9_completed_35hz": bool(left["all_held_out_completed"]),
        "all_9_completed_80hz": bool(right["all_held_out_completed"]),
        "coefficient_cosine_similarity": cosine,
        "coefficient_relative_l2_change_from_35hz": relative_l2,
      }
    )
  return output


def build_trial_pairs(
  configuration_rows: list[dict[str, object]],
  status_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
  """Pair identical held-out trials across matched 35 and 80 Hz models."""
  configuration_by_id = {
    int(row["configuration_index"]): row for row in configuration_rows
  }
  records: dict[
    tuple[tuple[int, int, int, int], int],
    dict[float, dict[str, str]],
  ] = defaultdict(dict)
  for row in status_rows:
    configuration = configuration_by_id[int(row["configuration_index"])]
    key = (pairing_key(configuration), int(row["test_trial_id"]))
    records[key][float(configuration["lowpass_hz"])] = row

  output = []
  for (parameters, trial_id), cutoff_rows in sorted(records.items()):
    if set(cutoff_rows) != {35.0, 80.0}:
      raise ValueError(
        f"Incomplete trial cutoff pair for parameters {parameters}, trial {trial_id}."
      )
    left = cutoff_rows[35.0]
    right = cutoff_rows[80.0]
    left_reached = float(left["reached_duration_s"]) / float(
      left["requested_duration_s"]
    )
    right_reached = float(right["reached_duration_s"]) / float(
      right["requested_duration_s"]
    )
    output.append(
      {
        "degree": parameters[0],
        "n_delays": parameters[1],
        "delay_samples": parameters[2],
        "smooth_window_samples": parameters[3],
        "test_trial_id": trial_id,
        "completed_35hz": left["simulation_status"] == "success",
        "completed_80hz": right["simulation_status"] == "success",
        "reached_fraction_35hz": left_reached,
        "reached_fraction_80hz": right_reached,
        "reached_fraction_difference_80_minus_35": right_reached - left_reached,
      }
    )
  return output


def grouped_cutoff_summary(
  configuration_rows: list[dict[str, object]],
  trial_pairs: list[dict[str, object]],
) -> list[dict[str, object]]:
  """Summarize each cutoff overall and separately by polynomial degree."""
  output = []
  scopes: list[tuple[str, int | None]] = [("overall", None), *[
    ("degree", degree) for degree in sorted({int(row["degree"]) for row in configuration_rows})
  ]]
  for scope, degree in scopes:
    configurations = [
      row
      for row in configuration_rows
      if degree is None or int(row["degree"]) == degree
    ]
    paired_trials = [
      row for row in trial_pairs if degree is None or int(row["degree"]) == degree
    ]
    for cutoff in (35.0, 80.0):
      cutoff_configurations = [
        row for row in configurations if float(row["lowpass_hz"]) == cutoff
      ]
      completed_key = f"completed_{int(cutoff)}hz"
      reached_key = f"reached_fraction_{int(cutoff)}hz"
      attempts = len(paired_trials)
      completed = sum(bool(row[completed_key]) for row in paired_trials)
      all_completed = sum(
        bool(row["all_held_out_completed"]) for row in cutoff_configurations
      )
      output.append(
        {
          "scope": scope,
          "degree": "all" if degree is None else degree,
          "lowpass_hz": cutoff,
          "configurations": len(cutoff_configurations),
          "all_9_completed_count": all_completed,
          "all_9_completed_fraction": all_completed / len(cutoff_configurations),
          "simulation_attempts": attempts,
          "completed_simulations": completed,
          "simulation_completion_fraction": completed / attempts,
          "mean_reached_fraction": float(
            np.mean([float(row[reached_key]) for row in paired_trials])
          ),
        }
      )
  return output


def exact_paired_pvalue(left_only: int, right_only: int) -> float:
  """Return a two-sided exact binomial test for discordant paired outcomes."""
  discordant = left_only + right_only
  if discordant == 0:
    return 1.0
  return float(binomtest(left_only, discordant, p=0.5).pvalue)


def paired_scope_summary(
  configuration_pairs: list[dict[str, object]],
  trial_pairs: list[dict[str, object]],
) -> list[dict[str, object]]:
  """Summarize paired cutoff differences overall and within each degree."""
  output = []
  scopes: list[tuple[str, int | None]] = [("overall", None), *[
    ("degree", degree)
    for degree in sorted({int(row["degree"]) for row in configuration_pairs})
  ]]
  for scope, degree in scopes:
    configurations = [
      row
      for row in configuration_pairs
      if degree is None or int(row["degree"]) == degree
    ]
    trials = [
      row for row in trial_pairs if degree is None or int(row["degree"]) == degree
    ]
    differences = np.asarray(
      [int(row["completed_trials_difference_80_minus_35"]) for row in configurations]
    )
    all9_35_only = sum(
      bool(row["all_9_completed_35hz"]) and not bool(row["all_9_completed_80hz"])
      for row in configurations
    )
    all9_80_only = sum(
      bool(row["all_9_completed_80hz"]) and not bool(row["all_9_completed_35hz"])
      for row in configurations
    )
    trial_35_only = sum(
      bool(row["completed_35hz"]) and not bool(row["completed_80hz"])
      for row in trials
    )
    trial_80_only = sum(
      bool(row["completed_80hz"]) and not bool(row["completed_35hz"])
      for row in trials
    )
    output.append(
      {
        "scope": scope,
        "degree": "all" if degree is None else degree,
        "matched_configuration_pairs": len(configurations),
        "configuration_wins_35hz": int(np.sum(differences < 0)),
        "configuration_wins_80hz": int(np.sum(differences > 0)),
        "configuration_ties": int(np.sum(differences == 0)),
        "mean_completed_trial_difference_80_minus_35": float(
          np.mean(differences)
        ),
        "all9_35hz_only": all9_35_only,
        "all9_80hz_only": all9_80_only,
        "all9_discordance_exact_p": exact_paired_pvalue(
          all9_35_only, all9_80_only
        ),
        "matched_trial_pairs": len(trials),
        "trial_completion_35hz_only": trial_35_only,
        "trial_completion_80hz_only": trial_80_only,
        "trial_completion_discordance_exact_p": exact_paired_pvalue(
          trial_35_only, trial_80_only
        ),
        "mean_reached_fraction_difference_80_minus_35": float(
          np.mean(
            [float(row["reached_fraction_difference_80_minus_35"]) for row in trials]
          )
        ),
      }
    )
  return output


def coefficient_parameter_summary(
  configuration_pairs: list[dict[str, object]],
) -> list[dict[str, object]]:
  """Summarize paired coefficient differences by each swept parameter."""
  output = []
  for parameter in COEFFICIENT_GROUP_PARAMETERS:
    groups: dict[object, list[dict[str, object]]] = defaultdict(list)
    for row in configuration_pairs:
      groups[row[parameter]].append(row)
    for value, rows in sorted(groups.items(), key=lambda item: item[0]):
      cosine = np.asarray(
        [float(row["coefficient_cosine_similarity"]) for row in rows]
      )
      relative_l2 = np.asarray(
        [
          float(row["coefficient_relative_l2_change_from_35hz"])
          for row in rows
        ]
      )
      output.append(
        {
          "parameter": parameter,
          "value": value,
          "matched_equation_pairs": len(rows),
          "mean_coefficient_cosine_similarity": float(np.mean(cosine)),
          "median_coefficient_cosine_similarity": float(np.median(cosine)),
          "minimum_coefficient_cosine_similarity": float(np.min(cosine)),
          "mean_relative_l2_change_from_35hz": float(np.mean(relative_l2)),
          "median_relative_l2_change_from_35hz": float(np.median(relative_l2)),
          "maximum_relative_l2_change_from_35hz": float(np.max(relative_l2)),
        }
      )
  return output


def overall_paired_summary(
  configuration_pairs: list[dict[str, object]],
  trial_pairs: list[dict[str, object]],
) -> dict[str, object]:
  """Calculate paired wins, discordance tests, and coefficient differences."""
  differences = np.asarray(
    [int(row["completed_trials_difference_80_minus_35"]) for row in configuration_pairs]
  )
  all9_35_only = sum(
    bool(row["all_9_completed_35hz"]) and not bool(row["all_9_completed_80hz"])
    for row in configuration_pairs
  )
  all9_80_only = sum(
    bool(row["all_9_completed_80hz"]) and not bool(row["all_9_completed_35hz"])
    for row in configuration_pairs
  )
  trial_35_only = sum(
    bool(row["completed_35hz"]) and not bool(row["completed_80hz"])
    for row in trial_pairs
  )
  trial_80_only = sum(
    bool(row["completed_80hz"]) and not bool(row["completed_35hz"])
    for row in trial_pairs
  )
  return {
    "matched_configuration_pairs": len(configuration_pairs),
    "matched_trial_pairs": len(trial_pairs),
    "configuration_pairs_35hz_more_completed_trials": int(np.sum(differences < 0)),
    "configuration_pairs_80hz_more_completed_trials": int(np.sum(differences > 0)),
    "configuration_pairs_tied": int(np.sum(differences == 0)),
    "mean_completed_trial_difference_80_minus_35": float(np.mean(differences)),
    "all9_35hz_only": all9_35_only,
    "all9_80hz_only": all9_80_only,
    "all9_discordance_exact_p": exact_paired_pvalue(all9_35_only, all9_80_only),
    "trial_completion_35hz_only": trial_35_only,
    "trial_completion_80hz_only": trial_80_only,
    "trial_completion_discordance_exact_p": exact_paired_pvalue(
      trial_35_only, trial_80_only
    ),
    "mean_reached_fraction_difference_80_minus_35": float(
      np.mean(
        [float(row["reached_fraction_difference_80_minus_35"]) for row in trial_pairs]
      )
    ),
    "mean_coefficient_cosine_similarity": float(
      np.mean(
        [float(row["coefficient_cosine_similarity"]) for row in configuration_pairs]
      )
    ),
    "median_coefficient_cosine_similarity": float(
      np.median(
        [float(row["coefficient_cosine_similarity"]) for row in configuration_pairs]
      )
    ),
    "median_coefficient_relative_l2_change_from_35hz": float(
      np.median(
        [
          float(row["coefficient_relative_l2_change_from_35hz"])
          for row in configuration_pairs
        ]
      )
    ),
  }


def plot_comparison(
  path: Path,
  grouped_rows: list[dict[str, object]],
  configuration_pairs: list[dict[str, object]],
) -> None:
  """Plot cutoff completion rates and paired successful-trial differences."""
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  labels = ["Overall", "Degree 1", "Degree 2", "Degree 3"]
  degree_values: list[str | int] = ["all", 1, 2, 3]
  x = np.arange(len(labels))
  width = 0.34
  colors = {35.0: "#176b57", 80.0: "#d97706"}

  figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
  for offset, cutoff in [(-width / 2, 35.0), (width / 2, 80.0)]:
    rows = {
      row["degree"]: row
      for row in grouped_rows
      if float(row["lowpass_hz"]) == cutoff
    }
    all9 = [100 * float(rows[degree]["all_9_completed_fraction"]) for degree in degree_values]
    trial = [
      100 * float(rows[degree]["simulation_completion_fraction"])
      for degree in degree_values
    ]
    axes[0].bar(x + offset, all9, width, label=f"{cutoff:g} Hz", color=colors[cutoff])
    axes[1].bar(x + offset, trial, width, label=f"{cutoff:g} Hz", color=colors[cutoff])

  for axis, title, ylabel in [
    (axes[0], "Equations completing all 9 trials", "Configurations (%)"),
    (axes[1], "Individual simulations completed", "Simulation attempts (%)"),
  ]:
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=20, ha="right")
    axis.set_ylim(0, 105)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()

  differences = [
    int(row["completed_trials_difference_80_minus_35"])
    for row in configuration_pairs
  ]
  axes[2].hist(
    differences,
    bins=np.arange(-9.5, 10.5, 1),
    color="#4f46e5",
    edgecolor="white",
  )
  axes[2].axvline(0, color="#202124", linewidth=1)
  axes[2].set_xlabel("Completed trials: 80 Hz minus 35 Hz")
  axes[2].set_ylabel("Matched configurations")
  axes[2].set_title("Paired completion difference")
  axes[2].grid(axis="y", alpha=0.25)

  figure.suptitle("35 Hz versus 80 Hz low-pass sweep comparison")
  figure.tight_layout(rect=(0, 0, 1, 0.94))
  path.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(path, dpi=180)
  plt.close(figure)


def plot_coefficient_differences(
  path: Path,
  configuration_pairs: list[dict[str, object]],
) -> None:
  """Plot matched coefficient change across every swept model parameter."""
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  figure, axes = plt.subplots(3, 3, figsize=(14, 11), sharex=True, sharey=True)
  colors = {1: "#176b57", 2: "#d97706", 5: "#4f46e5"}
  for degree_index, degree in enumerate((1, 2, 3)):
    for smoothing_index, smoothing in enumerate((0, 5, 9)):
      axis = axes[degree_index, smoothing_index]
      panel_rows = [
        row
        for row in configuration_pairs
        if int(row["degree"]) == degree
        and int(row["smooth_window_samples"]) == smoothing
      ]
      for delay in (1, 2, 5):
        delay_rows = sorted(
          [row for row in panel_rows if int(row["delay_samples"]) == delay],
          key=lambda row: int(row["n_delays"]),
        )
        axis.plot(
          [int(row["n_delays"]) for row in delay_rows],
          [
            100 * float(row["coefficient_relative_l2_change_from_35hz"])
            for row in delay_rows
          ],
          marker="o",
          color=colors[delay],
          label=f"Spacing {delay}",
        )
      if degree_index == 0:
        axis.set_title(f"Smoothing {smoothing} samples")
      if smoothing_index == 0:
        axis.set_ylabel(f"Degree {degree}\nRelative L2 change (%)")
      if degree_index == 2:
        axis.set_xlabel("Delay coordinates")
      axis.set_xticks([2, 4, 6, 8])
      axis.grid(alpha=0.25)
  axes[0, -1].legend(loc="upper left")
  figure.suptitle(
    "Equation coefficient change from 35 Hz to 80 Hz\n"
    "Matched degree, delays, spacing, and smoothing"
  )
  figure.tight_layout(rect=(0, 0, 1, 0.95))
  path.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(path, dpi=180)
  plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, object]:
  """Run the paired cutoff analysis and save tabular and visual outputs."""
  grid_rows = read_csv(args.grid_csv)
  status_rows = read_csv(args.status_csv)
  validate_inputs(grid_rows, status_rows, 216, 9)
  configurations = configuration_summary(
    grid_rows,
    status_rows,
    figures_dir=args.figures_dir,
  )
  configuration_pairs = build_configuration_pairs(configurations, grid_rows)
  trial_pairs = build_trial_pairs(configurations, status_rows)
  grouped_rows = grouped_cutoff_summary(configurations, trial_pairs)
  paired_scope_rows = paired_scope_summary(configuration_pairs, trial_pairs)
  coefficient_group_rows = coefficient_parameter_summary(configuration_pairs)
  overall = overall_paired_summary(configuration_pairs, trial_pairs)

  write_csv(
    args.output_dir / "lowpass_group_summary.csv",
    grouped_rows,
    GROUP_FIELDS,
  )
  write_csv(
    args.output_dir / "lowpass_coefficient_parameter_summary.csv",
    coefficient_group_rows,
    COEFFICIENT_GROUP_FIELDS,
  )
  args.output_dir.mkdir(parents=True, exist_ok=True)
  (args.output_dir / "lowpass_35_vs_80_summary.json").write_text(
    json.dumps(overall, indent=2) + "\n"
  )
  plot_comparison(
    args.output_dir / "lowpass_35_vs_80_comparison.png",
    grouped_rows,
    configuration_pairs,
  )
  plot_coefficient_differences(
    args.output_dir / "lowpass_equation_coefficient_difference.png",
    configuration_pairs,
  )
  return overall


def main() -> None:
  """Parse CLI arguments and compare the two low-pass sweep branches."""
  parser = argparse.ArgumentParser(
    description="Compare matched 35 Hz and 80 Hz raw-grid configurations."
  )
  parser.add_argument("--grid-csv", type=Path, default=DEFAULT_GRID)
  parser.add_argument(
    "--status-csv",
    type=Path,
    default=DEFAULT_SIMULATION_DIR / "simulation_status_merged.csv",
  )
  parser.add_argument(
    "--figures-dir",
    type=Path,
    default=DEFAULT_SIMULATION_DIR / "figures",
  )
  parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
  args = parser.parse_args()

  summary = run(args)
  print(
    "paired completion wins (35 Hz / 80 Hz / ties): "
    f"{summary['configuration_pairs_35hz_more_completed_trials']} / "
    f"{summary['configuration_pairs_80hz_more_completed_trials']} / "
    f"{summary['configuration_pairs_tied']}"
  )
  print(
    "mean completed-trial difference (80 - 35 Hz): "
    f"{summary['mean_completed_trial_difference_80_minus_35']:.3f}"
  )
  print(f"saved analyses: {args.output_dir}")


if __name__ == "__main__":
  main()
