"""Run the full post-sweep analysis pipeline in the correct order.

Usage (from repo root):
    .venv/bin/python scripts/pysindy/run_analysis.py
    .venv/bin/python scripts/pysindy/run_analysis.py --expected 216
    .venv/bin/python scripts/pysindy/run_analysis.py --skip-merge

Steps:
    1. merge_raw_grid          -- merge Slurm part CSVs into raw_grid_merged.csv
    2. summarize_simulations   -- merge per-configuration simulation status files
    3. add_simulation_rms      -- join pooled RMS back onto the merged grid
    4. analyze_grid            -- configuration-level summary stats and figures
    5. analyze_term_utilization -- term-count distribution across parameters
    6. analyze_smoothing       -- coefficient variance across smoothing windows
    7. analyze_delays          -- delay-spacing effect on coefficients
    8. compare_lowpass         -- paired 35 Hz vs 80 Hz statistical test
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYSINDY = ROOT / "scripts" / "pysindy"
SWEEP_ANALYSIS = PYSINDY / "sweep_analysis"

PYTHON = sys.executable


def run(label: str, script: Path, extra_args: list[str] | None = None) -> None:
  """Run one analysis script and exit loudly on failure.

  Args:
    label: Human-readable step name printed to the console.
    script: Absolute path to the Python script.
    extra_args: Optional additional CLI arguments passed to the script.
  """
  cmd = [PYTHON, str(script)] + (extra_args or [])
  print(f"\n--- {label} ---")
  print("  " + " ".join(cmd))
  result = subprocess.run(cmd, cwd=ROOT)
  if result.returncode != 0:
    print(f"\nFAILED at step: {label} (exit code {result.returncode})")
    sys.exit(result.returncode)
  print(f"  done.")


def main() -> None:
  """Run the full post-sweep analysis pipeline."""
  parser = argparse.ArgumentParser(
    description="Run all post-sweep analysis steps in order."
  )
  parser.add_argument(
    "--sweep-dir",
    type=Path,
    default=None,
    help=(
      "Base directory of a sweep output (e.g. outputs/pysindy/raw_grid_threshold1000). "
      "Overrides all default input/output paths. Defaults to outputs/pysindy/raw_grid."
    ),
  )
  parser.add_argument(
    "--expected",
    type=int,
    default=216,
    help="Expected number of configurations in the sweep (default: 216).",
  )
  parser.add_argument(
    "--expected-trials",
    type=int,
    default=9,
    help="Expected number of held-out trials per configuration (default: 9).",
  )
  parser.add_argument(
    "--skip-merge",
    action="store_true",
    help="Skip step 1 (merge_raw_grid). Use when parts are already merged.",
  )
  parser.add_argument(
    "--skip-summarize",
    action="store_true",
    help="Skip steps 1–3. Use when raw_grid_merged.csv already has RMS columns.",
  )
  parser.add_argument(
    "--allow-missing-figures",
    action="store_true",
    help="Pass --allow-missing-figures to summarize step; use when some simulation figures are absent.",
  )
  args = parser.parse_args()

  sweep_dir: Path = (
    args.sweep_dir.resolve() if args.sweep_dir else ROOT / "outputs" / "pysindy" / "raw_grid"
  )
  parts_dir = sweep_dir / "parts"
  grid_csv = sweep_dir / "raw_grid_merged.csv"
  sim_dir = sweep_dir / "simulations"
  sim_status_csv = sim_dir / "simulation_status_merged.csv"
  figures_dir = sim_dir / "figures"
  analysis_dir = sweep_dir / "analysis"

  expected = str(args.expected)
  expected_trials = str(args.expected_trials)

  if not args.skip_merge and not args.skip_summarize:
    run(
      "1/8  merge_raw_grid",
      PYSINDY / "merge_raw_grid.py",
      ["--input-dir", str(parts_dir), "--output-csv", str(grid_csv), "--expected", expected],
    )

  if not args.skip_summarize:
    summarize_extra = ["--allow-missing-figures"] if args.allow_missing_figures else []
    run(
      "2/8  summarize_raw_grid_simulations",
      PYSINDY / "summarize_raw_grid_simulations.py",
      ["--output-dir", str(sim_dir), "--grid-csv", str(grid_csv), "--expected", expected]
      + summarize_extra,
    )

    run(
      "3/8  add_simulation_rms_to_grid",
      SWEEP_ANALYSIS / "add_simulation_rms_to_grid.py",
      ["--grid-csv", str(grid_csv), "--status-csv", str(sim_status_csv)],
    )

  run(
    "4/8  analyze_simulation_grid",
    SWEEP_ANALYSIS / "analyze_simulation_grid.py",
    [
      "--grid-csv", str(grid_csv),
      "--status-csv", str(sim_status_csv),
      "--figures-dir", str(figures_dir),
      "--output-dir", str(analysis_dir),
      "--expected-configurations", expected,
      "--expected-trials", expected_trials,
    ],
  )

  run(
    "5/8  analyze_term_utilization",
    SWEEP_ANALYSIS / "analyze_term_utilization.py",
    ["--grid-csv", str(grid_csv), "--output-dir", str(analysis_dir)],
  )

  run(
    "6/8  analyze_smoothing_variability",
    SWEEP_ANALYSIS / "analyze_smoothing_variability.py",
    ["--grid-csv", str(grid_csv), "--output-dir", str(analysis_dir)],
  )

  run(
    "7/8  analyze_delay_parameters",
    SWEEP_ANALYSIS / "analyze_delay_parameters.py",
    ["--grid-csv", str(grid_csv), "--status-csv", str(sim_status_csv), "--output-dir", str(analysis_dir)],
  )

  run(
    "8/8  compare_lowpass_cutoffs",
    SWEEP_ANALYSIS / "compare_lowpass_cutoffs.py",
    [
      "--grid-csv", str(grid_csv),
      "--status-csv", str(sim_status_csv),
      "--figures-dir", str(figures_dir),
      "--output-dir", str(analysis_dir),
    ],
  )

  print("\nAll analysis steps completed.")
  print(f"Outputs in: {analysis_dir}")


if __name__ == "__main__":
  main()
