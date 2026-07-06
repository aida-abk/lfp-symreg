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
  args = parser.parse_args()

  expected = str(args.expected)
  expected_trials = str(args.expected_trials)

  if not args.skip_merge and not args.skip_summarize:
    run(
      "1/8  merge_raw_grid",
      PYSINDY / "merge_raw_grid.py",
      ["--expected", expected],
    )

  if not args.skip_summarize:
    run(
      "2/8  summarize_raw_grid_simulations",
      PYSINDY / "summarize_raw_grid_simulations.py",
      ["--expected", expected],
    )

    run(
      "3/8  add_simulation_rms_to_grid",
      SWEEP_ANALYSIS / "add_simulation_rms_to_grid.py",
    )

  run(
    "4/8  analyze_simulation_grid",
    SWEEP_ANALYSIS / "analyze_simulation_grid.py",
    ["--expected-configurations", expected, "--expected-trials", expected_trials],
  )

  run(
    "5/8  analyze_term_utilization",
    SWEEP_ANALYSIS / "analyze_term_utilization.py",
  )

  run(
    "6/8  analyze_smoothing_variability",
    SWEEP_ANALYSIS / "analyze_smoothing_variability.py",
  )

  run(
    "7/8  analyze_delay_parameters",
    SWEEP_ANALYSIS / "analyze_delay_parameters.py",
  )

  run(
    "8/8  compare_lowpass_cutoffs",
    SWEEP_ANALYSIS / "compare_lowpass_cutoffs.py",
  )

  print("\nAll analysis steps completed.")
  print(f"Outputs in: {ROOT / 'outputs' / 'pysindy' / 'raw_grid' / 'analysis'}")


if __name__ == "__main__":
  main()
