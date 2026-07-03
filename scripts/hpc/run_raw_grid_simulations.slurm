#!/bin/bash
#SBATCH --job-name=lfp_grid_sim
#SBATCH --array=1-216%12
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm-lfp-grid-sim-%A_%a.out
#SBATCH --error=slurm-lfp-grid-sim-%A_%a.err

set -euo pipefail

# Run from the repository submitted to Slurm.
cd "$SLURM_SUBMIT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
GRID_CSV="${GRID_CSV:-outputs/pysindy/raw_grid/raw_grid_merged.csv}"
METADATA_JSON="${METADATA_JSON:-outputs/pysindy/raw_grid/parts/part_lp35_degree1_metadata.json}"
SIMULATION_OUTPUT_DIR="${SIMULATION_OUTPUT_DIR:-outputs/pysindy/raw_grid/simulations}"
TRIAL_TIMEOUT_S="${TRIAL_TIMEOUT_S:-1800}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Each array task reconstructs one stored equation and simulates all nine
# held-out trials over their complete available embedded duration.
"$PYTHON_BIN" scripts/pysindy/visualize_raw_grid_simulations.py \
  --grid-csv "$GRID_CSV" \
  --metadata-json "$METADATA_JSON" \
  --output-dir "$SIMULATION_OUTPUT_DIR" \
  --configuration-index "$SLURM_ARRAY_TASK_ID" \
  --trial-timeout-s "$TRIAL_TIMEOUT_S"
