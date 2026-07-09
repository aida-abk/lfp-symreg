#!/bin/bash
#SBATCH --job-name=lfp_sim_deg57
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --array=1-128%32
#SBATCH --time=08:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL
#SBATCH --mail-user=aida_abkenova@brown.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm/slurm-lfp-sim-deg57-%A_%a.out
#SBATCH --error=slurm/slurm-lfp-sim-deg57-%A_%a.err

# Simulate all 144 deg57 configurations (2 degrees × 2 lowpass × 36 param combos).
# Run after merge: merge_raw_grid.py --input-dir outputs/pysindy/raw_grid_deg57_t1000/parts
#                                    --output-csv outputs/pysindy/raw_grid_deg57_t1000/raw_grid_merged.csv
#                                    --expected 144
# Each task reconstructs one stored equation, simulates all held-out trials, and
# writes a status CSV and figure (raw + lowpass-filtered + simulated x0).

set -euo pipefail

export SLURM_EXPORT_ENV=ALL

cd "$SLURM_SUBMIT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/oscar/data/tdesroch/dbuyukya/symreg_env/bin/python}"
GRID_CSV="${GRID_CSV:-outputs/pysindy/raw_grid_deg57_t1000/raw_grid_merged.csv}"
METADATA_JSON="${METADATA_JSON:-outputs/pysindy/raw_grid_deg57_t1000/parts/part_lp35_degree5_metadata.json}"
SIMULATION_OUTPUT_DIR="${SIMULATION_OUTPUT_DIR:-outputs/pysindy/raw_grid_deg57_t1000/simulations}"
TRIAL_TIMEOUT_S="${TRIAL_TIMEOUT_S:-1800}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

"$PYTHON_BIN" scripts/pysindy/visualize_raw_grid_simulations.py \
  --grid-csv "$GRID_CSV" \
  --metadata-json "$METADATA_JSON" \
  --output-dir "$SIMULATION_OUTPUT_DIR" \
  --configuration-index "$SLURM_ARRAY_TASK_ID" \
  --max-test-trials 4 \
  --per-trial-figures \
  --trial-timeout-s "$TRIAL_TIMEOUT_S"
