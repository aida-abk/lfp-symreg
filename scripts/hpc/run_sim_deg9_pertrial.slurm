#!/bin/bash
#SBATCH --job-name=lfp_sim_deg9
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --array=1-54%32
#SBATCH --time=04:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL
#SBATCH --mail-user=aida_abkenova@brown.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm/slurm-sim-deg9-%A_%a.out
#SBATCH --error=slurm/slurm-sim-deg9-%A_%a.err

# 54 configs = 2 lowpass × 3 n_delays (2,4,6) × 3 delays × 3 smooth_windows.
# Run after merging parts (--expected 54):
#   python scripts/pysindy/merge_raw_grid.py \
#     --input-dir <OUTPUT_DIR>/parts \
#     --output-csv <OUTPUT_DIR>/raw_grid_merged.csv \
#     --expected 54
#
# Usage (one sbatch per threshold):
#   GRID_CSV=outputs/pysindy/raw_grid_deg9_t1000/raw_grid_merged.csv \
#   METADATA_JSON=outputs/pysindy/raw_grid_deg9_t1000/parts/part_lp35_degree9_metadata.json \
#   SIMULATION_OUTPUT_DIR=outputs/pysindy/raw_grid_deg9_t1000/simulations_pertrial \
#   sbatch scripts/hpc/run_sim_deg9_pertrial.slurm

set -euo pipefail

export SLURM_EXPORT_ENV=ALL

cd "$SLURM_SUBMIT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/oscar/data/tdesroch/dbuyukya/symreg_env/bin/python}"
GRID_CSV="${GRID_CSV:-outputs/pysindy/raw_grid_deg9_t1000/raw_grid_merged.csv}"
METADATA_JSON="${METADATA_JSON:-outputs/pysindy/raw_grid_deg9_t1000/parts/part_lp35_degree9_metadata.json}"
SIMULATION_OUTPUT_DIR="${SIMULATION_OUTPUT_DIR:-outputs/pysindy/raw_grid_deg9_t1000/simulations_pertrial}"
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
