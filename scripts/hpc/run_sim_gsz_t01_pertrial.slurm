#!/bin/bash
#SBATCH --job-name=lfp_sim_gsz_t01
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --array=1-216%32
#SBATCH --time=04:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL
#SBATCH --mail-user=aida_abkenova@brown.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm/slurm-sim-gsz-t01-pertrial-%A_%a.out
#SBATCH --error=slurm/slurm-sim-gsz-t01-pertrial-%A_%a.err

set -euo pipefail

export SLURM_EXPORT_ENV=ALL

cd "$SLURM_SUBMIT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/oscar/data/tdesroch/dbuyukya/symreg_env/bin/python}"
GRID_CSV="${GRID_CSV:-outputs/pysindy/raw_grid_globalzscore/raw_grid_merged.csv}"
METADATA_JSON="${METADATA_JSON:-outputs/pysindy/raw_grid_globalzscore/parts/part_lp35_degree1_metadata.json}"
SIMULATION_OUTPUT_DIR="${SIMULATION_OUTPUT_DIR:-outputs/pysindy/raw_grid_globalzscore/simulations_pertrial}"
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
