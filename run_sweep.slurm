#!/bin/bash
#SBATCH --job-name=lfp_sindy
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --output=outputs/logs/lfp_sindy_%j.out
#SBATCH --error=outputs/logs/lfp_sindy_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

PYTHON=/oscar/data/tdesroch/dbuyukya/symreg_env/bin/python

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

"$PYTHON" scripts/pysindy/exploration_sweep.py \
  --source lfp \
  --mat-file raw_data/trialdata_v03_buzz_20231106_pre-0.100_post0.100.mat \
  --trial-type fixation \
  --channel 0 \
  --downsample 2 \
  --normalize zscore \
  --degree-list 1,2,3 \
  --lowpass-list 35,50,65,80 \
  --threshold-list 0.001,0.01,0.05,0.1,0.2 \
  --n-delays-list 2,4,6,8 \
  --delay-list 1,2,5 \
  --smooth-window-list 0,5,9 \
  --simulation-horizon 4 \
  --seed 0 \
  --out-csv outputs/pysindy/oscar_sweep.csv \
  --top-csv outputs/pysindy/oscar_sweep_top.csv \
  --equations-out outputs/pysindy/oscar_sweep_equations.txt
