#!/bin/bash
#SBATCH --job-name=lfp_forecast_skill
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --array=0-5
#SBATCH --time=4:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL
#SBATCH --mail-user=aida_abkenova@brown.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm/slurm-lfp-forecast-skill-%A_%a.out
#SBATCH --error=slurm/slurm-lfp-forecast-skill-%A_%a.err

# Forecast skill vs lead time, swept over embedding window and mode count.
#
# Settings are chosen from measured results rather than inherited:
#
#   max lead 1.0s   The signal's own autocorrelation reaches zero by ~350ms,
#                   so the previous 6s horizon was ~15x past the point where
#                   any number could mean anything. 1s covers the skill window
#                   plus enough noise floor to see where skill ends.
#   stride 0.5s     ~243 forecasts per configuration instead of 110, which
#                   narrows the +/-0.2 noise band that currently swamps the
#                   skill estimate at 200ms.
#   degree 1        Degrees 2 and 3 gained nothing in the HAVOK sweep, and
#                   degree 3 at 12 modes produced 5460 terms while completing
#                   only 5 of 9 simulations.
#   threshold 1000  Skill was flat across thresholds 0-10000; 1000 gives the
#                   sparser model at no measured cost.
#   modes 8-24      Reconstruction ceilings 0.77 / 0.91 / 0.96 / 0.99 / 0.999.
#                   The previous grid stopped at 12, which was its best point
#                   and therefore the edge of the sweep.
#   n_delays        The genuinely untested axis. 80 samples = 316ms was copied
#                   from the paper's input_dim and is nearly the whole
#                   predictability horizon; 40 and 120 bracket it.
#
# Skill at lead 0 is the reconstruction ceiling for that truncation, so each
# configuration measures its own ceiling with no separate probe.
#
# Array: 3 embedding windows x 2 degrees = 6 tasks, each sweeping 5 mode counts.
#   task = window_index * 2 + degree_index
#
# Walltime and memory come from the measured learning-curve job (100% CPU,
# 1.0GB of 32GB, TIMEOUT at 12h). Here simulations are 1s rather than 6s and
# the models are linear, so 4h and 4G are ample.
#
# Concatenate after the array finishes:
#   find outputs/pysindy/forecast_skill/parts -name forecast_skill.csv \
#     | xargs awk 'FNR==1 && NR!=1 {next} {print}' \
#     > outputs/pysindy/forecast_skill/forecast_skill_all.csv

set -euo pipefail

export SLURM_EXPORT_ENV=ALL

cd "$SLURM_SUBMIT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/oscar/data/tdesroch/dbuyukya/symreg_env/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/pysindy/forecast_skill/parts}"
MODES_LIST="${MODES_LIST:-8,12,16,20,24}"
THRESHOLD="${THRESHOLD:-1000}"
LOWPASS="${LOWPASS:-35}"
MAX_LEAD="${MAX_LEAD:-1.0}"
ORIGIN_STRIDE="${ORIGIN_STRIDE:-0.5}"

WINDOWS=(40 80 120)
DEGREES=(1 2)

WINDOW_INDEX=$(( SLURM_ARRAY_TASK_ID / ${#DEGREES[@]} ))
DEGREE_INDEX=$(( SLURM_ARRAY_TASK_ID % ${#DEGREES[@]} ))
N_DELAYS="${WINDOWS[$WINDOW_INDEX]}"
DEGREE="${DEGREES[$DEGREE_INDEX]}"

# Only the first task carries the delay-coordinate reference, so the baseline
# is fitted once rather than six times.
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
  DELAY_BASELINE="nd4"
else
  DELAY_BASELINE="none"
fi

OUT_DIR="$OUTPUT_ROOT/q${N_DELAYS}_deg${DEGREE}"
mkdir -p "$OUT_DIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "task ${SLURM_ARRAY_TASK_ID}: n_delays=${N_DELAYS} degree=${DEGREE}"

"$PYTHON_BIN" -u scripts/pysindy/forecast_skill.py \
  --n-delays-list "$N_DELAYS" \
  --modes-list "$MODES_LIST" \
  --degree-list "$DEGREE" \
  --threshold "$THRESHOLD" \
  --lowpass "$LOWPASS" \
  --max-lead "$MAX_LEAD" \
  --origin-stride "$ORIGIN_STRIDE" \
  --delay-baseline "$DELAY_BASELINE" \
  --out-dir "$OUT_DIR"
