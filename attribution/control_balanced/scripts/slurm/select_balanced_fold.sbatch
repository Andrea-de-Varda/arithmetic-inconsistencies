#!/bin/bash
#SBATCH -t 00:30:00
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH --job-name=balanced_fold
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Pick the 2 000-item balanced fold from the stage-1 responses.
# CPU-only, ~1 min. Run AFTER all 4 stage-1 jobs have completed.
#
# Usage (from `attribution/`):
#   sbatch control_balanced/scripts/slurm/select_balanced_fold.sbatch
# Optional: MODEL (default Llama-3.1-8B), N_CORRECT (default 1600),
#           N_INCORRECT (default 400), SEED (default 42).

set -e

# The tokenizer of the gated Llama model is loaded, so HF_TOKEN is needed.
: "${HF_TOKEN:?HF_TOKEN env var is required (see attribution/scripts/slurm/README.md)}"
export HF_TOKEN

cd "${SLURM_SUBMIT_DIR:?submit from the attribution/ directory}"
source scripts/slurm/env.sh

MODEL=${MODEL:-meta-llama/Llama-3.1-8B}
N_CORRECT=${N_CORRECT:-1600}
N_INCORRECT=${N_INCORRECT:-400}
SEED=${SEED:-42}

echo "====== balanced fold selection: model=$MODEL n_correct=$N_CORRECT n_incorrect=$N_INCORRECT seed=$SEED ======"

python control_balanced/scripts/select_balanced_fold.py \
    --model "$MODEL" \
    --responses-dir control_balanced/results/responses \
    --output-dir control_balanced/results/balanced_fold \
    --n-correct "$N_CORRECT" \
    --n-incorrect "$N_INCORRECT" \
    --seed "$SEED"
