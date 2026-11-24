#!/bin/bash
#SBATCH -t 01:00:00
#SBATCH -c 2
#SBATCH --mem=64G
#SBATCH --job-name=balanced_stage3
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Stage 3 (overlap and loading) for the control_balanced pipeline.
# CPU only. Runs after all four formats of stage 2 have completed.
#
# Usage (from `attribution/`):
#   sbatch control_balanced/scripts/slurm/stage3_overlap.sbatch
# Optional env var: MODEL (default Llama-3.1-8B)

set -e

cd "${SLURM_SUBMIT_DIR:?submit from the attribution/ directory}"
source scripts/slurm/env.sh

MODEL=${MODEL:-meta-llama/Llama-3.1-8B}

echo "====== control_balanced stage 3: model=$MODEL ======"

python scripts/run_overlap.py \
    --model "$MODEL" \
    --results-dir control_balanced/results \
    --components neuron
