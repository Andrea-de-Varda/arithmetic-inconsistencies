#!/bin/bash
#SBATCH -t 01:00:00
#SBATCH -c 2
#SBATCH --mem=64G
#SBATCH --job-name=arith_overlap
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Per-model: run stage 3 (overlap and loading analysis).
# CPU-only. MODEL is passed via --export.

set -e

cd "${SLURM_SUBMIT_DIR:?submit from the attribution/ directory}"
source scripts/slurm/env.sh

MODEL=${MODEL:?MODEL env var must be set}

echo "===== overlap model=$MODEL ====="
python scripts/run_overlap.py --model "$MODEL" --components neuron
