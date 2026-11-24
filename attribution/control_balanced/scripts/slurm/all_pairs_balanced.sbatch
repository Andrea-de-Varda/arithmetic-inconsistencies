#!/bin/bash
#SBATCH -t 00:30:00
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH --job-name=all_pairs_balanced
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# CPU-only: balanced-fold all-pairs item-level point-biserial.
# Loads four per-item .pt tensors (~1.8 GB each on Llama-3.1-8B) into RAM,
# extracts top-1% indices, computes 16 point-biserials, writes the CSV.
#
# Env vars (optional):
#   MODELS  default "meta-llama/Llama-3.1-8B"
#           (balanced data only exists for this model right now)

set -e

cd "${SLURM_SUBMIT_DIR:?submit from the attribution/ directory}"
source scripts/slurm/env.sh

MODELS=${MODELS:-meta-llama/Llama-3.1-8B}

echo "===================================================================="
echo "all_pairs balanced   models=$MODELS"
echo "===================================================================="

python control_balanced/scripts/all_pairs_item_loadings_balanced.py \
    --models $MODELS
