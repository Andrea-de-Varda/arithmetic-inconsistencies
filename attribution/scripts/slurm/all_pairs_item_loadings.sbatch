#!/bin/bash
#SBATCH -t 02:00:00
#SBATCH -c 2
#SBATCH --mem=128G
#SBATCH --job-name=arith_all_pairs
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Item-level all-pairs (anchor x target) point-biserial. Reads the per-item
# attribution tensors for all 13 models and writes
#   results/all_pairs/per_item_pointbiserial.csv
# (small, gettable via git pull). Pure CPU, runs in a few minutes; the 128G
# memory is generous because the 32B models' per-item tensors are ~7 GB each.

set -e

cd "${SLURM_SUBMIT_DIR:?submit from the attribution/ directory}"
source scripts/slurm/env.sh

python scripts/all_pairs_item_loadings.py
