#!/bin/bash
#SBATCH -t 06:00:00
#SBATCH -c 2
#SBATCH --mem=128G
#SBATCH --job-name=arith_crossfit
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Cross-fitted (held-out) numeric circuits for all 13 models. Reads the
# per-item attribution tensors and writes the small, git-tracked CSVs under
# heldout/results/. Pure CPU, no GPU and no HF_TOKEN needed: it reuses the
# attribution tensors the main sweep already produced, so there are no new
# forward or backward passes.
#
# 128G matches all_pairs_item_loadings.sbatch: the 32B models' per-item
# tensors are ~7 GB at fp16 and the script holds one at a time, plus a
# [k, n_units] float32 fold-sum buffer (~71 MB at k=10).

set -e

cd "${SLURM_SUBMIT_DIR:?submit from the attribution/ directory}"
source scripts/slurm/env.sh

python heldout/scripts/crossfit_circuits.py --k 10 --seed 42
