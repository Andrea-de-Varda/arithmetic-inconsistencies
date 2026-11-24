#!/bin/bash
#SBATCH -t 06:00:00
#SBATCH -c 2
#SBATCH --mem=128G
#SBATCH --job-name=arith_matched_baseline
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Matched random baseline for the item-level loading, all 13 models. Reads the
# per-item attribution tensors and writes the small, git-tracked CSVs under
# matched_baseline/results/. Pure CPU, no GPU and no HF_TOKEN needed: no new
# forward or backward passes.
#
# 128G matches crossfit.sbatch: the 32B models' per-item tensors are ~7 GB at
# fp16 and the script holds one at a time. Tested locally at Llama-3.1-8B
# dimensions: ~1 min and ~4 GB per model for 1,000 draws.

set -e

cd "${SLURM_SUBMIT_DIR:?submit from the attribution/ directory}"
source scripts/slurm/env.sh

python matched_baseline/scripts/selftest_matched.py
python matched_baseline/scripts/random_matched_loadings.py --n-draws 1000 --seed 42
