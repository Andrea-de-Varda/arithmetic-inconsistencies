#!/bin/bash
#SBATCH -t 03:00:00
#SBATCH -c 2
#SBATCH --mem=128G
#SBATCH --job-name=arith_threshold_robustness
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Item-level loading and r_pb at circuit thresholds 0.5/1/2/5%, all 13 models.
# Reads the per-item attribution tensors and writes the small, git-tracked
# CSVs under threshold_robustness/results/. Pure CPU, no GPU and no HF_TOKEN
# needed: no new forward or backward passes.
#
# 128G matches crossfit.sbatch and matched_baseline.sbatch: the 32B models'
# per-item tensors are ~7 GB at fp16 and the script holds one at a time.

set -e

cd "${SLURM_SUBMIT_DIR:?submit from the attribution/ directory}"
source scripts/slurm/env.sh

python threshold_robustness/scripts/selftest_item_level.py
python threshold_robustness/scripts/item_level_loadings.py
