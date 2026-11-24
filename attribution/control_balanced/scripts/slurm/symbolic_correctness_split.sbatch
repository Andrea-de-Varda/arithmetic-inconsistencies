#!/bin/bash
#SBATCH -t 00:30:00
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH --job-name=symb_split
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Symbolic-correctness conditional analysis. CPU-only, ~1–2 min.
# Reads control_balanced/results/attribution/<model>/numeric/neuron_attribution_per_item.pt
# (which is git-ignored and lives only on the cluster), splits it by
# clean_match_gold, rebuilds two derived symbolic circuits, and compares.
#
# Run from `attribution/`:
#   sbatch control_balanced/scripts/slurm/symbolic_correctness_split.sbatch

set -e

cd "${SLURM_SUBMIT_DIR:?submit from the attribution/ directory}"
source scripts/slurm/env.sh

MODEL=${MODEL:-meta-llama/Llama-3.1-8B}

echo "====== symbolic-correctness split: model=$MODEL ======"

python control_balanced/scripts/symbolic_correctness_split.py \
    --model "$MODEL" \
    --results-dir control_balanced/results \
    --top-k-fraction 0.01
