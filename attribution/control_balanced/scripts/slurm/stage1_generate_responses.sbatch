#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH -t 06:00:00
#SBATCH -c 1
#SBATCH --mem=40G
#SBATCH --job-name=balanced_stage1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Stage 1 for the control_balanced pipeline: greedy-decode meta-llama/Llama-3.1-8B
# on the full 20 000-item dataset for one verbal format. We need responses on
# ALL items (no fold subset) so we can later pick the 1600/400 balanced split.
#
# Required env var:
#   FORMAT      numeric | english | spanish | italian
# Optional:
#   MODEL       default meta-llama/Llama-3.1-8B
#   GBATCH      default 32
#
# Run 4 in parallel from `attribution/`:
#   for f in numeric english spanish italian; do
#       sbatch --export=ALL,FORMAT=$f control_balanced/scripts/slurm/stage1_generate_responses.sbatch
#   done

set -e

# HF_TOKEN must be set in the calling environment (do not hardcode here).
: "${HF_TOKEN:?HF_TOKEN env var is required (see attribution/scripts/slurm/README.md)}"
export HF_TOKEN

cd "${SLURM_SUBMIT_DIR:?submit from the attribution/ directory}"
source scripts/slurm/env.sh

MODEL=${MODEL:-meta-llama/Llama-3.1-8B}
FORMAT=${FORMAT:?FORMAT env var must be set (numeric|english|spanish|italian)}
GBATCH=${GBATCH:-32}

echo "====== control_balanced stage 1: model=$MODEL format=$FORMAT ======"

python scripts/generate_responses.py \
    --model "$MODEL" \
    --format "$FORMAT" \
    --use-all-items \
    --data-dir control_balanced/data \
    --results-dir control_balanced/results/responses \
    --batch-size "$GBATCH"
