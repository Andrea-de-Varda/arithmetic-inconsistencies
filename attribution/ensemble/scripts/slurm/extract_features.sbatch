#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH -t 04:00:00
#SBATCH -c 2
#SBATCH --mem=120G
#SBATCH --job-name=ensemble_extract
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Predictor comparison, stage A (GPU): one teacher-forced forward per item over
# prompt+model_response, capturing residual + MLP-input reps at the decision
# point, plus first-token entropy and own-response logprob. Covers the
# held-out numeric probe pool and the verbal test items in a single load.
# Writes ensemble/results/<model>/features/ (gitignored, deleted by stage B).
# The 120G host memory is for the 32B models (~7 GB fp16 per item group).
#
# Env vars:
#   MODEL   HF id (required)
#   BATCH   forward batch size (default 8)

set -e

: "${HF_TOKEN:?HF_TOKEN env var is required (export before sbatch)}"
export HF_TOKEN

cd "${SLURM_SUBMIT_DIR:?submit from the attribution/ directory}"
source scripts/slurm/env.sh

MODEL=${MODEL:?MODEL env var is required}
BATCH=${BATCH:-8}
export PYTHONUNBUFFERED=1

echo "===================================================================="
echo "ensemble extract_features   model=$MODEL  batch=$BATCH"
echo "===================================================================="

python -u ensemble/scripts/extract_features.py --model "$MODEL" --batch-size "$BATCH"
