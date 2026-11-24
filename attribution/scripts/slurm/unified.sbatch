#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH -t 16:00:00
#SBATCH -c 1
#SBATCH --mem=80G
#SBATCH --job-name=arith_pipeline
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Single-task pipeline: stage 1 (greedy decode) then stage 2 (attribution)
# for ONE (model, format) pair. Index decoded from SLURM_ARRAY_TASK_ID.
# The array launcher (launch_all_models.sh) sets `--array=0-N%4` so the global
# in-flight count never exceeds 4 A100s.

set -e

# HF_TOKEN must be set in the calling environment (do not hardcode here).
# The launcher script forwards it via `--export=ALL`; if you submit this
# sbatch directly, do:
#   HF_TOKEN=hf_xxxxx sbatch ... scripts/slurm/unified.sbatch
: "${HF_TOKEN:?HF_TOKEN env var is required (see attribution/scripts/slurm/README.md)}"
export HF_TOKEN

cd "${SLURM_SUBMIT_DIR:?submit from the attribution/ directory}"
source scripts/slurm/env.sh

# (model, batch_size) lookup, must match launch_all_models.sh exactly
MODELS=(
    "Qwen/Qwen3-0.6B"
    "meta-llama/Llama-3.2-1B"
    "google/gemma-2-2b"
    "meta-llama/Llama-3.2-3B"
    "Qwen/Qwen3-4B"
    "mistralai/Mistral-7B-v0.1"
    "allenai/Olmo-3-1025-7B"
    "meta-llama/Llama-3.1-8B"
    "Qwen/Qwen3-8B"
    "google/gemma-2-9b"
    "microsoft/phi-4"
    "Qwen/Qwen3-32B"
    "allenai/Olmo-3-1125-32B"
)
BATCHES=( 8 8 8 8 8 4 4 4 4 4 2 1 1 )
GENBATCHES=( 64 64 32 32 32 16 16 16 16 16 8 4 4 )
FORMATS=(numeric english spanish italian)

m_idx=$((SLURM_ARRAY_TASK_ID / 4))
f_idx=$((SLURM_ARRAY_TASK_ID % 4))
MODEL=${MODELS[$m_idx]}
FORMAT=${FORMATS[$f_idx]}
BATCH=${BATCHES[$m_idx]}
GBATCH=${GENBATCHES[$m_idx]}

echo "===================================================================="
echo "task $SLURM_ARRAY_TASK_ID  model=$MODEL  format=$FORMAT  batch=$BATCH"
echo "===================================================================="

MODEL_SAFE=$(echo "$MODEL" | tr '/.' '_-')
RESP="results/responses/${MODEL_SAFE}/${FORMAT}.json"
ATTR_AGG="results/attribution/${MODEL_SAFE}/${FORMAT}/neuron_attribution_aggregate.pt"

# Stage 1: greedy decode. Skip if a complete responses JSON already exists.
if [[ -f "$RESP" ]]; then
    echo "[stage 1] skip: $RESP already exists"
else
    python scripts/generate_responses.py \
        --model "$MODEL" --format "$FORMAT" --batch-size $GBATCH
fi

# Stage 2: attribution. Skip if the aggregate tensor already exists.
if [[ -f "$ATTR_AGG" ]]; then
    echo "[stage 2] skip: $ATTR_AGG already exists"
else
    python scripts/run_attribution.py \
        --model "$MODEL" --format "$FORMAT" --component neurons --batch-size $BATCH
fi

echo "task $SLURM_ARRAY_TASK_ID done"
