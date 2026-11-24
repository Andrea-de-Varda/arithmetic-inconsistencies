#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH -t 08:00:00
#SBATCH -c 1
#SBATCH --mem=60G
#SBATCH --job-name=pool_responses
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Stage 1 for the PROBE POOL of the predictor comparison, one model per array
# task. Greedy-decodes numeric problems from the control dataset
# (control_balanced/data, 20 000 items, generated with a different seed than
# the main dataset) so the supervised probes have a labelled training pool that
# shares no problem with the main fold they are later tested against.
#
# Two economies over a naive 20 000-item run per model:
#
#   --skip-corrupted   the probe only needs the clean prompt, its label and the
#                      model's own clean answer, so the sign-flipped decode is
#                      not run. Halves the cost. NEVER use this flag for the
#                      main or control pipelines, which need both directions.
#
#   --max-items N      N is set per model from its known numeric accuracy so
#                      that the pool is expected to contain ~400 incorrect
#                      items, which is what the probe needs to learn anything.
#                      A model at 47% accuracy reaches that in 2 500 items; one
#                      at 98.8% needs the full 20 000. Summing the per-model N
#                      costs ~92 k decoded items instead of 260 k.
#
# Output: ensemble/results/_pool_responses/<model>/numeric.json
# Nothing under control_balanced/ is written to; that pipeline's own 20 000-item
# Llama-3.1-8B responses stay exactly as published, and extract_features.py
# falls back to them, so array task 7 is a no-op if they are present.
#
# Submit from attribution/:
#   export HF_TOKEN=hf_...
#   sbatch --array=0-12%4 ensemble/scripts/slurm/stage1_pool_responses.sbatch

set -e

: "${HF_TOKEN:?HF_TOKEN env var is required (export before sbatch)}"
export HF_TOKEN

cd "${SLURM_SUBMIT_DIR:?submit from the attribution/ directory}"
source scripts/slurm/env.sh

export PYTHONUNBUFFERED=1

# The 13-model panel. MODELS / POOLITEMS / GENBATCHES are index-aligned;
# GENBATCHES mirrors scripts/slurm/unified.sbatch. POOLITEMS is
# ceil(400 / (1 - numeric_accuracy)) rounded up to 500, floored at 2500 and
# capped at 20000 (the size of the control dataset).
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
POOLITEMS=(  2500  2500  2500  6000  3000  2500  5000 14500  3000  3500 20000  7500 20000 )
GENBATCHES=(   64    64    32    32    32    16    16    16    16    16     8     4     4 )

# Normally one array task per model, indexed exactly as launch_all.sh's
# ALL_MODELS. Setting MODEL explicitly overrides that, for a single rerun
# without an array; the per-model item count is then looked up by name.
IDX=${SLURM_ARRAY_TASK_ID:-0}
if [[ -n "${MODEL:-}" ]]; then
    IDX=-1
    for j in "${!MODELS[@]}"; do
        [[ "${MODELS[$j]}" == "$MODEL" ]] && IDX=$j && break
    done
    if (( IDX < 0 )); then
        echo "MODEL=$MODEL is not in the panel" >&2
        exit 1
    fi
fi
MODEL=${MODELS[$IDX]}
NITEMS=${POOLITEMS[$IDX]}
GBATCH=${GENBATCHES[$IDX]}
MODEL_SAFE=$(echo "$MODEL" | tr '/.' '_-')

OUT="ensemble/results/_pool_responses/${MODEL_SAFE}/numeric.json"
FALLBACK="control_balanced/results/responses/${MODEL_SAFE}/numeric.json"

echo "===================================================================="
echo "probe-pool stage 1   model=$MODEL  max_items=$NITEMS  batch=$GBATCH"
echo "===================================================================="

if [[ -f "$OUT" ]]; then
    echo "$OUT exists; nothing to do."
    exit 0
fi
if [[ -f "$FALLBACK" ]]; then
    echo "$FALLBACK exists (control pipeline ran the full 20k for this model);"
    echo "extract_features.py falls back to it. Nothing to do."
    exit 0
fi

python -u scripts/generate_responses.py \
    --model "$MODEL" \
    --format numeric \
    --use-all-items \
    --max-items "$NITEMS" \
    --skip-corrupted \
    --data-dir control_balanced/data \
    --results-dir ensemble/results/_pool_responses \
    --batch-size "$GBATCH"

echo "done."
