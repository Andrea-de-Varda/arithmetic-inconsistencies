#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH -t 06:00:00
#SBATCH -c 2
#SBATCH --mem=100G
#SBATCH --job-name=order_cf
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Operand-ORDER counterfactual, one array task per model.
#
# The main pipeline flips the OPERATOR (44 + 22 vs 44 - 22), so a unit that
# merely reads the operator token can enter the circuit. Here the operator is
# held fixed and the operands are swapped: 70 - 33 vs 33 - 70, answers +37 and
# -37. An operator-reading unit gets exactly zero attribution, and because the
# magnitude of the answer is unchanged the contrast is a single sign decision,
# which is small enough for a 1% patch to carry.
#
# Per task, numeric format only, on the 925 eligible items:
#   1. stage 1  greedy decode on x and the order-swapped x'
#   2. stage 2  attribution -> the order circuit
#   3. overlap  order circuit vs the main operator circuit, against chance and
#               a split-half ceiling; plus whether the order circuit predicts
#               verbal accuracy (CPU, reads the per-item tensors)
#   4. transfer patch the OPERATOR circuit (from the main results) into the
#               order-swapped prompts, against size-matched random units:
#               does the circuit support arithmetic under a counterfactual it
#               was not selected on?
#   5. ceiling  the same patch with the ORDER circuit, the best circuit for
#               this contrast, plus its own random control. Step 4 is only
#               interpretable against this.
#
# Submit from attribution/, at most 6 GPUs in flight:
#   export HF_TOKEN=hf_...
#   python order_cf/scripts/make_order_dataset.py      # once, CPU, seconds
#   sbatch --array=0-12%6 order_cf/scripts/slurm/order_cf.sbatch
#
# Smoke test:  sbatch --array=7 order_cf/scripts/slurm/order_cf.sbatch

set -e

: "${HF_TOKEN:?HF_TOKEN env var is required (export before sbatch)}"
export HF_TOKEN

cd "${SLURM_SUBMIT_DIR:?submit from the attribution/ directory}"
source scripts/slurm/env.sh
export PYTHONUNBUFFERED=1

# index-aligned with scripts/slurm/unified.sbatch
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
BATCHES=(    8  8  8  8  8  4  4  4  4  4  2  1  1 )
GENBATCHES=( 64 64 32 32 32 16 16 16 16 16  8  4  4 )

IDX=${SLURM_ARRAY_TASK_ID:-0}
MODEL=${MODELS[$IDX]}
BATCH=${BATCHES[$IDX]}
GBATCH=${GENBATCHES[$IDX]}
SAFE=$(echo "$MODEL" | tr '/.' '_-')

RES=order_cf/results
DATA=order_cf/data

echo "===================================================================="
echo "operand-order counterfactual   model=$MODEL  batch=$BATCH  gen=$GBATCH"
echo "===================================================================="

if [[ ! -f "$DATA/arith_dataset_numeric_counterfactual.json" ]]; then
    echo "missing $DATA: run order_cf/scripts/make_order_dataset.py first" >&2
    exit 1
fi

# --- 1. stage 1 -------------------------------------------------------------
if [[ -f "$RES/responses/$SAFE/numeric.json" ]]; then
    echo "[1/5] responses exist, skipping"
else
    echo "[1/5] stage 1"
    python -u scripts/generate_responses.py \
        --model "$MODEL" --format numeric \
        --data-dir "$DATA" --results-dir "$RES/responses" \
        --use-all-items \
        --batch-size "$GBATCH"
fi

# --- 2. stage 2 -------------------------------------------------------------
if [[ -f "$RES/attribution/$SAFE/numeric/neuron_attribution_aggregate.pt" ]]; then
    echo "[2/5] attribution exists, skipping"
else
    echo "[2/5] stage 2"
    python -u scripts/run_attribution.py \
        --model "$MODEL" --format numeric --component neurons \
        --results-dir "$RES" --batch-size "$BATCH"
fi

# --- 3. overlap (CPU) -------------------------------------------------------
echo "[3/5] circuit overlap"
python -u order_cf/scripts/circuit_overlap.py --models "$MODEL" \
    --out-dir "$RES/per_model/$SAFE"

# --- 4. transfer: operator circuit into order-swapped prompts ---------------
echo "[4/5] patching transfer"
python -u patching/scripts/run_patching.py \
    --models "$MODEL" --formats numeric \
    --results-dir "$RES" \
    --circuit-results-dir results \
    --out-dir "$RES/patching" \
    --batch-size "$BATCH"

# --- 5. ceiling: order circuit into its own prompts -------------------------
echo "[5/5] patching, within-counterfactual ceiling"
python -u patching/scripts/run_patching.py \
    --models "$MODEL" --formats numeric \
    --results-dir "$RES" \
    --circuit-results-dir "$RES" \
    --out-dir "$RES/patching_own" \
    --batch-size "$BATCH"

echo "done."
