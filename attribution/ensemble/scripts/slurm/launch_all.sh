#!/bin/bash
# Submit the predictor comparison for the whole 13-model panel with SLURM
# dependencies:
#
#   crossfit (CPU, all models)          heldout/scripts/slurm/crossfit.sbatch
#   pool responses (GPU, array 0-12)    stage1_pool_responses.sbatch  (one task per model;
#                                        a task is a no-op if the responses exist)
#   per model:
#     stage A  feature extraction (GPU) extract_features.sbatch  (afterok: its pool task)
#     stage B  probes + stats (CPU)     analysis.sbatch          (afterok: A + crossfit)
#
# The pool array is throttled with %MAX_GPU; extraction jobs are throttled by
# chaining each one after the extraction of the model MAX_GPU places earlier,
# so at most ~2*MAX_GPU A100s are ever requested by this launcher.
#
# Usage, from attribution/ on the login node, with HF_TOKEN exported:
#   bash ensemble/scripts/slurm/launch_all.sh
#   MODELS="Qwen/Qwen3-0.6B meta-llama/Llama-3.2-1B" bash ensemble/scripts/slurm/launch_all.sh
#   SKIP_CROSSFIT=1 ...   if heldout/results/crossfit_item_loadings.csv already exists
#
# The model order below is the array index order of stage1_pool_responses.sbatch
# and MUST stay identical to it.

set -e

cd "$(dirname "$0")/../../.."   # -> attribution/
mkdir -p logs

: "${HF_TOKEN:?HF_TOKEN env var is required (see scripts/slurm/README.md)}"

ALL_MODELS=(
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
# forward batch for extract_features (mirrors scripts/slurm/unified.sbatch)
ALL_BATCH=( 8 8 8 8 8 4 4 4 4 4 2 1 1 )

MAX_GPU=${MAX_GPU:-4}

# --- which array indices to run ---
if [[ -n "$MODELS" ]]; then
    read -r -a WANT <<< "$MODELS"
    IDX=()
    for m in "${WANT[@]}"; do
        found=""
        for i in "${!ALL_MODELS[@]}"; do
            [[ "${ALL_MODELS[$i]}" == "$m" ]] && found=$i
        done
        [[ -z "$found" ]] && { echo "unknown model: $m"; exit 1; }
        IDX+=("$found")
    done
else
    IDX=( $(seq 0 $(( ${#ALL_MODELS[@]} - 1 ))) )
fi
ARRAY_SPEC=$(IFS=,; echo "${IDX[*]}")

# --- cross-fitted loadings (heldout package; one CPU job, all models) ---
if [[ "${SKIP_CROSSFIT:-0}" == "1" ]]; then
    CF_DEP=""
    echo "skipping crossfit job (SKIP_CROSSFIT=1)"
else
    CF_ID=$(sbatch --parsable heldout/scripts/slurm/crossfit.sbatch)
    CF_DEP=":${CF_ID}"
    echo "crossfit job: $CF_ID"
fi

# --- probe-pool responses: one array, one task per model ---
POOL_ID=$(sbatch --parsable --array="${ARRAY_SPEC}%${MAX_GPU}" \
    --export="ALL,HF_TOKEN=$HF_TOKEN" \
    ensemble/scripts/slurm/stage1_pool_responses.sbatch)
echo "pool responses array: $POOL_ID  (tasks $ARRAY_SPEC)"

# --- per model: extract (GPU) -> analysis (CPU) ---
declare -a A_IDS=()
n=0
for i in "${IDX[@]}"; do
    MODEL=${ALL_MODELS[$i]}
    b=${ALL_BATCH[$i]}
    deps="afterok:${POOL_ID}_${i}"
    if (( n >= MAX_GPU )); then
        deps="${deps},afterany:${A_IDS[$((n - MAX_GPU))]}"
    fi
    A_ID=$(sbatch --parsable --dependency="$deps" \
        --export="ALL,HF_TOKEN=$HF_TOKEN,MODEL=$MODEL,BATCH=$b" \
        ensemble/scripts/slurm/extract_features.sbatch)
    B_ID=$(sbatch --parsable --dependency="afterok:${A_ID}${CF_DEP}" \
        --export="ALL,MODEL=$MODEL" \
        ensemble/scripts/slurm/analysis.sbatch)
    A_IDS+=("$A_ID")
    printf "  %-30s pool=%s_%s extract=%s analysis=%s\n" "$MODEL" "$POOL_ID" "$i" "$A_ID" "$B_ID"
    n=$((n + 1))
done

echo
echo "all jobs submitted. Watch with:  squeue -u \$USER -a"
echo "when everything is done, on the laptop after git pull:"
echo "  python heldout/scripts/predictive_metrics.py"
echo "  python ensemble/scripts/stats_from_scores.py"
echo "  python ensemble/scripts/plot_ensemble.py"
