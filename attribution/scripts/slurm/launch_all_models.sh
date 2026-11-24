#!/bin/bash
# Submit the entire multi-model pipeline:
#   - one big SLURM array (one element per (model, format) pair) running
#     stage 1 + stage 2 sequentially. The %4 throttle on the array means
#     at most 4 A100s are ever in flight.
#   - per-model stage-3 (overlap) jobs that depend
#     on the relevant 4 array elements completing.
#
# Run this once on the login node:
#   bash scripts/slurm/launch_all_models.sh
#
# The model list and per-model batch sizes MUST stay in sync with
# unified.sbatch — both files duplicate the arrays so SLURM can index from
# either side. Edit them together.
#
# Models intentionally omitted:
#   - Qwen/Qwen3-30B-A3B-Base : MoE; src/attribution.py only hooks a single
#     mlp.down_proj per layer (no expert routing). Would silently attribute
#     over the wrong tensor. Run separately with a MoE-aware extension if
#     you want it.

set -e

cd "$(dirname "$0")/../.."   # -> attribution/
mkdir -p logs

# HF_TOKEN must be exported in the calling shell. sbatch inherits the
# environment by default, but we forward it explicitly via --export below
# to be safe.
: "${HF_TOKEN:?HF_TOKEN env var is required (see scripts/slurm/README.md)}"

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

N_MODELS=${#MODELS[@]}
N_FORMATS=4   # numeric english spanish italian
TOTAL=$(( N_MODELS * N_FORMATS ))

echo "submitting array job for $N_MODELS models × $N_FORMATS formats = $TOTAL tasks"
echo "throttle: %4 (at most 4 A100s in flight at any time)"

ARR_ID=$(sbatch --parsable \
    --array=0-$((TOTAL - 1))%4 \
    --export="ALL,HF_TOKEN=$HF_TOKEN" \
    scripts/slurm/unified.sbatch)
echo "stage 1+2 array job id: $ARR_ID  (sbatch name: arith_pipeline)"

# Per-model overlap job, depending on its 4 array tasks.
for ((m = 0; m < N_MODELS; m++)); do
    base=$(( m * N_FORMATS ))
    deps="afterok:${ARR_ID}_${base}"
    for ((k = 1; k < N_FORMATS; k++)); do
        deps="${deps}:${ARR_ID}_$((base + k))"
    done
    OP_ID=$(sbatch --parsable --dependency="$deps" \
        --export="ALL,MODEL=${MODELS[$m]}" \
        scripts/slurm/overlap.sbatch)
    printf "  overlap for %-40s -> jid %s\n" "${MODELS[$m]}" "$OP_ID"
done

echo
echo "all jobs submitted. Watch with:"
echo "  squeue -u \$USER -a"
echo
echo "stage 1+2 logs:   logs/arith_pipeline_${ARR_ID}_<task>.{out,err}"
echo "stage 3 logs  :   logs/arith_overlap_<jid>.{out,err}"
