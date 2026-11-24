#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH -t 48:00:00
#SBATCH -c 1
#SBATCH --mem=80G
#SBATCH --job-name=arith_patching
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Causal validation of the item-level attribution scores (patching/README.md).
# One job loops over ALL panel models: for each verbal item, overwrites the
# numeric-circuit units at the decision point of the sign-flipped run with the
# clean-run activations (plus a size-matched random control), and records the
# per-item recovery of the preference gap. Per-model batch sizes are built in;
# models already processed are skipped; models downloaded BY THIS RUN are
# purged from the HF cache after use (pre-existing downloads are kept).
#
# Env vars (all optional):
#   MODELS       default "all"
#   TOP_K_FRAC   default 0.01
#   FORMATS      default "english spanish italian"
#   LIMIT        if set, only first N items per format (smoke test)
#   SEED         default 0 (random-control units)

set -e

export PYTHONUNBUFFERED=1   # live progress in the log

: "${HF_TOKEN:?HF_TOKEN env var is required (export before sbatch)}"

cd "${SLURM_SUBMIT_DIR:?submit from the attribution/ directory}"
source scripts/slurm/env.sh

MODELS=${MODELS:-all}
TOP_K_FRAC=${TOP_K_FRAC:-0.01}
FORMATS=${FORMATS:-"english spanish italian"}
SEED=${SEED:-0}
LIMIT_ARG=""
if [[ -n "${LIMIT:-}" ]]; then
    LIMIT_ARG="--limit $LIMIT"
fi

echo "===================================================================="
echo "item-level causal patching  models=$MODELS  formats=$FORMATS"
echo "===================================================================="

python -u patching/scripts/run_patching.py \
    --models $MODELS \
    --top-k-fraction "$TOP_K_FRAC" \
    --formats $FORMATS \
    --seed "$SEED" \
    --purge-cache \
    $LIMIT_ARG
