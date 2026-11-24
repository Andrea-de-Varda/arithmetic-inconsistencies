#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --job-name=ensemble_analysis
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Predictor comparison, stage B (CPU): nested-CV probes (both substrates) ->
# x1a/x1b, join the cross-fitted circuit loading + confidence controls,
# per-predictor AUROC/AP + linear-model variance partition. Writes the small
# result CSVs under ensemble/results/<model>/, then DELETES that model's
# feature tensors (regeneratable via stage A). Features are memory-mapped and
# read one layer at a time, so 64G is enough even for the 32B models.
#
# Env vars:
#   MODEL          HF id (required)
#   KEEP_FEATURES  if "1", do not delete ensemble/results/<model>/features
#   XMAIN          crossfit (default) | full

set -e

cd "${SLURM_SUBMIT_DIR:?submit from the attribution/ directory}"
source scripts/slurm/env.sh

MODEL=${MODEL:?MODEL env var is required}
KEEP_FEATURES=${KEEP_FEATURES:-0}
XMAIN=${XMAIN:-crossfit}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

MODEL_SAFE=$(echo "$MODEL" | tr '/.' '_-')

echo "===================================================================="
echo "ensemble analysis   model=$MODEL  x_main=$XMAIN"
echo "===================================================================="

python -u ensemble/scripts/probe_and_controls.py --model "$MODEL" \
    --x-main-source "$XMAIN"

if [[ "$KEEP_FEATURES" != "1" ]]; then
    echo "deleting feature tensors (ensemble/results/${MODEL_SAFE}/features) ..."
    rm -rf "ensemble/results/${MODEL_SAFE}/features"
fi
echo "done."
