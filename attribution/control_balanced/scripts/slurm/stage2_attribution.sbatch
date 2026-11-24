#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH -t 06:00:00
#SBATCH -c 1
#SBATCH --mem=80G
#SBATCH --job-name=balanced_stage2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Stage 2 for the control_balanced pipeline: attribution patching on the 2 000
# balanced ids selected by select_balanced_fold.py. Reads
# control_balanced/results/balanced_fold/<model>/<fmt>.json to know which
# items to keep.
#
# Required env var: FORMAT
# Optional:        MODEL (default Llama-3.1-8B), BATCH (default 4)
#
# Run 4 in parallel from `attribution/`:
#   for f in numeric english spanish italian; do
#       sbatch --export=ALL,FORMAT=$f control_balanced/scripts/slurm/stage2_attribution.sbatch
#   done

set -e

# HF_TOKEN must be set in the calling environment (do not hardcode here).
: "${HF_TOKEN:?HF_TOKEN env var is required (see attribution/scripts/slurm/README.md)}"
export HF_TOKEN

cd "${SLURM_SUBMIT_DIR:?submit from the attribution/ directory}"
source scripts/slurm/env.sh

MODEL=${MODEL:-meta-llama/Llama-3.1-8B}
FORMAT=${FORMAT:?FORMAT env var must be set (numeric|english|spanish|italian)}
BATCH=${BATCH:-4}

MODEL_SAFE=$(echo "$MODEL" | tr '/.' '_-')
FILTER_IDS="control_balanced/results/balanced_fold/${MODEL_SAFE}/${FORMAT}.json"

if [[ ! -s "$FILTER_IDS" ]]; then
    echo "ERROR: $FILTER_IDS not found. Run select_balanced_fold.py first."
    exit 1
fi

echo "====== control_balanced stage 2: model=$MODEL format=$FORMAT batch=$BATCH ======"
echo "filter ids: $FILTER_IDS ($(python -c "import json; print(len(json.load(open('$FILTER_IDS'))))") ids)"

# We override --responses-dir indirectly through --results-dir: stage 2 reads
# responses from <results-dir>/responses/<model>/<fmt>.json (which lives under
# control_balanced/results/). Same convention as the main pipeline.
python scripts/run_attribution.py \
    --model "$MODEL" \
    --format "$FORMAT" \
    --component neurons \
    --batch-size "$BATCH" \
    --results-dir control_balanced/results \
    --filter-ids-file "$FILTER_IDS"
