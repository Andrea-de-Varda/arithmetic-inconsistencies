#!/bin/bash
#SBATCH -t 00:30:00
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH --job-name=balanced_datagen
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Generate a 20 000-item arithmetic dataset for the accuracy-balanced control
# analysis. CPU-only. Uses seed 43 so the new items don't collide with the
# existing 5 000-item dataset's fold (seed 42 there).
#
# Run from `attribution/`:
#   sbatch control_balanced/scripts/slurm/generate_data.sbatch

set -e

cd "${SLURM_SUBMIT_DIR:?submit from the attribution/ directory}"
source scripts/slurm/env.sh

# num2words is required by the generation scripts; install if missing.
python -c "import num2words" 2>/dev/null || pip install --user num2words

cd control_balanced/data
python ../../../data_generation/generate_all_formats.py \
    --num-problems 20000 \
    --seed 43 \
    --output-dir .

echo "data generation done"
ls -la *.json
