# Shared environment for every SLURM script in this repository.
#
# All job scripts are submitted from the attribution/ directory, cd into
# $SLURM_SUBMIT_DIR and source this file. Edit it to match your cluster.
#
# Partition: none is hardcoded. Pass `-p <partition>` to sbatch, or export
# SBATCH_PARTITION=<partition> before running a launcher script.

# Python environment (see environment.yml at the repository root). Either
# export CONDA_ENV before submitting, or replace this block with whatever
# your cluster needs (module load ..., source venv/bin/activate, ...).
CONDA_ENV="${CONDA_ENV:-arith-circuits}"
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

# HuggingFace cache. The 32B models need ~65 GB each; point HF_HOME at a
# large disk if your home quota is small.
# export HF_HOME=/path/to/hf_cache
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

mkdir -p logs
