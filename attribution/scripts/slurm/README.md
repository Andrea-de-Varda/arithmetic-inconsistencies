# SLURM scripts for the main pipeline

Job scripts for stages 1–3 of the main pipeline (13 models x 4 formats) and for the all-pairs item-level table. The analysis folders (`patching/`, `order_cf/`, `heldout/`, `matched_baseline/`, `threshold_robustness/`, `ensemble/`, `control_balanced/`) have their own job scripts under `<folder>/scripts/slurm/`, documented in each folder's README; they follow the same conventions.

## Conventions

- Submit every job from `attribution/`. Each script `cd`s into `$SLURM_SUBMIT_DIR` and sources `scripts/slurm/env.sh`, and writes logs to `logs/` (created by `env.sh`).
- `env.sh` is the only file you should need to edit. It activates the conda environment named by `CONDA_ENV` (default `arith-circuits`, from `environment.yml`), sets `HF_HUB_DISABLE_TELEMETRY=1` and `TOKENIZERS_PARALLELISM=false`, and has a commented `HF_HOME` line for pointing the HuggingFace cache at a large disk (the 32B models need about 65 GB each).
- No partition is hardcoded: pass `-p <partition>` to `sbatch`, or export `SBATCH_PARTITION` before running a launcher.
- Jobs that download models check `HF_TOKEN` at start-up and abort if it is unset. Llama, Gemma and Mistral are gated on the HuggingFace Hub. The token is read from the environment and never written to disk.
- GPU jobs request one A100 (`--gres=gpu:a100:1`). The 32B models run at batch size 1 and need an 80 GB card; `--mem` is host RAM, not GPU memory, so add a `--constraint` if your cluster mixes 40 GB and 80 GB A100s.

## Full sweep

```bash
cd attribution
export HF_TOKEN=...            # HuggingFace token with access to the gated models
export SBATCH_PARTITION=...    # your partition
bash scripts/slurm/launch_all_models.sh
```

`launch_all_models.sh` submits:

1. `unified.sbatch` as one array, `--array=0-51%4` (13 models x 4 formats, at most four GPUs at a time). Task `i` runs model `i / 4` on format `i % 4` (numeric, english, spanish, italian): stage 1 (`scripts/generate_responses.py`) then stage 2 (`scripts/run_attribution.py --component neurons`). A stage is skipped if its output (`results/responses/<model>/<format>.json`, resp. `results/attribution/<model>/<format>/neuron_attribution_aggregate.pt`) already exists, so resubmitting the array only fills gaps.
2. One `overlap.sbatch` per model, with an `afterok` dependency on that model's four array tasks. It runs stage 3, `scripts/run_overlap.py --model $MODEL --components neuron`.

The per-item tensors written by stage 2 stay on the cluster (they are git-ignored); stage 3 and every downstream analysis that reads them must run there. The cross-model figures are then rebuilt anywhere from the tracked CSVs (see [results/README.md](../../results/README.md)).

## Scripts

| Script | What it runs | GPU | CPUs | Host memory | Walltime | Env vars |
|---|---|---|---|---|---|---|
| `launch_all_models.sh` | submits the array and the per-model stage-3 jobs (run on the login node) | – | – | – | – | `HF_TOKEN` (required) |
| `unified.sbatch` | stage 1 + stage 2 for one (model, format) | 1 x A100 | 1 | 80G | 16 h | `HF_TOKEN` (required); `SLURM_ARRAY_TASK_ID` selects the cell |
| `overlap.sbatch` | stage 3 for one model | – | 2 | 64G | 1 h | `MODEL` (required) |
| `all_pairs_item_loadings.sbatch` | `scripts/all_pairs_item_loadings.py`, all 13 models | – | 2 | 128G | 2 h | – |

Batch sizes are set per model inside `unified.sbatch`: attribution 8 (models up to 4B), 4 (7–9B), 2 (Phi-4), 1 (32B); decoding 64 down to 4. The `MODELS` arrays in `launch_all_models.sh` and `unified.sbatch` must list the models in the same order, because the array index is decoded against both.

To rerun a single cell, submit the matching array index, e.g. Llama-3.1-8B (model index 7) on Spanish (format index 2): `sbatch --array=30 --export=ALL,HF_TOKEN=$HF_TOKEN scripts/slurm/unified.sbatch`. To rerun stage 3 for one model: `sbatch --export=ALL,MODEL=meta-llama/Llama-3.1-8B scripts/slurm/overlap.sbatch`.

`all_pairs_item_loadings.sbatch` needs stage 2 to have finished for all models and writes `results/all_pairs/per_item_pointbiserial.csv`. The 128G covers the 32B models' per-item tensors (about 7 GB each in fp16).

## Models

The order used by the array index: Qwen3-0.6B, Llama-3.2-1B, gemma-2-2b, Llama-3.2-3B, Qwen3-4B, Mistral-7B-v0.1, Olmo-3-1025-7B, Llama-3.1-8B, Qwen3-8B, gemma-2-9b, phi-4, Qwen3-32B, Olmo-3-1125-32B. Mixture-of-experts models are not supported: `src/attribution.py` hooks a single `mlp.down_proj` per layer and has no expert routing.
