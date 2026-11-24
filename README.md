# Shared circuits predict whether LLMs generalize across formats in arithmetic reasoning

Code and data for the EACL paper by Andrea Gregor de Varda, Sana Pandey, Pengrui Han, Jacob Andreas and Evelina Fedorenko.

LLMs solve numeric arithmetic (`44 + 22 =`) almost perfectly but are much less accurate on verbal renditions of the same problems (*forty-four plus twenty-two equals*, and its Spanish and Italian translations). For 13 base models (0.6B–32B, six families) we localize the units each model recruits for numeric, English, Spanish and Italian arithmetic with attribution patching, define the **numeric circuit** as the top 1% of units of the numeric aggregate, and test whether overlap with that circuit predicts verbal-format behavior. It does, at three levels: which formats are hardest, which models generalize best, and which individual items are solved correctly.

## Repository layout

```
.
├── arithmetic-dataset/         the 5,000 problems x 4 formats used in the main analyses
├── data_generation/            dataset generator (numeric + English/Spanish/Italian renditions + sign-flipped counterfactuals)
├── environment.yml
└── attribution/                all analyses
    ├── src/                    shared library: data loading, model loading, attribution patching, metrics
    ├── scripts/                main pipeline (stages 1-3), Figure 2 and Figure 3, SLURM launchers
    ├── data/                   the fixed 2,000-item fold used by every analysis
    ├── results/                main pipeline outputs (responses, aggregate attributions, overlap and loading CSVs)
    ├── ensemble/               circuit loading vs. probes and confidence measures (Section 3.4, Appendix G-H)
    ├── patching/               causal validation by activation patching (Appendix C)
    ├── order_cf/               operand-order counterfactual (Appendix D)
    ├── matched_baseline/       size- and layer-matched random unit sets (Appendix E)
    ├── threshold_robustness/   circuit threshold sweep 0.5-5% (Appendix F)
    ├── heldout/                cross-fitted (held-out) circuit identification (Appendix G)
    └── control_balanced/       accuracy-balanced control and all format pairs, Llama-3.1-8B (Appendix I-J)
```

Each analysis folder has its own README with the design, the exact commands and the output schema.

## Paper to code

| Paper | What | Produced by (run from `attribution/`) | Output |
|---|---|---|---|
| Fig. 2, §3.1 | per-format accuracy | `python scripts/make_cross_model_plots.py` | `results/plots/_cross_model/10_accuracy_bars_per_format_1x4_neuron.png` |
| §3.2 | format- and model-level overlap | same script | `results/plots/_cross_model/cross_model_table_neuron.csv` |
| Fig. 3, §3.3 | item-level loading vs. correctness | `python scripts/make_fig_item_combined.py` | `results/necessity/figures/item_violins_necessity_combined.png` |
| Fig. 4, §3.4 | variance partition (LMG) | `ensemble/` (see its README) | `ensemble/results/figures/C5_lmg_bars.png` |
| Fig. 5, App. C | causal validation | `python patching/scripts/plot_patching.py` | `patching/results/figures/causal_validation.png` |
| App. D | operand-order counterfactual | `order_cf/` | `order_cf/results/summary.csv` |
| App. E | matched random baseline | `matched_baseline/` | `matched_baseline/results/summary.{csv,txt}` |
| App. F | threshold robustness | `threshold_robustness/` | `threshold_robustness/results/*_summary.csv` |
| Fig. 6, App. G | held-out circuits, AUROC | `heldout/`, `ensemble/` | `heldout/results/predictive_metrics.csv`, `ensemble/results/figures/C4_auroc_bars.png` |
| App. H | probe and confidence details | `ensemble/` | `ensemble/results/<model>/` |
| Table 1, App. I | correct vs. incorrect numeric circuits | `control_balanced/scripts/symbolic_correctness_split.py` | `control_balanced/results/overlap/meta-llama_Llama-3-1-8B/symbolic_split_overview.txt` |
| Fig. 7, App. J | every anchor format vs. every target | `python control_balanced/scripts/plot_all_pairs_balanced.py` | `control_balanced/results/all_pairs/figures/F3_item_pointbiserial_heatmap_balanced.png` |

## Quickstart: rebuild the figures on a laptop

All outputs needed to rebuild the figures and summary tables are tracked in the repository, so this step needs no GPU and no model downloads.

```bash
conda env create -f environment.yml
conda activate arith-circuits
cd attribution
python scripts/make_cross_model_plots.py        # Figure 2 + cross-model table
python scripts/make_fig_item_combined.py        # Figure 3
```

The other figures and summaries are rebuilt by the CPU-only summary/plot script of each analysis folder; see the folder READMEs.

## Full rerun on a GPU cluster

The GPU stages were run on a SLURM cluster with one A100 (80 GB) per job. All job scripts are submitted from `attribution/` and source `attribution/scripts/slurm/env.sh`, the only file you should need to edit (Python environment, HuggingFace cache location). No partition is hardcoded: pass `-p <partition>` or export `SBATCH_PARTITION`. Gated models (Llama, Gemma, Mistral) need a HuggingFace token in the environment; it is never written to disk.

```bash
cd attribution
export HF_TOKEN=...                       # your HuggingFace token
export SBATCH_PARTITION=...               # your partition
bash scripts/slurm/launch_all_models.sh   # stages 1-3 for all 13 models x 4 formats
```

See [attribution/scripts/slurm/README.md](attribution/scripts/slurm/README.md) for resources per stage.

Stage 2 writes per-item attribution tensors (`neuron_attribution_per_item.pt`, up to ~2 GB per model and format). They are git-ignored and not distributed; the analyses that need them (stage 3, `heldout/`, `matched_baseline/`, `threshold_robustness/` item level, `order_cf/`, `control_balanced/`) regenerate them on the cluster. Everything derived from them is tracked.

## Data

`arithmetic-dataset/arith_dataset_<format>_counterfactual.json` holds 5,000 problems per format. Each problem has the form `a1 □ a2 (□ a3) =` over positive integers with `□ ∈ {+, −}`; operands have 2 or 3 digits and problems have 2 or 3 terms (balanced 50/50), and half require a carry. Each item carries its sign-flipped counterfactual: `x` and `y` are the prompt and gold answer, `x_prime` and `y_prime` the sign-flipped prompt and its gold answer. Item ids are aligned across the four formats, so the same id is the same arithmetic content in every format.

All main analyses use a fixed 2,000-item fold drawn once from these items with `np.random.default_rng(42)` and stored in `attribution/data/attribution_fold_ids.json`; the same ids are used for every model and format.

`data_generation/generate_all_formats.py` generates datasets with this structure. `python data_generation/generate_all_formats.py -n 20000 --seed 43 --output-dir attribution/control_balanced/data` reproduces the 20,000-item dataset of Appendices H-J byte for byte. The 5,000-item main dataset was generated with an earlier version of the same generator and is distributed as is.

## Method summary

1. **Responses** (`scripts/generate_responses.py`). Each prompt is a one-shot exemplar in the target format followed by the problem, with no chat template. The model greedily decodes an answer to `x` and to `x'`; correctness is scored against the gold answer.
2. **Attribution patching** (`scripts/run_attribution.py`, `src/attribution.py`). The metric is built from the model's own answers, not from gold answers: `m(z) = log P(ŷ | z) − log P(ŷ' | z)`, rescaled to 1 on `x` and 0 on `x'`. Each MLP unit at the last prompt position gets `(a(x) − a(x')) · ∇M`, with the gradient taken on `x'`. Items whose two prompts tokenize to different lengths, or whose two answers are identical, are dropped.
3. **Overlap and loading** (`scripts/run_overlap.py`). Item attributions are averaged into a format-level aggregate. The numeric circuit is the top 1% of units of the numeric aggregate by absolute attribution. For each verbal format we report its top-1% Jaccard overlap with the numeric aggregate and, per item, its loading: the sum of the item's attributions over the numeric circuit.

## Models

Qwen-3 0.6B, 4B, 8B, 32B; Llama-3.2 1B, 3B and Llama-3.1 8B; Gemma-2 2B, 9B; Mistral 7B v0.1; OLMo-3 7B, 32B; Phi-4 14B. All are base (non-instruction-tuned) checkpoints from the HuggingFace Hub.

## Citation

```bibtex
@inproceedings{devarda2027shared,
  title     = {Shared circuits predict whether {LLM}s generalize across formats in arithmetic reasoning},
  author    = {de Varda, Andrea Gregor and Pandey, Sana and Han, Pengrui and Andreas, Jacob and Fedorenko, Evelina},
  booktitle = {Proceedings of the Conference of the European Chapter of the Association for Computational Linguistics (EACL)},
  year      = {2027}
}
```

## License

Code is released under the MIT License (see `LICENSE`). The datasets in `arithmetic-dataset/` and `attribution/control_balanced/data/` are released under CC BY 4.0.
