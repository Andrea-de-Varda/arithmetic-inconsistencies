# Operand-order counterfactual (Appendix D)

The attribution metric contrasts each prompt with its sign-flipped twin (`70 − 33 =` vs. `70 + 33 =`). The units it selects could therefore merely read the operator token rather than take part in the arithmetic. This analysis identifies a second circuit with a counterfactual in which the operator is held fixed and the operands are swapped, and asks whether it is the same machinery as the main (operator-flip) circuit.

```
main pipeline (operator flip):   x = 70 - 33 =  ->  37     x' = 70 + 33 =  ->  103
this analysis (operand order):   x = 70 - 33 =  ->  37     x' = 33 - 70 =  ->  -37
```

## Design choices

- **The operator token is identical in `x` and `x'`**, so a unit that only reads the operator receives exactly zero attribution. Whatever this counterfactual selects concerns the operands.
- **`|y'| = |y|`**, so the contrast is a single sign decision. This matters for the causal test: overwriting 1% of units at one token position can carry a low-dimensional decision (which is why the operator flip is patchable at all), but not a whole number. A counterfactual that changes an operand's *value* (`44 + 22` vs. `44 + 25`) would also zero out operator-reading units, but its contrast requires transmitting a multi-digit answer through the patched units, and a 1% patch then recovers almost nothing even with that counterfactual's own circuit, leaving no usable reference point. The order swap keeps the decision as small as the operator flip's.
- **Scope.** Because `|y|` is unchanged, the order swap isolates operand order and the sign of the result, not the magnitude computation. It is evidence against an operator-reading account of the circuit, not a complete account of what the circuit computes.

## Items

All two-term subtraction problems with distinct operands in the 5,000-item main dataset: 925 items, of which 356 lie in the main 2,000-item fold. The whole dataset is used rather than only the fold because the circuit has to be estimated from as many items as possible. In this dataset every such item has `a1 > a2` and operands with the same number of digits, so `x'` is a token-level reordering of `x` and the two prompts align. `x` is unchanged, so these are the paper's problems. Numeric format only.

Stage 2's distinguishability filter drops items where the model gives the same answer to both orders. A model that outputs `|a1 − a2|` regardless of order therefore loses most of its items. This happens for Llama-3.2-1B (16 items kept; same answer under both orders on 98% of items) and Llama-3.2-3B (398 kept; 57%), which are excluded from the cross-model summaries and marked `*` in the printed tables.

## What is computed

1. **Overlap.** Top-1% Jaccard between the order circuit and the operator circuit, against two references: the exact hypergeometric chance level for two independent size-matched masks (mean and 95% interval, about 0.005), and a split-half ceiling, the Jaccard between circuits built from two disjoint halves of the items within one counterfactual. The order circuit rests on fewer items, so its own split-half value is the binding ceiling and is the one reported (`ceiling`, `fraction_of_ceiling`). Because the split-half circuits use half the items, the ceiling slightly understates the reliability of a full-data circuit.
2. **Item-level prediction.** The §3.3 result recomputed with the order circuit: each verbal item's loading is summed over the order circuit's units, with the main verbal per-item attributions unchanged, and correlated with correctness.
3. **Causal transfer.** The operator circuit is patched (as in `patching/`, at the decision point) into the order-swapped prompts, and the recovery of the preference for the answer to `x` is measured. Two references make the number interpretable: the order circuit patched into the same prompts (the ceiling for a 1% circuit under this contrast) and size-matched random units (the floor).

## How to run

```bash
# local, CPU, seconds: build and check the dataset (writes order_cf/data/, tracked)
python order_cf/scripts/make_order_dataset.py
python order_cf/scripts/selftest_order.py

# cluster, from attribution/: one array task per model
export HF_TOKEN=...
sbatch --array=0-12%6 order_cf/scripts/slurm/order_cf.sbatch
# smoke test on Llama-3.1-8B
sbatch --array=7 order_cf/scripts/slurm/order_cf.sbatch

# local, CPU, from tracked files
python order_cf/scripts/summarize_order.py
```

`order_cf.sbatch` (1 x A100, 2 CPUs, 100G host memory, 6 h per task; model order as in `scripts/slurm/unified.sbatch`) runs five steps for one model, skipping stages 1–2 if their outputs exist:

1. stage 1, `scripts/generate_responses.py --format numeric --use-all-items --data-dir order_cf/data --results-dir order_cf/results/responses`;
2. stage 2, `scripts/run_attribution.py --results-dir order_cf/results` → the order circuit;
3. `order_cf/scripts/circuit_overlap.py` (CPU), overlap and item-level prediction, written to `order_cf/results/per_model/<model>/`;
4. `patching/scripts/run_patching.py --results-dir order_cf/results --circuit-results-dir results --out-dir order_cf/results/patching`: operator circuit into order-swapped prompts, plus random units;
5. the same with `--circuit-results-dir order_cf/results --out-dir order_cf/results/patching_own`: the order circuit's own ceiling, plus its random control.

Step 3 reads the per-item attribution tensors of both runs (the order run's, written in step 2, and the main run's numeric and verbal tensors), which are git-ignored; it must run on the cluster after the main sweep. Steps 1, 2, 4 and 5 need a GPU.

Script options: `make_order_dataset.py --data-dir --out-dir --fold-only`; `circuit_overlap.py --main-results-dir --order-results-dir --out-dir --top-k-fraction --models`; `summarize_order.py --results-dir`.

## Outputs

```
order_cf/
├── data/arith_dataset_numeric_counterfactual.json     the 925 order-swap items, main dataset schema
└── results/
    ├── responses/, baselines/, attribution/<model>/numeric/   stages 1–2 of the order run (per-item tensor git-ignored)
    ├── per_model/<model>/
    │   ├── circuit_overlap.csv          model,n_units,K,jaccard_order_vs_operator,chance,chance_lo,chance_hi,ratio_to_chance,
    │   │                                split_half_operator,split_half_order,ceiling,fraction_of_ceiling,n_kept_order
    │   ├── item_level_transfer.csv      model,format,n,n_correct,low_n,rpb_order,p_order,auroc_order,rpb_operator,auroc_operator
    │   └── overlap_validation.txt       log
    ├── patching/<model>/denoise__{real,random}/numeric.json       transfer (operator circuit), patching/ schema
    ├── patching_own/<model>/denoise__{real,random}/numeric.json   ceiling (order circuit), patching/ schema
    └── summary.csv                      all per-model circuit_overlap.csv rows (summarize_order.py)
```

`summarize_order.py` prints the three tables (overlap, item-level prediction, causal test) and writes `summary.csv`.

## Results

Eleven models after the two exclusions above, from `summarize_order.py`:

- **Overlap.** The order circuit and the operator circuit share far more units than chance: median Jaccard 0.312 (range 0.170–0.396), a median 62 times the chance level, and a median 34% of the order circuit's split-half ceiling.
- **Item-level prediction.** Loading on the order circuit predicts verbal correctness about as well as loading on the operator circuit: median r_pb 0.386 vs. 0.387 (English), 0.345 vs. 0.320 (Spanish), 0.327 vs. 0.290 (Italian). It is significantly positive in 10/11, 11/11 and 9/9 models (cells with at least 30 correct items).
- **Causal transfer.** Patching the operator circuit into order-swapped prompts restores a mean 0.698 of the preference, against 0.706 for the order circuit's own ceiling and 0.002 for random units; per model the operator circuit reaches a median 99% of the ceiling (minimum 91%). The transfer is slightly but consistently below the ceiling (Wilcoxon p = 0.007) and far above random units (p = 0.001).
