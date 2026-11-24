"""Item-level confusion matrix: per-(model, anchor, target) point-biserial
between per-item loading on S^{model, anchor} and per-item correctness in
the target format, on the main 2,000-item fold. The numeric-anchor rows are
the attribution-based r_pb on the x axis of Figure 5B
(patching/scripts/plot_patching.py) and the reference values checked by
matched_baseline/scripts/summarize_matched.py. The balanced-sample version of
this analysis (Figure 7) is control_balanced/scripts/all_pairs_item_loadings_balanced.py.

Needs the per-item attribution tensors that are git-ignored. Runs on the
cluster (CPU, ~1-2 min per model).

For each model and each anchor format f_a:
  S^{m, f_a} = top-1% units of the f_a aggregate (by absolute attribution)

For each target format f_t and each item i in f_t:
  loading_i = sum of f_t's per-item attribution at the S^{m, f_a} indices
  accuracy_i = clean_match_gold

Then point-biserial r between loading_i and accuracy_i across kept items.

Output (one CSV, all (model, anchor, target) rows, all kept items):
  results/all_pairs/per_item_pointbiserial.csv     summary rows
  results/all_pairs/per_item_loadings.csv          per-item rows (optional)

Usage on the cluster:
  cd attribution/
  python scripts/all_pairs_item_loadings.py
  # or, batched:
  sbatch scripts/slurm/all_pairs_item_loadings.sbatch
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy.stats import pointbiserialr

from src.data import FORMATS


MODELS = [
    "Qwen/Qwen3-0.6B",
    "meta-llama/Llama-3.2-1B",
    "google/gemma-2-2b",
    "meta-llama/Llama-3.2-3B",
    "Qwen/Qwen3-4B",
    "mistralai/Mistral-7B-v0.1",
    "allenai/Olmo-3-1025-7B",
    "meta-llama/Llama-3.1-8B",
    "Qwen/Qwen3-8B",
    "google/gemma-2-9b",
    "microsoft/phi-4",
    "Qwen/Qwen3-32B",
    "allenai/Olmo-3-1125-32B",
]


def model_short(name):
    return name.replace("/", "_").replace(".", "-")


def load_aggregate(results_dir, mid, fmt):
    p = os.path.join(results_dir, "attribution", model_short(mid), fmt,
                     "neuron_attribution_aggregate.pt")
    if not os.path.exists(p):
        return None
    return torch.load(p).float().numpy()    # [L, dim]


def load_per_item(results_dir, mid, fmt):
    p = os.path.join(results_dir, "attribution", model_short(mid), fmt,
                     "neuron_attribution_per_item.pt")
    if not os.path.exists(p):
        return None
    return torch.load(p).float().numpy()    # [n_kept, L, dim]


def load_kept_ids(results_dir, mid, fmt):
    p = os.path.join(results_dir, "attribution", model_short(mid), fmt,
                     "kept_ids.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def load_responses(results_dir, mid, fmt):
    p = os.path.join(results_dir, "responses", model_short(mid), f"{fmt}.json")
    if not os.path.exists(p):
        return {}
    return {r["id"]: r for r in json.load(open(p))["records"]}


def topk_abs_indices(flat, K):
    return np.argpartition(np.abs(flat), -K)[-K:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--top-k-fraction", type=float, default=0.01)
    ap.add_argument("--save-per-item", action="store_true",
                    help="also write the full per-item CSV (large)")
    args = ap.parse_args()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.results_dir is None:
        args.results_dir = os.path.join(repo_root, "results")
    out_dir = os.path.join(args.results_dir, "all_pairs")
    os.makedirs(out_dir, exist_ok=True)

    summary_rows = []
    per_item_rows = []
    for mid in MODELS:
        # Pre-compute S^{m, f_a} for each anchor (only the flat indices we need)
        anchor_indices = {}
        anchor_K = {}
        for fa in FORMATS:
            agg = load_aggregate(args.results_dir, mid, fa)
            if agg is None:
                continue
            n_units = agg.size
            K = max(1, int(round(args.top_k_fraction * n_units)))
            anchor_indices[fa] = topk_abs_indices(agg.ravel(), K)
            anchor_K[fa] = K
        if not anchor_indices:
            print(f"  skip {mid}: no aggregates found")
            continue

        for ft in FORMATS:
            per_item = load_per_item(args.results_dir, mid, ft)
            kept_ids = load_kept_ids(args.results_dir, mid, ft)
            responses = load_responses(args.results_dir, mid, ft)
            if per_item is None or kept_ids is None or not responses:
                print(f"  skip {mid}/{ft}: per-item .pt or responses missing")
                continue
            # accuracy vector aligned with kept_ids
            accs = np.array([
                int(bool(responses[i]["clean_match_gold"]))
                if i in responses else 0
                for i in kept_ids
            ], dtype=int)
            flat = per_item.reshape(per_item.shape[0], -1)   # [n, L*dim]

            for fa, idx in anchor_indices.items():
                loadings = flat[:, idx].sum(axis=1).astype(float)
                if len(set(accs.tolist())) > 1:
                    r, p = pointbiserialr(accs.astype(float), loadings)
                    r, p = float(r), float(p)
                else:
                    r, p = float("nan"), float("nan")
                summary_rows.append({
                    "model": mid,
                    "anchor": fa,
                    "target": ft,
                    "K": anchor_K[fa],
                    "n_items": len(loadings),
                    "n_correct": int(accs.sum()),
                    "rpb": r,
                    "p": p,
                })
                if args.save_per_item:
                    for k, iid in enumerate(kept_ids):
                        per_item_rows.append({
                            "model": mid,
                            "anchor": fa,
                            "target": ft,
                            "id": iid,
                            "accuracy": int(accs[k]),
                            "loading": float(loadings[k]),
                        })
            print(f"  {mid}  target={ft}  done")

    summary_path = os.path.join(out_dir, "per_item_pointbiserial.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "model", "anchor", "target", "K", "n_items", "n_correct", "rpb", "p",
        ])
        w.writeheader()
        w.writerows(summary_rows)
    print(f"wrote {summary_path}  ({len(summary_rows)} rows)")

    if args.save_per_item and per_item_rows:
        per_item_path = os.path.join(out_dir, "per_item_loadings.csv")
        with open(per_item_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "model", "anchor", "target", "id", "accuracy", "loading",
            ])
            w.writeheader()
            w.writerows(per_item_rows)
        print(f"wrote {per_item_path}  ({len(per_item_rows)} rows)")


if __name__ == "__main__":
    main()
