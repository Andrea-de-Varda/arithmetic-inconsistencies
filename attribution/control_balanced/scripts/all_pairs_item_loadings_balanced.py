"""Balanced-fold version of the all-pairs item-level analysis.

The main-data version (`scripts/all_pairs_item_loadings.py`) is computed on the standard 2000-item attribution fold, whose per-format
accuracy varies wildly across the four formats (Llama-3.1-8B: ~97% numeric,
~38% English, ~33% Spanish, ~18% Italian). That means each anchor's circuit
is identified on a sample with a very different correct/incorrect mix,
which muddies the cross-anchor comparison.

This script computes the SAME analysis on the BALANCED fold instead — the
1600 correct + 400 incorrect items per format that the control_balanced
pipeline already used to produce the appendix's symbolic-correctness split
(Llama-3.1-8B only). Each anchor's circuit is now built from a sample with
identical accuracy (80%), and each target's items also have the 80% balanced
prior. The cross-anchor 4x4 is therefore apples-to-apples.

Pipeline (mirrors the main all_pairs script):
  S^{anchor} = top-1% of |aggregate| over the balanced fold
  for each target format and each kept item i:
      loading_i  = sum of per-item attribution at S^{anchor} indices
      accuracy_i = clean_match_gold
  r_pb(loadings, accuracies) across the 2000 kept items

Inputs (cluster-only; per-item .pt is gitignored):
  control_balanced/results/attribution/<model>/<fmt>/neuron_attribution_aggregate.pt
  control_balanced/results/attribution/<model>/<fmt>/neuron_attribution_per_item.pt
  control_balanced/results/attribution/<model>/<fmt>/kept_ids.json
  control_balanced/results/responses/<model>/<fmt>.json

Output:
  control_balanced/results/all_pairs/per_item_pointbiserial_balanced.csv

Usage on the cluster:
  cd attribution/
  python control_balanced/scripts/all_pairs_item_loadings_balanced.py
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np
import torch
from scipy.stats import pointbiserialr

from src.data import FORMATS


DEFAULT_MODELS = ["meta-llama/Llama-3.1-8B"]


def model_short(name):
    return name.replace("/", "_").replace(".", "-")


def load_aggregate(results_dir, mid, fmt):
    p = os.path.join(results_dir, "attribution", model_short(mid), fmt,
                     "neuron_attribution_aggregate.pt")
    if not os.path.exists(p):
        return None
    return torch.load(p).float().numpy()


def load_per_item(results_dir, mid, fmt):
    p = os.path.join(results_dir, "attribution", model_short(mid), fmt,
                     "neuron_attribution_per_item.pt")
    if not os.path.exists(p):
        return None
    return torch.load(p).float().numpy()


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
    ap.add_argument("--results-dir", default=None,
                    help="defaults to control_balanced/results next to this script")
    ap.add_argument("--top-k-fraction", type=float, default=0.01)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    help="space-separated model ids; balanced data only "
                         "exists for Llama-3.1-8B at the moment")
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.results_dir is None:
        args.results_dir = os.path.join(repo_root, "results")
    out_dir = os.path.join(args.results_dir, "all_pairs")
    os.makedirs(out_dir, exist_ok=True)

    summary_rows = []
    for mid in args.models:
        print(f"\n=== {mid} ===")
        anchor_indices = {}
        anchor_K = {}
        for fa in FORMATS:
            agg = load_aggregate(args.results_dir, mid, fa)
            if agg is None:
                print(f"  skip anchor {fa}: aggregate missing")
                continue
            n_units = agg.size
            K = max(1, int(round(args.top_k_fraction * n_units)))
            anchor_indices[fa] = topk_abs_indices(agg.ravel(), K)
            anchor_K[fa] = K
            print(f"  anchor {fa}: K = {K}")
        if not anchor_indices:
            print(f"  skip {mid}: no aggregates found")
            continue

        for ft in FORMATS:
            per_item = load_per_item(args.results_dir, mid, ft)
            kept_ids = load_kept_ids(args.results_dir, mid, ft)
            responses = load_responses(args.results_dir, mid, ft)
            if per_item is None or kept_ids is None or not responses:
                print(f"  skip target {ft}: per-item .pt or responses missing")
                continue
            accs = np.array([
                int(bool(responses[i]["clean_match_gold"]))
                if i in responses else 0
                for i in kept_ids
            ], dtype=int)
            flat = per_item.reshape(per_item.shape[0], -1)

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
            print(f"  target {ft}: n={len(accs)}  correct={int(accs.sum())}  "
                  f"(rpb computed for all 4 anchors)")

    summary_path = os.path.join(out_dir, "per_item_pointbiserial_balanced.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "model", "anchor", "target", "K", "n_items", "n_correct", "rpb", "p",
        ])
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nwrote {summary_path}  ({len(summary_rows)} rows)")


if __name__ == "__main__":
    main()
