"""Stage 3: format-level + item-level overlap analysis.

Reads per-format aggregate and per-item attribution tensors and the
stage-1 responses (for per-item gold accuracy). Computes:

  3a. Format-level overlap matrix (4x4) of MLP-unit attributions:
      - Pearson correlation of flattened aggregate attribution
      - Top-K Jaccard of |attribution| at K = 1%, 5%, 10%
      - Layer-wise versions of both
  3b. Item-level: does circuit overlap with the symbolic ('numeric') circuit
      predict per-item accuracy in each verbal format?
      - sum_on_S, cosine_with_S, jaccard_with_S
      - Pearson and point-biserial correlation against clean_match_gold

Outputs:
  results/overlap/<model>/format_overlap_summary.csv
  results/overlap/<model>/format_overlap_by_layer.csv
  results/overlap/<model>/item_level_predictions_neuron.csv
  results/overlap/<model>/item_level_correlations_neuron.txt
  plus per-layer breakdowns (*_by_layer*.csv, per_layer_*_neuron.csv)

Usage:
  python scripts/run_overlap.py --model meta-llama/Llama-3.1-8B
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy.stats import pearsonr, pointbiserialr

from src.data import FORMATS


def model_short(name):
    return name.replace("/", "_").replace(".", "-")


def load_attr(results_dir, model, fmt, component):
    """Load aggregate at fp32 (small, computed at fp32) and per_item at its
    saved precision (fp16 — the per-item tensor is the dominant memory
    cost; for a 32B model it is ~7 GB at fp16 and we keep up to four of
    them resident, so upcasting to fp32 here doubles the memory footprint
    and triggers OOMs on the 32B + Olmo-32B models). All downstream code
    uses np.abs, argpartition, sum, gather — all fp16-safe. Cosine
    similarity (which would underflow at fp16 due to ~1e-12 squared
    magnitudes) is computed via the cosine() helper which upcasts on
    the fly inside its slice."""
    base = os.path.join(results_dir, "attribution", model_short(model), fmt)
    agg = torch.load(os.path.join(base, f"{component}_attribution_aggregate.pt")).float().numpy()
    per_item = torch.load(os.path.join(base, f"{component}_attribution_per_item.pt")).numpy()
    with open(os.path.join(base, "kept_ids.json")) as f:
        kept_ids = json.load(f)
    return agg, per_item, kept_ids


def load_responses(results_dir, model, fmt):
    p = os.path.join(results_dir, "responses", model_short(model), f"{fmt}.json")
    with open(p) as f:
        b = json.load(f)
    by_id = {r["id"]: r for r in b["records"]}
    return by_id


def topk_indices_abs(vec, k):
    """Indices of top-k entries by absolute value."""
    flat = np.abs(vec).ravel()
    if k >= flat.size:
        return np.arange(flat.size)
    idx = np.argpartition(flat, -k)[-k:]
    return idx


def jaccard(a, b):
    sa, sb = set(a.tolist()), set(b.tolist())
    if not sa and not sb:
        return float("nan")
    return len(sa & sb) / len(sa | sb)


def pearson(a, b):
    if a.size == 0 or b.size == 0:
        return float("nan")
    a = a.ravel()
    b = b.ravel()
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def cosine(a, b):
    # Upcast to fp32 inside the helper because attribution magnitudes are
    # ~1e-6, so squared values are ~1e-12 — well below fp16 dynamic range.
    a = np.asarray(a).ravel().astype(np.float32, copy=False)
    b = np.asarray(b).ravel().astype(np.float32, copy=False)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def format_overlap_table(matrix, formats, label):
    """Pretty-print a 4x4 matrix to stdout."""
    print(f"\n{label}")
    header = "          " + " ".join(f"{f:>10}" for f in formats)
    print(header)
    for i, fr in enumerate(formats):
        cells = " ".join(f"{matrix[i, j]:>10.3f}" for j in range(len(formats)))
        print(f"{fr:>10} {cells}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--components", nargs="+", default=["neuron"],
                    choices=["neuron", "head"])
    ap.add_argument("--symbolic-format", default="numeric")
    ap.add_argument("--top-k-fractions", nargs="+", type=float,
                    default=[0.01, 0.05, 0.10])
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.results_dir is None:
        args.results_dir = os.path.join(repo_root, "results")

    out_dir = os.path.join(args.results_dir, "overlap", model_short(args.model))
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print(f"STAGE 3: overlap analysis — model={args.model}")
    print("=" * 70)

    summary_rows = []
    by_layer_rows = []

    for component in args.components:
        # Load per-format aggregate + per-item
        aggs, per_items, kept = {}, {}, {}
        for fmt in FORMATS:
            try:
                a, p, k = load_attr(args.results_dir, args.model, fmt, component)
            except FileNotFoundError as ex:
                print(f"  skipping {fmt}/{component}: {ex}")
                continue
            aggs[fmt] = a
            per_items[fmt] = p
            kept[fmt] = k
        if not aggs:
            print(f"  no data for component={component}, skipping")
            continue

        formats = [f for f in FORMATS if f in aggs]
        n_units = aggs[formats[0]].size  # num_layers * dim
        n_layers = aggs[formats[0]].shape[0]
        print(f"\n[{component}] formats loaded: {formats}, "
              f"shape per format = {aggs[formats[0]].shape} "
              f"(total units = {n_units})")

        # 3a. Format-level overlap (Pearson + Jaccard at multiple K)
        pearson_mat = np.full((len(formats), len(formats)), np.nan)
        for i, fi in enumerate(formats):
            for j, fj in enumerate(formats):
                pearson_mat[i, j] = pearson(aggs[fi], aggs[fj])
                summary_rows.append({
                    "component": component, "metric": "pearson",
                    "format_x": fi, "format_y": fj,
                    "K": "", "value": pearson_mat[i, j],
                })
        format_overlap_table(pearson_mat, formats,
                             f"[{component}] Pearson r — flattened aggregate")

        for kfrac in args.top_k_fractions:
            K = max(1, int(round(kfrac * n_units)))
            jac_mat = np.full((len(formats), len(formats)), np.nan)
            topks = {f: topk_indices_abs(aggs[f], K) for f in formats}
            for i, fi in enumerate(formats):
                for j, fj in enumerate(formats):
                    jac_mat[i, j] = jaccard(topks[fi], topks[fj])
                    summary_rows.append({
                        "component": component, "metric": f"jaccard_top{kfrac:.0%}",
                        "format_x": fi, "format_y": fj,
                        "K": K, "value": jac_mat[i, j],
                    })
            format_overlap_table(jac_mat, formats,
                                 f"[{component}] Jaccard top-{kfrac:.0%} (K={K})")

        # Layer-wise
        for li in range(n_layers):
            for i, fi in enumerate(formats):
                for j, fj in enumerate(formats):
                    pr = pearson(aggs[fi][li], aggs[fj][li])
                    by_layer_rows.append({
                        "component": component, "metric": "pearson",
                        "layer": li, "format_x": fi, "format_y": fj, "K": "",
                        "value": pr,
                    })
            for kfrac in args.top_k_fractions:
                dim = aggs[formats[0]].shape[1]
                K = max(1, int(round(kfrac * dim)))
                topks = {f: topk_indices_abs(aggs[f][li], K) for f in formats}
                for i, fi in enumerate(formats):
                    for j, fj in enumerate(formats):
                        by_layer_rows.append({
                            "component": component,
                            "metric": f"jaccard_top{kfrac:.0%}",
                            "layer": li, "format_x": fi, "format_y": fj, "K": K,
                            "value": jaccard(topks[fi], topks[fj]),
                        })

        # 3b. Item-level overlap with symbolic circuit (only on neuron component
        #     by default; head circuit is small so item-level Jaccard is noisy,
        #     but compute everything anyway).
        sym = args.symbolic_format
        if sym not in aggs:
            print(f"  symbolic format {sym} missing for {component}, skipping item-level")
            continue
        sym_agg = aggs[sym]
        kfrac_main = args.top_k_fractions[0]
        K_main = max(1, int(round(kfrac_main * sym_agg.size)))
        sym_topk = set(topk_indices_abs(sym_agg, K_main).tolist())
        sym_flat = sym_agg.ravel()

        item_rows = []
        item_correlations = []
        for fmt in formats:
            if fmt == sym:
                continue
            responses_by_id = load_responses(args.results_dir, args.model, fmt)
            kept_ids = kept[fmt]
            per_item = per_items[fmt]  # [n_kept, num_layers, dim]
            n_kept, _, _ = per_item.shape

            sums, cosines, jaccards, accs = [], [], [], []
            for i, iid in enumerate(kept_ids):
                rec = responses_by_id.get(iid)
                if rec is None:
                    continue
                vec = per_item[i].ravel()
                sum_S = float(vec[list(sym_topk)].sum())
                cos_S = cosine(vec, sym_flat)
                item_topk = set(topk_indices_abs(per_item[i], K_main).tolist())
                jac_S = (
                    len(item_topk & sym_topk) / len(item_topk | sym_topk)
                    if (item_topk or sym_topk) else float("nan")
                )
                acc = bool(rec["clean_match_gold"])
                sums.append(sum_S)
                cosines.append(cos_S)
                jaccards.append(jac_S)
                accs.append(acc)
                item_rows.append({
                    "component": component, "id": iid, "format": fmt,
                    "accuracy": int(acc),
                    "sum_on_S": sum_S, "cosine_with_S": cos_S,
                    "jaccard_with_S": jac_S, "K": K_main,
                })

            if sums:
                accs_arr = np.array(accs, dtype=float)
                sums_arr = np.array(sums, dtype=float)
                cos_arr = np.array(cosines, dtype=float)
                jac_arr = np.array(jaccards, dtype=float)
                # point-biserial = Pearson when one side is binary
                pb_sum = pointbiserialr(accs_arr, sums_arr) if len(set(accs)) > 1 else (float("nan"), float("nan"))
                pb_cos = pointbiserialr(accs_arr, cos_arr) if len(set(accs)) > 1 else (float("nan"), float("nan"))
                pb_jac = pointbiserialr(accs_arr, jac_arr) if len(set(accs)) > 1 else (float("nan"), float("nan"))
                item_correlations.append({
                    "component": component, "format": fmt, "n": len(sums),
                    "acc_rate": float(np.mean(accs_arr)),
                    "rpb_sum_on_S": pb_sum[0], "p_sum_on_S": pb_sum[1],
                    "rpb_cosine": pb_cos[0], "p_cosine": pb_cos[1],
                    "rpb_jaccard": pb_jac[0], "p_jaccard": pb_jac[1],
                })

        # 3c. Per-layer item-level point-biserial: for each layer L, compute
        #     per-item sum on the per-layer symbolic top-1% units, then correlate
        #     with accuracy. Saves a CSV that the plotting script can read
        #     without needing the per-item .pt files (which are git-ignored).
        per_layer_corr_rows = []
        dim_per_layer = sym_agg.shape[1]
        K_layer = max(1, int(round(kfrac_main * dim_per_layer)))
        sym_layer_topk = [
            np.array(sorted(topk_indices_abs(sym_agg[L], K_layer).tolist()))
            for L in range(n_layers)
        ]
        for fmt in formats:
            if fmt == sym:
                continue
            responses_by_id = load_responses(args.results_dir, args.model, fmt)
            kept_ids = kept[fmt]
            per_item = per_items[fmt]  # [n_kept, num_layers, dim]
            accs = np.array(
                [int(bool(responses_by_id[iid]["clean_match_gold"]))
                 for iid in kept_ids if iid in responses_by_id],
                dtype=float,
            )
            for L in range(n_layers):
                idx = sym_layer_topk[L]
                # per-item sum on layer-L symbolic top-1% units
                sums_L = per_item[:, L, :][:, idx].sum(axis=1).astype(float)
                if len(set(accs.astype(int).tolist())) > 1:
                    rpb, p = pointbiserialr(accs, sums_L)
                else:
                    rpb, p = float("nan"), float("nan")
                per_layer_corr_rows.append({
                    "component": component, "format": fmt, "layer": L,
                    "K_layer": K_layer, "n": len(sums_L),
                    "rpb_sum_on_S": float(rpb), "p_sum_on_S": float(p),
                })
        per_layer_corr_path = os.path.join(
            out_dir, f"item_level_correlations_by_layer_{component}.csv"
        )
        with open(per_layer_corr_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "component", "format", "layer", "K_layer", "n",
                "rpb_sum_on_S", "p_sum_on_S",
            ])
            w.writeheader()
            w.writerows(per_layer_corr_rows)

        # 3d. Per-layer signals split by correctness. Two CSVs are written:
        #
        #     (a) format_overlap_by_layer_correctness_<comp>.csv
        #         Jaccard top-1% with the symbolic circuit, computed
        #         within each layer separately.  This is rank-based within
        #         a layer, so it tells us about *unit identity* — but it
        #         turns out to be largely insensitive to the correct vs
        #         incorrect split, because the within-layer rank order is
        #         preserved across subsets (only magnitudes differ).
        #
        #     (b) per_layer_recruitment_by_correctness_<comp>.csv
        #         Mean (and SD) of per-item `sum at layer-L symbolic top-1%`
        #         within each subset.  This is *magnitude*-based: it tells
        #         us where in the network correct vs incorrect items
        #         actually diverge in their engagement of the symbolic
        #         units. This is what the per-layer point-biserial r is
        #         measuring; saving the raw means lets us visualise the
        #         two subsets directly without going through r.
        per_layer_recruitment_rows = []
        by_layer_correctness_rows = []
        for fmt in formats:
            if fmt == sym:
                continue
            responses_by_id = load_responses(args.results_dir, args.model, fmt)
            kept_ids = kept[fmt]
            per_item = per_items[fmt]  # [n_kept, num_layers, dim]
            accs_int = np.array(
                [int(bool(responses_by_id[iid]["clean_match_gold"]))
                 if iid in responses_by_id else 0
                 for iid in kept_ids],
                dtype=int,
            )
            subsets = {
                "all":       np.arange(per_item.shape[0]),
                "correct":   np.where(accs_int == 1)[0],
                "incorrect": np.where(accs_int == 0)[0],
            }
            for subset_name, idx in subsets.items():
                if len(idx) < 2:
                    for L in range(n_layers):
                        by_layer_correctness_rows.append({
                            "component": component, "format": fmt,
                            "subset": subset_name, "layer": L,
                            "K_layer": K_layer, "n_items": int(len(idx)),
                            "value": float("nan"),
                        })
                        per_layer_recruitment_rows.append({
                            "component": component, "format": fmt,
                            "subset": subset_name, "layer": L,
                            "K_layer": K_layer, "n_items": int(len(idx)),
                            "mean_sum_on_S": float("nan"),
                            "sd_sum_on_S": float("nan"),
                        })
                    continue

                sub_aggregate = per_item[idx].mean(axis=0)  # [num_layers, dim]
                for L in range(n_layers):
                    # (a) per-layer Jaccard with the symbolic top-1% at L,
                    #     based on the rank of units in the subset aggregate.
                    sub_topk = set(
                        topk_indices_abs(sub_aggregate[L], K_layer).tolist()
                    )
                    sym_topk_L = set(sym_layer_topk[L].tolist())
                    union = sub_topk | sym_topk_L
                    jac = (
                        len(sub_topk & sym_topk_L) / len(union)
                        if union else float("nan")
                    )
                    by_layer_correctness_rows.append({
                        "component": component, "format": fmt,
                        "subset": subset_name, "layer": L,
                        "K_layer": K_layer, "n_items": int(len(idx)),
                        "value": float(jac),
                    })
                    # (b) per-layer mean magnitude on the per-LAYER symbolic
                    #     top-1% units, averaged within the subset.  This is
                    #     the magnitude version of `sum_on_S` localised to
                    #     each layer.
                    layer_idx = sym_layer_topk[L]   # 1D array of unit indices
                    sums_L = per_item[idx][:, L, :][:, layer_idx].sum(axis=1)
                    per_layer_recruitment_rows.append({
                        "component": component, "format": fmt,
                        "subset": subset_name, "layer": L,
                        "K_layer": K_layer, "n_items": int(len(idx)),
                        "mean_sum_on_S": float(np.mean(sums_L)),
                        "sd_sum_on_S": float(np.std(sums_L, ddof=1))
                                         if len(sums_L) > 1 else 0.0,
                    })
        by_layer_correctness_path = os.path.join(
            out_dir, f"format_overlap_by_layer_correctness_{component}.csv"
        )
        with open(by_layer_correctness_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "component", "format", "subset", "layer",
                "K_layer", "n_items", "value",
            ])
            w.writeheader()
            w.writerows(by_layer_correctness_rows)
        print(f"  wrote {os.path.basename(by_layer_correctness_path)}")

        per_layer_recruitment_path = os.path.join(
            out_dir, f"per_layer_recruitment_by_correctness_{component}.csv"
        )
        with open(per_layer_recruitment_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "component", "format", "subset", "layer",
                "K_layer", "n_items", "mean_sum_on_S", "sd_sum_on_S",
            ])
            w.writeheader()
            w.writerows(per_layer_recruitment_rows)
        print(f"  wrote {os.path.basename(per_layer_recruitment_path)}")

        # ===================================================================
        # 3e. Item-level paired metrics + per-item aggregate-overlap.
        #     For each verbal item i, find the same id in the numeric kept set
        #     and compute, layer-by-layer:
        #         jaccard_paired   = J( top-1%(verbal_item[L]),
        #                               top-1%(numeric_item[L]) )
        #         sum_on_paired    = Σ verbal_item[L, top-1%(numeric_item[L])]
        #         jaccard_with_S   = J( top-1%(verbal_item[L]),
        #                               top-1%(numeric_aggregate[L]) )
        #     Subset means (correct / incorrect / all) saved to
        #         per_layer_correctness_metrics_<comp>.csv
        #     Network-level totals saved to
        #         item_level_paired_<comp>.csv  (one row per (item, format)).
        # ===================================================================
        per_layer_extra_rows = []
        item_paired_rows = []

        if sym in per_items:
            num_per_item = per_items[sym]              # [n_num, n_layers, dim]
            num_kept_ids = kept[sym]
            num_id_to_idx = {iid: i for i, iid in enumerate(num_kept_ids)}
            n_layers_local, dim_local = num_per_item.shape[1:]
            num_K_main = max(1, int(round(kfrac_main * n_layers_local * dim_local)))

            for fmt in formats:
                if fmt == sym:
                    continue
                ver_per_item = per_items[fmt]
                ver_kept_ids = kept[fmt]
                responses_by_id = load_responses(args.results_dir, args.model, fmt)

                paired = []   # (verbal_idx, numeric_idx, accuracy, id)
                for vi, iid in enumerate(ver_kept_ids):
                    ni = num_id_to_idx.get(iid)
                    if ni is None:
                        continue
                    rec = responses_by_id.get(iid)
                    if rec is None:
                        continue
                    paired.append((vi, ni,
                                    int(bool(rec["clean_match_gold"])), iid))
                if not paired:
                    continue

                v_idx = np.array([p[0] for p in paired])
                n_idx = np.array([p[1] for p in paired])
                accs = np.array([p[2] for p in paired])
                n_paired = len(paired)
                K_layer_local = max(1, int(round(kfrac_main * dim_local)))

                # Per-layer paired + aggregate-overlap, vectorised
                jac_paired_all = np.full((n_paired, n_layers_local), np.nan)
                sum_on_paired_all = np.full((n_paired, n_layers_local), np.nan)
                jac_with_S_agg_all = np.full((n_paired, n_layers_local), np.nan)

                arange_p = np.arange(n_paired)
                for L in range(n_layers_local):
                    ver_L = ver_per_item[v_idx, L, :]
                    num_L = num_per_item[n_idx, L, :]
                    # top-K_layer per item by absolute value
                    ver_top = np.argpartition(np.abs(ver_L), -K_layer_local,
                                                axis=1)[:, -K_layer_local:]
                    num_top = np.argpartition(np.abs(num_L), -K_layer_local,
                                                axis=1)[:, -K_layer_local:]
                    ver_mask = np.zeros((n_paired, dim_local), dtype=bool)
                    ver_mask[arange_p[:, None], ver_top] = True
                    num_mask = np.zeros((n_paired, dim_local), dtype=bool)
                    num_mask[arange_p[:, None], num_top] = True
                    inter_pn = (ver_mask & num_mask).sum(axis=1)
                    union_pn = (ver_mask | num_mask).sum(axis=1)
                    jac_paired_all[:, L] = inter_pn / np.maximum(union_pn, 1)
                    sum_on_paired_all[:, L] = np.take_along_axis(
                        ver_L, num_top, axis=1).sum(axis=1)
                    # aggregate symbolic top-1% at layer L
                    agg_top = sym_layer_topk[L]
                    agg_mask = np.zeros(dim_local, dtype=bool)
                    agg_mask[agg_top] = True
                    inter_pa = (ver_mask & agg_mask).sum(axis=1)
                    union_pa = ver_mask.sum(axis=1) + agg_mask.sum() - inter_pa
                    jac_with_S_agg_all[:, L] = inter_pa / np.maximum(union_pa, 1)

                # Subset means per layer
                subsets_idx = {
                    "all":       np.arange(n_paired),
                    "correct":   np.where(accs == 1)[0],
                    "incorrect": np.where(accs == 0)[0],
                }
                for subset_name, sidx in subsets_idx.items():
                    for L in range(n_layers_local):
                        if len(sidx) < 2:
                            row = {
                                "component": component, "format": fmt,
                                "subset": subset_name, "layer": L,
                                "K_layer": K_layer_local,
                                "n_items": int(len(sidx)),
                                "mean_sum_on_paired": float("nan"),
                                "sd_sum_on_paired": float("nan"),
                                "mean_jaccard_paired": float("nan"),
                                "sd_jaccard_paired": float("nan"),
                                "mean_jaccard_with_S_agg": float("nan"),
                                "sd_jaccard_with_S_agg": float("nan"),
                            }
                        else:
                            row = {
                                "component": component, "format": fmt,
                                "subset": subset_name, "layer": L,
                                "K_layer": K_layer_local,
                                "n_items": int(len(sidx)),
                                "mean_sum_on_paired": float(np.mean(sum_on_paired_all[sidx, L])),
                                "sd_sum_on_paired": float(np.std(sum_on_paired_all[sidx, L], ddof=1)),
                                "mean_jaccard_paired": float(np.mean(jac_paired_all[sidx, L])),
                                "sd_jaccard_paired": float(np.std(jac_paired_all[sidx, L], ddof=1)),
                                "mean_jaccard_with_S_agg": float(np.mean(jac_with_S_agg_all[sidx, L])),
                                "sd_jaccard_with_S_agg": float(np.std(jac_with_S_agg_all[sidx, L], ddof=1)),
                            }
                        per_layer_extra_rows.append(row)

                # Network-level paired metrics, computed ONE ITEM AT A TIME.
                # For a 32B model the flat per-item tensor is ~3.5 MB at fp16
                # (≈ 1.77M fp16 = 3.5 MB) — orders of magnitude smaller than
                # the [n_paired, total_units] tensor (≈ 7 GB) the previous
                # vectorised approach materialised. The per-item path keeps
                # peak working memory bounded to a handful of MB per
                # iteration regardless of model size.
                for k, (vi, ni, acc, iid) in enumerate(paired):
                    ver_item = ver_per_item[vi].ravel()      # fp16, ~3.5 MB
                    num_item = num_per_item[ni].ravel()
                    abs_v = np.abs(ver_item)
                    abs_n = np.abs(num_item)
                    ver_top_idx = np.argpartition(abs_v, -num_K_main)[-num_K_main:]
                    num_top_idx = np.argpartition(abs_n, -num_K_main)[-num_K_main:]
                    vt = set(int(x) for x in ver_top_idx)
                    nt = set(int(x) for x in num_top_idx)
                    inter = len(vt & nt)
                    union = len(vt) + len(nt) - inter
                    jac_total = inter / max(union, 1)
                    sum_total = float(ver_item[num_top_idx].sum())
                    cos_total = cosine(ver_item, num_item)
                    item_paired_rows.append({
                        "component": component, "id": iid,
                        "format": fmt, "accuracy": acc, "K": num_K_main,
                        "sum_on_paired": sum_total,
                        "jaccard_paired": jac_total,
                        "cosine_paired": cos_total,
                    })

        per_layer_extra_path = os.path.join(
            out_dir, f"per_layer_correctness_metrics_{component}.csv"
        )
        with open(per_layer_extra_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "component", "format", "subset", "layer",
                "K_layer", "n_items",
                "mean_sum_on_paired", "sd_sum_on_paired",
                "mean_jaccard_paired", "sd_jaccard_paired",
                "mean_jaccard_with_S_agg", "sd_jaccard_with_S_agg",
            ])
            w.writeheader()
            w.writerows(per_layer_extra_rows)
        print(f"  wrote {os.path.basename(per_layer_extra_path)}")

        item_paired_path = os.path.join(
            out_dir, f"item_level_paired_{component}.csv"
        )
        with open(item_paired_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "component", "id", "format", "accuracy", "K",
                "sum_on_paired", "jaccard_paired", "cosine_paired",
            ])
            w.writeheader()
            w.writerows(item_paired_rows)
        print(f"  wrote {os.path.basename(item_paired_path)}")

        # write item-level outputs
        item_pred_path = os.path.join(out_dir, f"item_level_predictions_{component}.csv")
        with open(item_pred_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "component", "id", "format", "accuracy",
                "sum_on_S", "cosine_with_S", "jaccard_with_S", "K",
            ])
            w.writeheader()
            w.writerows(item_rows)

        corr_path = os.path.join(out_dir, f"item_level_correlations_{component}.txt")
        with open(corr_path, "w") as f:
            f.write(
                f"Item-level correlations between circuit-overlap-with-symbolic and clean_match_gold\n"
                f"  symbolic = {sym}, K (fraction) = {kfrac_main}, "
                f"K (units) = {K_main}\n\n"
            )
            for row in item_correlations:
                line = (
                    f"  format={row['format']:<8} n={row['n']:<5} "
                    f"acc={row['acc_rate']:.2%}   "
                    f"rpb(sum_on_S)={row['rpb_sum_on_S']:+.3f} (p={row['p_sum_on_S']:.3g})   "
                    f"rpb(cos)={row['rpb_cosine']:+.3f} (p={row['p_cosine']:.3g})   "
                    f"rpb(jac)={row['rpb_jaccard']:+.3f} (p={row['p_jaccard']:.3g})\n"
                )
                f.write(line)
                print(f"[{component}] {line.strip()}")

    # 3a outputs
    summary_path = os.path.join(out_dir, "format_overlap_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "component", "metric", "format_x", "format_y", "K", "value"
        ])
        w.writeheader()
        w.writerows(summary_rows)
    by_layer_path = os.path.join(out_dir, "format_overlap_by_layer.csv")
    with open(by_layer_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "component", "metric", "layer", "format_x", "format_y", "K", "value"
        ])
        w.writeheader()
        w.writerows(by_layer_rows)

    print(f"\nwrote {summary_path}")
    print(f"wrote {by_layer_path}")


if __name__ == "__main__":
    main()
