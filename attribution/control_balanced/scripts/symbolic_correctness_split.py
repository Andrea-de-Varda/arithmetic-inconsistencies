"""Conditional analysis: split the balanced numeric (symbolic) per-item
attribution tensor by clean_match_gold and rebuild two symbolic circuits —
one from the 1 600 correctly-solved numeric items, one from the 400
incorrectly-solved ones. Then compare each to every verbal-format aggregate
and recompute per-item overlap metrics under each.

Question being tested: in the balanced sample, is the 26 % of top-1 % units
that differs between the original symbolic and balanced symbolic circuit the
same as the units that differ between correct-numeric and incorrect-numeric
items? If so, the symbolic circuit has a separable "correctness flavour" on
top of a "pure arithmetic" component.

Reads (all under control_balanced/results/ for this model):
  attribution/<model>/numeric/neuron_attribution_per_item.pt    [n_kept, L, dim]
  attribution/<model>/numeric/kept_ids.json
  attribution/<model>/<fmt>/neuron_attribution_aggregate.pt     for fmt in {english,spanish,italian}
  attribution/<model>/<fmt>/neuron_attribution_per_item.pt
  attribution/<model>/<fmt>/kept_ids.json
  responses/<model>/numeric.json                                (to label numeric items by accuracy)
  responses/<model>/<fmt>.json                                  (to label verbal items by accuracy)

Writes (under control_balanced/results/overlap/<model>/):
  symbolic_split_overview.txt
  symbolic_split_per_item.csv
  symbolic_split_format_overlap.csv

Usage on the cluster (CPU, ~1–2 min):
  cd attribution/
  sbatch control_balanced/scripts/slurm/symbolic_correctness_split.sbatch
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


PEER_FORMATS = ["english", "spanish", "italian"]


def model_short(name):
    return name.replace("/", "_").replace(".", "-")


def load_per_item(results_dir, model_id, fmt):
    p = os.path.join(results_dir, "attribution", model_short(model_id), fmt,
                     "neuron_attribution_per_item.pt")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return torch.load(p).float().numpy()   # [n_kept, L, dim]


def load_aggregate(results_dir, model_id, fmt):
    p = os.path.join(results_dir, "attribution", model_short(model_id), fmt,
                     "neuron_attribution_aggregate.pt")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return torch.load(p).float().numpy()   # [L, dim]


def load_kept_ids(results_dir, model_id, fmt):
    p = os.path.join(results_dir, "attribution", model_short(model_id), fmt,
                     "kept_ids.json")
    with open(p, "r") as f:
        return json.load(f)


def load_responses(results_dir, model_id, fmt):
    p = os.path.join(results_dir, "responses", model_short(model_id),
                     f"{fmt}.json")
    with open(p, "r") as f:
        d = json.load(f)
    return {r["id"]: r for r in d["records"]}


def topk_set(flat_vec, k):
    idx = np.argpartition(flat_vec, -k)[-k:]
    return set(idx.tolist())


def jac(a, b):
    return len(a & b) / len(a | b) if (a or b) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--results-dir", default="control_balanced/results")
    ap.add_argument("--top-k-fraction", type=float, default=0.01)
    args = ap.parse_args()

    short = model_short(args.model)
    out_dir = os.path.join(args.results_dir, "overlap", short)
    os.makedirs(out_dir, exist_ok=True)

    # --- 1. Load numeric per-item attribution and split by correctness ---
    num_per_item = load_per_item(args.results_dir, args.model, "numeric")
    num_kept_ids = load_kept_ids(args.results_dir, args.model, "numeric")
    num_responses = load_responses(args.results_dir, args.model, "numeric")

    n_kept, n_layers, dim = num_per_item.shape
    n_units = n_layers * dim
    K = max(1, int(round(args.top_k_fraction * n_units)))

    accs = np.array([
        int(bool(num_responses[i]["clean_match_gold"])) for i in num_kept_ids
    ])
    correct_idx = np.where(accs == 1)[0]
    incorrect_idx = np.where(accs == 0)[0]
    n_c, n_i = len(correct_idx), len(incorrect_idx)
    print(f"numeric: n_kept={n_kept}  n_correct={n_c}  n_incorrect={n_i}")

    sym_full_agg     = num_per_item.mean(axis=0)
    sym_correct_agg  = num_per_item[correct_idx].mean(axis=0) if n_c else None
    sym_incorr_agg   = num_per_item[incorrect_idx].mean(axis=0) if n_i else None

    sym_full_top   = topk_set(np.abs(sym_full_agg.ravel()), K)
    sym_correct_top = topk_set(np.abs(sym_correct_agg.ravel()), K) if sym_correct_agg is not None else set()
    sym_incorr_top  = topk_set(np.abs(sym_incorr_agg.ravel()), K) if sym_incorr_agg is not None else set()

    # --- 2. Three-way Jaccard among the three symbolic circuits ---
    print()
    print("Jaccard among numeric-derived top-1% circuits:")
    pair_results = []
    for a_name, a in [("symbolic_balanced", sym_full_top),
                       ("symbolic_correct", sym_correct_top),
                       ("symbolic_incorrect", sym_incorr_top)]:
        for b_name, b in [("symbolic_balanced", sym_full_top),
                           ("symbolic_correct", sym_correct_top),
                           ("symbolic_incorrect", sym_incorr_top)]:
            if a is None or b is None:
                continue
            v = jac(a, b)
            pair_results.append((a_name, b_name, v))
            if a_name <= b_name:
                print(f"  J({a_name}, {b_name}) = {v:.3f}")

    # Difference between the two derived circuits
    n_correct_only = len(sym_correct_top - sym_incorr_top)
    n_incorr_only  = len(sym_incorr_top - sym_correct_top)
    n_both         = len(sym_correct_top & sym_incorr_top)
    print(f"  units only in symbolic_correct   : {n_correct_only:>6d}")
    print(f"  units only in symbolic_incorrect : {n_incorr_only:>6d}")
    print(f"  units in both                    : {n_both:>6d}")

    # --- 3. Compare each verbal aggregate to each of the three symbolic circuits ---
    print()
    print("Jaccard top-1% between verbal aggregates and numeric-derived circuits:")
    format_rows = []
    for fmt in PEER_FORMATS:
        fmt_agg = load_aggregate(args.results_dir, args.model, fmt)
        fmt_top = topk_set(np.abs(fmt_agg.ravel()), K)
        row = {"format": fmt}
        for anchor_name, anchor_top in [
            ("symbolic_balanced", sym_full_top),
            ("symbolic_correct", sym_correct_top),
            ("symbolic_incorrect", sym_incorr_top),
        ]:
            if not anchor_top:
                row[f"J_{anchor_name}"] = float("nan")
                continue
            v = jac(fmt_top, anchor_top)
            row[f"J_{anchor_name}"] = v
            print(f"  J({fmt:8s}, {anchor_name:<19s}) = {v:.3f}")
        format_rows.append(row)

    # --- 4. Per-item sum_on_S using each of the three symbolic circuits;
    #         r_pb against per-item accuracy ---
    print()
    print("r_pb(sum on each symbolic circuit, accuracy) — per verbal format:")
    per_item_rows = []
    rpb_summary = []
    for fmt in PEER_FORMATS:
        f_per_item = load_per_item(args.results_dir, args.model, fmt)
        f_kept_ids = load_kept_ids(args.results_dir, args.model, fmt)
        f_resp     = load_responses(args.results_dir, args.model, fmt)
        f_accs = np.array([
            int(bool(f_resp[i]["clean_match_gold"])) for i in f_kept_ids
        ])
        n_kept_f = f_per_item.shape[0]
        # flatten each item's vector for indexing
        flat = f_per_item.reshape(n_kept_f, -1)  # [n, L*dim]

        idx_full      = np.array(sorted(sym_full_top))
        idx_correct   = np.array(sorted(sym_correct_top)) if sym_correct_top else np.array([], dtype=int)
        idx_incorrect = np.array(sorted(sym_incorr_top))  if sym_incorr_top  else np.array([], dtype=int)

        sum_full   = flat[:, idx_full].sum(axis=1)
        sum_corr_s = flat[:, idx_correct].sum(axis=1) if idx_correct.size else np.zeros(n_kept_f)
        sum_inc_s  = flat[:, idx_incorrect].sum(axis=1) if idx_incorrect.size else np.zeros(n_kept_f)

        def rpb(sums):
            if len(set(f_accs.tolist())) < 2:
                return float("nan"), float("nan")
            r, p = pointbiserialr(f_accs.astype(float), sums)
            return float(r), float(p)

        r_full,  p_full  = rpb(sum_full)
        r_corr,  p_corr  = rpb(sum_corr_s)
        r_inc,   p_inc   = rpb(sum_inc_s)
        print(f"  {fmt:8s}  symbolic_balanced  r_pb={r_full:+.3f}  p={p_full:.2g}")
        print(f"  {fmt:8s}  symbolic_correct   r_pb={r_corr:+.3f}  p={p_corr:.2g}")
        print(f"  {fmt:8s}  symbolic_incorrect r_pb={r_inc:+.3f}  p={p_inc:.2g}")
        rpb_summary.append((fmt, r_full, p_full, r_corr, p_corr, r_inc, p_inc,
                             n_kept_f, int(f_accs.sum())))

        for k, iid in enumerate(f_kept_ids):
            per_item_rows.append({
                "format": fmt,
                "id": iid,
                "accuracy": int(f_accs[k]),
                "sum_balanced":  float(sum_full[k]),
                "sum_correct":   float(sum_corr_s[k]),
                "sum_incorrect": float(sum_inc_s[k]),
            })

    # --- 5. Write outputs ---
    overview = os.path.join(out_dir, "symbolic_split_overview.txt")
    with open(overview, "w") as f:
        f.write(f"Symbolic-correctness split analysis — model={args.model}\n")
        f.write(f"Top-K = {K} units (fraction {args.top_k_fraction:.0%}); "
                f"numeric n_kept={n_kept} (n_correct={n_c}, n_incorrect={n_i})\n\n")
        f.write("== Jaccard top-1% among numeric-derived circuits ==\n")
        for a, b, v in pair_results:
            if a <= b:
                f.write(f"  J({a:<19s}, {b:<19s}) = {v:.3f}\n")
        f.write(f"\n  units only in symbolic_correct   : {n_correct_only}\n")
        f.write(f"  units only in symbolic_incorrect : {n_incorr_only}\n")
        f.write(f"  units in both                    : {n_both}\n\n")

        f.write("== Verbal aggregate vs numeric-derived top-1% ==\n")
        for row in format_rows:
            f.write(f"  {row['format']:8s}  "
                    f"J(symbolic_balanced)={row['J_symbolic_balanced']:.3f}  "
                    f"J(symbolic_correct)={row['J_symbolic_correct']:.3f}  "
                    f"J(symbolic_incorrect)={row['J_symbolic_incorrect']:.3f}\n")

        f.write("\n== Per-item r_pb(sum_on_S, accuracy) under each symbolic anchor ==\n")
        for fmt, r1, p1, r2, p2, r3, p3, nk, nc in rpb_summary:
            f.write(f"  {fmt:8s}  n={nk}  acc={nc/nk:.0%}\n"
                    f"      symbolic_balanced   r_pb={r1:+.3f}  p={p1:.2g}\n"
                    f"      symbolic_correct    r_pb={r2:+.3f}  p={p2:.2g}\n"
                    f"      symbolic_incorrect  r_pb={r3:+.3f}  p={p3:.2g}\n")
    print(f"\nwrote {overview}")

    fmt_csv = os.path.join(out_dir, "symbolic_split_format_overlap.csv")
    with open(fmt_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "format", "J_symbolic_balanced",
            "J_symbolic_correct", "J_symbolic_incorrect",
        ])
        w.writeheader(); w.writerows(format_rows)
    print(f"wrote {fmt_csv}")

    per_item_csv = os.path.join(out_dir, "symbolic_split_per_item.csv")
    with open(per_item_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "format", "id", "accuracy",
            "sum_balanced", "sum_correct", "sum_incorrect",
        ])
        w.writeheader(); w.writerows(per_item_rows)
    print(f"wrote {per_item_csv}")


if __name__ == "__main__":
    main()
