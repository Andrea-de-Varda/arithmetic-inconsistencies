"""Item-level loading on the numeric circuit at each threshold -> r_pb.

CLUSTER (CPU): reads the per-item attribution tensors, which are git-ignored
and are regenerated on the cluster. No forward or backward passes.

For each model, threshold q and verbal format:
  S_q         = top q of units of the numeric aggregate by |attribution|
                (K = round(q * L * d_MLP), argpartition; run_overlap.py)
  loading_i,q = sum of item i's attribution over S_q, computed with the
                fp16 gather-and-sum of run_overlap.py, so that at q = 1% it
                reproduces the published sum_on_S exactly
  r_pb(q)     = point-biserial r between loading and clean_match_gold

The same loading is also summed in float64 as a check: fp16 accumulation is
coarse and its error grows with K, so the r_pb of both paths is logged and
written for every cell.

Outputs (small, git-tracked), under threshold_robustness/results/:
  item_loadings.csv     model,format,id,accuracy,loading_q0.5,loading_q1,
                        loading_q2,loading_q5          (fp16 path)
  cell_rpb.csv          model,format,q,K,n,n_correct,rpb,p,rpb_float64
  item_level_validation.txt

Usage:
  python threshold_robustness/scripts/item_level_loadings.py
  python threshold_robustness/scripts/item_level_loadings.py --models meta-llama/Llama-3.1-8B
"""

import argparse
import os
import sys

import numpy as np
from scipy.stats import pointbiserialr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (DEFAULT_OUT_DIR, DEFAULT_RESULTS_DIR, MODELS,
                    PUBLISHED_Q, THRESHOLDS, VERBAL, load_aggregate,
                    load_kept_ids, load_per_item, load_published_items,
                    load_responses, n_units_for, q_label, topk_indices_abs,
                    write_csv)


def loading_col(q):
    return f"loading_q{q * 100:g}"


ITEM_FIELDS = (["model", "format", "id", "accuracy"]
               + [loading_col(q) for q in THRESHOLDS])
CELL_FIELDS = ["model", "format", "q", "K", "n", "n_correct", "rpb", "p",
               "rpb_float64"]


def circuit_orders(agg, thresholds):
    """q -> (K, index array in the element order run_overlap.py sums in).

    run_overlap.py keeps the circuit as a Python set and gathers with
    list(set), so the fp16 sum runs in set-iteration order. The order is
    built once per threshold here, the same way."""
    n_units = agg.size
    out = {}
    for q in thresholds:
        K = n_units_for(q, n_units)
        idx = topk_indices_abs(agg, K)
        out[q] = (K, np.array(list(set(idx.tolist())), dtype=np.int64))
    return out


def loadings(flat_fp16, order):
    """[n] fp16 and float64 sums of each item's attribution over `order`.

    Each fp16 sum is taken over a contiguous 1-D row, as vec[list(S)].sum()
    is in run_overlap.py, so it follows the same pairwise summation."""
    n = flat_fp16.shape[0]
    fp16 = np.empty(n, dtype=np.float64)
    f64 = np.empty(n, dtype=np.float64)
    for i in range(n):
        g = flat_fp16[i][order]
        fp16[i] = float(g.sum())
        f64[i] = float(g.astype(np.float64).sum())
    return fp16, f64


def rpb(acc, x):
    if len(set(acc.tolist())) < 2:
        return float("nan"), float("nan")
    r, p = pointbiserialr(acc.astype(float), x)
    return float(r), float(p)


def run_model(mid, results_dir, thresholds, log):
    agg = load_aggregate(results_dir, mid, "numeric")
    if agg is None:
        log(f"  skip {mid}: numeric aggregate missing")
        return None
    orders = circuit_orders(agg, thresholds)
    log(f"  L={agg.shape[0]} d={agg.shape[1]} units={agg.size}; K = "
        + ", ".join(f"{orders[q][0]} ({q_label(q)})" for q in thresholds))
    # the top sets are nested by construction; checked, not assumed
    qs = sorted(thresholds)
    nested = all(set(orders[a][1].tolist()) <= set(orders[b][1].tolist())
                 for a, b in zip(qs, qs[1:]))
    log(f"  [check] circuits nested across thresholds = {nested}")

    published = load_published_items(results_dir, mid)
    item_rows, cell_rows = [], []
    for fmt in VERBAL:
        per_item = load_per_item(results_dir, mid, fmt)
        kept = load_kept_ids(results_dir, mid, fmt)
        resp = load_responses(results_dir, mid, fmt)
        if per_item is None or kept is None or resp is None:
            log(f"  skip {mid}/{fmt}: per-item tensor, kept_ids or responses "
                f"missing")
            continue
        resp = {r["id"]: r for r in resp["records"]}
        flat = per_item.reshape(per_item.shape[0], -1)
        # items without a response record are skipped, as in run_overlap.py
        rows = [i for i, iid in enumerate(kept) if iid in resp]
        if len(rows) < len(kept):
            flat = flat[rows]
        ids = [kept[i] for i in rows]
        acc = np.array([int(bool(resp[i]["clean_match_gold"])) for i in ids])
        log(f"  {fmt}: n={len(ids)} correct={int(acc.sum())}")

        per_q = {}
        for q in thresholds:
            K, order = orders[q]
            l16, l64 = loadings(flat, order)
            per_q[q] = l16
            r16, p16 = rpb(acc, l16)
            r64, _ = rpb(acc, l64)
            log(f"    {q_label(q):>4} K={K}: r_pb = {r16:+.4f} (p={p16:.2e}); "
                f"float64 path {r64:+.4f}; max|fp16 - float64| loading = "
                f"{float(np.max(np.abs(l16 - l64))):.2e} (mean |loading| "
                f"{float(np.mean(np.abs(l64))):.2e})")
            if q == PUBLISHED_Q:
                pub = published.get(fmt, {})
                diffs = [abs(l16[k] - pub[iid]) for k, iid in enumerate(ids)
                         if iid in pub]
                log(f"    [check] {q_label(q)} loading reproduces published "
                    f"sum_on_S on {len(diffs)}/{len(ids)} items, max|diff| = "
                    f"{(max(diffs) if diffs else float('nan')):.3e}")
            cell_rows.append({
                "model": mid, "format": fmt, "q": q, "K": K, "n": len(ids),
                "n_correct": int(acc.sum()), "rpb": r16, "p": p16,
                "rpb_float64": r64,
            })
        item_rows.extend(
            dict({"model": mid, "format": fmt, "id": iid,
                  "accuracy": int(acc[k])},
                 **{loading_col(q): float(per_q[q][k]) for q in thresholds})
            for k, iid in enumerate(ids))
        del per_item, flat
    return item_rows, cell_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--models", nargs="+", default=None)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    log_lines = []
    val_path = os.path.join(args.out_dir, "item_level_validation.txt")

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    def flush_log():
        with open(val_path, "w") as f:
            f.write("\n".join(log_lines) + "\n")

    log("threshold robustness, item level: thresholds "
        + ", ".join(q_label(q) for q in THRESHOLDS))
    paths = {k: os.path.join(args.out_dir, f"{k}.csv")
             for k in ("item_loadings", "cell_rpb")}
    # rewrite from scratch; each model is appended as soon as it finishes so
    # a crash on a late (large) model keeps the earlier ones
    write_csv(paths["item_loadings"], ITEM_FIELDS, [])
    write_csv(paths["cell_rpb"], CELL_FIELDS, [])

    n_done = 0
    for mid in args.models or MODELS:
        log(f"\n{mid}")
        out = run_model(mid, args.results_dir, THRESHOLDS, log)
        if out is None:
            continue
        items, cells = out
        write_csv(paths["item_loadings"], ITEM_FIELDS, items, "a")
        write_csv(paths["cell_rpb"], CELL_FIELDS, cells, "a")
        n_done += bool(cells)
        flush_log()

    log(f"\n{n_done} models written to {args.out_dir}")
    flush_log()


if __name__ == "__main__":
    main()
