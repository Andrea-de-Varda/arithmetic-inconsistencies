"""Compare the ORDER-SWAP circuit with the main OPERATOR circuit, and test
whether it carries the paper's predictive signal.

Runs on the cluster: it reads the per-item attribution tensors, which are
git-ignored. Pure CPU, a couple of minutes per model.

Per model, on the numeric format:

1. OVERLAP. Top-1% Jaccard between the order circuit and the operator
   circuit, against two reference points:
     - chance, the exact hypergeometric expectation for two independent
       size-matched masks (mean and 95% interval; no simulation);
     - a split-half ceiling, the Jaccard between circuits built from two
       disjoint halves of the items within each counterfactual. This says how
       reproducible a circuit of this kind is at all given item-sampling
       noise, so it is the upper reference for a between-counterfactual
       overlap. It is computed on half the items, so it slightly understates
       the reliability of a full-data circuit, which flatters the reported
       fraction-of-ceiling; the summary says so.
   The order circuit is estimated from far fewer items than the operator one
   (925 eligible, minus filter losses, against 2,000), so its own split-half
   ceiling is the relevant yardstick rather than the operator circuit's.

2. TRANSFER OF THE PREDICTIVE SIGNAL. The paper's item-level result is that
   loading on the numeric circuit predicts verbal accuracy. We recompute it
   with the ORDER circuit: the verbal per-item attributions are the main
   run's, unchanged, only the unit set changes. If the order circuit is the
   same machinery, it should predict verbal correctness too.

Outputs (small, git-tracked):
  circuit_overlap.csv      per model
  item_level_transfer.csv  per (model, verbal format)
  overlap_validation.txt   log

Usage:
  python order_cf/scripts/circuit_overlap.py --models meta-llama/Llama-3.1-8B
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
from scipy.stats import hypergeom, pointbiserialr
from sklearn.metrics import roc_auc_score

VERBAL = ("english", "spanish", "italian")

MODELS = [
    "Qwen/Qwen3-0.6B", "meta-llama/Llama-3.2-1B", "google/gemma-2-2b",
    "meta-llama/Llama-3.2-3B", "Qwen/Qwen3-4B", "mistralai/Mistral-7B-v0.1",
    "allenai/Olmo-3-1025-7B", "meta-llama/Llama-3.1-8B", "Qwen/Qwen3-8B",
    "google/gemma-2-9b", "microsoft/phi-4", "Qwen/Qwen3-32B",
    "allenai/Olmo-3-1125-32B",
]


def model_short(name):
    return name.replace("/", "_").replace(".", "-")


def topk_abs(vec, k):
    """run_overlap.py's rule: top-k flat indices by |value|."""
    flat = np.abs(vec).ravel()
    if k >= flat.size:
        return np.arange(flat.size)
    return np.argpartition(flat, -k)[-k:]


def jaccard(a, b):
    sa, sb = set(a.tolist()), set(b.tolist())
    return len(sa & sb) / len(sa | sb) if (sa or sb) else float("nan")


def chance_jaccard(N, K):
    """Exact null for two independent size-K masks: |A n B| ~ Hypergeom."""
    rv = hypergeom(N, K, K)
    j = lambda i: i / (2 * K - i)
    return j(rv.mean()), j(rv.ppf(0.025)), j(rv.ppf(0.975))


def attr_file(results_dir, mid, fmt, name):
    return os.path.join(results_dir, "attribution", model_short(mid), fmt, name)


def load_flat_per_item(results_dir, mid, fmt):
    p = attr_file(results_dir, mid, fmt, "neuron_attribution_per_item.pt")
    if not os.path.exists(p):
        return None
    t = torch.load(p).numpy()
    return t.reshape(t.shape[0], -1)


def split_half_ceiling(flat, K, seed=0):
    """Jaccard between top-K circuits built from two disjoint halves."""
    n = flat.shape[0]
    idx = np.random.default_rng(seed).permutation(n)
    a, b = idx[: n // 2], idx[n // 2:]
    ma = flat[a].astype(np.float32).mean(axis=0)
    mb = flat[b].astype(np.float32).mean(axis=0)
    return jaccard(topk_abs(ma, K), topk_abs(mb, K))


def run_model(mid, main_dir, ord_dir, kfrac, log):
    a_op = attr_file(main_dir, mid, "numeric", "neuron_attribution_aggregate.pt")
    a_or = attr_file(ord_dir, mid, "numeric", "neuron_attribution_aggregate.pt")
    if not (os.path.exists(a_op) and os.path.exists(a_or)):
        log(f"  skip {mid}: missing aggregate "
            f"(operator={os.path.exists(a_op)}, order={os.path.exists(a_or)})")
        return None, []

    agg_op = torch.load(a_op).float().numpy().ravel()
    agg_or = torch.load(a_or).float().numpy().ravel()
    N = agg_op.size
    assert agg_or.size == N, "aggregate shapes differ between runs"
    K = max(1, int(round(kfrac * N)))
    S_op, S_or = topk_abs(agg_op, K), topk_abs(agg_or, K)
    j = jaccard(S_op, S_or)
    ch, lo, hi = chance_jaccard(N, K)

    ceil_op = ceil_or = float("nan")
    f_op = load_flat_per_item(main_dir, mid, "numeric")
    if f_op is not None:
        ceil_op = split_half_ceiling(f_op, K)
        del f_op
    f_or = load_flat_per_item(ord_dir, mid, "numeric")
    n_kept_or = 0 if f_or is None else f_or.shape[0]
    if f_or is not None:
        ceil_or = split_half_ceiling(f_or, K)
        del f_or

    row = {
        "model": mid, "n_units": N, "K": K,
        "jaccard_order_vs_operator": j,
        "chance": ch, "chance_lo": lo, "chance_hi": hi,
        "ratio_to_chance": j / ch if ch else float("nan"),
        "split_half_operator": ceil_op, "split_half_order": ceil_or,
        # the order circuit is the noisier of the two, so its own reliability
        # is the binding constraint on how high the overlap could go
        "ceiling": ceil_or,
        "fraction_of_ceiling": j / ceil_or if ceil_or == ceil_or else float("nan"),
        "n_kept_order": n_kept_or,
    }
    log(f"  Jaccard(order, operator) = {j:.3f}  "
        f"(chance {ch:.4f} [{lo:.4f}, {hi:.4f}], {row['ratio_to_chance']:.0f}x; "
        f"split-half ceiling {ceil_or:.3f} order / {ceil_op:.3f} operator; "
        f"{100 * row['fraction_of_ceiling']:.0f}% of the order ceiling; "
        f"n_kept={n_kept_or})")

    item_rows = []
    S_op_l, S_or_l = sorted(S_op.tolist()), sorted(S_or.tolist())
    for fmt in VERBAL:
        flat = load_flat_per_item(main_dir, mid, fmt)
        kp = attr_file(main_dir, mid, fmt, "kept_ids.json")
        rp = os.path.join(main_dir, "responses", model_short(mid), f"{fmt}.json")
        if flat is None or not (os.path.exists(kp) and os.path.exists(rp)):
            continue
        kept = json.load(open(kp))
        resp = {r["id"]: r for r in json.load(open(rp))["records"]}
        y, l_or, l_op = [], [], []
        for i, iid in enumerate(kept):
            r = resp.get(iid)
            if r is None:
                continue
            v = flat[i]
            y.append(int(bool(r["clean_match_gold"])))
            l_or.append(float(v[S_or_l].sum()))
            l_op.append(float(v[S_op_l].sum()))
        del flat
        y = np.array(y)
        if len(set(y.tolist())) < 2:
            continue
        r_or = pointbiserialr(y, l_or)
        r_op = pointbiserialr(y, l_op)
        item_rows.append({
            "model": mid, "format": fmt, "n": len(y), "n_correct": int(y.sum()),
            "low_n": int(y.sum() < 30),
            "rpb_order": float(r_or[0]), "p_order": float(r_or[1]),
            "auroc_order": float(roc_auc_score(y, l_or)),
            "rpb_operator": float(r_op[0]),
            "auroc_operator": float(roc_auc_score(y, l_op)),
        })
        log(f"  {fmt}: n={len(y)} correct={int(y.sum())}  "
            f"r_pb order={r_or[0]:+.3f} vs operator={r_op[0]:+.3f}")
    return row, item_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-results-dir", default=None,
                    help="default attribution/results (the operator run)")
    ap.add_argument("--order-results-dir", default=None,
                    help="default order_cf/results")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--top-k-fraction", type=float, default=0.01)
    ap.add_argument("--models", nargs="+", default=None)
    args = ap.parse_args()

    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    attr_root = os.path.dirname(pkg_root)
    main_dir = args.main_results_dir or os.path.join(attr_root, "results")
    ord_dir = args.order_results_dir or os.path.join(pkg_root, "results")
    out_dir = args.out_dir or ord_dir
    os.makedirs(out_dir, exist_ok=True)

    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    rows, item_rows = [], []
    for mid in (args.models or MODELS):
        log(f"\n{mid}")
        r, ir = run_model(mid, main_dir, ord_dir, args.top_k_fraction, log)
        if r:
            rows.append(r)
            item_rows.extend(ir)

    if not rows:
        log("\nnothing computed (are the per-item tensors present?)")
        return

    def write(path, fields, data):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(data)
        log(f"wrote {path} ({len(data)} rows)")

    write(os.path.join(out_dir, "circuit_overlap.csv"), list(rows[0]), rows)
    if item_rows:
        write(os.path.join(out_dir, "item_level_transfer.csv"),
              list(item_rows[0]), item_rows)
    with open(os.path.join(out_dir, "overlap_validation.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
