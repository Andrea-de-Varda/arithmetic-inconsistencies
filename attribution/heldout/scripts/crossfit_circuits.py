"""Cross-fitted (held-out) numeric circuits -> per-item verbal loadings.

In the main analysis the numeric circuit is identified on the *same* 2,000
problems whose verbal renditions it then scores. Here the circuit that scores
an item is identified without that item (paper Appendix G).

Design (k-fold cross-fitting; no new forward/backward passes)
------------------------------------------------------------
The 2,000 ids of the main attribution fold are partitioned into k folds
(seeded). For fold j:

  S_j  = top-1% |units| of the numeric aggregate rebuilt from the numeric
         per-item attributions of the items NOT in fold j
  test = the verbal items that ARE in fold j, scored on S_j

so the numeric rendition of a scored item never contributes to the circuit
used to score it. Concatenating the k held-out blocks gives one fully
out-of-sample loading per verbal item, directly comparable to the published
`sum_on_S` (which is also written out, as `loading_full`, for validation).

Circuit and loading follow `scripts/run_overlap.py` exactly: K = round(1% of
L*d_MLP), top-K by |attribution| over the flattened aggregate via argpartition,
and the loading is the fp16 gather-and-sum `float(vec[list(S)].sum())`. The
fp16 path is deliberate: it reproduces the published loadings to within ~2e-4 (exactly for some
models), which the script checks against results/overlap/<model>/item_level_predictions_neuron.csv.

Reads the per-item attribution tensors (git-ignored, cluster-only), so this
runs on the cluster. Pure CPU, a few minutes per model.

Outputs (small, git-tracked), under heldout/results/:
  crossfit_item_loadings.csv     model,format,id,fold,accuracy,
                                 loading_full,loading_crossfit
  crossfit_circuit_stability.csv model,fold,K,n_estimation,jaccard_with_full
  crossfit_validation.txt        per-model self-checks

Usage:
  python heldout/scripts/crossfit_circuits.py
  python heldout/scripts/crossfit_circuits.py --models meta-llama/Llama-3.1-8B
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

VERBAL = ("english", "spanish", "italian")

# Column schema, shared by the combined CSVs and the per-model copies, so that
# the predictor comparison and the summarizer read exactly the same fields.
LOADING_FIELDS = ["model", "format", "id", "fold", "accuracy",
                  "loading_full", "loading_crossfit"]
STABILITY_FIELDS = ["model", "fold", "K", "n_estimation", "n_heldout_numeric",
                    "jaccard_with_full"]

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


# --- helpers copied verbatim from scripts/run_overlap.py -------------------

def topk_indices_abs(vec, k):
    """Indices of top-k entries by absolute value. (run_overlap.py:70)"""
    flat = np.abs(vec).ravel()
    if k >= flat.size:
        return np.arange(flat.size)
    idx = np.argpartition(flat, -k)[-k:]
    return idx


def jaccard_sets(a, b):
    return len(a & b) / len(a | b) if (a or b) else float("nan")


def reference_loading(vec_fp16, circuit_set):
    """The published loading: fp16 gather + fp16 sum. (run_overlap.py:245)"""
    return float(vec_fp16[list(circuit_set)].sum())


# --- io --------------------------------------------------------------------

def attr_dir(results_dir, mid, fmt):
    return os.path.join(results_dir, "attribution", model_short(mid), fmt)


def load_per_item(results_dir, mid, fmt):
    """Per-item tensor at its saved fp16 precision (memory), as [n, L*d]."""
    p = os.path.join(attr_dir(results_dir, mid, fmt),
                     "neuron_attribution_per_item.pt")
    if not os.path.exists(p):
        return None
    t = torch.load(p).numpy()
    return t.reshape(t.shape[0], -1)


def load_aggregate(results_dir, mid, fmt):
    p = os.path.join(attr_dir(results_dir, mid, fmt),
                     "neuron_attribution_aggregate.pt")
    if not os.path.exists(p):
        return None
    return torch.load(p).float().numpy().ravel()


def load_kept_ids(results_dir, mid, fmt):
    p = os.path.join(attr_dir(results_dir, mid, fmt), "kept_ids.json")
    return json.load(open(p)) if os.path.exists(p) else None


def load_responses(results_dir, mid, fmt):
    p = os.path.join(results_dir, "responses", model_short(mid), f"{fmt}.json")
    if not os.path.exists(p):
        return {}
    return {r["id"]: r for r in json.load(open(p))["records"]}


def load_published_loadings(results_dir, mid):
    """format -> {id: sum_on_S} from the published overlap CSV, for validation."""
    p = os.path.join(results_dir, "overlap", model_short(mid),
                     "item_level_predictions_neuron.csv")
    out = {}
    if not os.path.exists(p):
        return out
    for r in csv.DictReader(open(p)):
        out.setdefault(r["format"], {})[r["id"]] = float(r["sum_on_S"])
    return out


# --- core ------------------------------------------------------------------

def assign_folds(fold_ids, k, seed):
    """id -> fold index, over the 2,000 main-fold ids (deterministic)."""
    ids = sorted(fold_ids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    return {ids[int(p)]: i % k for i, p in enumerate(perm)}


def fold_sums(flat_fp16, row_fold, k, chunk):
    """One pass over the per-item tensor -> [k, n_units] float32 fold sums.

    Accumulating in float32 matches run_attribution.py's
    `per_item_t.float().mean(dim=0)`. The [k, n_units] buffer is small
    (10 x 1.77M x 4B = 71 MB even for the 32B models).
    """
    n, n_units = flat_fp16.shape
    sums = np.zeros((k, n_units), dtype=np.float32)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = flat_fp16[start:stop].astype(np.float32)
        folds_here = row_fold[start:stop]
        for f in np.unique(folds_here):
            if f < 0:
                continue
            sums[f] += block[folds_here == f].sum(axis=0)
        del block
    return sums


def run_model(mid, results_dir, fold_of, k, chunk, log):
    """Returns (loading_rows, stability_rows)."""
    num_kept = load_kept_ids(results_dir, mid, "numeric")
    agg_disk = load_aggregate(results_dir, mid, "numeric")
    if num_kept is None or agg_disk is None:
        log(f"  skip {mid}: numeric aggregate/kept_ids missing")
        return [], []

    n_units = agg_disk.size
    K = max(1, int(round(0.01 * n_units)))
    S_full = set(topk_indices_abs(agg_disk, K).tolist())

    flat = load_per_item(results_dir, mid, "numeric")
    if flat is None:
        log(f"  skip {mid}: numeric per-item tensor missing")
        return [], []
    if flat.shape[0] != len(num_kept) or flat.shape[1] != n_units:
        log(f"  skip {mid}: shape mismatch {flat.shape} vs "
            f"({len(num_kept)}, {n_units})")
        return [], []

    # fold index of every numeric row (-1 = id outside the main fold; should
    # not happen, all kept ids are drawn from it, but stay defensive)
    row_fold = np.array([fold_of.get(i, -1) for i in num_kept], dtype=np.int64)
    n_outside = int((row_fold < 0).sum())

    fs = fold_sums(flat, row_fold, k, chunk)
    total = fs.sum(axis=0)
    n_total = int((row_fold >= 0).sum()) + n_outside

    # self-check 1: aggregate rebuilt from per-item == aggregate on disk
    agg_rebuilt = total / max(n_total, 1)
    max_abs = float(np.max(np.abs(agg_rebuilt - agg_disk)))
    S_rebuilt = set(topk_indices_abs(agg_rebuilt, K).tolist())
    jac_rebuilt = jaccard_sets(S_rebuilt, S_full)
    log(f"  [check] aggregate rebuild: max|diff| = {max_abs:.3e}, "
        f"top-{K} Jaccard vs disk = {jac_rebuilt:.4f}")

    # per-fold held-out circuits
    counts = np.array([int((row_fold == f).sum()) for f in range(k)])
    S_by_fold, stability_rows = {}, []
    for f in range(k):
        n_est = n_total - int(counts[f])
        mean_minus_f = (total - fs[f]) / max(n_est, 1)
        S_by_fold[f] = set(topk_indices_abs(mean_minus_f, K).tolist())
        stability_rows.append({
            "model": mid, "fold": f, "K": K, "n_estimation": n_est,
            "n_heldout_numeric": int(counts[f]),
            "jaccard_with_full": jaccard_sets(S_by_fold[f], S_full),
        })
    jl = [r["jaccard_with_full"] for r in stability_rows]
    log(f"  [check] held-out circuit vs published circuit: "
        f"Jaccard {min(jl):.3f}-{max(jl):.3f} (mean {np.mean(jl):.3f})")

    del flat, fs
    published = load_published_loadings(results_dir, mid)

    loading_rows = []
    for fmt in VERBAL:
        vflat = load_per_item(results_dir, mid, fmt)
        vkept = load_kept_ids(results_dir, mid, fmt)
        resp = load_responses(results_dir, mid, fmt)
        if vflat is None or vkept is None or not resp:
            log(f"  skip {mid}/{fmt}: per-item tensor or responses missing")
            continue

        pub = published.get(fmt, {})
        worst = 0.0
        n_checked = 0
        for i, iid in enumerate(vkept):
            rec = resp.get(iid)
            f = fold_of.get(iid, -1)
            if rec is None or f < 0:
                continue
            vec = vflat[i]
            l_full = reference_loading(vec, S_full)
            l_cf = reference_loading(vec, S_by_fold[f])
            if iid in pub:
                worst = max(worst, abs(l_full - pub[iid]))
                n_checked += 1
            loading_rows.append({
                "model": mid, "format": fmt, "id": iid, "fold": f,
                "accuracy": int(bool(rec["clean_match_gold"])),
                "loading_full": l_full, "loading_crossfit": l_cf,
            })
        # self-check 2: loading_full reproduces the published sum_on_S
        log(f"  {fmt}: {len(vkept)} kept; reproduced published sum_on_S on "
            f"{n_checked} items, max|diff| = {worst:.3e}")
        del vflat

    return loading_rows, stability_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=None,
                    help="main results dir (default: attribution/results)")
    ap.add_argument("--out-dir", default=None,
                    help="default: attribution/heldout/results")
    ap.add_argument("--k", type=int, default=10, help="number of folds")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk", type=int, default=256,
                    help="rows per block when accumulating fold sums")
    ap.add_argument("--models", nargs="+", default=None)
    args = ap.parse_args()

    attribution_root = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    results_dir = args.results_dir or os.path.join(attribution_root, "results")
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(out_dir, exist_ok=True)

    fold_ids = json.load(open(os.path.join(attribution_root, "data",
                                           "attribution_fold_ids.json")))
    fold_of = assign_folds(fold_ids, args.k, args.seed)
    sizes = np.bincount(np.array(list(fold_of.values())), minlength=args.k)

    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    log(f"{len(fold_ids)} main-fold ids -> {args.k} folds "
        f"(seed {args.seed}), sizes {sizes.tolist()}")

    models = args.models or MODELS
    all_loadings, all_stability = [], []
    for mid in models:
        log(f"\n{mid}")
        lr, sr = run_model(mid, results_dir, fold_of, args.k, args.chunk, log)
        all_loadings.extend(lr)
        all_stability.extend(sr)

    if not all_loadings:
        log("\nno per-item tensors found; nothing written "
            "(this script must run where the *_per_item.pt files live)")
        return

    p1 = os.path.join(out_dir, "crossfit_item_loadings.csv")
    with open(p1, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOADING_FIELDS)
        w.writeheader()
        w.writerows(all_loadings)
    log(f"\nwrote {p1}  ({len(all_loadings)} rows)")

    p2 = os.path.join(out_dir, "crossfit_circuit_stability.csv")
    with open(p2, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=STABILITY_FIELDS)
        w.writeheader()
        w.writerows(all_stability)
    log(f"wrote {p2}  ({len(all_stability)} rows)")

    # Per-model copies, which is the layout the predictor comparison reads
    # (ensemble/scripts/probe_and_controls.py --crossfit-dir) and the
    # summarizer globs. Same rows, split by model: one implementation of the
    # cross-fitting, so the two analyses can never drift apart.
    for mid in sorted({r["model"] for r in all_loadings}):
        mdir = os.path.join(out_dir, model_short(mid))
        os.makedirs(mdir, exist_ok=True)
        with open(os.path.join(mdir, "loadings.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=LOADING_FIELDS)
            w.writeheader()
            w.writerows([r for r in all_loadings if r["model"] == mid])
        with open(os.path.join(mdir, "fold_stability.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=STABILITY_FIELDS)
            w.writeheader()
            w.writerows([r for r in all_stability if r["model"] == mid])
    log(f"wrote per-model loadings.csv + fold_stability.csv for "
        f"{len({r['model'] for r in all_loadings})} models")

    p3 = os.path.join(out_dir, "crossfit_validation.txt")
    with open(p3, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"wrote {p3}")


if __name__ == "__main__":
    main()
