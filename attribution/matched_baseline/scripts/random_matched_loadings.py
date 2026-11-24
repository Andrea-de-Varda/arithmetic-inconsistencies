"""Matched random baseline for the item-level loading -> per-draw r_pb.

Loading is an unnormalized sum over the circuit units, so an item with larger
attributions throughout the network could score higher regardless of where
they are concentrated. This compares the circuit against size- and
layer-matched random unit sets (paper Appendix E).

Design (no new forward/backward passes)
---------------------------------------
For each model:
  S       = top-1% |units| of the numeric aggregate, exactly as in
            scripts/run_overlap.py (K = round(1% of L*d_MLP), argpartition)
  c_L     = number of units of S in layer L
  R_r     = r-th random set (r = 1..R): in every layer L, c_L units drawn
            uniformly without replacement from the units of L NOT in S.
            Same size as S, same layer profile, disjoint from S.

The R random sets are drawn once per model (seeded) and reused for all three
verbal formats. For each verbal format and each item i:
  loading_real_i = sum of i's attribution over S   (the published sum_on_S)
  loading_r_i    = sum of i's attribution over R_r
and per (model, format) we compute r_pb(loading, accuracy) for S and for every
random draw. If loading only reflected the overall size of an item's
attributions, the random sets would predict accuracy as well as S does.

The real loading is computed with the fp16 gather-and-sum of run_overlap.py,
so it reproduces the published sum_on_S (checked against
results/overlap/<model>/item_level_predictions_neuron.csv). The random
loadings go through one sparse [R+1, n_units] indicator product in float32;
row 0 of that matrix is S itself, so the float32 path is also checked against
the fp16 reference.

Reads the per-item attribution tensors (git-ignored, cluster-only). Pure CPU.

Outputs (small, git-tracked), under matched_baseline/results/:
  random_draws_rpb.csv   model,format,draw,rpb           (one row per draw)
  cell_real.csv          model,format,n,n_correct,K,n_draws,rpb_real,p_real,
                         rpb_real_float32
  item_loadings.csv      model,format,id,accuracy,loading_real,
                         loading_random_mean,loading_random_sd
  layer_profile.csv      model,layer,d,n_in_S
  validation.txt         per-model self-checks

Usage:
  python matched_baseline/scripts/random_matched_loadings.py
  python matched_baseline/scripts/random_matched_loadings.py --models meta-llama/Llama-3.1-8B --n-draws 200
"""

import argparse
import csv
import json
import os

import numpy as np
import scipy.sparse as sp
import torch
from scipy.stats import pointbiserialr

VERBAL = ("english", "spanish", "italian")

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

DRAW_FIELDS = ["model", "format", "draw", "rpb"]
CELL_FIELDS = ["model", "format", "n", "n_correct", "K", "n_draws",
               "rpb_real", "p_real", "rpb_real_float32"]
ITEM_FIELDS = ["model", "format", "id", "accuracy", "loading_real",
               "loading_random_mean", "loading_random_sd"]
LAYER_FIELDS = ["model", "layer", "d", "n_in_S"]


def model_short(name):
    return name.replace("/", "_").replace(".", "-")


# --- helpers matching scripts/run_overlap.py -------------------------------

def topk_indices_abs(vec, k):
    """Indices of top-k entries by absolute value. (run_overlap.py:70)"""
    flat = np.abs(vec).ravel()
    if k >= flat.size:
        return np.arange(flat.size)
    return np.argpartition(flat, -k)[-k:]


def reference_loading(vec_fp16, circuit_idx):
    """The published loading: fp16 gather + fp16 sum. (run_overlap.py:245)"""
    return float(vec_fp16[circuit_idx].sum())


# --- core ------------------------------------------------------------------

def draw_matched_sets(S_idx, n_layers, d, n_draws, rng):
    """R per-layer count-matched random sets, disjoint from S.

    Returns (sets [n_draws, K] int64 flat indices, counts [n_layers],
    n_short) where n_short counts layers whose complement was too small to
    match (never expected at a 1% cut; reported in the validation log).

    Per layer: draw c_L + s_L distinct units uniformly (shuffled), drop the
    ones in S, keep the first c_L. A shuffled uniform sample without
    replacement restricted to the complement is a uniform sample from the
    complement, so this is exact and avoids materializing the complement.
    """
    layer_of = S_idx // d
    counts = np.bincount(layer_of, minlength=n_layers)
    S_by_layer = [np.sort(S_idx[layer_of == L] % d) for L in range(n_layers)]
    K = int(counts.sum())
    sets = np.empty((n_draws, K), dtype=np.int64)
    n_short = 0
    for r in range(n_draws):
        pos = 0
        for L in range(n_layers):
            c = int(counts[L])
            if c == 0:
                continue
            s_in = S_by_layer[L]
            if 2 * c > d:
                # complement smaller than c: take all of it (unmatched)
                comp = np.setdiff1d(np.arange(d), s_in, assume_unique=True)
                pick = rng.permutation(comp)[:c]
                if r == 0:
                    n_short += 1
                if pick.size < c:   # pad by repeating: keeps the matrix shape
                    pick = np.resize(pick, c)
            else:
                cand = rng.choice(d, size=c + s_in.size, replace=False,
                                  shuffle=True)
                keep = cand[~np.isin(cand, s_in, assume_unique=True)]
                pick = keep[:c]
            sets[r, pos:pos + c] = L * d + pick
            pos += c
    return sets, counts, n_short


def indicator_matrix(rows_of_idx, n_units):
    """[n_sets, n_units] csr 0/1 matrix, one row per index set."""
    n_sets, K = rows_of_idx.shape
    indptr = np.arange(0, n_sets * K + 1, K, dtype=np.int64)
    data = np.ones(n_sets * K, dtype=np.float32)
    return sp.csr_matrix((data, rows_of_idx.ravel(), indptr),
                         shape=(n_sets, n_units))


def set_loadings(flat_fp16, M, chunk):
    """[n_items, n_sets] float32 sums of each item's attribution per set."""
    n = flat_fp16.shape[0]
    out = np.empty((n, M.shape[0]), dtype=np.float32)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = np.ascontiguousarray(flat_fp16[start:stop].astype(np.float32).T)
        out[start:stop] = (M @ block).T
        del block
    return out


def rpb_columns(acc, loadings):
    """Point-biserial r (= Pearson) between binary acc [n] and each column."""
    a = acc.astype(np.float64)
    a = a - a.mean()
    X = loadings.astype(np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    num = a @ X
    den = np.sqrt((a @ a) * (X * X).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


# --- io --------------------------------------------------------------------

def attr_dir(results_dir, mid, fmt):
    return os.path.join(results_dir, "attribution", model_short(mid), fmt)


def load_per_item(results_dir, mid, fmt):
    p = os.path.join(attr_dir(results_dir, mid, fmt),
                     "neuron_attribution_per_item.pt")
    if not os.path.exists(p):
        return None
    return torch.load(p).numpy()          # [n, L, d] at saved fp16


def load_aggregate(results_dir, mid, fmt):
    p = os.path.join(attr_dir(results_dir, mid, fmt),
                     "neuron_attribution_aggregate.pt")
    if not os.path.exists(p):
        return None
    return torch.load(p).float().numpy()  # [L, d]


def load_kept_ids(results_dir, mid, fmt):
    p = os.path.join(attr_dir(results_dir, mid, fmt), "kept_ids.json")
    return json.load(open(p)) if os.path.exists(p) else None


def load_responses(results_dir, mid, fmt):
    p = os.path.join(results_dir, "responses", model_short(mid), f"{fmt}.json")
    if not os.path.exists(p):
        return {}
    return {r["id"]: r for r in json.load(open(p))["records"]}


def load_published_loadings(results_dir, mid):
    """format -> {id: sum_on_S} from the published overlap CSV."""
    p = os.path.join(results_dir, "overlap", model_short(mid),
                     "item_level_predictions_neuron.csv")
    out = {}
    if not os.path.exists(p):
        return out
    for r in csv.DictReader(open(p)):
        out.setdefault(r["format"], {})[r["id"]] = float(r["sum_on_S"])
    return out


# --- driver ----------------------------------------------------------------

def run_model(mid, model_idx, results_dir, n_draws, seed, chunk, log):
    agg = load_aggregate(results_dir, mid, "numeric")
    if agg is None:
        log(f"  skip {mid}: numeric aggregate missing")
        return None
    n_layers, d = agg.shape
    n_units = n_layers * d
    K = max(1, int(round(0.01 * n_units)))
    S_idx = np.sort(topk_indices_abs(agg.ravel(), K)).astype(np.int64)

    rng = np.random.default_rng([seed, model_idx])
    sets, counts, n_short = draw_matched_sets(S_idx, n_layers, d, n_draws, rng)
    log(f"  L={n_layers} d={d} K={K}; S spans {int((counts > 0).sum())} "
        f"layers (max {int(counts.max())} units in one layer)")

    # self-checks on the draws (first draw exhaustively, all draws on counts)
    S_set = set(S_idx.tolist())
    overlap0 = len(S_set & set(sets[0].tolist()))
    dup0 = K - len(set(sets[0].tolist()))
    cnt_ok = all(np.array_equal(np.bincount(sets[r] // d, minlength=n_layers),
                                counts) for r in range(n_draws))
    log(f"  [check] draws: per-layer counts match S in all {n_draws} = "
        f"{cnt_ok}; draw 0 overlap with S = {overlap0}, duplicates = {dup0}; "
        f"layers too dense to match = {n_short}")
    between = len(set(sets[0].tolist()) & set(sets[1].tolist())) / K \
        if n_draws > 1 else float("nan")
    # two independent draws share sum_L c_L^2 / (d - c_L) units in expectation
    expected = float(np.sum(counts ** 2 / (d - counts))) / K
    log(f"  [check] draw 0 vs draw 1 shared units = {between:.4f} "
        f"(expected ~ {expected:.4f})")

    M = indicator_matrix(np.vstack([S_idx[None, :], sets]), n_units)
    published = load_published_loadings(results_dir, mid)

    draw_rows, cell_rows, item_rows = [], [], []
    for fmt in VERBAL:
        per_item = load_per_item(results_dir, mid, fmt)
        kept = load_kept_ids(results_dir, mid, fmt)
        resp = load_responses(results_dir, mid, fmt)
        if per_item is None or kept is None or not resp:
            log(f"  skip {mid}/{fmt}: per-item tensor, kept_ids or responses "
                f"missing")
            continue
        flat = per_item.reshape(per_item.shape[0], -1)
        # items without a response record are skipped, as in run_overlap.py
        rows = [i for i, iid in enumerate(kept) if iid in resp]
        if len(rows) < len(kept):
            flat = flat[rows]
        ids = [kept[i] for i in rows]
        acc = np.array([int(bool(resp[i]["clean_match_gold"])) for i in ids])

        L = set_loadings(flat, M, chunk)                 # [n, 1 + R]
        real_fp16 = np.array([reference_loading(flat[i], S_idx)
                              for i in range(flat.shape[0])])

        pub = published.get(fmt, {})
        diffs = [abs(real_fp16[k] - pub[iid]) for k, iid in enumerate(ids)
                 if iid in pub]
        log(f"  {fmt}: n={len(ids)} correct={int(acc.sum())}; "
            f"reproduced published sum_on_S on {len(diffs)} items, max|diff| = "
            f"{(max(diffs) if diffs else float('nan')):.3e}; float32 vs fp16 "
            f"real loading max|diff| = "
            f"{float(np.max(np.abs(L[:, 0] - real_fp16))):.3e}")

        if len(set(acc.tolist())) > 1:
            r_real, p_real = pointbiserialr(acc.astype(float), real_fp16)
            r_cols = rpb_columns(acc, L)
            r_real32, r_draws = float(r_cols[0]), r_cols[1:]
        else:
            r_real = p_real = r_real32 = float("nan")
            r_draws = np.full(n_draws, np.nan)
        log(f"    r_pb real = {float(r_real):+.3f} (float32 path "
            f"{r_real32:+.3f}); random draws mean = "
            f"{np.nanmean(r_draws) if np.isfinite(r_draws).any() else float('nan'):+.3f}, "
            f"95% = [{np.nanpercentile(r_draws, 2.5) if np.isfinite(r_draws).any() else float('nan'):+.3f}, "
            f"{np.nanpercentile(r_draws, 97.5) if np.isfinite(r_draws).any() else float('nan'):+.3f}]")

        cell_rows.append({
            "model": mid, "format": fmt, "n": len(ids),
            "n_correct": int(acc.sum()), "K": K, "n_draws": n_draws,
            "rpb_real": float(r_real), "p_real": float(p_real),
            "rpb_real_float32": r_real32,
        })
        draw_rows.extend({"model": mid, "format": fmt, "draw": r,
                          "rpb": float(r_draws[r])} for r in range(n_draws))
        rand = L[:, 1:].astype(np.float64)
        rmean, rsd = rand.mean(axis=1), rand.std(axis=1, ddof=1)
        item_rows.extend({
            "model": mid, "format": fmt, "id": iid, "accuracy": int(acc[k]),
            "loading_real": float(real_fp16[k]),
            "loading_random_mean": float(rmean[k]),
            "loading_random_sd": float(rsd[k]),
        } for k, iid in enumerate(ids))
        del per_item, flat, L, rand

    layer_rows = [{"model": mid, "layer": L_, "d": d, "n_in_S": int(counts[L_])}
                  for L_ in range(n_layers)]
    return draw_rows, cell_rows, item_rows, layer_rows


def write_csv(path, fields, rows, mode="w"):
    new = mode == "w" or not os.path.exists(path)
    with open(path, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=None,
                    help="main results dir (default: attribution/results)")
    ap.add_argument("--out-dir", default=None,
                    help="default: attribution/matched_baseline/results")
    ap.add_argument("--n-draws", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk", type=int, default=64,
                    help="items per block in the sparse product")
    ap.add_argument("--models", nargs="+", default=None)
    args = ap.parse_args()

    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    attribution_root = os.path.dirname(pkg_root)
    results_dir = args.results_dir or os.path.join(attribution_root, "results")
    out_dir = args.out_dir or os.path.join(pkg_root, "results")
    os.makedirs(out_dir, exist_ok=True)

    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    log(f"matched random baseline: {args.n_draws} per-layer count-matched "
        f"draws per model, disjoint from S, seed {args.seed}")

    models = args.models or MODELS
    paths = {k: os.path.join(out_dir, f"{k}.csv") for k in
             ("random_draws_rpb", "cell_real", "item_loadings",
              "layer_profile")}
    # rewrite from scratch; each model is appended as soon as it finishes so
    # a crash on a late (large) model keeps the earlier ones
    for k, fields in (("random_draws_rpb", DRAW_FIELDS),
                      ("cell_real", CELL_FIELDS),
                      ("item_loadings", ITEM_FIELDS),
                      ("layer_profile", LAYER_FIELDS)):
        write_csv(paths[k], fields, [])

    n_done = 0
    for mid in models:
        log(f"\n{mid}")
        model_idx = MODELS.index(mid) if mid in MODELS else len(MODELS)
        out = run_model(mid, model_idx, results_dir, args.n_draws, args.seed,
                        args.chunk, log)
        if out is None:
            continue
        draws, cells, items, layers = out
        write_csv(paths["random_draws_rpb"], DRAW_FIELDS, draws, "a")
        write_csv(paths["cell_real"], CELL_FIELDS, cells, "a")
        write_csv(paths["item_loadings"], ITEM_FIELDS, items, "a")
        write_csv(paths["layer_profile"], LAYER_FIELDS, layers, "a")
        n_done += bool(cells)
        with open(os.path.join(out_dir, "validation.txt"), "w") as f:
            f.write("\n".join(log_lines) + "\n")

    log(f"\n{n_done} models written to {out_dir}")
    with open(os.path.join(out_dir, "validation.txt"), "w") as f:
        f.write("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()
