"""Causal validation of the item-level attribution scores (all models, GPU).

The main analyses show, correlationally, that an item's attribution loading on
the numeric top-1% circuit predicts whether the model solves it. This script
runs the intervention that attribution patching only approximates: while the
model reads the sign-flipped prompt x', it OVERWRITES the activations of the
numeric-circuit units at the decision point (the last prompt token, where
attribution was scored) with the values they take on the original prompt x,
and records the per-item recovery of the preference gap:

    recovery_i = (m_patch(x') - m_i(x')) / (m_i(x) - m_i(x'))

where m(z) = logP(y_hat|z) - logP(y_hat'|z) (teacher-forced, model's own
answers). The unpatched per-item baselines m_i(x), m_i(x') are already stored
by stage 2 (results/baselines/<model>/<fmt>.json). The item-level test is then
r_pb(recovery, correctness) per (model, format): items the model solves should
be the items where the circuit has the largest causal effect.

Conditions:
  real     S = top-1% units of |numeric aggregate| (the paper's circuit S^m)
  random   per-layer count-matched random units, disjoint from S (seeded)

Runs ALL panel models in one job: loads each model, runs the verbal formats,
frees the GPU, and (with --purge-cache) deletes the model's HF cache directory
afterwards -- but only if the model was NOT already cached before this run, so
pre-existing downloads are never touched. Existing output JSONs are skipped
(--no-skip-existing to force).

Outputs: patching/results/<model_short>/denoise__<condition>/<fmt>.json
  {"summary": {...}, "records": [...]}

Usage on the cluster:
  cd .../arithmetic-inconsistencies/attribution
  python patching/scripts/run_patching.py                 # all models
  python patching/scripts/run_patching.py --models meta-llama/Llama-3.1-8B \
      --limit 32                                          # smoke test
"""

import argparse
import gc
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))   # attribution/ -> `import src.*`

import numpy as np
import torch

from src.model import load_model_hf
from src.data import make_prompt, prepare_sequence_inputs, pad_sequence_inputs
from src.attribution import _build_attention_mask, _hook_targets
from src.metrics import compute_sequence_log_prob


VERBAL = ("english", "spanish", "italian")

# The paper's 13-model panel with per-model teacher-forcing batch sizes
# (mirrors scripts/slurm/unified.sbatch BATCHES).
MODEL_BATCH = {
    "Qwen/Qwen3-0.6B": 8,
    "meta-llama/Llama-3.2-1B": 8,
    "google/gemma-2-2b": 8,
    "meta-llama/Llama-3.2-3B": 8,
    "Qwen/Qwen3-4B": 8,
    "mistralai/Mistral-7B-v0.1": 4,
    "allenai/Olmo-3-1025-7B": 4,
    "meta-llama/Llama-3.1-8B": 4,
    "Qwen/Qwen3-8B": 4,
    "google/gemma-2-9b": 4,
    "microsoft/phi-4": 2,
    "Qwen/Qwen3-32B": 1,
    "allenai/Olmo-3-1125-32B": 1,
}


def model_short(name):
    return name.replace("/", "_").replace(".", "-")


def hf_cache_dir(model_name):
    """The HF hub cache directory of a model, e.g.
    $HF_HOME/hub/models--meta-llama--Llama-3.1-8B."""
    home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    return os.path.join(home, "hub", "models--" + model_name.replace("/", "--"))


# ---------------------------------------------------------------------------
# circuit construction
# ---------------------------------------------------------------------------

def load_numeric_circuit(results_dir, mid, top_k_fraction):
    """Top-K flat indices of |numeric aggregate|. Returns (flat_idx, L, dim)."""
    p = os.path.join(results_dir, "attribution", model_short(mid), "numeric",
                     "neuron_attribution_aggregate.pt")
    agg = torch.load(p).float().numpy()          # [L, dim]
    L, dim = agg.shape
    K = max(1, int(round(top_k_fraction * agg.size)))
    flat = np.abs(agg.ravel())
    idx = np.argpartition(flat, -K)[-K:]
    return np.sort(idx), L, dim


def per_layer_units(flat_idx, L, dim):
    """flat -> {layer: sorted unit indices}, skipping empty layers."""
    out = {}
    layers = flat_idx // dim
    units = flat_idx % dim
    for li in range(L):
        u = np.sort(units[layers == li])
        if len(u):
            out[int(li)] = u
    return out


def random_matched_circuit(real_by_layer, dim, seed):
    """Random units, per-layer count-matched to the real circuit, disjoint
    from it within each layer (so the control never touches circuit units)."""
    rng = np.random.default_rng(seed)
    out = {}
    for li, u in real_by_layer.items():
        pool = np.setdiff1d(np.arange(dim), u, assume_unique=False)
        out[li] = np.sort(rng.choice(pool, size=len(u), replace=False))
    return out


# ---------------------------------------------------------------------------
# hooks
# ---------------------------------------------------------------------------

class DecisionPointCapturer:
    """Captures the full down_proj-input vector at a per-row position, for a
    given set of layers. captures[layer] -> [B, dim] (on device)."""

    def __init__(self, modules_by_layer):
        self.modules_by_layer = modules_by_layer   # {layer_idx: nn.Module}
        self.positions = None                      # LongTensor [B] (device)
        self.captures = {}
        self._handles = []

    def set_positions(self, positions):
        self.positions = positions
        self.captures = {}

    def __enter__(self):
        for li, mod in self.modules_by_layer.items():
            self._handles.append(mod.register_forward_pre_hook(self._hook(li)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []

    def _hook(self, li):
        def hook(_module, inputs):
            x = inputs[0]                                     # [B, T, dim]
            B = x.shape[0]
            rows = torch.arange(B, device=x.device)
            self.captures[li] = x[rows, self.positions, :].detach().clone()
            return None
        return hook


class DecisionPointPatcher:
    """Overwrites selected units of the down_proj input with donor values at a
    per-row position (teacher-forced, right-padded batches). Donor values are
    set per batch via set(); the same registered instance is reused."""

    def __init__(self, modules_by_layer, units_by_layer):
        self.modules_by_layer = modules_by_layer
        self.units_by_layer = {li: torch.as_tensor(u, dtype=torch.long)
                               for li, u in units_by_layer.items()}
        self.positions = None      # LongTensor [B]
        self.donors = None         # {layer: [B, n_units_layer]} (device)
        self._handles = []

    def set(self, donors_by_layer, positions):
        self.donors = donors_by_layer
        self.positions = positions

    def __enter__(self):
        for li, mod in self.modules_by_layer.items():
            self._handles.append(mod.register_forward_pre_hook(self._hook(li)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []

    def _hook(self, li):
        def hook(_module, inputs):
            if self.donors is None:
                return None
            x = inputs[0]                                     # [B, T, dim]
            units = self.units_by_layer[li].to(x.device)
            donor = self.donors[li].to(x.device, dtype=x.dtype)  # [B, n_units]
            B = x.shape[0]
            rows = torch.arange(B, device=x.device)
            # broadcast: rows [B,1] x positions [B,1] x units [1,n] -> [B,n]
            x[rows.unsqueeze(1), self.positions.unsqueeze(1),
              units.unsqueeze(0)] = donor
            return None
        return hook


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def load_format_data(results_dir, mid, fmt):
    short = model_short(mid)
    kept = json.load(open(os.path.join(results_dir, "attribution", short, fmt,
                                       "kept_ids.json")))
    resp = json.load(open(os.path.join(results_dir, "responses", short,
                                       f"{fmt}.json")))["records"]
    by_id = {r["id"]: r for r in resp}
    base = json.load(open(os.path.join(results_dir, "baselines", short,
                                       f"{fmt}.json")))
    assert base["kept_ids"] == kept, "baselines/kept_ids misalignment"
    items = []
    for i, iid in enumerate(kept):
        r = by_id[iid]
        m_clean = base["clean_correct_lps"][i] - base["clean_incorrect_lps"][i]
        m_corr = base["corr_correct_lps"][i] - base["corr_incorrect_lps"][i]
        items.append({
            "id": iid, "x": r["x"], "x_prime": r["x_prime"],
            "y_hat": r["model_clean_response"],
            "y_hat_prime": r["model_corrupted_response"],
            "clean_match_gold": bool(r["clean_match_gold"]),
            "m_clean": float(m_clean), "m_corr": float(m_corr),
        })
    return items


# ---------------------------------------------------------------------------
# core computation
# ---------------------------------------------------------------------------

def teacher_forced_lps(model, seqs, plens, pad_id, patcher=None,
                       donors=None, positions=None):
    """Forward a right-padded batch of (prompt+answer) sequences; return [B]
    numpy of answer log-probs. If patcher given (already registered), set
    donors/positions on it for this forward."""
    device = model.device
    seq_lens = [t.shape[0] for t in seqs]
    padded = pad_sequence_inputs(seqs, pad_token_id=pad_id).to(device)
    amask = _build_attention_mask(padded, seq_lens, pad_id).to(device)
    if patcher is not None:
        patcher.set(donors, positions)
    with torch.no_grad():
        logits = model(input_ids=padded, attention_mask=amask).logits
    out = []
    for i in range(len(seqs)):
        lp = compute_sequence_log_prob(
            logits[i, :seq_lens[i], :].float().cpu(), seqs[i].cpu(), plens[i])
        out.append(float(lp))
    del logits
    return np.array(out)


def run_format(model, tokenizer, items, fmt, circuits, capt_modules,
               batch_size, self_check_n=0):
    """Denoising only: patch clean-run donors into the corrupted run at the
    numeric-circuit units. Returns {condition: list of records}."""
    device = model.device
    pad_id = tokenizer.pad_token_id
    results = {c: [] for c in circuits}
    patchers = {c: DecisionPointPatcher(
        {li: capt_modules[li] for li in circuits[c]}, circuits[c])
        for c in circuits}
    capt_layers = sorted({li for c in circuits.values() for li in c})
    capturer = DecisionPointCapturer({li: capt_modules[li]
                                      for li in capt_layers})

    self_checked = 0
    n = len(items)
    for s in range(0, n, batch_size):
        chunk = items[s:s + batch_size]
        prompts_clean = [make_prompt(it["x"], fmt) for it in chunk]
        prompts_corr = [make_prompt(it["x_prime"], fmt) for it in chunk]
        y_hat = [it["y_hat"] for it in chunk]
        y_hat_p = [it["y_hat_prime"] for it in chunk]

        # teacher-forced sequences of the patched (corrupted) run
        rc, rc_pl = prepare_sequence_inputs(prompts_corr, y_hat, tokenizer)
        ri, ri_pl = prepare_sequence_inputs(prompts_corr, y_hat_p, tokenizer)
        pos_corr = torch.tensor([p - 1 for p in rc_pl], device=device)

        # donor capture on the clean prompt (prompt-only forward: the
        # activation at the decision point is independent of the answer)
        ids = [torch.tensor(tokenizer(p, add_special_tokens=True)["input_ids"],
                            dtype=torch.long) for p in prompts_clean]
        seq_lens = [t.shape[0] for t in ids]
        pos_clean = torch.tensor([l - 1 for l in seq_lens], device=device)
        padded = pad_sequence_inputs(ids, pad_token_id=pad_id).to(device)
        amask = _build_attention_mask(padded, seq_lens, pad_id).to(device)
        with capturer:
            capturer.set_positions(pos_clean)
            with torch.no_grad():
                model(input_ids=padded, attention_mask=amask)
            donors_clean = dict(capturer.captures)

        for cond, units in circuits.items():
            donor_units = {li: donors_clean[li][:, torch.as_tensor(
                units[li], dtype=torch.long, device=device)] for li in units}
            with patchers[cond] as P:
                lp_c = teacher_forced_lps(model, rc, rc_pl, pad_id,
                                          P, donor_units, pos_corr)
                lp_i = teacher_forced_lps(model, ri, ri_pl, pad_id,
                                          P, donor_units, pos_corr)
            m_patch = lp_c - lp_i
            for j, it in enumerate(chunk):
                denom = it["m_clean"] - it["m_corr"]
                effect = (m_patch[j] - it["m_corr"]) / denom \
                    if abs(denom) > 1e-9 else float("nan")
                results[cond].append({
                    "id": it["id"],
                    "m_patched": float(m_patch[j]),
                    "m_clean": it["m_clean"], "m_corr": it["m_corr"],
                    "effect": float(effect),
                    "small_denominator": bool(abs(denom) < 1.0),
                    "clean_match_gold": it["clean_match_gold"],
                })

        # self-check on first items: (1) hook exactness (self-patching the
        # corrupted run with its OWN donors == unpatched forward); (2) stage-2
        # alignment (unpatched m == stored baseline, fp16 noise)
        if self_checked < self_check_n:
            ids_c = [torch.tensor(tokenizer(p, add_special_tokens=True)
                                  ["input_ids"], dtype=torch.long)
                     for p in prompts_corr]
            lens_c = [t.shape[0] for t in ids_c]
            padded_c = pad_sequence_inputs(ids_c, pad_token_id=pad_id).to(device)
            amask_c = _build_attention_mask(padded_c, lens_c, pad_id).to(device)
            with capturer:
                capturer.set_positions(pos_corr)
                with torch.no_grad():
                    model(input_ids=padded_c, attention_mask=amask_c)
                donors_self = dict(capturer.captures)
            cond = "real"
            donor_units = {li: donors_self[li][:, torch.as_tensor(
                circuits[cond][li], dtype=torch.long, device=device)]
                for li in circuits[cond]}
            with patchers[cond] as P:
                lp_c = teacher_forced_lps(model, rc, rc_pl, pad_id, P,
                                          donor_units, pos_corr)
                lp_i = teacher_forced_lps(model, ri, ri_pl, pad_id, P,
                                          donor_units, pos_corr)
            lp_c0 = teacher_forced_lps(model, rc, rc_pl, pad_id)
            lp_i0 = teacher_forced_lps(model, ri, ri_pl, pad_id)
            d_hook = np.abs((lp_c - lp_i) - (lp_c0 - lp_i0))
            d_base = np.abs((lp_c0 - lp_i0)
                            - np.array([it["m_corr"] for it in chunk]))
            print(f"  [self-check] hook exactness max={d_hook.max():.5f} "
                  f"(must be ~0); stage-2 drift max={d_base.max():.4f}",
                  flush=True)
            self_checked += len(chunk)

        done = min(s + batch_size, n)
        if done % (batch_size * 25) < batch_size or done == n:
            print(f"  [{fmt}] {done}/{n} items", flush=True)
    return results


# ---------------------------------------------------------------------------

def run_model(mid, args, results_dir, out_root, circuit_dir=None):
    short = model_short(mid)
    fmts = [f for f in args.formats
            if not (args.skip_existing and all(os.path.exists(os.path.join(
                out_root, short, f"denoise__{c}", f"{f}.json"))
                for c in args.conditions))]
    if not fmts:
        print(f"[{mid}] all outputs exist, skipping", flush=True)
        return
    # The circuit may come from a different run than the prompts: the operand
    # counterfactual patches the OPERATOR-derived circuit while the model reads
    # operand-changed prompts, which is a transfer test across counterfactuals.
    try:
        flat, L, dim = load_numeric_circuit(circuit_dir or results_dir, mid,
                                            args.top_k_fraction)
    except FileNotFoundError as ex:
        print(f"[{mid}] skip: {ex}", flush=True)
        return
    real = per_layer_units(flat, L, dim)
    circuits = {}
    if "real" in args.conditions:
        circuits["real"] = real
    if "random" in args.conditions:
        circuits["random"] = random_matched_circuit(real, dim, args.seed)
    n_units = sum(len(u) for u in real.values())

    cached_before = os.path.isdir(hf_cache_dir(mid))
    batch = args.batch_size or MODEL_BATCH.get(mid, 4)
    print(f"\n########## {mid}  (batch={batch}, circuit={n_units} units, "
          f"cached_before={cached_before}) ##########", flush=True)
    model, tokenizer, config = load_model_hf(mid)
    mlp_modules = _hook_targets(model, config["family"], want_mlp=True)
    capt_modules = {li: mlp_modules[li] for li in range(L)}

    for fmt in fmts:
        try:
            items = load_format_data(results_dir, mid, fmt)
        except (FileNotFoundError, AssertionError) as ex:
            print(f"  [{fmt}] skip: {ex}", flush=True)
            continue
        if args.limit:
            items = items[:args.limit]
        print(f"  [{fmt}] {len(items)} items", flush=True)
        res = run_format(model, tokenizer, items, fmt, circuits, capt_modules,
                         batch, self_check_n=args.self_check)
        for cond, records in res.items():
            eff = np.array([r["effect"] for r in records
                            if not r["small_denominator"]
                            and np.isfinite(r["effect"])])
            summary = {
                "model": mid, "format": fmt, "direction": "denoise",
                "condition": cond, "top_k_fraction": args.top_k_fraction,
                "n_units": n_units, "seed": args.seed,
                "patch_position": "last_prompt_token",
                "n_items": len(records), "n_used_for_mean": int(len(eff)),
                "mean_effect": float(np.mean(eff)) if len(eff) else None,
            }
            out_dir = os.path.join(out_root, short, f"denoise__{cond}")
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, f"{fmt}.json"), "w") as f:
                json.dump({"summary": summary, "records": records}, f)
            print(f"  wrote {out_dir}/{fmt}.json  mean recovery ({cond}) "
                  f"= {summary['mean_effect']}", flush=True)

    # free the GPU and, if requested, the disk (only for models this run
    # downloaded itself; pre-existing cache entries are never deleted)
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if args.purge_cache and not cached_before:
        cache = hf_cache_dir(mid)
        if os.path.isdir(cache):
            shutil.rmtree(cache, ignore_errors=True)
            print(f"  purged HF cache {cache}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["all"],
                    help='"all" (default) or explicit model ids')
    ap.add_argument("--top-k-fraction", type=float, default=0.01)
    ap.add_argument("--formats", nargs="+", default=list(VERBAL))
    ap.add_argument("--conditions", nargs="+", default=["real", "random"],
                    choices=["real", "random"])
    ap.add_argument("--batch-size", type=int, default=None,
                    help="override the per-model default")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None,
                    help="only first N items per format (smoke test)")
    ap.add_argument("--self-check", type=int, default=8)
    ap.add_argument("--purge-cache", action="store_true",
                    help="after each model, delete its HF cache dir "
                         "(only if it was not cached before this run)")
    ap.add_argument("--no-skip-existing", dest="skip_existing",
                    action="store_false", default=True)
    ap.add_argument("--results-dir", default=None,
                    help="where prompts, responses and baselines come from")
    ap.add_argument("--circuit-results-dir", default=None,
                    help="where the numeric circuit comes from; defaults to "
                         "--results-dir. Set it to the main results to patch "
                         "the operator-flip circuit into another run's prompts.")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo_root = os.path.dirname(pkg_root)
    results_dir = args.results_dir or os.path.join(repo_root, "results")
    out_root = args.out_dir or os.path.join(pkg_root, "results")

    models = list(MODEL_BATCH) if args.models == ["all"] else args.models
    print(f"{len(models)} model(s); formats={args.formats}; "
          f"conditions={args.conditions}; purge_cache={args.purge_cache}",
          flush=True)
    circuit_dir = args.circuit_results_dir or results_dir
    if circuit_dir != results_dir:
        print(f"circuit from {circuit_dir}\nprompts from {results_dir}", flush=True)
    for mid in models:
        run_model(mid, args, results_dir, out_root, circuit_dir)
    print("\nall done.", flush=True)


if __name__ == "__main__":
    main()
