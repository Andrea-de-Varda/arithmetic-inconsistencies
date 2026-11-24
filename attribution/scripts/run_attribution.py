"""Stage 2: attribution patching using model-driven answers.

Reads responses produced by generate_responses.py, applies two filters
(token alignment + distinguishability), computes baselines, and runs
node-level attribution for MLP neurons (default) and/or attention heads.

Outputs per (model, format) under attribution/results/:
  baselines/<model>/<format>.json
  attribution/<model>/<format>/
    kept_ids.json
    neuron_attribution_per_item.pt   [n_kept, num_layers, intermediate_size] fp16
    neuron_attribution_aggregate.pt  [num_layers, intermediate_size]         fp32
    head_attribution_per_item.pt     (only if --component in {heads,both})
    head_attribution_aggregate.pt

Usage:
  python scripts/run_attribution.py \
      --model meta-llama/Llama-3.1-8B \
      --format spanish --component neurons --batch-size 4
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.data import prepare_sequence_inputs, make_prompt
from src.model import load_model_hf
from src.attribution import (
    compute_baselines, run_neuron_attribution, run_head_attribution,
)


def model_short(name):
    return name.replace("/", "_").replace(".", "-")


def filter_records(records, tokenizer, format_name):
    """Apply token-alignment + distinguishability filters.

    Token-alignment is checked on the full prompt (one-shot exemplar + x),
    matching how stage 1 built the input — otherwise positions wouldn't be
    comparable between clean and corrupted runs.

    Returns:
      kept_records: list of records that survived both filters
      counts: dict with n_total, n_after_alignment, n_after_distinguishability, n_kept
    """
    n_total = len(records)
    aligned = []
    for r in records:
        c_full = make_prompt(r["x"], format_name)
        rp_full = make_prompt(r["x_prime"], format_name)
        c_len = len(tokenizer(c_full, add_special_tokens=True)["input_ids"])
        r_len = len(tokenizer(rp_full, add_special_tokens=True)["input_ids"])
        if c_len == r_len:
            aligned.append(r)
    n_after_alignment = len(aligned)

    distinguishable = [
        r for r in aligned
        if r["model_clean_response"] != r["model_corrupted_response"]
    ]
    n_after_dist = len(distinguishable)

    return distinguishable, {
        "n_total": n_total,
        "n_after_alignment": n_after_alignment,
        "n_after_distinguishability": n_after_dist,
        "n_kept": n_after_dist,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--format", required=True)
    ap.add_argument("--component", default="neurons",
                    choices=["neurons", "heads", "both"])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--filter-ids-file", default=None,
                    help="optional JSON file containing a list of prob_xxxxxx "
                         "ids; if provided, restrict stage 2 to these ids "
                         "(intersected with the items that survive the "
                         "alignment + distinguishability filters). Used by "
                         "the control_balanced pipeline to restrict to the "
                         "balanced 2k-item subset.")
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.results_dir is None:
        args.results_dir = os.path.join(repo_root, "results")

    responses_path = os.path.join(
        args.results_dir, "responses", model_short(args.model), f"{args.format}.json"
    )
    if not os.path.exists(responses_path):
        raise FileNotFoundError(
            f"{responses_path} not found. Run generate_responses.py first."
        )

    baselines_dir = os.path.join(args.results_dir, "baselines", model_short(args.model))
    attr_dir = os.path.join(
        args.results_dir, "attribution", model_short(args.model), args.format
    )
    os.makedirs(baselines_dir, exist_ok=True)
    os.makedirs(attr_dir, exist_ok=True)

    print("=" * 70)
    print(f"STAGE 2: attribution — model={args.model}, format={args.format}, "
          f"component={args.component}")
    print("=" * 70)

    with open(responses_path, "r") as f:
        bundle = json.load(f)
    records = bundle["records"]

    model, tokenizer, model_config = load_model_hf(args.model)
    pad_id = tokenizer.pad_token_id

    kept, counts = filter_records(records, tokenizer, args.format)
    print(f"  filter survival: "
          f"n_total={counts['n_total']} -> "
          f"alignment={counts['n_after_alignment']} -> "
          f"distinguishable={counts['n_after_distinguishability']} -> "
          f"kept={counts['n_kept']}")

    # Optional id-list intersection (used by control_balanced to restrict to
    # the pre-selected balanced 2k-item subset).
    if args.filter_ids_file is not None:
        with open(args.filter_ids_file, "r") as f:
            allow_ids = set(json.load(f))
        n_before = len(kept)
        kept = [r for r in kept if r["id"] in allow_ids]
        counts["n_filter_ids_file"] = len(allow_ids)
        counts["n_kept"] = len(kept)
        print(f"  --filter-ids-file: {len(allow_ids)} requested ids; "
              f"kept {len(kept)} after intersecting with surviving items "
              f"({n_before - len(kept)} dropped)")

    if counts["n_kept"] == 0:
        print("  no items survive filters; aborting.")
        return

    kept_ids = [r["id"] for r in kept]
    with open(os.path.join(attr_dir, "kept_ids.json"), "w") as f:
        json.dump(kept_ids, f, indent=2)

    # Use the same one-shot-prefixed prompts that stage 1 generated against,
    # otherwise activations and baselines reflect a different scenario.
    clean_prompts = [make_prompt(r["x"], args.format) for r in kept]
    corr_prompts = [make_prompt(r["x_prime"], args.format) for r in kept]
    clean_resps = [r["model_clean_response"] for r in kept]
    corr_resps = [r["model_corrupted_response"] for r in kept]

    # Tokenize four (prompt, answer) pairings
    cc_ids, cc_plens = prepare_sequence_inputs(clean_prompts, clean_resps, tokenizer)
    ci_ids, ci_plens = prepare_sequence_inputs(clean_prompts, corr_resps, tokenizer)
    rc_ids, rc_plens = prepare_sequence_inputs(corr_prompts, clean_resps, tokenizer)
    ri_ids, ri_plens = prepare_sequence_inputs(corr_prompts, corr_resps, tokenizer)

    print(f"  example clean+correct token preview: "
          f"{tokenizer.decode(cc_ids[0][:60])!r}...")

    print("computing baselines...")
    baselines = compute_baselines(
        model, model_config,
        cc_ids, cc_plens, ci_ids, ci_plens,
        rc_ids, rc_plens, ri_ids, ri_plens,
        batch_size=args.batch_size, pad_id=pad_id,
    )
    clean_b = baselines["clean_baseline"]
    corr_b = baselines["corrupted_baseline"]
    print(f"  clean_baseline      = {clean_b:.4f}  (expected > 0)")
    print(f"  corrupted_baseline  = {corr_b:.4f}  (expected < 0)")
    if clean_b <= 0:
        print("  WARNING: clean_baseline <= 0 — model does NOT prefer its clean answer "
              "on its clean prompt. Metric may be ill-conditioned.")
    if corr_b >= 0:
        print("  WARNING: corrupted_baseline >= 0 — corruption did not flip the "
              "model. Metric may be ill-conditioned.")
    if clean_b == corr_b:
        print("  ERROR: clean_baseline == corrupted_baseline; aborting.")
        return

    baselines_out = {
        "model": args.model,
        "format": args.format,
        "filter_counts": counts,
        "kept_ids": kept_ids,
        "clean_baseline": clean_b,
        "corrupted_baseline": corr_b,
        "clean_correct_lps": baselines["clean_correct_lps"],
        "clean_incorrect_lps": baselines["clean_incorrect_lps"],
        "corr_correct_lps": baselines["corr_correct_lps"],
        "corr_incorrect_lps": baselines["corr_incorrect_lps"],
    }
    with open(os.path.join(baselines_dir, f"{args.format}.json"), "w") as f:
        json.dump(baselines_out, f, indent=2)

    if args.component in ("neurons", "both"):
        print("running neuron attribution...")
        per_item = run_neuron_attribution(
            model, model_config,
            cc_ids, cc_plens, rc_ids, rc_plens, ri_ids, ri_plens,
            clean_b, corr_b, batch_size=args.batch_size, pad_id=pad_id,
        )
        per_item_t = torch.from_numpy(np.stack(per_item, axis=0)).to(torch.float16)
        torch.save(per_item_t, os.path.join(attr_dir, "neuron_attribution_per_item.pt"))
        agg = per_item_t.float().mean(dim=0)
        torch.save(agg, os.path.join(attr_dir, "neuron_attribution_aggregate.pt"))
        print(f"  neuron per-item: {tuple(per_item_t.shape)} fp16")
        print(f"  neuron aggregate: {tuple(agg.shape)} fp32")

    if args.component in ("heads", "both"):
        print("running head attribution...")
        per_item_h = run_head_attribution(
            model, model_config,
            cc_ids, cc_plens, rc_ids, rc_plens, ri_ids, ri_plens,
            clean_b, corr_b, batch_size=args.batch_size, pad_id=pad_id,
        )
        per_item_ht = torch.from_numpy(np.stack(per_item_h, axis=0)).float()
        torch.save(per_item_ht, os.path.join(attr_dir, "head_attribution_per_item.pt"))
        agg = per_item_ht.mean(dim=0)
        torch.save(agg, os.path.join(attr_dir, "head_attribution_aggregate.pt"))
        print(f"  head per-item: {tuple(per_item_ht.shape)} fp32")
        print(f"  head aggregate: {tuple(agg.shape)} fp32")

    print("done.")


if __name__ == "__main__":
    main()
