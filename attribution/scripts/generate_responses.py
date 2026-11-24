"""Stage 1: greedy-decode the model's response for each clean and corrupted prompt.

For each item in the attribution fold:
  K_i = max(token_len(y_i), token_len(y_prime_i))
  model_clean_response     = greedy_decode(x_i, K_i)
  model_corrupted_response = greedy_decode(x'_i, K_i)

Outputs:
  attribution/results/responses/<model>/<format>.json
  - per-item records (id, x, x_prime, y, y_prime, both responses, both correctness flags, K_i, ...)
  - top-level summary (n_total, clean_accuracy, corrupted_accuracy, n_distinguishable)

Also prints to stdout:
  - format-level accuracies
  - 10 sampled clean errors and 10 sampled corrupted errors with prompt+gold+model_response

Usage:
  python scripts/generate_responses.py \
      --model meta-llama/Llama-3.1-8B \
      --format numeric --batch-size 32
"""

import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.data import (
    DEFAULT_DATA_DIR, FORMATS,
    build_attribution_fold, load_format, subset_to_fold, make_prompt,
)
from src.generate import greedy_decode_fixed_k, k_for_item
from src.model import load_model_hf


def model_short(name):
    return name.replace("/", "_").replace(".", "-")


def normalize_numeric(s):
    """Extract first signed integer; tolerant of trailing punctuation/garbage."""
    if s is None:
        return None
    m = re.search(r"-?\d+", str(s))
    if not m:
        return None
    try:
        return int(m.group())
    except ValueError:
        return None


def normalize_verbal(s):
    s = s.lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)
    return " ".join(s.split())


def is_match(model_response, gold, format_name):
    """Item-level correctness:
       - numeric: regex-extract first integer, compare as int
       - verbal: lowercase + strip punctuation, allow prefix match (since K_i may
         be longer than the gold and add trailing tokens)."""
    if format_name == "numeric":
        a = normalize_numeric(model_response)
        b = normalize_numeric(str(gold))
        return a is not None and b is not None and a == b
    pred = normalize_verbal(model_response)
    exp = normalize_verbal(str(gold))
    return pred == exp or pred.startswith(exp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--format", required=True, choices=list(FORMATS))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--results-dir", default=None,
                    help="defaults to attribution/results/responses")
    ap.add_argument("--n-items", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-spotcheck", type=int, default=10)
    ap.add_argument("--use-all-items", action="store_true",
                    help="skip the deterministic fold and process every item "
                         "in the data file. Use this for the control_balanced "
                         "pipeline where we need responses on the full dataset "
                         "before selecting the balanced subset.")
    ap.add_argument("--max-items", type=int, default=None,
                    help="with --use-all-items, stop after the first N items "
                         "of the data file (file order, so deterministic). "
                         "Used by the probe-pool stage 1, which only needs "
                         "enough items to collect a target number of errors.")
    ap.add_argument("--skip-corrupted", action="store_true",
                    help="decode only the clean prompt x, not the sign-flipped "
                         "x'. Halves stage-1 cost. Only valid when the "
                         "consumer needs clean responses and labels alone "
                         "(the probe pool); attribution needs both, so never "
                         "use this for the main or control pipelines. "
                         "Corrupted fields are written as null.")
    args = ap.parse_args()

    if args.max_items is not None and not args.use_all_items:
        ap.error("--max-items requires --use-all-items")

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.results_dir is None:
        args.results_dir = os.path.join(repo_root, "results", "responses")

    out_dir = os.path.join(args.results_dir, model_short(args.model))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.format}.json")

    print("=" * 70)
    print(f"STAGE 1: generate responses — model={args.model}, format={args.format}")
    print("=" * 70)

    if args.use_all_items:
        items = load_format(args.format, args.data_dir)
        if args.max_items is not None:
            items = items[: args.max_items]
            print(f"using the first {len(items)} items in {args.format} "
                  f"(--max-items {args.max_items})")
        else:
            print(f"using all items in {args.format}: {len(items)}")
    else:
        fold_ids = build_attribution_fold(
            n_items=args.n_items, seed=args.seed, data_dir=args.data_dir,
        )
        print(f"fold size: {len(fold_ids)}")
        items = subset_to_fold(load_format(args.format, args.data_dir), fold_ids)

    model, tokenizer, _ = load_model_hf(args.model)

    # Compute K_i per item
    print("computing K_i...")
    ks = [k_for_item(tokenizer, it["y"], it["y_prime"]) for it in items]
    # Prepend the per-format one-shot exemplar; without it Llama-3.1-8B base
    # has no idea what format to emit.
    clean_prompts = [make_prompt(it["x"], args.format) for it in items]
    corr_prompts = [make_prompt(it["x_prime"], args.format) for it in items]

    clean_resps = greedy_decode_fixed_k(
        model, tokenizer, clean_prompts, ks, args.batch_size, desc="clean decode"
    )
    if args.skip_corrupted:
        print("--skip-corrupted: not decoding x'; corrupted fields are null")
        corr_resps = [None] * len(items)
    else:
        corr_resps = greedy_decode_fixed_k(
            model, tokenizer, corr_prompts, ks, args.batch_size,
            desc="corrupted decode"
        )

    records = []
    n_clean_correct = 0
    n_corr_correct = 0
    n_distinguishable = 0
    for it, k, cr, rr in zip(items, ks, clean_resps, corr_resps):
        clean_match = is_match(cr, it["y"], args.format)
        corr_match = None if rr is None else is_match(rr, it["y_prime"], args.format)
        n_clean_correct += int(clean_match)
        n_corr_correct += int(bool(corr_match))
        n_distinguishable += int(rr is not None and cr != rr)
        records.append({
            "id": it["id"],
            "x": it["x"],
            "x_prime": it["x_prime"],
            "y": it["y"],
            "y_prime": it["y_prime"],
            "ground_truth": it.get("ground_truth"),
            "ground_truth_prime": it.get("ground_truth_prime"),
            "model_clean_response": cr,
            "model_corrupted_response": rr,
            "clean_match_gold": clean_match,
            "corrupted_match_gold": corr_match,
            "K_i": k,
            "n_terms": it.get("n_terms"),
            "n_digits": it.get("n_digits"),
            "has_carry": it.get("has_carry"),
            "split": it.get("split"),
        })

    n_total = len(records)
    summary = {
        "model": args.model,
        "format": args.format,
        "n_total": n_total,
        "clean_accuracy": n_clean_correct / n_total,
        "corrupted_accuracy": None if args.skip_corrupted else n_corr_correct / n_total,
        "n_distinguishable": None if args.skip_corrupted else n_distinguishable,
        "fold_seed": args.seed,
        "fold_n_items": args.n_items,
        "skip_corrupted": bool(args.skip_corrupted),
        "max_items": args.max_items,
    }

    with open(out_path, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2,
                  ensure_ascii=False)

    print()
    print(f"wrote {out_path}")
    print(f"  n_total              = {n_total}")
    print(f"  clean_accuracy       = {summary['clean_accuracy']:.2%}")
    if args.skip_corrupted:
        print("  corrupted_accuracy   = (skipped)")
        print("  n_distinguishable    = (skipped)")
    else:
        print(f"  corrupted_accuracy   = {summary['corrupted_accuracy']:.2%}")
        print(f"  n_distinguishable    = {n_distinguishable} ({n_distinguishable/n_total:.2%})")

    rng = random.Random(args.seed)
    clean_errs = [r for r in records if not r["clean_match_gold"]]
    corr_errs = ([] if args.skip_corrupted
                 else [r for r in records if not r["corrupted_match_gold"]])

    def show(label, errs, gold_field, resp_field, prompt_field):
        print(f"\n  --- {len(errs)} {label} errors; sampling {min(args.n_spotcheck, len(errs))} ---")
        for r in rng.sample(errs, min(args.n_spotcheck, len(errs))):
            print(f"    {r['id']}  prompt: {r[prompt_field]!r}  "
                  f"gold: {r[gold_field]!r}  model: {r[resp_field]!r}")

    show("clean", clean_errs, "y", "model_clean_response", "x")
    show("corrupted", corr_errs, "y_prime", "model_corrupted_response", "x_prime")


if __name__ == "__main__":
    main()
