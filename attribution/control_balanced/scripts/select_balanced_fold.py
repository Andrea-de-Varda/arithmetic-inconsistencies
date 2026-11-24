"""Pick the balanced fold for the control analysis.

For one model and four formats, read the stage-1 responses (must already
exist for all four formats), apply the same alignment + distinguishability
filters that stage 2 will apply, then sample
  n_correct = 1600 items where clean_match_gold == True
  n_incorrect = 400 items where clean_match_gold == False
independently per format. The samples differ across formats — the only
constraint is the within-format 80/20 ratio.

Outputs one JSON file per (model, format) containing the 2000 prob_xxxxxx
ids that stage 2 should process.

Usage:
  python control_balanced/scripts/select_balanced_fold.py \
      --model meta-llama/Llama-3.1-8B \
      --responses-dir control_balanced/results/responses \
      --output-dir control_balanced/results/balanced_fold \
      --n-correct 1600 --n-incorrect 400 --seed 42
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))  # so `from src.data import FORMATS` works

import numpy as np
from transformers import AutoTokenizer

from src.data import FORMATS, make_prompt


def model_short(name):
    return name.replace("/", "_").replace(".", "-")


def filter_records(records, tokenizer, format_name):
    """Replicate the filter logic from scripts/run_attribution.py so the
    balanced fold only contains items that stage 2 will actually keep."""
    aligned = []
    for r in records:
        c_full = make_prompt(r["x"], format_name)
        rp_full = make_prompt(r["x_prime"], format_name)
        c_len = len(tokenizer(c_full, add_special_tokens=True)["input_ids"])
        r_len = len(tokenizer(rp_full, add_special_tokens=True)["input_ids"])
        if c_len == r_len:
            aligned.append(r)
    distinguishable = [
        r for r in aligned
        if r["model_clean_response"] != r["model_corrupted_response"]
    ]
    return distinguishable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--responses-dir", required=True,
                    help="directory containing <model>/<fmt>.json from stage 1")
    ap.add_argument("--output-dir", required=True,
                    help="directory to write <model>/<fmt>.json with 2000 ids")
    ap.add_argument("--n-correct", type=int, default=1600)
    ap.add_argument("--n-incorrect", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    short = model_short(args.model)
    out_dir = os.path.join(args.output_dir, short)
    os.makedirs(out_dir, exist_ok=True)

    print(f"loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    rng = np.random.default_rng(args.seed)
    summary = {}
    fatal = False
    for fmt in FORMATS:
        responses_path = os.path.join(args.responses_dir, short, f"{fmt}.json")
        if not os.path.exists(responses_path):
            print(f"[{fmt}]  ERROR  responses file missing: {responses_path}")
            fatal = True
            continue
        with open(responses_path, "r") as f:
            bundle = json.load(f)
        n_total = len(bundle["records"])

        survivors = filter_records(bundle["records"], tokenizer, fmt)
        correct = [r for r in survivors if r["clean_match_gold"]]
        incorrect = [r for r in survivors if not r["clean_match_gold"]]

        n_have_c, n_have_i = len(correct), len(incorrect)
        n_need_c, n_need_i = args.n_correct, args.n_incorrect
        print(f"[{fmt}]  n_total={n_total}  surviving_filters={len(survivors)}  "
              f"correct={n_have_c}  incorrect={n_have_i}  "
              f"need={n_need_c}/{n_need_i}")

        if n_have_c < n_need_c or n_have_i < n_need_i:
            print(f"[{fmt}]  FATAL  not enough items to fill the balanced fold "
                  f"(have {n_have_c} correct < {n_need_c} OR "
                  f"{n_have_i} incorrect < {n_need_i}). "
                  f"Generate more items or pick a different model.")
            fatal = True
            continue

        sampled_c_idx = rng.choice(len(correct), n_need_c, replace=False)
        sampled_i_idx = rng.choice(len(incorrect), n_need_i, replace=False)
        chosen_ids = sorted(
            [correct[i]["id"] for i in sampled_c_idx]
            + [incorrect[i]["id"] for i in sampled_i_idx]
        )

        out_path = os.path.join(out_dir, f"{fmt}.json")
        with open(out_path, "w") as f:
            json.dump(chosen_ids, f, indent=2)
        print(f"[{fmt}]  wrote {out_path}  ({len(chosen_ids)} ids)")

        summary[fmt] = {
            "n_total": n_total,
            "n_after_filters": len(survivors),
            "n_correct_available": n_have_c,
            "n_incorrect_available": n_have_i,
            "n_correct_sampled": n_need_c,
            "n_incorrect_sampled": n_need_i,
        }

    summary_path = os.path.join(out_dir, "_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "model": args.model,
            "n_correct": args.n_correct,
            "n_incorrect": args.n_incorrect,
            "seed": args.seed,
            "per_format": summary,
        }, f, indent=2)
    print(f"wrote {summary_path}")

    if fatal:
        sys.exit(1)


if __name__ == "__main__":
    main()
