"""Wrapper that generates the problem set once and writes all four format
JSONs (numeric / english / spanish / italian) with matched ids, in the
format of arithmetic-dataset/arith_dataset_<fmt>_counterfactual.json.

The bundled `generate_arithmetic_counterfactual.py` only saves the numeric
format; this script reuses its problem generator and corruption function
but adds verbal formatting via `DatasetFormatter.format_verbal(..., lang)`.

Usage (reproduces attribution/control_balanced/data/ byte for byte):
    python data_generation/generate_all_formats.py \
        --num-problems 20000 --seed 43 \
        --output-dir attribution/control_balanced/data
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from generate_arithmetic_data import (
    ArithmeticProblemGenerator,
    DatasetFormatter,
    split_dataset,
)
from generate_arithmetic_counterfactual import corrupt_problem_by_sign_switching


FORMATS = [
    ("numeric", None),
    ("english", "en"),
    ("spanish", "es"),
    ("italian", "it"),
]


def format_one(problem, fmt_key, lang, formatter):
    if fmt_key == "numeric":
        return formatter.format_numeric(problem)
    return formatter.format_verbal(problem, lang)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num-problems", type=int, default=20000)
    ap.add_argument("--terms", type=float, nargs=2, default=[0.5, 0.5])
    ap.add_argument("--digits", type=float, nargs=2, default=[0.5, 0.5])
    ap.add_argument("--carry-percentage", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--output-dir", default=".",
                    help="directory to write the four arith_dataset_<fmt>_counterfactual.json")
    args = ap.parse_args()

    print(f"generating {args.num_problems} problems, seed={args.seed}")
    gen = ArithmeticProblemGenerator(
        terms_distribution=tuple(args.terms),
        digits_distribution=tuple(args.digits),
        carry_percentage=args.carry_percentage,
        seed=args.seed,
    )
    problems = gen.generate_problems(args.num_problems)
    problems = split_dataset(problems)
    for i, p in enumerate(problems):
        p["id"] = f"prob_{i:06d}"
    print(f"  produced {len(problems)} problems")

    formatter = DatasetFormatter()
    os.makedirs(args.output_dir, exist_ok=True)

    for fmt_key, lang in FORMATS:
        entries = []
        for p in problems:
            x, y = format_one(p, fmt_key, lang, formatter)
            cp = corrupt_problem_by_sign_switching(p)
            xp, yp = format_one(cp, fmt_key, lang, formatter)
            entries.append({
                "id": p["id"],
                "x": x,
                "x_prime": xp,
                "y": y,
                "y_prime": yp,
                "ground_truth": p["result"],
                "ground_truth_prime": cp["result"],
                "split": p["split"],
                "has_carry": p["has_carry"],
                "n_terms": p["n_terms"],
                "n_digits": p["n_digits"],
            })
        out_path = os.path.join(
            args.output_dir,
            f"arith_dataset_{fmt_key}_counterfactual.json",
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        print(f"  wrote {out_path}  ({len(entries)} entries)")


if __name__ == "__main__":
    main()
