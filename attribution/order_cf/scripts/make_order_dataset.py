"""Build the OPERAND-ORDER counterfactual dataset.

Why
---
The main pipeline identifies circuits by flipping the OPERATOR (`44 + 22 =`
vs `44 - 22 =`). The selected units need not implement
arithmetic: they could merely encode the contrast between a prompt and its
sign-flipped twin, i.e. read the operator token.

Here the operator is held FIXED and the two operands are swapped instead:

    x  = 70 - 33 =     ->  y  =  37
    x' = 33 - 70 =     ->  y' = -37

Two properties make this the right contrast.

* The operator token is identical in x and x', so a unit that only reads the
  operator has an attribution of exactly zero. Whatever this counterfactual
  selects is about the operands.
* |y'| = |y| exactly, so the contrast is a single sign decision. That matters
  for the causal test: patching 1% of units at one position can carry a
  low-dimensional decision, which is why the operator flip is patchable at
  all. An earlier attempt that changed an operand's *value* (44+22 vs 44+25)
  had to transmit a whole number through those units and recovered nothing,
  not even when patched with its own circuit, so it could not serve as a
  reference point. This design keeps the decision as small as the operator
  flip's.

Eligibility: two-term subtraction problems with distinct operands. In this
dataset every such item has a1 > a2 and both operands have the same number of
digits, so x' is a token-level reordering of x and the two prompts align.
There are 925 of them in the 5,000-item dataset, and we take all of them
rather than only the 356 inside the main attribution fold, because the
circuit has to be estimated from as many items as possible. The items are
still the paper's problems, and `x` is untouched.

What this contrast cannot test: since |y| is unchanged, it isolates operand
order and the sign of the result, not the magnitude computation. It is
evidence against an operator-reading account, not a complete account of the
circuit. The README says so.

Output: <out-dir>/arith_dataset_numeric_counterfactual.json, in the schema the
rest of the pipeline expects.

Usage (local, CPU, seconds):
  python order_cf/scripts/make_order_dataset.py
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from src.data import DEFAULT_DATA_DIR, DEFAULT_FOLD_PATH

TERM_RE = re.compile(r"^\s*(\d+)((?:\s*[+-]\s*\d+)+)\s*=\s*$")
OP_RE = re.compile(r"([+-])\s*(\d+)")


def parse_problem(x):
    """'70 - 33 =' -> ([70, 33], ['-'])."""
    m = TERM_RE.match(x)
    if not m:
        raise ValueError(f"cannot parse problem: {x!r}")
    operands = [int(m.group(1))]
    operators = []
    for op, val in OP_RE.findall(m.group(2)):
        operators.append(op)
        operands.append(int(val))
    return operands, operators


def render(operands, operators):
    s = str(operands[0])
    for op, val in zip(operators, operands[1:]):
        s += f" {op} {val}"
    return s + " ="


def eligible(operands, operators):
    """Two-term subtraction with distinct operands: swapping negates the answer."""
    return (len(operands) == 2 and operators == ["-"]
            and operands[0] != operands[1])


def build_item(rec):
    operands, operators = parse_problem(rec["x"])
    swapped = [operands[1], operands[0]]
    y = operands[0] - operands[1]
    y_prime = swapped[0] - swapped[1]
    assert y_prime == -y, f"{rec['id']}: swap is not a negation"

    out = dict(rec)
    out["x_prime"] = render(swapped, operators)
    out["y"] = f" {y}"
    out["y_prime"] = f" {y_prime}"
    out["ground_truth"] = y
    out["ground_truth_prime"] = y_prime
    out["cf_kind"] = "operand_order_swap"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", default=None, help="default: order_cf/data")
    ap.add_argument("--fold-only", action="store_true",
                    help="restrict to the 2,000 main-fold ids (356 items); "
                         "by default every eligible item in the dataset is "
                         "used, to estimate the circuit from more data")
    args = ap.parse_args()

    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = args.out_dir or os.path.join(pkg_root, "data")
    os.makedirs(out_dir, exist_ok=True)

    src = os.path.join(args.data_dir, "arith_dataset_numeric_counterfactual.json")
    items = json.load(open(src))
    if args.fold_only:
        fold = set(json.load(open(DEFAULT_FOLD_PATH)))
        items = [it for it in items if it["id"] in fold]

    out = []
    for rec in items:
        operands, operators = parse_problem(rec["x"])
        if eligible(operands, operators):
            out.append(build_item(rec))

    path = os.path.join(out_dir, "arith_dataset_numeric_counterfactual.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    fold = set(json.load(open(DEFAULT_FOLD_PATH)))
    n_digits_equal = sum(
        len(str(parse_problem(r["x"])[0][0])) == len(str(parse_problem(r["x"])[0][1]))
        for r in out)
    print(f"wrote {path}  ({len(out)} of {len(items)} items eligible)")
    print(f"  operands have equal digit count in {n_digits_equal}/{len(out)}")
    print(f"  {sum(r['id'] in fold for r in out)} of them are in the main "
          f"attribution fold")
    print(f"  example: {out[0]['x']!r} -> {out[0]['x_prime']!r} "
          f"(y {out[0]['y']!r} -> {out[0]['y_prime']!r})")


if __name__ == "__main__":
    main()
