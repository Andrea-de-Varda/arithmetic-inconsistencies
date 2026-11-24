"""Self-test for the operand-order counterfactual (local, no cluster data).

The argument rests on two properties of the dataset, so these are checked on
every item: the operator is identical in x and x' (an operator-reading unit
gets zero attribution), and the swap negates the answer exactly (the contrast
is a single sign decision, small enough for a 1% patch to carry). The rest
checks eligibility, correctness of the answers, determinism, and the overlap
machinery in circuit_overlap.py.

Usage (local):  python order_cf/scripts/selftest_order.py
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))

from make_order_dataset import parse_problem, render, eligible, build_item
from circuit_overlap import chance_jaccard, split_half_ceiling, topk_abs, jaccard
from src.data import DEFAULT_DATA_DIR

FAIL = []


def check(name, cond, detail=""):
    print(f"  [{'ok ' if cond else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAIL.append(name)


def main():
    pkg_root = os.path.dirname(HERE)
    path = os.path.join(pkg_root, "data",
                        "arith_dataset_numeric_counterfactual.json")
    if not os.path.exists(path):
        print(f"missing {path}; run make_order_dataset.py first")
        sys.exit(1)
    items = json.load(open(path))
    main_items = {r["id"]: r for r in json.load(open(os.path.join(
        DEFAULT_DATA_DIR, "arith_dataset_numeric_counterfactual.json")))}

    print("1. the operator never changes (an operator-reading unit gets zero)")
    ops_equal = all(parse_problem(r["x"])[1] == parse_problem(r["x_prime"])[1]
                    for r in items)
    check("operators identical in x and x'", ops_equal, f"n={len(items)}")
    check("every item is a single subtraction",
          all(parse_problem(r["x"])[1] == ["-"] for r in items))

    print("2. the swap negates the answer exactly (a one-bit contrast)")
    check("y' == -y for every item",
          all(r["ground_truth_prime"] == -r["ground_truth"] for r in items))
    check("|y'| == |y|, so only the sign differs",
          all(abs(r["ground_truth_prime"]) == abs(r["ground_truth"]) for r in items))
    check("y is positive and y' negative in every item",
          all(r["ground_truth"] > 0 > r["ground_truth_prime"] for r in items))

    print("3. x' is a reordering of x")
    ok_swap, same_len = True, 0
    for r in items:
        a = parse_problem(r["x"])[0]
        b = parse_problem(r["x_prime"])[0]
        ok_swap &= (b == [a[1], a[0]])
        same_len += len(r["x"]) == len(r["x_prime"])
    check("x' swaps the two operands of x", ok_swap)
    check("x and x' have the same character length", same_len == len(items),
          f"{same_len}/{len(items)}")

    print("4. answers are correct and items are eligible")
    check("y is the true answer to x",
          all(parse_problem(r["x"])[0][0] - parse_problem(r["x"])[0][1]
              == r["ground_truth"] for r in items))
    check("y' is the true answer to x'",
          all(parse_problem(r["x_prime"])[0][0] - parse_problem(r["x_prime"])[0][1]
              == r["ground_truth_prime"] for r in items))
    check("the y strings match the integers",
          all(r["y"] == f" {r['ground_truth']}" and
              r["y_prime"] == f" {r['ground_truth_prime']}" for r in items))
    check("no item has equal operands (which would give y' == y)",
          all(parse_problem(r["x"])[0][0] != parse_problem(r["x"])[0][1]
              for r in items))

    print("5. provenance")
    check("x is unchanged from the main dataset",
          all(r["x"] == main_items[r["id"]]["x"] for r in items))
    check("x' differs from the main (operator-flipped) x'",
          all(r["x_prime"] != main_items[r["id"]]["x_prime"] for r in items))
    check("every eligible item in the source dataset is included",
          len(items) == sum(1 for r in main_items.values()
                            if eligible(*parse_problem(r["x"]))),
          f"n={len(items)}")
    check("rebuilding an item is deterministic",
          all(build_item(main_items[r["id"]])["x_prime"] == r["x_prime"]
              for r in items[:200]))
    check("render(parse(x)) round-trips",
          all(render(*parse_problem(r["x"])) == r["x"].strip() for r in items))

    print("6. overlap machinery")
    rng = np.random.default_rng(0)
    N, K = 200000, 2000
    mean, lo, hi = chance_jaccard(N, K)
    sim = []
    for _ in range(200):
        a = rng.choice(N, K, replace=False)
        b = rng.choice(N, K, replace=False)
        i = len(np.intersect1d(a, b))
        sim.append(i / (2 * K - i))
    check("analytic chance matches simulation",
          abs(mean - np.mean(sim)) < 0.0005, f"{mean:.4f} vs {np.mean(sim):.4f}")
    check("simulated values fall inside the analytic 95% interval",
          np.mean((np.array(sim) >= lo) & (np.array(sim) <= hi)) > 0.9)

    n_units = 4096
    K2 = max(1, int(round(0.01 * n_units)))
    noise = rng.normal(0, 1, (400, n_units)).astype(np.float16)
    structured = noise.astype(np.float32).copy()
    structured[:, np.arange(100, 160)] += (100.0 - np.arange(60))[None, :]
    c_noise = split_half_ceiling(noise, K2, seed=0)
    c_struct = split_half_ceiling(structured.astype(np.float16), K2, seed=0)
    check("split-half ceiling is near chance for pure noise", c_noise < 0.05,
          f"{c_noise:.3f}")
    check("split-half ceiling is high for a structured aggregate", c_struct > 0.9,
          f"{c_struct:.3f}")
    check("jaccard of a set with itself is 1",
          jaccard(topk_abs(structured.mean(0), K2),
                  topk_abs(structured.mean(0), K2)) == 1.0)

    print()
    if FAIL:
        print(f"FAILED: {len(FAIL)} check(s): {FAIL}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
