"""Self-test for the cross-fitting logic, on synthetic tensors.

The real script reads per-item attribution tensors that only exist on the
cluster, so this test validates the logic locally before a cluster run:

  1. assign_folds partitions the ids exactly once each, near-evenly.
  2. fold_sums decomposes the aggregate exactly: total/n reproduces the plain
     mean, and (total - fold_f)/n_est reproduces the leave-fold-out mean.
  3. reference_loading matches a literal transcription of run_overlap.py:245.
  4. Behavioural: a unit driven into the circuit by items of ONE fold appears
     in the full-data circuit but NOT in that fold's held-out circuit, i.e.
     held-out items genuinely cannot influence the circuit that scores them.

Usage (local): python heldout/scripts/selftest_crossfit.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crossfit_circuits import (assign_folds, fold_sums, reference_loading,
                               topk_indices_abs, jaccard_sets)

FAIL = []


def check(name, cond, detail=""):
    print(f"  [{'ok ' if cond else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAIL.append(name)


def main():
    rng = np.random.default_rng(0)
    k, n_ids, n_units = 10, 2000, 4096
    ids = [f"prob_{i:06d}" for i in range(n_ids)]

    print("1. fold assignment")
    fold_of = assign_folds(ids, k, seed=42)
    sizes = np.bincount(np.array(list(fold_of.values())), minlength=k)
    check("every id assigned exactly once", len(fold_of) == n_ids)
    check("folds near-even", sizes.max() - sizes.min() <= 1, f"sizes={sizes.tolist()}")
    check("assignment is deterministic", fold_of == assign_folds(ids, k, seed=42))

    print("2. fold-sum decomposition")
    kept = ids[:1700]                       # numeric kept subset, as in real data
    flat = rng.normal(0, 1, (len(kept), n_units)).astype(np.float16)
    row_fold = np.array([fold_of[i] for i in kept], dtype=np.int64)
    fs = fold_sums(flat, row_fold, k, chunk=256)
    total = fs.sum(axis=0)

    direct_mean = flat.astype(np.float32).mean(axis=0)
    check("total/n == plain mean",
          np.allclose(total / len(kept), direct_mean, rtol=1e-4, atol=1e-6),
          f"max|diff|={np.max(np.abs(total / len(kept) - direct_mean)):.2e}")

    worst = 0.0
    for f in range(k):
        mask = row_fold != f
        want = flat[mask].astype(np.float32).mean(axis=0)
        got = (total - fs[f]) / int(mask.sum())
        worst = max(worst, float(np.max(np.abs(want - got))))
    check("leave-fold-out mean exact", worst < 1e-3, f"max|diff|={worst:.2e}")

    print("3. reference loading reproduces run_overlap.py")
    agg = rng.normal(0, 1, n_units).astype(np.float32)
    K = max(1, int(round(0.01 * n_units)))
    S = set(topk_indices_abs(agg, K).tolist())
    vec = flat[0]
    literal = float(vec[list(S)].sum())          # run_overlap.py:245, verbatim
    check("loading matches literal transcription",
          reference_loading(vec, S) == literal, f"value={literal:.6f}")
    check("circuit has exactly K units", len(S) == K, f"K={K}")

    print("4. held-out items cannot influence their own circuit")
    # The aggregate must be WELL-DETERMINED for this check to mean anything:
    # on i.i.d. noise the top-1% of a mean is arbitrary, so any two subsamples
    # would disagree by construction and a Jaccard test would measure nothing.
    # So plant a stable signal shared by all items -- 60 "true circuit" units
    # with separated magnitudes 100, 99, ... 41 -- on top of the noise. The
    # top-K of that signal is stable under dropping any one fold.
    n_true = 60
    true_units = np.arange(200, 200 + n_true)
    structured = flat.astype(np.float32).copy()
    structured[:, true_units] += (100.0 - np.arange(n_true))[None, :]

    # unit U is driven ONLY by the items of fold 3. Size the plant so that its
    # all-items mean is exactly 90, which lands mid-rank (~11th) inside the
    # top-K of the true signal, while its leave-fold-3-out mean is 0.
    U = 123
    target_rows = np.where(row_fold == 3)[0]
    structured[target_rows, U] += 90.0 * len(kept) / len(target_rows)
    planted = structured.astype(np.float16)

    fs2 = fold_sums(planted, row_fold, k, chunk=256)
    total2 = fs2.sum(axis=0)
    S_full = set(topk_indices_abs(total2 / len(kept), K).tolist())
    n_est = int((row_fold != 3).sum())
    S_heldout = set(topk_indices_abs((total2 - fs2[3]) / n_est, K).tolist())
    check("planted unit IS in the full-data circuit", U in S_full)
    check("planted unit is NOT in fold 3's held-out circuit", U not in S_heldout)
    other = set(topk_indices_abs((total2 - fs2[5]) / int((row_fold != 5).sum()),
                                 K).tolist())
    check("planted unit IS in another fold's circuit (control)", U in other)
    jac = jaccard_sets(S_full, S_heldout)
    check("held-out circuit otherwise close to full circuit", jac > 0.8,
          f"Jaccard={jac:.3f}")
    check("held-out circuit is the true circuit minus the planted unit",
          S_heldout == set(true_units[:K].tolist()),
          f"|S_heldout & true|={len(S_heldout & set(true_units.tolist()))}/{K}")

    print()
    if FAIL:
        print(f"FAILED: {len(FAIL)} check(s): {FAIL}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
