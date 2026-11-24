"""Self-test for the matched random baseline, on synthetic tensors.

The real script reads per-item attribution tensors that only exist on the
cluster, so this validates the logic locally before a cluster run:

  1. draw_matched_sets: every draw has S's per-layer counts, no duplicates,
     no unit of S, and is uniform over the complement (per-unit inclusion
     frequency matches c_L / (d - c_L)).
  2. set_loadings (sparse float32 product) equals a literal gather-and-sum,
     and its row 0 equals reference_loading on S.
  3. rpb_columns equals scipy.stats.pointbiserialr column by column.
  4. Behavioural, the two cases the control must tell apart:
     a. localized: correct items get extra attribution on S only
        -> real r_pb high, random draws ~0.
     b. global scale: correct items have all attributions scaled up
        (the size confound) -> random draws predict accuracy too.

Usage (local): python matched_baseline/scripts/selftest_matched.py
"""

import os
import sys

import numpy as np
from scipy.stats import chisquare, pointbiserialr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from random_matched_loadings import (draw_matched_sets, indicator_matrix,
                                     set_loadings, rpb_columns,
                                     reference_loading, topk_indices_abs)

FAIL = []


def check(name, cond, detail=""):
    print(f"  [{'ok ' if cond else 'FAIL'}] {name}"
          f"{'  ' + detail if detail else ''}")
    if not cond:
        FAIL.append(name)


def make_S(rng, n_layers, d, K):
    # layer-skewed importance so S is unevenly spread across layers
    scale = np.linspace(0.3, 3.0, n_layers)[:, None]
    agg = rng.standard_normal((n_layers, d)) * scale
    return np.sort(topk_indices_abs(agg.ravel(), K)).astype(np.int64)


def main():
    rng = np.random.default_rng(0)
    n_layers, d = 8, 500
    n_units = n_layers * d
    K = int(round(0.01 * n_units)) * 5          # 200 units, denser than 1%
    S = make_S(rng, n_layers, d, K)
    counts_S = np.bincount(S // d, minlength=n_layers)

    print("1. sampler")
    n_draws = 4000
    sets, counts, n_short = draw_matched_sets(S, n_layers, d, n_draws,
                                              np.random.default_rng(1))
    check("shape", sets.shape == (n_draws, K))
    check("counts equal S's per-layer counts", np.array_equal(counts, counts_S))
    check("every draw layer-matched",
          all(np.array_equal(np.bincount(s // d, minlength=n_layers), counts_S)
              for s in sets))
    check("no duplicates", all(np.unique(s).size == K for s in sets))
    check("disjoint from S", not np.isin(sets, S).any())
    check("no unmatched layers", n_short == 0)
    freq = np.bincount(sets.ravel(), minlength=n_units)
    in_S = np.isin(np.arange(n_units), S)
    pvals = []
    for L in range(n_layers):
        if counts_S[L] == 0:
            continue
        comp = np.arange(L * d, (L + 1) * d)[~in_S[L * d:(L + 1) * d]]
        pvals.append(chisquare(freq[comp]).pvalue)
    check("uniform over complement (per-layer chi-square)",
          min(pvals) > 0.001 / len(pvals),
          f"min p = {min(pvals):.3f} over {len(pvals)} layers")
    check("layers without S units never sampled",
          freq[counts_S[np.arange(n_units) // d] == 0].sum() == 0)
    same_seed, _, _ = draw_matched_sets(S, n_layers, d, 5,
                                        np.random.default_rng(1))
    check("deterministic given seed", np.array_equal(same_seed, sets[:5]))

    print("2. loadings")
    n_items = 300
    X = (rng.standard_normal((n_items, n_units)) * 0.01).astype(np.float16)
    M = indicator_matrix(np.vstack([S[None, :], sets[:50]]), n_units)
    Lm = set_loadings(X, M, chunk=37)
    X32 = X.astype(np.float32)
    explicit = np.stack([X32[:, S].sum(1)] +
                        [X32[:, s].sum(1) for s in sets[:50]], axis=1)
    check("sparse product == gather-and-sum",
          np.allclose(Lm, explicit, atol=1e-5),
          f"max diff {np.abs(Lm - explicit).max():.2e}")
    ref = np.array([reference_loading(X[i], S) for i in range(n_items)])
    check("row 0 == fp16 reference loading on S",
          np.allclose(Lm[:, 0], ref, atol=1e-3),
          f"max diff {np.abs(Lm[:, 0] - ref).max():.2e}")

    print("3. r_pb")
    acc = (rng.random(n_items) < 0.4).astype(int)
    r_vec = rpb_columns(acc, Lm)
    r_sci = np.array([pointbiserialr(acc, Lm[:, j])[0]
                      for j in range(Lm.shape[1])])
    check("vectorized == scipy", np.allclose(r_vec, r_sci, atol=1e-10),
          f"max diff {np.abs(r_vec - r_sci).max():.2e}")

    print("4. behaviour")
    n_items = 800
    acc = (rng.random(n_items) < 0.4).astype(int)
    M = indicator_matrix(np.vstack([S[None, :], sets[:500]]), n_units)
    base = rng.standard_normal((n_items, n_units)).astype(np.float32) * 0.01 \
        + 0.002                                   # small clean-direction bias
    # a. localized: correct items get extra attribution on S only
    loc = base.copy()
    loc[np.ix_(acc == 1, S)] += 0.003
    r = rpb_columns(acc, set_loadings(loc.astype(np.float16), M, 128))
    null = r[1:]
    check("localized: real r_pb high", r[0] > 0.3, f"r_real = {r[0]:+.3f}")
    check("localized: random draws ~ 0", abs(null.mean()) < 0.05,
          f"null mean = {null.mean():+.3f}, 97.5% = "
          f"{np.percentile(null, 97.5):+.3f}")
    # b. global scale: every attribution of correct items is scaled up
    glob = base.copy()
    glob[acc == 1] *= 1.6
    r = rpb_columns(acc, set_loadings(glob.astype(np.float16), M, 128))
    null = r[1:]
    check("global scale: random draws also predict accuracy",
          null.mean() > 0.3,
          f"r_real = {r[0]:+.3f}, null mean = {null.mean():+.3f}")
    check("global scale: real r_pb inside the null",
          np.percentile(null, 2.5) <= r[0] <= np.percentile(null, 97.5) + 0.02)

    print(f"\n{'ALL CHECKS PASSED' if not FAIL else 'FAILED: ' + ', '.join(FAIL)}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
