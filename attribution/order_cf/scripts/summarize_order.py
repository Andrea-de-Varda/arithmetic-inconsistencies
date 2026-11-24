"""Summarize the operand-order counterfactual (local; small CSVs only).

Three tables:
  1. overlap of the order circuit with the main operator circuit, against
     chance and against the split-half ceiling;
  2. whether the order circuit predicts verbal accuracy as the operator
     circuit does (the paper's item-level result, recomputed);
  3. the causal test: how much of the model's preference each circuit
     restores when patched into order-swapped prompts. The operator circuit
     is the transfer of interest; the order circuit is the ceiling, i.e. what
     the best circuit for this contrast achieves; random units are the floor.
     Without the ceiling the transfer number cannot be read.

Usage (local):  python order_cf/scripts/summarize_order.py
"""

import argparse
import csv
import glob
import json
import os

import numpy as np
from scipy.stats import wilcoxon

# A model that answers `33 - 70 =` with `37` has the same behaviour under both
# orders, so the swap changes nothing and there is no preference to restore.
# This is not inaccuracy: those models are at ~100% on the clean prompt, they
# simply never emit a negative number. Their circuit is then estimated from a
# handful of surviving items and is uninterpretable, so they are excluded from
# the aggregates and reported separately.
MAX_SAME_ANSWER = 0.20

SOURCES = [("patching", "real", "operator circuit"),
           ("patching_own", "real", "order circuit (ceiling)"),
           ("patching", "random", "random units")]
COLS = [s[2] for s in SOURCES]


def read_many(pattern):
    rows = []
    for p in sorted(glob.glob(pattern)):
        rows.extend(list(csv.DictReader(open(p))))
    return rows


def f(row, key):
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return float("nan")


def behaviour(res):
    """Per model: how often the swap leaves the answer unchanged, and how often
    the model emits a negative number at all."""
    out = {}
    for p in sorted(glob.glob(os.path.join(res, "responses", "*", "numeric.json"))):
        m = p.split(os.sep)[-2]
        d = json.load(open(p))
        r = d["records"]
        out[m] = {
            "same": sum(1 for x in r
                        if x["model_clean_response"] == x["model_corrupted_response"]) / len(r),
            "neg": sum(1 for x in r if "-" in str(x["model_corrupted_response"])) / len(r),
            "clean_acc": d["summary"]["clean_accuracy"],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args()
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    res = args.results_dir or os.path.join(pkg_root, "results")

    ov = read_many(os.path.join(res, "per_model", "*", "circuit_overlap.csv")) \
        or read_many(os.path.join(res, "circuit_overlap.csv"))
    it = read_many(os.path.join(res, "per_model", "*", "item_level_transfer.csv")) \
        or read_many(os.path.join(res, "item_level_transfer.csv"))

    beh = behaviour(res)
    def excluded(model_id):
        m = model_id.replace("/", "_").replace(".", "-")
        return beh.get(m, {}).get("same", 0.0) > MAX_SAME_ANSWER

    if beh:
        bad = [m for m in beh if beh[m]["same"] > MAX_SAME_ANSWER]
        if bad:
            print("\n" + "=" * 92)
            print("Excluded: the order swap does not change the model's answer")
            print("=" * 92)
            for m in sorted(bad):
                print(f"  {m:28s} same answer under both orders "
                      f"{100*beh[m]['same']:5.1f}%   emits a minus sign on "
                      f"{100*beh[m]['neg']:5.1f}% of swapped prompts   "
                      f"(clean accuracy {100*beh[m]['clean_acc']:.1f}%)")

    if ov:
        print("\n" + "=" * 92)
        print("Order circuit vs operator circuit (numeric, top 1% of MLP units)")
        print("=" * 92)
        print(f"{'model':28s} {'items':>6s} {'Jaccard':>8s} {'chance':>7s} "
              f"{'x chance':>9s} {'ceiling':>8s} {'% of ceiling':>12s}")
        for r in ov:
            mark = " *" if excluded(r["model"]) else ""
            print(f"{r['model']+mark:28s} {int(f(r,'n_kept_order')):6d} "
                  f"{f(r,'jaccard_order_vs_operator'):8.3f} {f(r,'chance'):7.4f} "
                  f"{f(r,'ratio_to_chance'):8.0f}x {f(r,'ceiling'):8.3f} "
                  f"{100*f(r,'fraction_of_ceiling'):11.0f}%")
        inc = [r for r in ov if not excluded(r["model"])]
        j = np.array([f(r, "jaccard_order_vs_operator") for r in inc])
        ch = np.array([f(r, "chance") for r in inc])
        fc = np.array([f(r, "fraction_of_ceiling") for r in inc])
        print(f"\n  median Jaccard {np.nanmedian(j):.3f} "
              f"(range {np.nanmin(j):.3f}-{np.nanmax(j):.3f}); "
              f"median {np.nanmedian(j/ch):.0f}x chance; "
              f"median {100*np.nanmedian(fc):.0f}% of the split-half ceiling "
              f"(n={len(inc)} models; * excluded)")

    if it:
        print("\n" + "=" * 92)
        print("Item-level prediction of verbal accuracy: order vs operator circuit")
        print("=" * 92)
        for fmt in ("english", "spanish", "italian"):
            cells = [r for r in it if r["format"] == fmt and not int(r["low_n"])]
            if not cells:
                continue
            a = np.array([f(r, "rpb_order") for r in cells])
            b = np.array([f(r, "rpb_operator") for r in cells])
            ua = np.array([f(r, "auroc_order") for r in cells])
            sig = sum(f(r, "p_order") < 0.05 and f(r, "rpb_order") > 0 for r in cells)
            p = wilcoxon(a, b).pvalue if len(a) > 1 else float("nan")
            print(f"  {fmt:8s} n_models={len(cells):2d}  median r_pb order "
                  f"{np.median(a):+.3f} vs operator {np.median(b):+.3f} "
                  f"(Wilcoxon p={p:.3f})  median AUROC {np.median(ua):.3f}  "
                  f"significant positive {sig}/{len(cells)}")

    by = {}
    for sub, cond, label in SOURCES:
        for p in sorted(glob.glob(os.path.join(
                res, sub, "*", f"denoise__{cond}", "numeric.json"))):
            d = json.load(open(p))
            eff = np.array([r["effect"] for r in d["records"]
                            if not r["small_denominator"] and np.isfinite(r["effect"])])
            by.setdefault(d["summary"]["model"], {})[label] = \
                float(eff.mean()) if len(eff) else float("nan")
    if by:
        print("\n" + "=" * 92)
        print("Causal test: preference restored when patching into order-swapped prompts")
        print("=" * 92)
        print(f"{'model':28s}" + "".join(f"{c:>26s}" for c in COLS))
        for m in sorted(by):
            mark = " *" if excluded(m) else ""
            print(f"{m+mark:28s}" + "".join(
                f"{by[m].get(c, float('nan')):26.3f}" for c in COLS))
        both = [m for m in by
                if COLS[0] in by[m] and COLS[1] in by[m] and not excluded(m)]
        if both:
            tr = np.array([by[m][COLS[0]] for m in both])
            ce = np.array([by[m][COLS[1]] for m in both])
            fl = np.array([by[m].get(COLS[2], np.nan) for m in both])
            print(f"\n  transfer {tr.mean():.3f} vs ceiling {ce.mean():.3f} vs "
                  f"random {np.nanmean(fl):+.3f}  (n={len(both)} models)")
            print(f"  the operator circuit reaches a median "
                  f"{100*np.median(tr/ce):.0f}% of the ceiling (min "
                  f"{100*(tr/ce).min():.0f}%)")
            if len(both) > 1:
                print(f"  transfer vs ceiling: Wilcoxon p="
                      f"{wilcoxon(tr, ce).pvalue:.3f};  vs random: p="
                      f"{wilcoxon(tr, fl).pvalue:.4f}")
            if ce.mean() < 0.05:
                print("  WARNING: the ceiling itself is near zero, so this contrast "
                      "saturates the\n           patching method and the transfer "
                      "number carries no information.")

    if ov:
        out = os.path.join(res, "summary.csv")
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(ov[0]), extrasaction="ignore")
            w.writeheader()
            w.writerows(ov)
        print(f"\nwrote {os.path.relpath(out)}")
    elif not by:
        print("nothing found; run the cluster array first")


if __name__ == "__main__":
    main()
