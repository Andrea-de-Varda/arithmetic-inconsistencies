"""Summarize the cross-fitted loadings (local; reads only the small CSVs).

Per (model, format): item counts, point-biserial r of accuracy with the
published (full-S) and the cross-fitted loading, and the direct predictive
metrics for the cross-fitted loading as a parameter-free score of
correctness on held-out verbal problems: AUROC and average precision
(with the base rate alongside, since AP is read against it). Also the mean
Jaccard between the per-fold circuits and the full circuit.

Cells with fewer than 30 correct items are kept but flagged (`low_n`), the
same rule as the main text.

Writes heldout/results/summary.csv and prints the table plus cross-model
medians. Usage (local):
  python heldout/scripts/summarize_crossfit.py [--results-dir heldout/results]
"""

import argparse
import csv
import glob
import os

import numpy as np
from scipy.stats import pointbiserialr
from sklearn.metrics import roc_auc_score, average_precision_score

VERBAL_FORMATS = ("english", "spanish", "italian")
MIN_CORRECT = 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args()
    cf_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    res = args.results_dir or os.path.join(cf_root, "results")

    out_rows = []
    for ldir in sorted(glob.glob(os.path.join(res, "*", "loadings.csv"))):
        mdir = os.path.dirname(ldir)
        rows = list(csv.DictReader(open(ldir)))
        stab = list(csv.DictReader(open(os.path.join(mdir, "fold_stability.csv"))))
        mean_jac = float(np.mean([float(r["jaccard_with_full"]) for r in stab]))
        mid = rows[0]["model"]
        for fmt in VERBAL_FORMATS:
            sub = [r for r in rows if r["format"] == fmt]
            if not sub:
                continue
            y = np.array([int(r["accuracy"]) for r in sub])
            lf = np.array([float(r["loading_full"]) for r in sub])
            lc = np.array([float(r["loading_crossfit"]) for r in sub])
            n, nc = len(y), int(y.sum())
            row = {"model": mid, "format": fmt, "n": n, "n_correct": nc,
                   "base_rate": nc / n, "low_n": int(nc < MIN_CORRECT),
                   "mean_fold_jaccard": mean_jac}
            if 0 < nc < n:
                rf, pf = pointbiserialr(y, lf)
                rc, pc = pointbiserialr(y, lc)
                row.update({"rpb_full": rf, "p_full": pf, "rpb_crossfit": rc, "p_crossfit": pc,
                            "auroc_full": roc_auc_score(y, lf),
                            "auroc_crossfit": roc_auc_score(y, lc),
                            "ap_crossfit": average_precision_score(y, lc)})
            else:
                row.update({k: float("nan") for k in
                            ("rpb_full", "p_full", "rpb_crossfit", "p_crossfit", "auroc_full",
                             "auroc_crossfit", "ap_crossfit")})
            out_rows.append(row)

    fields = ["model", "format", "n", "n_correct", "base_rate", "low_n",
              "rpb_full", "p_full", "rpb_crossfit", "p_crossfit", "auroc_full", "auroc_crossfit",
              "ap_crossfit", "mean_fold_jaccard"]
    outp = os.path.join(res, "summary.csv")
    with open(outp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {os.path.relpath(outp)}  ({len(out_rows)} rows)\n")

    print(f"{'model':28s} {'fmt':8s} {'n':>5s} {'corr':>5s}  {'r_full':>7s} {'r_cv':>7s}  "
          f"{'AUC_full':>8s} {'AUC_cv':>7s} {'AP_cv':>6s} {'base':>6s}  {'Jac':>5s}")
    for r in out_rows:
        flag = " *" if r["low_n"] else ""
        print(f"{r['model']:28s} {r['format']:8s} {r['n']:5d} {r['n_correct']:5d}  "
              f"{r['rpb_full']:+7.3f} {r['rpb_crossfit']:+7.3f}  {r['auroc_full']:8.3f} "
              f"{r['auroc_crossfit']:7.3f} {r['ap_crossfit']:6.3f} {r['base_rate']:6.3f}  "
              f"{r['mean_fold_jaccard']:5.3f}{flag}")
    print("\n* fewer than 30 correct items")

    print("\ncross-model medians over cells with >= 30 correct items:")
    for fmt in VERBAL_FORMATS:
        ok = [r for r in out_rows if r["format"] == fmt and not r["low_n"]
              and np.isfinite(r["rpb_crossfit"])]
        if not ok:
            continue
        med = lambda k: float(np.median([r[k] for r in ok]))
        n_sig = sum(r["p_crossfit"] < 0.05 and r["rpb_crossfit"] > 0 for r in ok)
        print(f"  {fmt:8s} n_models={len(ok):2d}  r_full={med('rpb_full'):+.3f}  "
              f"r_cv={med('rpb_crossfit'):+.3f}  AUROC_cv={med('auroc_crossfit'):.3f}  "
              f"AP_cv={med('ap_crossfit'):.3f} (base {med('base_rate'):.3f})  "
              f"sig+ {n_sig}/{len(ok)}")


if __name__ == "__main__":
    main()
