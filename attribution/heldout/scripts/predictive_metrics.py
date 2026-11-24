"""Held-out predictive metrics for circuit loading (local, CPU).

Turns the cross-fitted loadings into direct predictive metrics (point-biserial
r, AUROC, average precision), alongside the published (in-sample-circuit) numbers so the two are
directly comparable.

Estimators, per (model, verbal format):
  published   loading on the published numeric circuit, built on ALL 2,000
              numeric items (the paper's `sum_on_S`)
  crossfit    loading on a numeric circuit built WITHOUT the item's own fold
              (fully out-of-sample; heldout/results/crossfit_item_loadings.csv)
  balanced    Llama-3.1-8B only: the accuracy-balanced control (paper
              Appendices I-J), whose numeric and verbal folds are sampled
              independently from a separate 20,000-item dataset
  balanced_disjoint
              the same, restricted to verbal items whose id is NOT in that
              model's numeric balanced fold (a second, independent-dataset
              route to a disjoint arithmetic set)

Metrics: point-biserial r (the paper's statistic), AUROC and average precision
(AP) against the item's base rate. AUROC is the direct discrimination metric;
AP is reported alongside because Italian base rates fall below 10%, where AUROC
alone is easy to over-read. `reliable` flags cells with >= 30 correct items,
the threshold already used in the main text.

Reads only small git-tracked CSVs. Writes heldout/results/predictive_metrics.csv
and prints a summary table.

Usage (local): python heldout/scripts/predictive_metrics.py
"""

import argparse
import csv
import os

import numpy as np
from scipy.stats import pointbiserialr
from sklearn.metrics import average_precision_score, roc_auc_score

VERBAL = ("english", "spanish", "italian")
MIN_CORRECT = 30
FIELDS = ["model", "format", "estimator", "n", "n_correct", "base_rate",
          "rpb", "p", "auroc", "ap", "ap_over_base_rate", "reliable"]


def cell_metrics(model, fmt, estimator, y, x):
    y = np.asarray(y, dtype=int)
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    y, x = y[ok], x[ok]
    n_correct = int(y.sum())
    base = float(y.mean()) if len(y) else float("nan")
    row = {"model": model, "format": fmt, "estimator": estimator,
           "n": len(y), "n_correct": n_correct, "base_rate": base,
           "reliable": int(n_correct >= MIN_CORRECT and len(y) - n_correct >= 1)}
    if len(set(y.tolist())) < 2:
        row.update({k: float("nan")
                    for k in ("rpb", "p", "auroc", "ap", "ap_over_base_rate")})
        return row
    r, p = pointbiserialr(y, x)
    ap = float(average_precision_score(y, x))
    row.update({"rpb": float(r), "p": float(p),
                "auroc": float(roc_auc_score(y, x)), "ap": ap,
                "ap_over_base_rate": ap / base if base > 0 else float("nan")})
    return row


def rows_from_crossfit(path):
    by = {}
    for r in csv.DictReader(open(path)):
        key = (r["model"], r["format"])
        by.setdefault(key, {"y": [], "full": [], "cf": []})
        by[key]["y"].append(int(r["accuracy"]))
        by[key]["full"].append(float(r["loading_full"]))
        by[key]["cf"].append(float(r["loading_crossfit"]))
    out = []
    for (mid, fmt), d in sorted(by.items()):
        out.append(cell_metrics(mid, fmt, "published", d["y"], d["full"]))
        out.append(cell_metrics(mid, fmt, "crossfit", d["y"], d["cf"]))
    return out


def rows_from_balanced(attribution_root, model="meta-llama/Llama-3.1-8B"):
    short = model.replace("/", "_").replace(".", "-")
    base = os.path.join(attribution_root, "control_balanced", "results")
    pred = os.path.join(base, "overlap", short,
                        "item_level_predictions_neuron.csv")
    numf = os.path.join(base, "balanced_fold", short, "numeric.json")
    if not os.path.exists(pred):
        return []
    import json
    numeric_ids = set(json.load(open(numf))) if os.path.exists(numf) else set()
    by = {}
    for r in csv.DictReader(open(pred)):
        if r["format"] not in VERBAL:
            continue
        by.setdefault(r["format"], []).append(
            (r["id"], int(r["accuracy"]), float(r["sum_on_S"])))
    out = []
    for fmt in VERBAL:
        recs = by.get(fmt)
        if not recs:
            continue
        out.append(cell_metrics(model, fmt, "balanced",
                                [a for _, a, _ in recs],
                                [s for _, _, s in recs]))
        dis = [(a, s) for i, a, s in recs if i not in numeric_ids]
        if dis:
            out.append(cell_metrics(model, fmt, "balanced_disjoint",
                                    [a for a, _ in dis], [s for _, s in dis]))
    return out


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--results-dir", default=None,
                     help="default: attribution/heldout/results")
    args = ap_.parse_args()

    heldout_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    attribution_root = os.path.dirname(heldout_root)
    res = args.results_dir or os.path.join(heldout_root, "results")

    rows = []
    cf_path = os.path.join(res, "crossfit_item_loadings.csv")
    if os.path.exists(cf_path):
        rows.extend(rows_from_crossfit(cf_path))
    else:
        print(f"note: {cf_path} not found -- run crossfit_circuits.py on the "
              f"cluster first; reporting the balanced control only")
    rows.extend(rows_from_balanced(attribution_root))

    if not rows:
        print("nothing to report")
        return

    out = os.path.join(res, "predictive_metrics.csv")
    os.makedirs(res, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}  ({len(rows)} rows)\n")

    # --- summary: published vs crossfit, reliable cells only ---------------
    for est in ("published", "crossfit"):
        sel = [r for r in rows if r["estimator"] == est and r["reliable"]]
        if not sel:
            continue
        print(f"{est:>10s}  (n = {len(sel)} reliable cells)")
        for fmt in VERBAL:
            g = [r for r in sel if r["format"] == fmt]
            if not g:
                continue
            print(f"    {fmt:8s}  median r_pb = "
                  f"{np.median([r['rpb'] for r in g]):+.3f}   "
                  f"median AUROC = {np.median([r['auroc'] for r in g]):.3f}   "
                  f"median AP = {np.median([r['ap'] for r in g]):.3f}   "
                  f"(n = {len(g)})")
    # --- paired published vs crossfit -------------------------------------
    pub = {(r["model"], r["format"]): r for r in rows
           if r["estimator"] == "published"}
    cf = {(r["model"], r["format"]): r for r in rows
          if r["estimator"] == "crossfit"}
    keys = [k for k in pub if k in cf and pub[k]["reliable"] and cf[k]["reliable"]]
    if keys:
        d_r = np.array([cf[k]["rpb"] - pub[k]["rpb"] for k in keys])
        d_a = np.array([cf[k]["auroc"] - pub[k]["auroc"] for k in keys])
        print(f"\npaired crossfit - published over {len(keys)} reliable cells")
        print(f"    delta r_pb : mean {d_r.mean():+.4f}  "
              f"max|delta| {np.abs(d_r).max():.4f}")
        print(f"    delta AUROC: mean {d_a.mean():+.4f}  "
              f"max|delta| {np.abs(d_a).max():.4f}")
        if len(keys) > 5:
            from scipy.stats import pearsonr, wilcoxon
            r, p = pearsonr([pub[k]["rpb"] for k in keys],
                            [cf[k]["rpb"] for k in keys])
            print(f"    r(published, crossfit) over cells = {r:.3f} (p = {p:.2g})")
            print(f"    paired Wilcoxon on r_pb: p = "
                  f"{wilcoxon(d_r).pvalue:.3g} (want: not significant)")

    bal = [r for r in rows if r["estimator"].startswith("balanced")]
    if bal:
        print("\naccuracy-balanced control (Llama-3.1-8B, separate dataset)")
        for r in bal:
            print(f"    {r['estimator']:18s} {r['format']:8s} n = {r['n']:5d}  "
                  f"r_pb = {r['rpb']:+.3f}  AUROC = {r['auroc']:.3f}  "
                  f"AP = {r['ap']:.3f}")


if __name__ == "__main__":
    main()
