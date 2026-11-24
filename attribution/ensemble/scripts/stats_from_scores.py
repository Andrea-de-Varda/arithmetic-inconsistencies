"""Recompute all predictor-comparison statistics locally from the per-model
per_item_scores.csv files, and build the cross-model summary tables.

per_item_scores.csv (model,format,id,y,x_main,x1a,x1b,x2,x3) under
ensemble/results/<model>/ is the single source of truth, so the full
statistical analysis can be (re)run on a laptop without the cluster.

Per model, writes / overwrites:
  <model>/variance_partition.csv   per (format, predictor): AUROC, AP,
                                   zero-order r, standardized coef + p,
                                   unique ("last") R^2, LMG R^2 share
  <model>/partial_corr.csv         per format: x_main partial r | controls
                                   (+ CI, p), model totals

Across models, writes:
  cross_model_predictors.csv       all per-(model, format, predictor) rows
  cross_model_xmain.csv            all per-(model, format) model rows
  cross_model_summary.csv          per (format, predictor): median AUROC, AP and
                                   LMG share over models with >= 30 correct items,
                                   number of models where the predictor is
                                   significant, and how often the circuit
                                   matches or beats each control on AUROC

Usage (local):
  python ensemble/scripts/stats_from_scores.py
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stats import (analyze_format, PREDICTORS, PER_PRED_FIELDS, MODEL_FIELDS)

VERBAL_FORMATS = ("english", "spanish", "italian")
PRED_NAME = {"x_main": "circuit", "x1a": "probe(resid)", "x1b": "probe(MLP)",
             "x2": "seq logP", "x3": "entropy"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    ens_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    res = args.results_dir or os.path.join(ens_root, "results")

    all_pp, all_mr = [], []
    for scores in sorted(glob.glob(os.path.join(res, "*", "per_item_scores.csv"))):
        mdir = os.path.dirname(scores)
        rows = list(csv.DictReader(open(scores)))
        if not rows or "model" not in rows[0]:
            continue    # skip files without the per-model schema
        mid = rows[0]["model"]
        by_fmt = {}
        for r in rows:
            by_fmt.setdefault(r["format"], []).append(r)

        per_pred_rows, model_rows = [], []
        for fmt in VERBAL_FORMATS:
            if fmt not in by_fmt:
                continue
            recs = by_fmt[fmt]
            y = [int(r["y"]) for r in recs]
            cols = {p: [float(r[p]) for r in recs] for p in PREDICTORS}
            pp, mr = analyze_format(fmt, cols, y, model=mid)
            per_pred_rows.extend(pp)
            model_rows.append(mr)
        _write(os.path.join(mdir, "variance_partition.csv"), PER_PRED_FIELDS,
               per_pred_rows)
        _write(os.path.join(mdir, "partial_corr.csv"), MODEL_FIELDS, model_rows)
        all_pp.extend(per_pred_rows)
        all_mr.extend(model_rows)
        if not args.quiet:
            _print_model(mid, per_pred_rows, model_rows)

    if not all_pp:
        print("no per-model per_item_scores.csv found")
        return
    _write(os.path.join(res, "cross_model_predictors.csv"), PER_PRED_FIELDS, all_pp)
    _write(os.path.join(res, "cross_model_xmain.csv"), MODEL_FIELDS, all_mr)
    summary = cross_model_summary(all_pp)
    _write(os.path.join(res, "cross_model_summary.csv"), SUMMARY_FIELDS, summary)
    _print_summary(summary, all_mr)


SUMMARY_FIELDS = ["format", "predictor", "n_models", "median_auroc",
                  "median_avg_precision", "median_base_rate", "median_lmg_r2",
                  "n_sig_coef", "n_circuit_ge_this"]


def cross_model_summary(all_pp):
    """Medians over the cells with >= 30 correct items."""
    out = []
    for fmt in VERBAL_FORMATS:
        cells = [r for r in all_pp if r["format"] == fmt and not int(r["low_n"])]
        models = sorted({r["model"] for r in cells})
        if not models:
            continue
        by = {(r["model"], r["predictor"]): r for r in cells}
        for p in PREDICTORS:
            rows = [by[(m, p)] for m in models if (m, p) in by]
            circ = [by[(m, "x_main")] for m in models if (m, p) in by]
            ge = sum(float(c["auroc"]) >= float(r["auroc"]) - 1e-9
                     for c, r in zip(circ, rows))
            out.append({
                "format": fmt, "predictor": p, "n_models": len(rows),
                "median_auroc": float(np.nanmedian([float(r["auroc"]) for r in rows])),
                "median_avg_precision": float(np.nanmedian(
                    [float(r["avg_precision"]) for r in rows])),
                "median_base_rate": float(np.median(
                    [float(r["n_correct"]) / float(r["n"]) for r in rows])),
                "median_lmg_r2": float(np.nanmedian(
                    [float(r["lmg_r2_linear"]) for r in rows])),
                "n_sig_coef": int(sum(float(r["lin_p"]) < 0.05 and
                                      float(r["lin_coef"]) * (-1 if p == "x3" else 1) > 0
                                      for r in rows)),
                "n_circuit_ge_this": int(ge),
            })
    return out


def _print_model(mid, per_pred_rows, model_rows):
    print("\n" + "=" * 96)
    print(f"{mid}   (* p<.05, ** p<.01, *** p<.001; AUROC/AP are for the "
          f"predictor alone, oriented)")
    print("=" * 96)
    for mr in model_rows:
        fmt = mr["format"]
        print(f"\n[{fmt}]  n={mr['n']}  correct={mr['n_correct']}"
              f"{'  LOW-N' if mr['low_n'] else ''}   "
              f"linear R2_full={mr['r2_full_linear']:.3f}   "
              f"partial r(x_main|controls)={mr['partial_r_xmain']:+.3f} "
              f"(p={mr['partial_p_xmain']:.1e})")
        print("  " + "predictor".ljust(14) + "AUROC".ljust(8) + "AP".ljust(8)
              + "coef (p)".ljust(24) + "uniqR2".ljust(9) + "lmgR2")
        for r in [r for r in per_pred_rows if r["format"] == fmt]:
            lin = f"{r['lin_coef']:+.3f} ({_stars(r['lin_p'])})"
            print("  " + PRED_NAME[r["predictor"]].ljust(14)
                  + f"{r['auroc']:.3f}".ljust(8) + f"{r['avg_precision']:.3f}".ljust(8)
                  + lin.ljust(24) + f"{r['unique_r2_linear']:.4f}".ljust(9)
                  + f"{r['lmg_r2_linear']:.4f}")


def _print_summary(summary, all_mr):
    print("\n" + "=" * 96)
    print("Cross-model summary (medians over models with >= 30 correct items)")
    print("=" * 96)
    for fmt in VERBAL_FORMATS:
        rows = [r for r in summary if r["format"] == fmt]
        if not rows:
            continue
        print(f"\n[{fmt}]  n_models={rows[0]['n_models']}  "
              f"median base rate={rows[0]['median_base_rate']:.3f}")
        print("  " + "predictor".ljust(14) + "med AUROC".ljust(11) + "med AP".ljust(9)
              + "med LMG".ljust(9) + "sig coef".ljust(10) + "circuit >= this")
        for r in rows:
            print("  " + PRED_NAME[r["predictor"]].ljust(14)
                  + f"{r['median_auroc']:.3f}".ljust(11)
                  + f"{r['median_avg_precision']:.3f}".ljust(9)
                  + f"{r['median_lmg_r2']:.3f}".ljust(9)
                  + f"{r['n_sig_coef']}/{r['n_models']}".ljust(10)
                  + f"{r['n_circuit_ge_this']}/{r['n_models']}")
        ok = [m for m in all_mr if m["format"] == fmt and not m["low_n"]]
        n_sig = sum(m["partial_p_xmain"] < 0.05 and m["partial_r_xmain"] > 0 for m in ok)
        print(f"  circuit partial r | controls: median "
              f"{np.median([m['partial_r_xmain'] for m in ok]):+.3f}, "
              f"significant positive in {n_sig}/{len(ok)} models")
    print()


def _stars(p):
    if p != p:
        return "n/a"
    s = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"
    return f"{p:.1e}{s}"


def _write(path, fields, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {os.path.relpath(path)}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
