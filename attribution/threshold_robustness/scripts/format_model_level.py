"""Format- and model-level results of the paper at each circuit threshold.

Runs locally: it needs only the aggregate tensors and the response summaries,
which are in the local results tree. For each model and each threshold q:

  overlap_f(q) = Jaccard between the top-q units of the numeric aggregate and
                 the top-q units of the aggregate of verbal format f
                 (the paper's "overlap", run_overlap.py 3a)

Format level (paper §4.2): mean overlap per format across models, and the
number of models in which English > Spanish > Italian holds strictly, and in
which English is the highest of the three.
Model level (paper §4.2): Pearson r across the 13 models between overlap and
clean accuracy, per format.

Jaccard between two top-q sets has a chance level that grows with q
(~ q / (2 - q)), so a chance-corrected overlap (J - J0) / (1 - J0) is also
written. Within a threshold J0 is the same for every model and format, so
the correction does not change the ordering counts or the cross-model r; it
only makes means comparable across thresholds.

Outputs, under threshold_robustness/results/:
  format_overlap.csv          model,q,K,n_units,format,accuracy,jaccard,
                              jaccard_chance,jaccard_corrected
  format_level_summary.csv    q,format,n_models,mean_jaccard,sd_jaccard,
                              mean_jaccard_corrected,sd_jaccard_corrected,
                              n_models_order_holds,n_models_english_highest
  model_level_summary.csv     q,format,n_models,r,p
  format_model_validation.txt reproduction of the published 1% numbers

Usage:
  python threshold_robustness/scripts/format_model_level.py
"""

import argparse
import os
import sys

import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (DEFAULT_OUT_DIR, DEFAULT_RESULTS_DIR, MODELS,
                    PUBLISHED_Q, THRESHOLDS, VERBAL, chance_jaccard,
                    corrected_jaccard, jaccard, load_aggregate,
                    load_overlap_summary, load_responses, n_units_for,
                    q_label, topk_indices_abs, write_csv)

OVERLAP_FIELDS = ["model", "q", "K", "n_units", "format", "accuracy",
                  "jaccard", "jaccard_chance", "jaccard_corrected"]
FORMAT_FIELDS = ["q", "format", "n_models", "mean_jaccard", "sd_jaccard",
                 "mean_jaccard_corrected", "sd_jaccard_corrected",
                 "n_models_order_holds", "n_models_english_highest"]
MODEL_FIELDS = ["q", "format", "n_models", "r", "p"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    rows = []
    for mid in MODELS:
        aggs = {f: load_aggregate(args.results_dir, mid, f)
                for f in ("numeric",) + VERBAL}
        if any(a is None for a in aggs.values()):
            log(f"skip {mid}: aggregate missing")
            continue
        acc = {f: load_responses(args.results_dir, mid, f)["summary"]
               ["clean_accuracy"] for f in VERBAL}
        n_units = aggs["numeric"].size
        published = load_overlap_summary(args.results_dir, mid)
        for q in THRESHOLDS:
            K = n_units_for(q, n_units)
            top_num = topk_indices_abs(aggs["numeric"], K)
            j0 = chance_jaccard(K, n_units)
            for f in VERBAL:
                j = jaccard(top_num, topk_indices_abs(aggs[f], K))
                rows.append({
                    "model": mid, "q": q, "K": K, "n_units": n_units,
                    "format": f, "accuracy": acc[f], "jaccard": j,
                    "jaccard_chance": j0,
                    "jaccard_corrected": corrected_jaccard(j, j0),
                })
                key = (f"jaccard_top{q:.0%}", "numeric", f)
                if q in (0.01, 0.05) and key in published:
                    log(f"[check] {mid} {f} {q_label(q)}: jaccard {j:.6f} vs "
                        f"published {published[key]:.6f} "
                        f"(|diff| = {abs(j - published[key]):.1e})")

    write_csv(os.path.join(args.out_dir, "format_overlap.csv"),
              OVERLAP_FIELDS, rows)

    fmt_rows, model_rows = [], []
    for q in THRESHOLDS:
        cell = [r for r in rows if r["q"] == q]
        by_model = {}
        for r in cell:
            by_model.setdefault(r["model"], {})[r["format"]] = r["jaccard"]
        n_order = sum(v["english"] > v["spanish"] > v["italian"]
                      for v in by_model.values())
        n_en = sum(v["english"] > max(v["spanish"], v["italian"])
                   for v in by_model.values())
        for f in VERBAL:
            sub = [r for r in cell if r["format"] == f]
            j = np.array([r["jaccard"] for r in sub])
            jc = np.array([r["jaccard_corrected"] for r in sub])
            a = np.array([r["accuracy"] for r in sub])
            fmt_rows.append({
                "q": q, "format": f, "n_models": len(sub),
                "mean_jaccard": j.mean(), "sd_jaccard": j.std(ddof=1),
                "mean_jaccard_corrected": jc.mean(),
                "sd_jaccard_corrected": jc.std(ddof=1),
                "n_models_order_holds": n_order,
                "n_models_english_highest": n_en,
            })
            r, p = pearsonr(j, a)
            model_rows.append({"q": q, "format": f, "n_models": len(sub),
                               "r": r, "p": p})

    write_csv(os.path.join(args.out_dir, "format_level_summary.csv"),
              FORMAT_FIELDS, fmt_rows)
    write_csv(os.path.join(args.out_dir, "model_level_summary.csv"),
              MODEL_FIELDS, model_rows)

    log("\nFormat level: mean overlap (chance-corrected) across models; "
        "models with English > Spanish > Italian; models with English highest")
    for q in THRESHOLDS:
        parts = [f"{r['format'][:2]} {r['mean_jaccard']:.3f} "
                 f"({r['mean_jaccard_corrected']:.3f})"
                 for r in fmt_rows if r["q"] == q]
        n_order = next(r["n_models_order_holds"] for r in fmt_rows
                       if r["q"] == q)
        n_en = next(r["n_models_english_highest"] for r in fmt_rows
                    if r["q"] == q)
        n_mod = next(r["n_models"] for r in fmt_rows if r["q"] == q)
        log(f"  q={q_label(q):>4}: " + ", ".join(parts)
            + f"; order holds in {n_order}/{n_mod}, English highest in "
              f"{n_en}/{n_mod}")
    log("\nModel level: Pearson r(overlap, accuracy) across models")
    for q in THRESHOLDS:
        parts = [f"{r['format'][:2]} r={r['r']:+.2f} (p={r['p']:.3f})"
                 for r in model_rows if r["q"] == q]
        log(f"  q={q_label(q):>4}: " + ", ".join(parts))
    log(f"\nPaper at {q_label(PUBLISHED_Q)}: overlap en 0.145, es 0.078, "
        f"it 0.069; r en 0.75 (p=0.003), es 0.58 (p=0.039), it 0.47 (p=0.10)")

    with open(os.path.join(args.out_dir, "format_model_validation.txt"),
              "w") as f:
        f.write("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()
