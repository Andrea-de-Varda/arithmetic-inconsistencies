"""Item-level summary at each threshold, in the paper's §4.3 terms.

Local, after pulling results/cell_rpb.csv back from the cluster. Per
threshold and verbal format, across the 13 models:

  - n_sig_pos: cells with r_pb > 0 and p < 0.05 (paper: 12/13, 11/13, 9/13)
  - median and range of r_pb over those cells (the paper's convention: at
    1% this reproduces medians 0.38 / 0.32 / 0.29)
  - n_min30 / n_pos_min30: cells with at least 30 correct items, and how many
    of them have r_pb > 0 (paper: "in every combination with at least 30
    correct items, correct items loaded more")
  - median r_pb over all 13 cells, for reference
  - max |r_pb fp16 - r_pb float64|, the precision check

At 1% the r_pb of every cell is also compared against the published value in
results/overlap/<model>/item_level_correlations_neuron.txt (3 decimals).

Outputs, under threshold_robustness/results/:
  item_level_summary.csv   q,format,n_models,n_sig_pos,median_rpb_sig,
                           min_rpb_sig,max_rpb_sig,n_min30,n_pos_min30,
                           median_rpb_all,max_abs_fp16_float64
  item_level_summary.txt

Usage:
  python threshold_robustness/scripts/summarize_item_level.py
"""

import argparse
import csv
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (DEFAULT_OUT_DIR, DEFAULT_RESULTS_DIR, PUBLISHED_Q,
                    THRESHOLDS, VERBAL, model_short, q_label, write_csv)

SUMMARY_FIELDS = ["q", "format", "n_models", "n_sig_pos", "median_rpb_sig",
                  "min_rpb_sig", "max_rpb_sig", "n_min30", "n_pos_min30",
                  "median_rpb_all", "max_abs_fp16_float64"]
MIN_CORRECT = 30


def published_rpb(results_dir, mid):
    p = os.path.join(results_dir, "overlap", model_short(mid),
                     "item_level_correlations_neuron.txt")
    out = {}
    if not os.path.exists(p):
        return out
    for line in open(p):
        m = re.search(r"format=(\w+).*?rpb\(sum_on_S\)=([+-]?[\d.]+)", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    cells = list(csv.DictReader(open(os.path.join(args.out_dir,
                                                  "cell_rpb.csv"))))
    for c in cells:
        for k in ("q", "rpb", "p", "rpb_float64"):
            c[k] = float(c[k])
        c["n_correct"] = int(c["n_correct"])

    lines = []
    n_checked, worst = 0, 0.0
    for c in cells:
        if c["q"] != PUBLISHED_Q:
            continue
        pub = published_rpb(args.results_dir, c["model"]).get(c["format"])
        if pub is not None:
            n_checked += 1
            worst = max(worst, abs(round(c["rpb"], 3) - pub))
    lines.append(f"[check] {q_label(PUBLISHED_Q)} r_pb vs published "
                 f"(3 decimals): {n_checked} cells, max|diff| = {worst:.3f}")

    rows = []
    for q in THRESHOLDS:
        for f in VERBAL:
            sub = [c for c in cells if c["q"] == q and c["format"] == f]
            r = np.array([c["rpb"] for c in sub])
            p = np.array([c["p"] for c in sub])
            sig = (r > 0) & (p < 0.05)
            big = np.array([c["n_correct"] >= MIN_CORRECT for c in sub])
            rows.append({
                "q": q, "format": f, "n_models": len(sub),
                "n_sig_pos": int(sig.sum()),
                "median_rpb_sig": float(np.median(r[sig])) if sig.any() else float("nan"),
                "min_rpb_sig": float(r[sig].min()) if sig.any() else float("nan"),
                "max_rpb_sig": float(r[sig].max()) if sig.any() else float("nan"),
                "n_min30": int(big.sum()),
                "n_pos_min30": int((r[big] > 0).sum()),
                "median_rpb_all": float(np.nanmedian(r)),
                "max_abs_fp16_float64": float(np.max(np.abs(
                    r - np.array([c["rpb_float64"] for c in sub])))),
            })
    write_csv(os.path.join(args.out_dir, "item_level_summary.csv"),
              SUMMARY_FIELDS, rows)

    lines.append("\nItem level: significant positive r_pb (median, range over "
                 "those); cells with >= 30 correct items with r_pb > 0")
    for q in THRESHOLDS:
        lines.append(f"  q={q_label(q)}")
        for r in rows:
            if r["q"] != q:
                continue
            lines.append(
                f"    {r['format']:<8} {r['n_sig_pos']}/{r['n_models']} "
                f"(median {r['median_rpb_sig']:.2f}, {r['min_rpb_sig']:.2f} "
                f"to {r['max_rpb_sig']:.2f}); positive in "
                f"{r['n_pos_min30']}/{r['n_min30']} cells with >= 30 correct; "
                f"median over all {r['median_rpb_all']:.2f}; "
                f"fp16 vs float64 r_pb max|diff| {r['max_abs_fp16_float64']:.1e}")
    lines.append("\nPaper at 1%: 12/13 (median 0.38, 0.11 to 0.56), "
                 "11/13 (0.32, 0.06 to 0.42), 9/13 (0.29, 0.19 to 0.50)")
    print("\n".join(lines))
    with open(os.path.join(args.out_dir, "item_level_summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
