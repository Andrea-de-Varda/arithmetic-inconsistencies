"""Matched random baseline: summary statistics (local, CPU).

Reads the CSVs written on the cluster by random_matched_loadings.py and
writes, under matched_baseline/results/:

  summary.csv     per (model, format): n, n_correct, low_n flag, r_pb on S,
                  its parametric p, the null distribution of r_pb over the
                  random draws (mean, sd, 2.5/97.5%, max), z of the real r_pb
                  against the null, and a one-sided permutation-style p
                  (1 + #{null >= real}) / (R + 1)
  summary.txt     headline numbers per format (including a paired Wilcoxon
                  test across models of the circuit r_pb vs the mean r_pb of
                  the random draws), per-cell numbers, and the check against
                  the published r_pb (results/all_pairs/per_item_pointbiserial.csv,
                  skipped when that file is absent)

validation.txt in the same folder is written by random_matched_loadings.py,
not by this script.

Usage (local): python matched_baseline/scripts/summarize_matched.py
"""

import argparse
import csv
import os

import numpy as np
from scipy.stats import wilcoxon

VERBAL = ("english", "spanish", "italian")
FORMAT_LABELS = {"english": "English", "spanish": "Spanish",
                 "italian": "Italian"}
MIN_CORRECT = 30   # threshold used in the main text

MODELS = [
    "Qwen/Qwen3-0.6B", "meta-llama/Llama-3.2-1B", "google/gemma-2-2b",
    "meta-llama/Llama-3.2-3B", "Qwen/Qwen3-4B", "mistralai/Mistral-7B-v0.1",
    "allenai/Olmo-3-1025-7B", "meta-llama/Llama-3.1-8B", "Qwen/Qwen3-8B",
    "google/gemma-2-9b", "microsoft/phi-4", "Qwen/Qwen3-32B",
    "allenai/Olmo-3-1125-32B",
]

SUMMARY_FIELDS = ["model", "format", "n", "n_correct", "low_n", "n_draws",
                  "rpb_real", "p_real", "null_mean", "null_sd", "null_q025",
                  "null_q975", "null_max", "z", "p_perm"]


# --- statistics ------------------------------------------------------------

def load_inputs(res_dir):
    cells = {}
    for r in csv.DictReader(open(os.path.join(res_dir, "cell_real.csv"))):
        cells[(r["model"], r["format"])] = r
    draws = {}
    for r in csv.DictReader(open(os.path.join(res_dir,
                                              "random_draws_rpb.csv"))):
        draws.setdefault((r["model"], r["format"]), []).append(float(r["rpb"]))
    return cells, {k: np.array(v) for k, v in draws.items()}


def cell_summary(cell, null):
    real = float(cell["rpb_real"])
    null = null[np.isfinite(null)]
    out = {"model": cell["model"], "format": cell["format"],
           "n": int(cell["n"]), "n_correct": int(cell["n_correct"]),
           "low_n": int(int(cell["n_correct"]) < MIN_CORRECT),
           "n_draws": int(null.size), "rpb_real": real,
           "p_real": float(cell["p_real"])}
    if null.size == 0 or not np.isfinite(real):
        out.update({k: float("nan") for k in SUMMARY_FIELDS if k not in out})
        return out
    sd = float(null.std(ddof=1))
    out.update({
        "null_mean": float(null.mean()), "null_sd": sd,
        "null_q025": float(np.percentile(null, 2.5)),
        "null_q975": float(np.percentile(null, 97.5)),
        "null_max": float(null.max()),
        "z": (real - float(null.mean())) / sd if sd > 0 else float("nan"),
        "p_perm": float((1 + np.sum(null >= real)) / (null.size + 1)),
    })
    return out


def load_published(attribution_root):
    p = os.path.join(attribution_root, "results", "all_pairs",
                     "per_item_pointbiserial.csv")
    if not os.path.exists(p):
        return {}
    return {(r["model"], r["target"]): float(r["rpb"])
            for r in csv.DictReader(open(p))
            if r["anchor"] == "numeric" and r["target"] in VERBAL}


def paired_wilcoxon(summ):
    """Per verbal format, paired Wilcoxon signed-rank test across models of
    the mean r_pb of the random draws vs the circuit r_pb on S. Returns
    {format: (n_models, p)} for formats with at least two usable cells."""
    wil = {}
    for fmt in VERBAL:
        rows = [s for s in summ if s["format"] == fmt
                and np.isfinite(s["rpb_real"]) and np.isfinite(s["null_mean"])]
        if len(rows) < 2:
            continue
        rnd = np.array([s["null_mean"] for s in rows])
        cir = np.array([s["rpb_real"] for s in rows])
        wil[fmt] = (len(rnd), float(wilcoxon(rnd, cir).pvalue))
    if not wil:
        print("  not enough cells for the paired Wilcoxon test")
    return wil


# --- main ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=None,
                    help="default: attribution/matched_baseline/results")
    args = ap.parse_args()
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    attribution_root = os.path.dirname(pkg_root)
    res_dir = args.results_dir or os.path.join(pkg_root, "results")

    if not os.path.exists(os.path.join(res_dir, "cell_real.csv")):
        print(f"no cell_real.csv under {res_dir}; run "
              f"random_matched_loadings.py on the cluster first")
        return
    cells, draws = load_inputs(res_dir)
    order = {m: i for i, m in enumerate(MODELS)}
    keys = sorted(cells, key=lambda k: (order.get(k[0], 99),
                                        VERBAL.index(k[1])))
    summ = [cell_summary(cells[k], draws.get(k, np.array([]))) for k in keys]

    with open(os.path.join(res_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(summ)
    print(f"wrote {os.path.join(res_dir, 'summary.csv')}  ({len(summ)} rows)")

    wil = paired_wilcoxon(summ)

    lines = []
    published = load_published(attribution_root)
    if published:
        d = [abs(s["rpb_real"] - published[(s["model"], s["format"])])
             for s in summ if (s["model"], s["format"]) in published
             and np.isfinite(s["rpb_real"])]
        lines.append(f"check: r_pb on S vs published all_pairs r_pb, "
                     f"{len(d)} cells, max|diff| = "
                     f"{(max(d) if d else float('nan')):.2e}")
    lines.append(f"\nper format ({summ[0]['n_draws'] if summ else 0} draws "
                 f"per cell; one-sided p_perm; low_n = fewer than "
                 f"{MIN_CORRECT} correct items)")
    for fmt in VERBAL:
        rows = [s for s in summ if s["format"] == fmt
                and np.isfinite(s["rpb_real"])]
        if not rows:
            continue
        above = sum(s["p_perm"] < 0.05 for s in rows)
        above_max = sum(s["rpb_real"] > s["null_max"] for s in rows)
        nm = np.array([s["null_mean"] for s in rows])
        nq = np.array([s["null_q975"] for s in rows])
        rr = np.array([s["rpb_real"] for s in rows])
        wl = wil.get(fmt)
        lines.append(
            f"  {FORMAT_LABELS[fmt]:8s} cells={len(rows)}  "
            f"circuit r_pb median {np.median(rr):+.2f} "
            f"[{rr.min():+.2f}, {rr.max():+.2f}]  "
            f"random mean r_pb median {np.median(nm):+.2f} "
            f"[{nm.min():+.2f}, {nm.max():+.2f}], 97.5% up to {nq.max():+.2f}  "
            f"above null p<0.05: {above}/{len(rows)}  "
            f"above every draw: {above_max}/{len(rows)}"
            + (f"  Wilcoxon (n={wl[0]}) p = {wl[1]:.2g}" if wl else ""))
    lines.append("\nper cell: model format n_correct r_real "
                 "null_mean [2.5%, 97.5%] z p_perm")
    for s in summ:
        lines.append(
            f"  {s['model']:28s} {s['format']:8s} {s['n_correct']:5d}"
            f"{'*' if s['low_n'] else ' '} {s['rpb_real']:+.3f}  "
            f"{s['null_mean']:+.3f} [{s['null_q025']:+.3f}, "
            f"{s['null_q975']:+.3f}]  z={s['z']:6.1f}  p={s['p_perm']:.4f}")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(res_dir, "summary.txt"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
