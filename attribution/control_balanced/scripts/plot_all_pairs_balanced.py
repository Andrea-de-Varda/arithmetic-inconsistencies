"""Plot the balanced-fold version of F3 (item-level point-biserial 4x4).

Reads control_balanced/results/all_pairs/per_item_pointbiserial_balanced.csv
(produced on the cluster by all_pairs_item_loadings_balanced.py) and writes:

  control_balanced/results/all_pairs/figures/
      F3_item_pointbiserial_heatmap_balanced.png
      F3_item_pointbiserial_heatmap_balanced.svg

The figure mirrors the main F3 style (RdBu_r diverging heatmap, symmetric
vmax, off-diagonal annotations only, no top/right frame). Difference: this
is a single-model figure on the balanced 80/20 fold, so cell values are
the raw r_pb (no cross-model averaging), and significant cells get a small
asterisk after the value (Holm-corrected over the 12 off-diagonal cells).

Run locally once the CSV is on the laptop:
    python control_balanced/scripts/plot_all_pairs_balanced.py
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from src.data import FORMATS


FORMAT_LABELS = {"numeric": "Numeric", "english": "English",
                 "spanish": "Spanish", "italian": "Italian"}
N_FORMATS = len(FORMATS)


mpl.rcParams.update({
    "svg.fonttype": "none",
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.06,
})


def save_fig(fig, base_path_no_ext):
    fig.savefig(base_path_no_ext + ".png", dpi=300)
    fig.savefig(base_path_no_ext + ".svg")
    plt.close(fig)
    print(f"  wrote {os.path.basename(base_path_no_ext)}.[png|svg]")


def holm_mask(pvals):
    """Holm-Bonferroni at alpha=0.05; returns boolean mask over input order."""
    p = np.asarray(pvals, dtype=float)
    finite = np.isfinite(p)
    order = np.argsort(np.where(finite, p, np.inf))
    m = finite.sum()
    sig = np.zeros_like(p, dtype=bool)
    for rank, k in enumerate(order):
        if not finite[k]:
            break
        thresh = 0.05 / (m - rank)
        if p[k] <= thresh:
            sig[k] = True
        else:
            break
    return sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=None,
                    help="defaults to control_balanced/results")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.results_dir is None:
        args.results_dir = os.path.join(repo_root, "results")
    out_dir = os.path.join(args.results_dir, "all_pairs")
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "per_item_pointbiserial_balanced.csv")
    if not os.path.exists(csv_path):
        print(f"missing {csv_path}; run all_pairs_item_loadings_balanced.py "
              f"on the cluster first")
        return

    rows = [r for r in csv.DictReader(open(csv_path)) if r["model"] == args.model]
    if not rows:
        print(f"no rows for model={args.model} in {csv_path}")
        return

    R = np.full((N_FORMATS, N_FORMATS), np.nan)
    P = np.full((N_FORMATS, N_FORMATS), np.nan)
    N = np.zeros((N_FORMATS, N_FORMATS), dtype=int)
    NCORR = np.zeros((N_FORMATS, N_FORMATS), dtype=int)
    for r in rows:
        fa, ft = r["anchor"], r["target"]
        if fa not in FORMATS or ft not in FORMATS:
            continue
        i, j = FORMATS.index(fa), FORMATS.index(ft)
        R[i, j] = float(r["rpb"])
        P[i, j] = float(r["p"])
        N[i, j] = int(r["n_items"])
        NCORR[i, j] = int(r["n_correct"])

    # Holm correction over the 12 off-diagonal cells
    off_diag = [(i, j) for i in range(N_FORMATS) for j in range(N_FORMATS) if i != j]
    pvec = [P[i, j] for (i, j) in off_diag]
    sig_vec = holm_mask(pvec)
    SIG = np.zeros((N_FORMATS, N_FORMATS), dtype=bool)
    for (i, j), s in zip(off_diag, sig_vec):
        SIG[i, j] = bool(s)

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    M_vis = R.copy()
    np.fill_diagonal(M_vis, np.nan)
    cmap = mpl.colormaps.get_cmap("RdBu_r").copy()
    cmap.set_bad("#f0f0f0")
    vmax = max(0.2, float(np.nanmax(np.abs(M_vis))))
    im = ax.imshow(M_vis, cmap=cmap, vmin=-vmax, vmax=vmax)

    for i in range(N_FORMATS):
        for j in range(N_FORMATS):
            if i == j:
                ax.text(j, i, "–", ha="center", va="center",
                        color="#666", fontsize=10)
            else:
                v = R[i, j]
                if np.isfinite(v):
                    label = f"{v:+.2f}"
                    if SIG[i, j]:
                        label += "*"
                    ax.text(j, i, label, ha="center", va="center",
                            color="black", fontsize=9)

    ax.set_xticks(range(N_FORMATS))
    ax.set_xticklabels([FORMAT_LABELS[f] for f in FORMATS], rotation=30, ha="right")
    ax.set_yticks(range(N_FORMATS))
    ax.set_yticklabels([FORMAT_LABELS[f] for f in FORMATS])
    ax.set_xlabel("target format")
    ax.set_ylabel("anchor format")

    # No figure title: the caption lives in the paper. Context (balanced fold,
    # n per cell, Holm marking) goes in the LaTeX caption instead.

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"$r_{pb}$", fontsize=10)
    cb.outline.set_linewidth(0.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    save_fig(fig, os.path.join(fig_dir, "F3_item_pointbiserial_heatmap_balanced"))


if __name__ == "__main__":
    main()
