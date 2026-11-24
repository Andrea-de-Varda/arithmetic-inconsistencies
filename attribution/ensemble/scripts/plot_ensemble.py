"""Plot the cross-model predictor-comparison figures (local; reads only the
small CSVs written by stats_from_scores.py).

Reads ensemble/results/cross_model_predictors.csv and writes, under
ensemble/results/figures/:

  C4_auroc_bars.{png,svg}  held-out AUROC per predictor (paper Fig. 6)
  C5_lmg_bars.{png,svg}    LMG R^2 share per predictor (paper Fig. 4)

Both use the same bar canvas: one panel per verbal test format, the five
predictors on x, bar = mean across models, black bar = SE across models.
Only cells with >= 30 correct items enter.

House style: editable-text SVG, DejaVu Sans, left+bottom spines only, thin
dashed y-grid, .png + .svg.

Usage (local):
  python ensemble/scripts/plot_ensemble.py [--results-dir DIR]
"""

import argparse
import csv
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

VERBAL_FORMATS = ("english", "spanish", "italian")
PREDICTORS = ["x_main", "x1a", "x1b", "x2", "x3"]
PRED_LABELS = {
    "x_main": "Circuit", "x1a": "Probe (residual)", "x1b": "Probe (MLP)",
    "x2": "Sequence logP", "x3": "Entropy",
}
FORMAT_LABELS = {"english": "English", "spanish": "Spanish", "italian": "Italian"}
XMAIN_COLOR = "#D55E00"     # the circuit predictor
CTRL_COLOR = "#9aa0a6"      # gray for the control predictors

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


def save_fig(fig, base):
    fig.savefig(base + ".png", dpi=300)
    fig.savefig(base + ".svg")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(base)}.[png|svg]")


def read_rows(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def style(ax, spine_w=1.5):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(spine_w)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)


def plot_prediction_bars(pred_rows, fig_dir, metric, ylabel, out_name):
    """One per-predictor metric, averaged across models.

    One panel per verbal format, the five predictors on x. Per predictor: a
    faint bar at the mean across models and a black SE bar (across models).
    Only cells with >= 30 correct items enter. For AUROC the bars rise from
    chance (0.5); for the LMG share they rise from 0.
    """

    formats = [f for f in VERBAL_FORMATS
               if any(r["format"] == f for r in pred_rows)]
    fig, axes = plt.subplots(1, len(formats), figsize=(1.9 * len(formats), 2.4 * 0.72),
                             sharey=True)
    if len(formats) == 1:
        axes = [axes]
    base = 0.5 if metric == "auroc" else 0.0
    lo_all, hi_all = 1.0, 0.0
    xs = np.arange(len(PREDICTORS))
    for ax, fmt in zip(axes, formats):
        cells = [r for r in pred_rows if r["format"] == fmt
                 and not int(r.get("low_n", 0)) and np.isfinite(float(r[metric]))]
        for xi, p in enumerate(PREDICTORS):
            rows = [r for r in cells if r["predictor"] == p]
            if not rows:
                continue
            vals = np.array([float(r[metric]) for r in rows])
            mean = float(vals.mean())
            n = len(vals)
            # standard error across models
            half = float(vals.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
            color = XMAIN_COLOR if p == "x_main" else CTRL_COLOR
            # faint face, solid black edge (alpha on the face only)
            face = mpl.colors.to_rgba(color, 0.40 if p == "x_main" else 0.45)
            ax.bar(xi, mean - base, bottom=base, width=0.72, color=face,
                   edgecolor="black", linewidth=0.8, zorder=2)
            ax.errorbar(xi, mean, yerr=half, fmt="none", ecolor="black",
                        elinewidth=1.2, capsize=2.5, capthick=1.2, zorder=4)
            lo_all = min(lo_all, mean - half)
            hi_all = max(hi_all, mean + half)
        ax.set_xticks(xs)
        ax.set_xticklabels([PRED_LABELS[p] for p in PREDICTORS], fontsize=10,
                           rotation=45, ha="right")
        for tick, p in zip(ax.get_xticklabels(), PREDICTORS):
            if p == "x_main":
                tick.set_fontweight("bold")
        ax.set_xlim(-0.6, len(PREDICTORS) - 0.4)
        ax.set_title(FORMAT_LABELS[fmt], color="black", fontsize=11)
        style(ax)
    if metric == "auroc":
        axes[0].set_ylim(min(0.5, lo_all - 0.02), min(1.0, hi_all + 0.02))
    else:
        axes[0].set_ylim(0, hi_all * 1.06)
    # two-decimal ticks so the two figures share the same tight-bbox canvas
    axes[0].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))
    axes[0].set_ylabel(ylabel)
    save_fig(fig, os.path.join(fig_dir, out_name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=None,
                    help="default ensemble/results; the dir holding "
                         "cross_model_predictors.csv")
    args = ap.parse_args()
    ens_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    res = args.results_dir or os.path.join(ens_root, "results")

    pred_rows = read_rows(os.path.join(res, "cross_model_predictors.csv"))
    if not pred_rows:
        print("no cross_model_predictors.csv: run stats_from_scores.py first")
        return
    fig_dir = os.path.join(res, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plot_prediction_bars(pred_rows, fig_dir, "auroc", "held-out AUROC",
                         "C4_auroc_bars")
    plot_prediction_bars(pred_rows, fig_dir, "lmg_r2_linear",
                         r"$R^2$ share (LMG)", "C5_lmg_bars")
    print("done.")


if __name__ == "__main__":
    main()
