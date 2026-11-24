"""Paper figure: item-level violins + necessity diagnostics, in one figure.

Layout (single figure, meant for \\textwidth):
  A  (left, ~3/4 width, full height)   item-level loading on the numeric
     circuit (z-scored within model and format), incorrect vs correct items,
     one violin pair per model, three rows = English / Spanish / Italian, with
     the point-biserial r per (model, format) printed under each pair
     ("n.s." and grey when p >= 0.05)
  B  (right-top)     dose-response: P(correct) vs loading percentile, with
                     95% Wilson intervals
  C  (right-bottom)  outcome uncertainty: Bernoulli entropy H(correct) vs
                     loading percentile

Panels B and C pool the 13 models: loading is converted to a percentile rank
within each (model, format) so scales are comparable, then items are pooled
per verbal format and binned into 20 equal-width percentile bins.

Inputs: results/overlap/<model_short>/item_level_predictions_neuron.csv (per
item: format, accuracy, sum_on_S = loading on the numeric circuit) plus the
model registry and loaders of scripts/make_cross_model_plots.py.

Output:
  results/necessity/figures/item_violins_necessity_combined.{png,svg}

Usage (from attribution/):
  python scripts/make_fig_item_combined.py
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from make_cross_model_plots import (MODELS, build_cross_model,
                                    load_item_predictions, model_short,
                                    FAMILY_COLORS, FAMILY_ORDER, FORMAT_LABELS,
                                    PEER_FORMATS)


mpl.rcParams.update({
    "svg.fonttype": "none",
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.06,
})

FORMAT_COLORS = {"english": "#0072B2", "spanish": "#009E73", "italian": "#E69F00"}


def style(ax, spine_w=1.5):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(spine_w)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)


def save_fig(fig, base):
    fig.savefig(base + ".png", dpi=300)
    fig.savefig(base + ".svg")
    plt.close(fig)
    print(f"  wrote {os.path.basename(base)}.[png|svg]")


# ---------------------------------------------------------------------------
# Necessity diagnostics (panels B and C)
# ---------------------------------------------------------------------------

def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion; returns (lo, hi)."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - half) / d, (c + half) / d)


def bern_entropy(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def load_all(results_dir):
    """Return dict fmt -> {'pct', 'acc'} pooled across models, where pct is
    the within-(model, format) percentile rank of loading in [0, 1].
    (model, format) cells with fewer than 30 items are skipped."""
    pooled = {f: {"pct": [], "acc": []} for f in PEER_FORMATS}
    for mid in MODELS:
        p = os.path.join(results_dir, "overlap", model_short(mid),
                         "item_level_predictions_neuron.csv")
        if not os.path.exists(p):
            print(f"  skip {mid}: no item_level file")
            continue
        rows = [r for r in csv.DictReader(open(p))
                if r.get("component", "neuron") == "neuron"]
        by_fmt = {}
        for r in rows:
            by_fmt.setdefault(r["format"], []).append(r)
        for fmt in PEER_FORMATS:
            fr = by_fmt.get(fmt, [])
            if len(fr) < 30:
                continue
            load = np.array([float(r["sum_on_S"]) for r in fr])
            acc = np.array([int(r["accuracy"]) for r in fr])
            order = load.argsort()
            ranks = np.empty(len(load)); ranks[order] = np.arange(len(load))
            pct = ranks / max(1, len(load) - 1)
            pooled[fmt]["pct"].append(pct)
            pooled[fmt]["acc"].append(acc)
    for f in PEER_FORMATS:
        for k in pooled[f]:
            pooled[f][k] = np.concatenate(pooled[f][k]) if pooled[f][k] else np.array([])
    return pooled


def binned(pct, acc, nbins):
    edges = np.linspace(0, 1, nbins + 1)
    centers, mean, lo, hi, H = [], [], [], [], []
    for b in range(nbins):
        m = (pct >= edges[b]) & (pct <= edges[b + 1] if b == nbins - 1
                                 else pct < edges[b + 1])
        n = int(m.sum())
        if n == 0:
            continue
        k = int(acc[m].sum())
        p = k / n
        centers.append((edges[b] + edges[b + 1]) / 2)
        mean.append(p)
        l, h = wilson(k, n)
        lo.append(l); hi.append(h); H.append(bern_entropy(p))
    return (np.array(centers), np.array(mean), np.array(lo),
            np.array(hi), np.array(H))


def draw_dose_response(ax, pooled, nbins=20):
    """Panel: P(correct) vs loading percentile, per verbal format."""
    for fmt in PEER_FORMATS:
        if not pooled[fmt]["acc"].size:
            continue
        c, m, lo, hi, _ = binned(pooled[fmt]["pct"], pooled[fmt]["acc"], nbins)
        col = FORMAT_COLORS[fmt]
        ax.fill_between(c * 100, lo, hi, color=col, alpha=0.18, zorder=2)
        ax.plot(c * 100, m, "-o", color=col, ms=3, lw=1.6,
                label=FORMAT_LABELS[fmt], zorder=3)
    ax.set_ylabel("P(correct)")
    ax.set_ylim(-0.02, 1.0)
    ax.set_title("Accuracy", fontsize=10)
    style(ax)


def draw_uncertainty(ax, pooled, nbins=20):
    """Panel: Bernoulli entropy of correctness vs loading percentile."""
    for fmt in PEER_FORMATS:
        if not pooled[fmt]["acc"].size:
            continue
        c, _, _, _, H = binned(pooled[fmt]["pct"], pooled[fmt]["acc"], nbins)
        ax.plot(c * 100, H, "-o", color=FORMAT_COLORS[fmt], ms=3, lw=1.6,
                zorder=3)
    ax.set_ylabel("H(correct)")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Uncertainty", fontsize=10)
    style(ax)


# ---------------------------------------------------------------------------
# Violin block (panel A), drawn into a provided list of three axes
# ---------------------------------------------------------------------------

def draw_violin_block(axes, per_model, results_dir, component="neuron",
                      sig_alpha=0.05, tick_fs=7.5, r_fs=7.0):
    from scipy.stats import pointbiserialr

    item_data = {}
    for mid in per_model:
        d = load_item_predictions(results_dir, mid, component=component)
        if d:
            item_data[mid] = d

    def sort_key(mid):
        rec = per_model[mid]
        return (FAMILY_ORDER.index(rec["family"])
                if rec["family"] in FAMILY_ORDER else 99, rec["size"])
    model_ids = sorted([m for m in per_model if m in item_data], key=sort_key)

    family_gap = 0.45
    x_pos, x_labels = [], []
    cursor, prev_family = 0.0, None
    for mid in model_ids:
        fam = per_model[mid]["family"]
        if prev_family is not None and fam != prev_family:
            cursor += family_gap
        x_pos.append(cursor)
        x_labels.append(per_model[mid]["label"])
        cursor += 1.0
        prev_family = fam
    x_pos = np.asarray(x_pos)

    NS_GREY, INC_GREY = "#bcbcbc", "#9aa1a6"

    panel_data = {fmt: [] for fmt in PEER_FORMATS}
    median_max, median_min = 0.0, 0.0
    for mid in model_ids:
        rec = per_model[mid]
        for fmt in PEER_FORMATS:
            entry = {"rec": rec, "ok": False}
            pairs = item_data[mid].get(fmt, [])
            if len(pairs) >= 4:
                sums = np.array([p[0] for p in pairs], dtype=float)
                accs = np.array([p[1] for p in pairs], dtype=int)
                sd = float(np.std(sums))
                if sd > 0 and len(set(accs.tolist())) > 1:
                    z = (sums - sums.mean()) / sd
                    inc, cor = z[accs == 0], z[accs == 1]
                    if len(inc) >= 2 and len(cor) >= 2:
                        rpb, p_val = pointbiserialr(accs, sums)
                        entry.update({"ok": True, "inc": inc, "cor": cor,
                                      "rpb": float(rpb), "p": float(p_val),
                                      "sig": p_val < sig_alpha})
                        for arr in (inc, cor):
                            median_max = max(median_max, float(np.median(arr)))
                            median_min = min(median_min, float(np.median(arr)))
            panel_data[fmt].append(entry)

    y_top = max(3.3, median_max + 1.0)
    y_bot = min(-3.4, median_min - 0.6)
    label_y = y_bot + 0.45

    for ax, fmt in zip(axes, PEER_FORMATS):
        for xi, entry in zip(x_pos, panel_data[fmt]):
            if not entry["ok"]:
                continue
            rec, sig = entry["rec"], entry["sig"]
            cor_color = rec["color"] if sig else NS_GREY
            inc_color = INC_GREY if sig else NS_GREY
            cor_alpha = 0.82 if sig else 0.22
            inc_alpha = 0.55 if sig else 0.22
            median_color = "#1f1f1f" if sig else "#b8b8b8"
            offset = 0.22

            v_inc = ax.violinplot([entry["inc"]], positions=[xi - offset],
                                  widths=0.36, showmeans=False,
                                  showmedians=True, showextrema=False)
            for pc in v_inc["bodies"]:
                pc.set_facecolor(inc_color); pc.set_alpha(inc_alpha)
                pc.set_edgecolor("none")
            v_inc["cmedians"].set_color(median_color)
            v_inc["cmedians"].set_linewidth(1.0)

            v_cor = ax.violinplot([entry["cor"]], positions=[xi + offset],
                                  widths=0.36, showmeans=False,
                                  showmedians=True, showextrema=False)
            for pc in v_cor["bodies"]:
                pc.set_facecolor(cor_color); pc.set_alpha(cor_alpha)
                pc.set_edgecolor("none")
            v_cor["cmedians"].set_color(median_color)
            v_cor["cmedians"].set_linewidth(1.0)

            # short label (no "r=" prefix, no sign: all r are positive here):
            # the block is narrower than the standalone violin figure and the
            # full labels collide between models
            label = f"{entry['rpb']:.2f}" if sig else "n.s."
            ax.text(xi, label_y, label, ha="center", va="bottom", fontsize=r_fs,
                    color="#1f1f1f" if sig else "#a8a8a8")

        ax.axhline(0, color="#888", linewidth=0.5, linestyle="--", zorder=0)
        ax.set_xticks(x_pos)
        if ax is axes[-1]:
            ax.set_xticklabels(x_labels, rotation=45, ha="right",
                               fontsize=tick_fs)
        else:
            ax.set_xticklabels([])
        ax.tick_params(axis="y", labelsize=8)
        ax.set_title(FORMAT_LABELS[fmt], loc="left", pad=2, fontweight="bold",
                     fontsize=9.5)
        ax.set_ylim(y_bot, y_top)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    handles = [
        Line2D([0], [0], marker="s", color=INC_GREY, linestyle="",
               markersize=7, alpha=0.55, label="incorrect"),
        Line2D([0], [0], marker="s", color=NS_GREY, linestyle="",
               markersize=7, alpha=0.30, label=f"n.s. (p ≥ {sig_alpha})"),
    ] + [
        Line2D([0], [0], marker="s", color=FAMILY_COLORS[fam], linestyle="",
               markersize=7, alpha=0.82, label=fam)
        for fam in FAMILY_ORDER
        if any(per_model[m]["family"] == fam for m in model_ids)
    ]
    return handles


# ---------------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = args.results_dir or os.path.join(repo_root, "results")
    fig_dir = os.path.join(results_dir, "necessity", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    per_model = build_cross_model(results_dir)
    pooled = load_all(results_dir)

    fig = plt.figure(figsize=(9.6, 4.8))
    # two independent gridspecs: the violin block keeps a deep bottom margin
    # (rotated model labels + legend); the right column extends further down
    # and is roughly half its former width
    gs_v = fig.add_gridspec(3, 1, left=0.065, right=0.775, top=0.93,
                            bottom=0.16, hspace=0.32)
    gs_r = fig.add_gridspec(2, 1, left=0.86, right=0.99, top=0.93,
                            bottom=0.10, hspace=0.55)

    v_axes = [fig.add_subplot(gs_v[i]) for i in range(3)]
    axB = fig.add_subplot(gs_r[0])
    axC = fig.add_subplot(gs_r[1])

    handles = draw_violin_block(v_axes, per_model, results_dir)
    v_axes[1].set_ylabel("Loading on numeric circuit (z-scored)",
                         fontsize=9.5)
    # legend under the violin block, flush with its left edge
    fig.legend(handles=handles, loc="lower left", ncol=len(handles),
               bbox_to_anchor=(0.005, -0.015), frameon=False,
               handletextpad=0.3, columnspacing=0.8, fontsize=7.5)

    draw_dose_response(axB, pooled)
    axB.legend(fontsize=6.5, loc="upper left")
    draw_uncertainty(axC, pooled)
    axB.tick_params(labelsize=8); axC.tick_params(labelsize=8)
    xlab = "loading on numeric circuit\n(percentile)"
    axB.set_xlabel(xlab, fontsize=8.5)
    axC.set_xlabel(xlab, fontsize=8.5)

    # panel letters
    fig.text(0.012, 0.965, "A", fontsize=14, fontweight="bold")
    posB = axB.get_position(); posC = axC.get_position()
    fig.text(posB.x0 - 0.065, 0.965, "B", fontsize=14, fontweight="bold")
    fig.text(posC.x0 - 0.065, posC.y1 + 0.035, "C", fontsize=14,
             fontweight="bold")

    save_fig(fig, os.path.join(fig_dir, "item_violins_necessity_combined"))
    print("done.")


if __name__ == "__main__":
    main()
