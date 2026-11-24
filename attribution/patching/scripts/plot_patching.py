"""Item-level causal validation: summary + figure (local, CPU).

Reads patching/results/<model>/denoise__<condition>/<fmt>.json for every panel
model and writes, under patching/results/:

  summary.csv           per (model, condition, format): n, mean recovery,
                        r_pb(recovery, correctness), p
  figures/causal_validation.{png,svg}
                        paper Fig. 5. A: paired boxplot per verbal format
                        across models, causal r_pb for random-unit patching
                        (gray) vs numeric-circuit patching (format color),
                        paired-Wilcoxon brackets. B: attribution-based vs
                        causal r_pb, one dot per (model, format), Pearson r.

It also prints a table of the real-circuit r_pb per model and format.

Panel B reads the attribution-based r_pb from
results/all_pairs/per_item_pointbiserial.csv (anchor numeric, verbal targets).

Items flagged small_denominator (|m(x)-m(x')| < 1) or with non-finite recovery
are excluded, as in the attribution analyses.

Usage (local): python patching/scripts/plot_patching.py
"""

import argparse
import csv
import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

VERBAL = ("english", "spanish", "italian")
FORMAT_LABELS = {"english": "English", "spanish": "Spanish",
                 "italian": "Italian"}
FORMAT_COLORS = {"english": "#0072B2", "spanish": "#009E73",
                 "italian": "#E69F00"}
RANDOM_COLOR = "#b6bcc2"

# panel order (small -> large, families grouped as in the paper's figures)
MODELS = [
    "Qwen/Qwen3-0.6B", "meta-llama/Llama-3.2-1B", "google/gemma-2-2b",
    "meta-llama/Llama-3.2-3B", "Qwen/Qwen3-4B", "mistralai/Mistral-7B-v0.1",
    "allenai/Olmo-3-1025-7B", "meta-llama/Llama-3.1-8B", "Qwen/Qwen3-8B",
    "google/gemma-2-9b", "microsoft/phi-4", "Qwen/Qwen3-32B",
    "allenai/Olmo-3-1125-32B",
]

mpl.rcParams.update({
    "svg.fonttype": "none",
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.06,
})


def model_short(name):
    return name.replace("/", "_").replace(".", "-")


def style(ax, w=1.5):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(w)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)


def save_fig(fig, base):
    fig.savefig(base + ".png", dpi=300)
    fig.savefig(base + ".svg")
    plt.close(fig)
    print(f"  wrote {os.path.basename(base)}.[png|svg]")


def load_cell(out_root, mid, cond, fmt):
    p = os.path.join(out_root, model_short(mid), f"denoise__{cond}",
                     f"{fmt}.json")
    if not os.path.exists(p):
        return None
    recs = json.load(open(p))["records"]
    return [r for r in recs
            if not r["small_denominator"] and np.isfinite(r["effect"])]


def cell_stats(recs):
    from scipy.stats import pointbiserialr
    eff = np.array([r["effect"] for r in recs])
    acc = np.array([int(r["clean_match_gold"]) for r in recs])
    out = {"n": len(recs), "mean_recovery": float(eff.mean())}
    if len(set(acc.tolist())) > 1 and len(recs) >= 10:
        r, p = pointbiserialr(acc, eff)
        out["rpb"], out["p"] = float(r), float(p)
    else:
        out["rpb"], out["p"] = float("nan"), float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args()
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_root = args.results_dir or os.path.join(pkg_root, "results")
    fig_dir = os.path.join(out_root, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    rows = []
    stats = {}   # (mid, cond, fmt) -> stats
    for mid in MODELS:
        for cond in ("real", "random"):
            for fmt in VERBAL:
                recs = load_cell(out_root, mid, cond, fmt)
                if recs is None:
                    continue
                st = cell_stats(recs)
                stats[(mid, cond, fmt)] = st
                rows.append({"model": mid, "condition": cond, "format": fmt,
                             **st})

    if not rows:
        print(f"no result JSONs under {out_root}; run run_patching.py first")
        return

    with open(os.path.join(out_root, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "condition", "format", "n",
                                          "mean_recovery", "rpb", "p"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {os.path.join(out_root, 'summary.csv')}  ({len(rows)} rows)")

    # stdout table (real condition)
    print(f"\n{'model':28s}" + "".join(f"{FORMAT_LABELS[f]:>12s}" for f in VERBAL)
          + "   (r_pb recovery vs accuracy, real circuit; * p<0.05)")
    for mid in MODELS:
        cells = []
        for fmt in VERBAL:
            st = stats.get((mid, "real", fmt))
            if st is None or not np.isfinite(st["rpb"]):
                cells.append(f"{'-':>12s}")
            else:
                star = "*" if st["p"] < 0.05 else " "
                cells.append(f"{st['rpb']:+.2f}{star:s}".rjust(12))
        print(f"{mid:28s}" + "".join(cells))

    plot_causal_validation(stats, pkg_root, fig_dir)
    print("done.")


def _stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 \
        else "n.s."


def plot_causal_validation(stats, pkg_root, fig_dir):
    """Two-panel single-column figure (paper Fig. 5).

    A  paired boxplot per verbal format across the 13 models: random units
       (gray) vs numeric circuit (format color), causal r_pb, with per-model
       pairing lines and paired-Wilcoxon brackets at a common height.
    B  scatter of attribution-based vs causal r_pb, one dot per
       (model, format), identity line, r/p box.
    """
    from scipy.stats import pearsonr, wilcoxon
    attr_csv = os.path.join(os.path.dirname(pkg_root), "results", "all_pairs",
                            "per_item_pointbiserial.csv")
    attr = {}
    if os.path.exists(attr_csv):
        for r in csv.DictReader(open(attr_csv)):
            if r["anchor"] == "numeric" and r["target"] in VERBAL:
                attr[(r["model"], r["target"])] = float(r["rpb"])

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(5.4 * 0.81, 2.5 * 0.81),
        gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.42})

    # ---- A: paired boxplot (random vs circuit) per format -----------------
    off = 0.22
    pairs_by_fmt = {}
    for fmt in VERBAL:
        pairs = []
        for mid in MODELS:
            s_r = stats.get((mid, "random", fmt))
            s_c = stats.get((mid, "real", fmt))
            if s_r and s_c and np.isfinite(s_r["rpb"]) \
                    and np.isfinite(s_c["rpb"]):
                pairs.append((s_r["rpb"], s_c["rpb"]))
        if pairs:
            pairs_by_fmt[fmt] = (np.array([p[0] for p in pairs]),
                                 np.array([p[1] for p in pairs]))
    y_max = max(max(rnd.max(), cir.max()) for rnd, cir in pairs_by_fmt.values())
    y_min = min(min(rnd.min(), cir.min()) for rnd, cir in pairs_by_fmt.values())
    bracket_y = y_max + 0.06   # all brackets share this height

    for i, fmt in enumerate(VERBAL):
        if fmt not in pairs_by_fmt:
            continue
        rnd, cir = pairs_by_fmt[fmt]
        # faint pairing lines
        for a, b in zip(rnd, cir):
            axA.plot([i - off, i + off], [a, b], color="#4a4a4a", lw=0.6,
                     alpha=0.18, zorder=2)
            axA.scatter([i - off, i + off], [a, b], s=6, color="#4a4a4a",
                        alpha=0.2, zorder=2)
        bp = axA.boxplot([rnd, cir], positions=[i - off, i + off],
                         widths=0.32, patch_artist=True, showfliers=False,
                         medianprops={"color": "black", "linewidth": 1.1},
                         boxprops={"linewidth": 1.0},
                         whiskerprops={"linewidth": 1.0},
                         capprops={"linewidth": 1.0})
        for patch, c in zip(bp["boxes"], (RANDOM_COLOR, FORMAT_COLORS[fmt])):
            patch.set_facecolor(c)
            patch.set_alpha(0.85)
        axA.scatter([i - off, i + off], [rnd.mean(), cir.mean()], marker="s",
                    s=14, color="#222222", zorder=4)
        # paired Wilcoxon bracket at the common height
        p_w = wilcoxon(rnd, cir).pvalue
        axA.plot([i - off, i - off, i + off, i + off],
                 [bracket_y - 0.012, bracket_y, bracket_y, bracket_y - 0.012],
                 color="#333333", lw=1.2)
        axA.text(i, bracket_y + 0.005, _stars(p_w), ha="center", va="bottom",
                 fontsize=8)

    axA.axhline(0, color="black", lw=0.8, ls="--", alpha=0.6, zorder=1)
    axA.set_xticks(range(len(VERBAL)))
    axA.set_xticklabels([FORMAT_LABELS[f] for f in VERBAL], fontsize=9)
    axA.set_ylabel(r"causal $r_{pb}$", fontsize=9)
    axA.set_ylim(min(y_min, 0) - 0.04, bracket_y + 0.28)
    # keep the negative data region but drop negative tick labels (the minus
    # sign eats horizontal margin); only non-negative ticks within range
    axA.set_yticks([t for t in (0.0, 0.2, 0.4, 0.6, 0.8)
                    if t <= axA.get_ylim()[1]])
    from matplotlib.patches import Patch
    axA.legend(handles=[
        Patch(facecolor=RANDOM_COLOR, alpha=0.85, label="random units"),
        Patch(facecolor="#666", alpha=0.85, label="numeric circuit")],
        fontsize=6.8, loc="upper left", handlelength=1.4, handletextpad=0.5,
        labelspacing=0.35, borderaxespad=0.2)
    style(axA)
    axA.tick_params(labelsize=8)

    # ---- B: attribution vs causal scatter ---------------------------------
    xs, ys, cols = [], [], []
    for (mid, cond, fmt), st in stats.items():
        if cond != "real" or not np.isfinite(st["rpb"]) \
                or (mid, fmt) not in attr:
            continue
        xs.append(attr[(mid, fmt)])
        ys.append(st["rpb"])
        cols.append(FORMAT_COLORS[fmt])
    xs, ys = np.array(xs), np.array(ys)
    lim = (min(xs.min(), ys.min()) - 0.05, max(xs.max(), ys.max()) + 0.05)
    axB.plot(lim, lim, color="#cccccc", lw=1.0, ls="--", zorder=1)
    axB.scatter(xs, ys, c=cols, s=22, edgecolor="black", linewidth=0.5,
                zorder=3)
    r, p = pearsonr(xs, ys)
    p_txt = "p < 0.0001" if p < 1e-4 else f"p = {p:.3f}"
    axB.text(0.05, 0.95, f"$r = {r:.2f}$\n${p_txt}$", transform=axB.transAxes,
             ha="left", va="top", fontsize=8,
             bbox=dict(facecolor="white", edgecolor="gray",
                       boxstyle="round,pad=0.3"))
    axB.set_xlim(lim); axB.set_ylim(lim)
    axB.set_yticks([t for t in (0.0, 0.2, 0.4, 0.6) if lim[0] <= t <= lim[1]])
    axB.set_xlabel(r"attribution $r_{pb}$", fontsize=9)
    axB.set_ylabel(r"causal $r_{pb}$", fontsize=9)
    style(axB)
    axB.tick_params(labelsize=8)

    save_fig(fig, os.path.join(fig_dir, "causal_validation"))


if __name__ == "__main__":
    main()
