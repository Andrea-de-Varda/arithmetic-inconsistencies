"""Cross-model accuracy figure and the per-(model, format) data table.

Reads the per-model overlap CSVs (results/overlap/<model_short>/) and response
summaries (results/responses/<model_short>/<format>.json), all git-tracked,
and writes to results/plots/_cross_model/:
  - cross_model_table_neuron.csv
        one row per (model, format): clean accuracy, n_distinguishable,
        top-1% Jaccard overlap with the numeric circuit, and the item-level
        point-biserial r between loading on the numeric circuit and accuracy
  - 10_accuracy_bars_per_format_1x4_neuron.{png,svg}
        clean accuracy per model, one panel per format (Numeric | English,
        Spanish, Italian grouped under "Verbal"), bars colored by family

The registry, palette and loaders defined here are also imported by
scripts/make_fig_item_combined.py.

Usage (from attribution/):
  python scripts/make_cross_model_plots.py
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.data import FORMATS

# ---------------------------------------------------------------------------
# Model registry: family colors and display labels.
# ---------------------------------------------------------------------------

FAMILY_COLORS = {
    "Mistral":  "tomato",
    "Gemma":    "gold",
    "Llama 3":  "steelblue",
    "Qwen 3":   "#9bb7d6",   # light steel blue
    "Olmo 3":   "#8c564b",   # brown
    "Phi-4":    "#708090",   # slate gray
}

# Explicit family display order (Mistral → Gemma → Llama → Qwen → Olmo → Phi).
FAMILY_ORDER = ["Mistral", "Gemma", "Llama 3", "Qwen 3", "Olmo 3", "Phi-4"]

# HF model id -> (display label, family, approximate parameter count in B).
MODELS = {
    "Qwen/Qwen3-0.6B":         ("Qwen-3$_{0.6B}$",     "Qwen 3",   0.6),
    "meta-llama/Llama-3.2-1B": ("Llama-3.2$_{1B}$",     "Llama 3",  1.0),
    "google/gemma-2-2b":       ("Gemma-2$_{2B}$",       "Gemma",    2.0),
    "meta-llama/Llama-3.2-3B": ("Llama-3.2$_{3B}$",     "Llama 3",  3.0),
    "Qwen/Qwen3-4B":           ("Qwen-3$_{4B}$",        "Qwen 3",   4.0),
    "mistralai/Mistral-7B-v0.1": ("Mistral$_{7B}$",     "Mistral",  7.0),
    "allenai/Olmo-3-1025-7B":  ("Olmo-3$_{7B}$",        "Olmo 3",   7.0),
    "meta-llama/Llama-3.1-8B": ("Llama-3.1$_{8B}$",     "Llama 3",  8.0),
    "Qwen/Qwen3-8B":           ("Qwen-3$_{8B}$",        "Qwen 3",   8.0),
    "google/gemma-2-9b":       ("Gemma-2$_{9B}$",       "Gemma",    9.0),
    "microsoft/phi-4":         ("Phi-4",                 "Phi-4",   14.0),
    "Qwen/Qwen3-32B":          ("Qwen-3$_{32B}$",       "Qwen 3",  32.0),
    "allenai/Olmo-3-1125-32B": ("Olmo-3$_{32B}$",       "Olmo 3",  32.0),
}

FORMAT_LABELS = {"numeric": "Numeric", "english": "English",
                  "spanish": "Spanish", "italian": "Italian"}
PEER_FORMATS = ["english", "spanish", "italian"]

MAIN_METRIC = "jaccard_top1%"

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

mpl.rcParams.update({
    "svg.fonttype": "none",
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.06,
})


def model_short(name):
    return name.replace("/", "_").replace(".", "-")


def save_fig(fig, out_path):
    fig.savefig(out_path + ".png", dpi=300)
    fig.savefig(out_path + ".svg")
    plt.close(fig)
    print(f"  wrote {os.path.basename(out_path)}.[png|svg]")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_csv(path, casts=None):
    rows = list(csv.DictReader(open(path)))
    if casts:
        for r in rows:
            for k, fn in casts.items():
                if k in r and r[k] != "":
                    r[k] = fn(r[k])
    return rows


def load_per_model_data(results_dir, model_id):
    """Return dict with summary_rows, item_corr, response_summaries, or None
    if the model has no format_overlap_summary.csv."""
    short = model_short(model_id)
    overlap_dir = os.path.join(results_dir, "overlap", short)
    summary_path = os.path.join(overlap_dir, "format_overlap_summary.csv")
    item_corr_path = os.path.join(overlap_dir, "item_level_correlations_neuron.txt")

    if not os.path.exists(summary_path):
        return None

    summary_rows = load_csv(summary_path, {"value": float, "K": lambda x: int(x) if x else None})

    item_corr = {}
    if os.path.exists(item_corr_path):
        for line in open(item_corr_path):
            line = line.strip()
            if not line.startswith("format="): continue
            for pf in PEER_FORMATS:
                if line.startswith(f"format={pf}"):
                    if "rpb(sum_on_S)=" in line:
                        v = line.split("rpb(sum_on_S)=")[1].split()[0]
                        try:
                            item_corr[pf] = float(v)
                        except ValueError:
                            pass

    response_summaries = {}
    for fmt in FORMATS:
        rp = os.path.join(results_dir, "responses", short, f"{fmt}.json")
        if os.path.exists(rp):
            try:
                response_summaries[fmt] = json.load(open(rp))["summary"]
            except Exception:
                pass

    return {
        "summary_rows": summary_rows,
        "item_corr": item_corr,
        "response_summaries": response_summaries,
    }


def lookup_overlap(summary_rows, component, metric, fx, fy):
    for r in summary_rows:
        if (r["component"] == component and r["metric"] == metric
                and r["format_x"] == fx and r["format_y"] == fy):
            return r["value"]
    return None


def load_item_predictions(results_dir, model_id, component="neuron"):
    """Returns dict[format] -> list of (sum_on_S, accuracy)."""
    p = os.path.join(results_dir, "overlap", model_short(model_id),
                     f"item_level_predictions_{component}.csv")
    if not os.path.exists(p):
        return {}
    rows = load_csv(p, {"accuracy": int, "sum_on_S": float})
    by_fmt = defaultdict(list)
    for r in rows:
        if r.get("component") != component:
            continue
        by_fmt[r["format"]].append((r["sum_on_S"], r["accuracy"]))
    return by_fmt


# ---------------------------------------------------------------------------
# Cross-model assembly
# ---------------------------------------------------------------------------

def build_cross_model(results_dir, component="neuron"):
    """Returns per_model: dict[model_id] = {id, label, family, size, color,
    accuracy, n_distinguishable, overlap_with_numeric, item_corr_sum_on_S}."""
    per_model = {}
    for mid, (label, family, size) in MODELS.items():
        d = load_per_model_data(results_dir, mid)
        if d is None:
            continue
        rec = {
            "id": mid, "label": label, "family": family, "size": size,
            "color": FAMILY_COLORS[family],
            "accuracy": {fmt: d["response_summaries"].get(fmt, {}).get("clean_accuracy")
                          for fmt in FORMATS},
            "n_distinguishable": {fmt: d["response_summaries"].get(fmt, {}).get("n_distinguishable")
                                   for fmt in FORMATS},
            "overlap_with_numeric": {pf: lookup_overlap(
                d["summary_rows"], component, MAIN_METRIC, "numeric", pf)
                for pf in PEER_FORMATS},
            "item_corr_sum_on_S": d["item_corr"],
        }
        per_model[mid] = rec
    return per_model


# ---------------------------------------------------------------------------
# Accuracy figure
# ---------------------------------------------------------------------------

def plot_accuracy_bars_per_format(per_model, out_dir):
    """1x4 per-format accuracy bars (Numeric | English, Spanish, Italian, with
    a "Verbal" bracket over the three verbal panels). Bar style: bars
    edgecolor=".2" lw=2.2, ylim 0-110, seaborn 'talk'-like font sizes; family
    ordering Mistral→Gemma→Llama→Qwen→Olmo→Phi with a 0.6-unit gap between
    families."""
    def sort_key(mid):
        rec = per_model[mid]
        return (FAMILY_ORDER.index(rec["family"])
                if rec["family"] in FAMILY_ORDER else 99,
                rec["size"])
    model_ids = sorted(per_model.keys(), key=sort_key)
    if not model_ids:
        return

    family_gap = 0.6
    x_pos, x_labels, colors = [], [], []
    cursor = 1.0
    prev_family = None
    for mid in model_ids:
        rec = per_model[mid]
        if prev_family is not None and rec["family"] != prev_family:
            cursor += family_gap
        x_pos.append(cursor)
        x_labels.append(rec["label"])
        colors.append(rec["color"])
        cursor += 1.0
        prev_family = rec["family"]
    x_pos = np.asarray(x_pos)

    # rcParams override scoped to this plot only — equivalent to seaborn's
    # set_context("talk") that the reference uses (thicker axes + ticks).
    talk_rc = {
        "font.size":          13,
        "axes.labelsize":     18,
        "axes.titlesize":     18,
        "xtick.labelsize":    15,
        "ytick.labelsize":    15,
        "axes.linewidth":      2.0,
        "xtick.major.width":   2.0,
        "ytick.major.width":   2.0,
        "xtick.major.size":    8.0,
        "ytick.major.size":    8.0,
    }

    formats_to_plot = ["numeric"] + PEER_FORMATS

    with mpl.rc_context(talk_rc):
        fig, axes = plt.subplots(
            1, 4, figsize=(22, 3.57), dpi=200, sharey=True,
            gridspec_kw={"wspace": 0.08},
        )
        for i, (ax, fmt) in enumerate(zip(axes, formats_to_plot)):
            accs_pct = [
                (per_model[m]["accuracy"].get(fmt) or 0.0) * 100.0
                for m in model_ids
            ]
            ax.bar(x_pos, accs_pct, color=colors, edgecolor=".2", lw=2.2)
            ax.set_ylim(0, 110)
            if i == 0:
                ax.set_ylabel("Accuracy (%)")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels, rotation=55, ha="right")
            # The numeric panel's label lives in the bracket row above, so it
            # is not repeated as a panel title.
            if fmt != "numeric":
                ax.set_title(FORMAT_LABELS[fmt], fontweight="bold", pad=12)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

        # Hierarchical grouping row above the panels:
        #   Numeric  |  Verbal -> {English, Spanish, Italian}
        fig.subplots_adjust(top=0.74)
        pos = [ax.get_position() for ax in axes]
        y = 0.93          # bracket height in figure coords
        drop = 0.045      # length of the bracket's downward ticks

        def _group_bracket(x0, x1, label):
            fig.add_artist(Line2D(
                [x0, x0, x1, x1], [y - drop, y, y, y - drop],
                transform=fig.transFigure, color="#333333", lw=2.0,
                clip_on=False))
            fig.text((x0 + x1) / 2, y + 0.03, label, ha="center",
                     va="bottom", fontsize=20, fontweight="bold")

        _group_bracket(pos[0].x0, pos[0].x1, "Numeric")
        _group_bracket(pos[1].x0, pos[3].x1, "Verbal")

        save_fig(fig, os.path.join(out_dir,
                                   "10_accuracy_bars_per_format_1x4_neuron"))


# ---------------------------------------------------------------------------
# Data table
# ---------------------------------------------------------------------------

def write_cross_model_table(per_model, out_dir, component="neuron"):
    """Long-format CSV with one row per (model, format)."""
    rows = []
    for mid, rec in per_model.items():
        for fmt in FORMATS:
            rows.append({
                "model_id": mid,
                "model_label": rec["label"],
                "family": rec["family"],
                "size_b": rec["size"],
                "format": fmt,
                "clean_accuracy": rec["accuracy"].get(fmt),
                "n_distinguishable": rec["n_distinguishable"].get(fmt),
                "overlap_with_numeric": (
                    rec["overlap_with_numeric"].get(fmt)
                    if fmt != "numeric" else None
                ),
                "rpb_sum_on_S": rec["item_corr_sum_on_S"].get(fmt)
                                 if fmt != "numeric" else None,
            })
    out = os.path.join(out_dir, f"cross_model_table_{component}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  wrote {os.path.basename(out)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.results_dir is None:
        args.results_dir = os.path.join(repo_root, "results")

    out_dir = os.path.join(args.results_dir, "plots", "_cross_model")
    os.makedirs(out_dir, exist_ok=True)
    print(f"writing cross-model outputs to {out_dir}")

    per_model = build_cross_model(args.results_dir)
    if not per_model:
        print("  no per-model data found; nothing to do.")
        return
    print(f"  loaded {len(per_model)} models: "
          f"{[rec['label'] for rec in per_model.values()]}")

    write_cross_model_table(per_model, out_dir)
    plot_accuracy_bars_per_format(per_model, out_dir)
    print("done.")


if __name__ == "__main__":
    main()
