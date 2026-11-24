"""Shared definitions for the threshold-robustness package.

The circuit at threshold q is S_q = top q of units of the numeric aggregate by
|attribution|, with K = max(1, round(q * L * d_MLP)) and argpartition, exactly
as in scripts/run_overlap.py. The paper uses q = 1%.
"""

import csv
import json
import os

import numpy as np
import torch

THRESHOLDS = (0.005, 0.01, 0.02, 0.05)
PUBLISHED_Q = 0.01
VERBAL = ("english", "spanish", "italian")

# same order as matched_baseline/ and heldout/
MODELS = [
    "Qwen/Qwen3-0.6B",
    "meta-llama/Llama-3.2-1B",
    "google/gemma-2-2b",
    "meta-llama/Llama-3.2-3B",
    "Qwen/Qwen3-4B",
    "mistralai/Mistral-7B-v0.1",
    "allenai/Olmo-3-1025-7B",
    "meta-llama/Llama-3.1-8B",
    "Qwen/Qwen3-8B",
    "google/gemma-2-9b",
    "microsoft/phi-4",
    "Qwen/Qwen3-32B",
    "allenai/Olmo-3-1125-32B",
]

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATTRIBUTION_ROOT = os.path.dirname(PKG_ROOT)
DEFAULT_RESULTS_DIR = os.path.join(ATTRIBUTION_ROOT, "results")
DEFAULT_OUT_DIR = os.path.join(PKG_ROOT, "results")


def q_label(q):
    return f"{q * 100:g}%"


def model_short(name):
    return name.replace("/", "_").replace(".", "-")


def n_units_for(q, n_units):
    """K as in run_overlap.py."""
    return max(1, int(round(q * n_units)))


def topk_indices_abs(vec, k):
    """Indices of top-k entries by absolute value. (run_overlap.py:70)"""
    flat = np.abs(vec).ravel()
    if k >= flat.size:
        return np.arange(flat.size)
    return np.argpartition(flat, -k)[-k:]


def jaccard(a, b):
    sa, sb = set(a.tolist()), set(b.tolist())
    if not sa and not sb:
        return float("nan")
    return len(sa & sb) / len(sa | sb)


def chance_jaccard(K, n_units):
    """Expected Jaccard of two independent uniform K-subsets of n_units units,
    to first order: E|A∩B| = K^2/N, so J ≈ I / (2K - I) = q / (2 - q)."""
    inter = K * K / n_units
    return inter / (2 * K - inter)


def corrected_jaccard(j, j_chance):
    return (j - j_chance) / (1 - j_chance)


def published_loading(vec_fp16, circuit_idx):
    """The published loading, run_overlap.py:245: the circuit is held as a
    Python set and summed in fp16 over list(set), so the element order (and
    hence the fp16 rounding) is that of the set."""
    return float(vec_fp16[list(set(circuit_idx.tolist()))].sum())


def attr_dir(results_dir, mid, fmt):
    return os.path.join(results_dir, "attribution", model_short(mid), fmt)


def load_aggregate(results_dir, mid, fmt):
    p = os.path.join(attr_dir(results_dir, mid, fmt),
                     "neuron_attribution_aggregate.pt")
    if not os.path.exists(p):
        return None
    return torch.load(p).float().numpy()  # [L, d]


def load_per_item(results_dir, mid, fmt):
    p = os.path.join(attr_dir(results_dir, mid, fmt),
                     "neuron_attribution_per_item.pt")
    if not os.path.exists(p):
        return None
    return torch.load(p).numpy()          # [n, L, d] at saved fp16


def load_kept_ids(results_dir, mid, fmt):
    p = os.path.join(attr_dir(results_dir, mid, fmt), "kept_ids.json")
    return json.load(open(p)) if os.path.exists(p) else None


def load_responses(results_dir, mid, fmt):
    p = os.path.join(results_dir, "responses", model_short(mid), f"{fmt}.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def load_overlap_summary(results_dir, mid):
    """{(metric, format_x, format_y): value} from format_overlap_summary.csv."""
    p = os.path.join(results_dir, "overlap", model_short(mid),
                     "format_overlap_summary.csv")
    if not os.path.exists(p):
        return {}
    return {(r["metric"], r["format_x"], r["format_y"]): float(r["value"])
            for r in csv.DictReader(open(p)) if r["component"] == "neuron"}


def load_published_items(results_dir, mid):
    """format -> {id: sum_on_S} from item_level_predictions_neuron.csv."""
    p = os.path.join(results_dir, "overlap", model_short(mid),
                     "item_level_predictions_neuron.csv")
    out = {}
    if not os.path.exists(p):
        return out
    for r in csv.DictReader(open(p)):
        out.setdefault(r["format"], {})[r["id"]] = float(r["sum_on_S"])
    return out


def write_csv(path, fields, rows, mode="w"):
    new = mode == "w" or not os.path.exists(path)
    with open(path, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)
