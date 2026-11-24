"""Self-test for item_level_loadings.py, on a synthetic results tree.

The real script reads per-item tensors that only exist on the cluster, so this
builds a small fake results tree with the same layout (aggregates, fp16
per-item tensors, kept_ids, responses, and a published
item_level_predictions_neuron.csv written by a literal transcription of
run_overlap.py) and checks:

  1. K per threshold equals round(q * L * d) and the circuits are nested.
  2. At q = 1% the fp16 loading equals the transcribed published sum_on_S
     bit-for-bit, for every item, including items without a response record
     being skipped.
  3. r_pb written for each cell equals scipy.stats.pointbiserialr on the
     written loadings, and the float64 path agrees closely.
  4. Behavioural: a signal planted on the units of the 0.5% circuit of
     correct items gives a positive r_pb at every threshold, which shrinks
     as q grows and more signal-free units enter the sum.

Usage (local): python threshold_robustness/scripts/selftest_item_level.py
"""

import csv
import json
import os
import sys
import tempfile

import numpy as np
import torch
from scipy.stats import pointbiserialr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import THRESHOLDS, VERBAL, model_short, topk_indices_abs
from item_level_loadings import CELL_FIELDS, loading_col, run_model

FAIL = []
MID = "synthetic/Model-1"


def check(name, cond, detail=""):
    print(f"  [{'ok ' if cond else 'FAIL'}] {name}"
          f"{'  ' + detail if detail else ''}")
    if not cond:
        FAIL.append(name)


def build_tree(root, rng, n_layers=6, d=700, n_items=300):
    """Returns (agg_numeric, planted S_0.5% flat indices)."""
    scale = np.linspace(0.5, 2.0, n_layers)[:, None]
    agg = (rng.standard_normal((n_layers, d)) * scale * 1e-3).astype(np.float32)
    n_units = agg.size
    S_small = topk_indices_abs(agg, int(round(0.005 * n_units)))

    base = os.path.join(root, "attribution", model_short(MID))
    os.makedirs(os.path.join(root, "responses", model_short(MID)))
    os.makedirs(os.path.join(root, "overlap", model_short(MID)))
    os.makedirs(os.path.join(base, "numeric"))
    torch.save(torch.from_numpy(agg),
               os.path.join(base, "numeric", "neuron_attribution_aggregate.pt"))

    # literal transcription of run_overlap.py 3b
    sym_topk = set(topk_indices_abs(agg, max(1, int(round(0.01 * agg.size))))
                   .tolist())
    pub_rows = []
    for fi, fmt in enumerate(VERBAL):
        os.makedirs(os.path.join(base, fmt))
        acc = rng.random(n_items) < (0.5 - 0.1 * fi)
        x = rng.standard_normal((n_items, n_units)) * 1e-3
        x[np.ix_(acc, S_small)] += 1.5e-3               # planted signal
        per_item = torch.from_numpy(
            x.reshape(n_items, n_layers, d).astype(np.float16))
        torch.save(per_item,
                   os.path.join(base, fmt, "neuron_attribution_per_item.pt"))
        ids = [f"prob_{fi}_{i:04d}" for i in range(n_items)]
        json.dump(ids, open(os.path.join(base, fmt, "kept_ids.json"), "w"))
        # one kept id has no response record and must be skipped
        records = [{"id": iid, "clean_match_gold": bool(a)}
                   for iid, a in zip(ids, acc)][1:]
        json.dump({"summary": {}, "records": records},
                  open(os.path.join(root, "responses", model_short(MID),
                                    f"{fmt}.json"), "w"))
        pi = per_item.numpy()
        by_id = {r["id"]: r for r in records}
        for i, iid in enumerate(ids):
            if by_id.get(iid) is None:
                continue
            vec = pi[i].ravel()
            pub_rows.append({"component": "neuron", "id": iid, "format": fmt,
                             "sum_on_S": float(vec[list(sym_topk)].sum())})
    with open(os.path.join(root, "overlap", model_short(MID),
                           "item_level_predictions_neuron.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=["component", "id", "format",
                                          "sum_on_S"])
        w.writeheader()
        w.writerows(pub_rows)
    return agg, pub_rows


def main():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as root:
        agg, pub_rows = build_tree(root, rng)
        lines = []
        items, cells = run_model(MID, root, THRESHOLDS, lines.append)
        print("\n".join("    " + l for l in lines))

        print("1. thresholds")
        for q in THRESHOLDS:
            Ks = {c["K"] for c in cells if c["q"] == q}
            check(f"K at {q:g}", Ks == {int(round(q * agg.size))}, str(Ks))
        check("circuits nested", any("nested across thresholds = True" in l
                                     for l in lines))

        print("2. reproduction of the published loading at 1%")
        pub = {(r["format"], r["id"]): r["sum_on_S"] for r in pub_rows}
        got = {(r["format"], r["id"]): r[loading_col(0.01)] for r in items}
        check("same item set (response-less item skipped)",
              set(pub) == set(got), f"{len(got)} items")
        check("bit-for-bit equal",
              all(got[k] == pub[k] for k in pub),
              f"max|diff| = {max(abs(got[k] - pub[k]) for k in pub):.1e}")

        print("3. r_pb")
        check("cell fields", all(set(c) == set(CELL_FIELDS) for c in cells))
        worst, worst64 = 0.0, 0.0
        for c in cells:
            sub = [r for r in items if r["format"] == c["format"]]
            a = np.array([r["accuracy"] for r in sub], dtype=float)
            x = np.array([r[loading_col(c["q"])] for r in sub])
            worst = max(worst, abs(pointbiserialr(a, x)[0] - c["rpb"]))
            worst64 = max(worst64, abs(c["rpb"] - c["rpb_float64"]))
        check("equals scipy", worst < 1e-12, f"max|diff| = {worst:.1e}")
        check("float64 path agrees", worst64 < 1e-3,
              f"max|diff| = {worst64:.1e}")

        print("4. behavioural")
        for fmt in VERBAL:
            r = [next(c["rpb"] for c in cells
                      if c["format"] == fmt and c["q"] == q)
                 for q in THRESHOLDS]
            check(f"{fmt}: positive at every q, decreasing with q",
                  all(v > 0 for v in r) and all(np.diff(r) < 0),
                  " ".join(f"{v:+.2f}" for v in r))

    print(f"\n{'ALL CHECKS PASSED' if not FAIL else f'{len(FAIL)} FAILED'}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
