"""Predictor comparison (CPU), one model per call: does the cross-fitted
circuit loading (x_main) predict per-item correctness on held-out verbal
problems beyond numeric-trained probes + confidence controls?

Consumes the features written by extract_features.py and the cross-fitted
loadings written by heldout/scripts/crossfit_circuits.py, then per verbal
test format assembles the per-item design matrix

    y, x_main, x1a, x1b, x2, x3

and computes the shared statistics in stats.py.

Predictors
  x_main : cross-fitted loading on the numeric circuit
           (heldout/results/crossfit_item_loadings.csv, column
           loading_crossfit: the circuit that scores item i was identified on
           numeric problems disjoint from i's fold). --x-main-source full
           uses the published all-items circuit instead (loading_full).
  x1a    : numeric-trained residual-stream probe, applied to the test item
  x1b    : numeric-trained MLP(down_proj-input) probe, applied to the test item
  x2     : mean log P(model's own response | prompt)      (features sidecar)
  x3     : entropy of the first response token             (features sidecar)

Probe (per substrate, numeric probe pool -> verbal test):
  inner stratified k-fold CV over the pool to pick (layer*, C*) by mean OOF AUC,
  class_weight="balanced"; refit on the full pool; predict P(correct) on test.
  The probe pool is disjoint by content from the test problems (see
  extract_features.py), so both x_main and the probes are held out.

Outputs (small, committed) under ensemble/results/<model>/:
  per_item_scores.csv     model,format,id,y,x_main,x1a,x1b,x2,x3
  layer_cv_auc.csv        substrate,layer,oof_auc
  probe_info.json         selected layer / C / OOF AUC per substrate, pool size
  variance_partition.csv  per (format, predictor): AUROC, AP, zero-order r,
                          standardized coef + p, unique R^2, LMG share
  partial_corr.csv        per format: x_main partial r | controls, model totals

Usage on the cluster (after extract_features.py and crossfit_circuits.py):
  python ensemble/scripts/probe_and_controls.py --model meta-llama/Llama-3.1-8B
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for `stats`

import numpy as np

from stats import (analyze_format, compute_auc, PREDICTORS, PER_PRED_FIELDS,
                   MODEL_FIELDS)

VERBAL_FORMATS = ("english", "spanish", "italian")
SUBSTRATES = {"x1a": "resid", "x1b": "mlp"}


def model_short(name):
    return name.replace("/", "_").replace(".", "-")


# ---------------------------------------------------------------------------
# probe: nested-CV layer/C selection, refit, transfer
# ---------------------------------------------------------------------------

def _cv_auc(Xl, y, folds, C, max_iter, tol):
    """OOF AUC for one layer at one C, with per-fold standardization (which
    makes lbfgs converge far faster on the high-dim probe features)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    oof = np.zeros(len(y))
    for tr, va in folds:
        sc = StandardScaler().fit(Xl[tr])
        clf = LogisticRegression(C=C, max_iter=max_iter, tol=tol, solver="lbfgs",
                                 class_weight="balanced")
        clf.fit(sc.transform(Xl[tr]), y[tr])
        oof[va] = clf.predict_proba(sc.transform(Xl[va]))[:, 1]
    return compute_auc(oof, y)


def _layer(X, layer):
    """One layer of a (possibly memory-mapped) [N, n_layers, width] array as a
    contiguous float32 matrix; only that layer is read from disk."""
    return np.ascontiguousarray(X[:, layer, :]).astype(np.float32)


def fit_probe(pool_X, pool_y, test_X_by_fmt, c_grid, n_folds, seed, substrate,
              default_C=0.1, max_iter=500, tol=1e-3):
    """pool_X: [N_pool, n_layers, width]; test_X_by_fmt: fmt -> [N_test, n_layers, width]
    (memory-mapped arrays are fine: one layer is materialized at a time).

    Two-phase nested-CV selection to keep the fit count manageable:
      phase 1 -- sweep every layer at default_C, pick the best layer by OOF AUC
      phase 2 -- tune C (over c_grid) only at that best layer
    Features are standardized inside each fold. Returns
    (test_proba_by_fmt, layer_oof_rows, best_layer, best_C, best_auc)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    n_layers = pool_X.shape[1]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = list(skf.split(np.zeros(len(pool_y)), pool_y))

    print(f"    [{substrate}] phase 1: layer sweep over {n_layers} layers "
          f"(C={default_C}) ...", flush=True)
    layer_oof_rows, layer_auc = [], []
    for layer in range(n_layers):
        Xl = _layer(pool_X, layer)
        auc = _cv_auc(Xl, pool_y, folds, default_C, max_iter, tol)
        layer_auc.append(auc)
        layer_oof_rows.append({"substrate": substrate, "layer": layer,
                               "oof_auc": auc})
        print(f"    [{substrate}] layer {layer:2d}/{n_layers-1}  "
              f"OOF AUC = {auc:.3f}", flush=True)
    best_layer = int(np.nanargmax(layer_auc))

    print(f"    [{substrate}] phase 2: tuning C at layer {best_layer} ...",
          flush=True)
    Xl = _layer(pool_X, best_layer)
    best_C, best_auc = default_C, layer_auc[best_layer]
    for C in c_grid:
        auc = layer_auc[best_layer] if C == default_C else \
            _cv_auc(Xl, pool_y, folds, C, max_iter, tol)
        print(f"    [{substrate}] layer {best_layer}  C={C:<6}  "
              f"OOF AUC = {auc:.3f}", flush=True)
        if auc > best_auc:
            best_auc, best_C = auc, C

    sc = StandardScaler().fit(Xl)
    clf = LogisticRegression(C=best_C, max_iter=max_iter, tol=tol, solver="lbfgs",
                             class_weight="balanced")
    clf.fit(sc.transform(Xl), pool_y)
    test_proba = {}
    for fmt, TX in test_X_by_fmt.items():
        test_proba[fmt] = clf.predict_proba(sc.transform(_layer(TX, best_layer)))[:, 1]
    print(f"    [{substrate}] selected layer {best_layer}  C={best_C}  "
          f"OOF AUC={best_auc:.3f}", flush=True)
    return test_proba, layer_oof_rows, best_layer, best_C, float(best_auc)


# ---------------------------------------------------------------------------

def load_features(feat_dir):
    """Feature arrays are memory-mapped; probe_and_controls reads one layer at
    a time so the 32B models' multi-GB tensors never sit in RAM at once."""
    labels = list(csv.DictReader(open(os.path.join(feat_dir, "labels_conf.csv"))))
    lab_by_key = {(r["format"], r["id"]): r for r in labels}
    probe_ids = json.load(open(os.path.join(feat_dir, "probe", "numeric_ids.json")))
    probe = {
        "resid": np.load(os.path.join(feat_dir, "probe", "numeric_resid.npy"),
                         mmap_mode="r"),
        "mlp": np.load(os.path.join(feat_dir, "probe", "numeric_mlp.npy"),
                       mmap_mode="r"),
        "ids": probe_ids,
    }
    probe_y = np.array([int(lab_by_key[("numeric", i)]["y"]) for i in probe_ids])
    test = {}
    for fmt in VERBAL_FORMATS:
        idp = os.path.join(feat_dir, "test", f"{fmt}_ids.json")
        if not os.path.exists(idp):
            continue
        test[fmt] = {
            "resid": np.load(os.path.join(feat_dir, "test", f"{fmt}_resid.npy"),
                             mmap_mode="r"),
            "mlp": np.load(os.path.join(feat_dir, "test", f"{fmt}_mlp.npy"),
                           mmap_mode="r"),
            "ids": json.load(open(idp)),
        }
    return lab_by_key, probe, probe_y, test


def load_x_main(crossfit_path, mid, source):
    """x_main keyed by (format, id) from heldout's crossfit_item_loadings.csv
    (all models in one file): loading_crossfit (source='crossfit') or
    loading_full (source='full', the published all-items circuit)."""
    if not os.path.exists(crossfit_path):
        raise FileNotFoundError(
            f"{crossfit_path} missing: run heldout/scripts/crossfit_circuits.py")
    col = {"crossfit": "loading_crossfit", "full": "loading_full"}[source]
    out = {(r["format"], r["id"]): float(r[col])
           for r in csv.DictReader(open(crossfit_path)) if r["model"] == mid}
    if not out:
        raise ValueError(f"no rows for {mid} in {crossfit_path}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--c-grid", type=float, nargs="+",
                    default=[0.001, 0.01, 0.1, 1.0])
    ap.add_argument("--x-main-source", choices=["crossfit", "full"],
                    default="crossfit")
    ap.add_argument("--crossfit-path", default=None,
                    help="default attribution/heldout/results/"
                         "crossfit_item_loadings.csv")
    ap.add_argument("--results-dir", default=None,
                    help="default attribution/ensemble/results")
    args = ap.parse_args()

    ens_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ensemble/
    attr_root = os.path.dirname(ens_root)                                   # attribution/
    results_dir = args.results_dir or os.path.join(ens_root, "results")
    model_dir = os.path.join(results_dir, model_short(args.model))
    feat_dir = os.path.join(model_dir, "features")
    crossfit_path = args.crossfit_path or os.path.join(
        attr_root, "heldout", "results", "crossfit_item_loadings.csv")

    print(f"probe_and_controls  model={args.model}  x_main={args.x_main_source}",
          flush=True)
    lab_by_key, probe, probe_y, test = load_features(feat_dir)
    x_main_map = load_x_main(crossfit_path, args.model, args.x_main_source)
    print(f"  probe reps resid{probe['resid'].shape} mlp{probe['mlp'].shape}; "
          f"test formats={list(test)}", flush=True)
    print(f"  probe pool n={len(probe_y)} "
          f"({int(probe_y.sum())} correct / {int((1 - probe_y).sum())} incorrect)",
          flush=True)

    # --- probes (both substrates) -> x1a / x1b on the test items ---
    layer_rows = []
    proba = {}
    probe_info = {"model": args.model, "n_pool": int(len(probe_y)),
                  "n_pool_correct": int(probe_y.sum()), "substrates": {}}
    for pred, sub in SUBSTRATES.items():
        print(f"  fitting probe substrate={sub} ({pred}) ...", flush=True)
        test_X = {f: test[f][sub] for f in test}
        tp, lrows, bl, bc, ba = fit_probe(probe[sub], probe_y, test_X,
                                          args.c_grid, args.n_folds, args.seed, sub)
        proba[pred] = tp
        layer_rows.extend(lrows)
        probe_info["substrates"][sub] = {"predictor": pred, "layer": bl, "C": bc,
                                         "oof_auc": ba}

    # --- assemble per-item design matrix per format ---
    per_item_rows, design = [], {}
    for fmt in test:
        ids = test[fmt]["ids"]
        missing = [i for i in ids if (fmt, i) not in x_main_map]
        if missing:
            print(f"  [{fmt}] {len(missing)} test ids have no x_main; dropped",
                  flush=True)
        keep = [k for k, i in enumerate(ids) if (fmt, i) in x_main_map]
        ids = [ids[k] for k in keep]
        y = np.array([int(lab_by_key[(fmt, i)]["y"]) for i in ids], float)
        cols = {
            "x_main": np.array([x_main_map[(fmt, i)] for i in ids], float),
            "x1a": np.asarray(proba["x1a"][fmt], float)[keep],
            "x1b": np.asarray(proba["x1b"][fmt], float)[keep],
            "x2": np.array([float(lab_by_key[(fmt, i)]["x2"]) for i in ids], float),
            "x3": np.array([float(lab_by_key[(fmt, i)]["x3"]) for i in ids], float),
        }
        design[fmt] = (cols, y)
        for k, i in enumerate(ids):
            per_item_rows.append({"model": args.model, "format": fmt, "id": i,
                                  "y": int(y[k]),
                                  **{c: float(cols[c][k]) for c in PREDICTORS}})

    # Write the expensive outputs FIRST. The probe fits take hours and every
    # statistic is a pure function of per_item_scores.csv, so a failure in the
    # statistics below (as happened in the first v2 run, a missing statsmodels)
    # must never throw them away: stats_from_scores.py recomputes everything
    # locally from these files.
    _write_csv(os.path.join(model_dir, "per_item_scores.csv"),
               ["model", "format", "id", "y"] + PREDICTORS, per_item_rows)
    _write_csv(os.path.join(model_dir, "layer_cv_auc.csv"),
               ["substrate", "layer", "oof_auc"], layer_rows)
    probe_info["x_main_source"] = args.x_main_source
    json.dump(probe_info, open(os.path.join(model_dir, "probe_info.json"), "w"),
              indent=2)
    print("  probe outputs saved", flush=True)

    # --- statistics (recomputable locally with stats_from_scores.py) ---
    partial_rows, var_rows = [], []
    for fmt, (cols, y) in design.items():
        pp_rows, model_row = analyze_format(fmt, {c: cols[c] for c in PREDICTORS},
                                            y, model=args.model)
        var_rows.extend(pp_rows)
        partial_rows.append(model_row)
        aucs = "  ".join(f"{r['predictor']}={r['auroc']:.3f}" for r in pp_rows)
        print(f"  [{fmt}] n={model_row['n']} correct={model_row['n_correct']}"
              f"{' (low_n)' if model_row['low_n'] else ''}  AUROC: {aucs}", flush=True)
        print(f"          partial r(x_main|controls)={model_row['partial_r_xmain']:+.3f} "
              f"(p={model_row['partial_p_xmain']:.2g})  "
              f"LMG R2(x_main)={model_row['lmg_r2_xmain']:.4f}", flush=True)

    _write_csv(os.path.join(model_dir, "variance_partition.csv"),
               PER_PRED_FIELDS, var_rows)
    _write_csv(os.path.join(model_dir, "partial_corr.csv"),
               MODEL_FIELDS, partial_rows)
    print("done.", flush=True)


def _write_csv(path, fields, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {os.path.relpath(path)}  ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
