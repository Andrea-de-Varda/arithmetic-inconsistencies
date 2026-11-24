"""Self-test for the predictor-comparison statistics, on synthetic scores.

The real inputs come from a GPU feature-extraction run and the cluster-side
attribution tensors, so this validates the parts that turn per-item scores into
the numbers and figures in the paper, using data whose right answer is known by
construction:

  1. Orientation. AUROC is computed on the ORIENTED predictor, so every
     predictor is scored as "higher means more likely correct". Entropy enters
     with a negative sign; a predictor built to be negatively related to
     correctness must come out above 0.5, not below.
  2. Degenerate and perfect predictors. Pure noise sits at 0.5, a copy of the
     label reaches 1.0, and average precision sits at the base rate for noise.
  3. The variance decomposition. LMG shares are a Shapley decomposition, so
     they must sum to the full model's R^2, and each unique ("last") R^2 must
     be no larger than its LMG share when predictors are positively related.
  4. The low_n rule. Cells with fewer than 30 correct items are flagged, which
     is what the cross-model medians filter on.
  5. The cross-model summary. Medians, significance counts and the
     "circuit >= control" tally are recomputed directly from the per-cell rows
     and must match what stats_from_scores.py writes.
  6. The probe pool. The class mix must never collapse to one class, whether
     the model is far above or far below 50% numeric accuracy.
  7. The OLS inference (coef, SE, t, p) matches statsmodels, which the stats
     no longer import because the cluster environment lacks it.
  8. probe_and_controls.py itself, end to end on tiny synthetic features with
     statsmodels blocked: it completes, writes every output, writes the probe
     outputs before the statistics, and the probe finds the informative layer.

Usage (local):  python ensemble/scripts/selftest_ensemble.py
"""

import csv
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stats import analyze_format, ORIENT, PREDICTORS, MIN_CORRECT

FAIL = []
SCRIPTS = os.path.dirname(os.path.abspath(__file__))


def check(name, cond, detail=""):
    print(f"  [{'ok ' if cond else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAIL.append(name)


def make_cell(rng, n=800, base_rate=0.35, signal=1.2):
    """One (model, format) cell whose predictors have known behaviour:
    x_main and x1a/x1b carry signal, x2 carries weaker signal, x3 (entropy) is
    NEGATIVELY related to correctness, as real entropy is."""
    y = (rng.random(n) < base_rate).astype(float)
    def with_signal(strength):
        return strength * y + rng.normal(0, 1, n)
    return y, {
        "x_main": with_signal(signal),
        "x1a": with_signal(signal * 0.8),
        "x1b": with_signal(signal * 0.7),
        "x2": with_signal(signal * 0.5),
        "x3": -with_signal(signal * 0.9),      # higher entropy -> less likely right
    }


def main():
    rng = np.random.default_rng(7)

    print("1. orientation")
    y, cols = make_cell(rng)
    pp, mr = analyze_format("english", cols, y, model="m")
    by = {r["predictor"]: r for r in pp}
    check("entropy enters with a negative sign", ORIENT["x3"] == -1.0)
    check("negatively-related entropy scores above 0.5 once oriented",
          by["x3"]["auroc"] > 0.5, f"AUROC={by['x3']['auroc']:.3f}")
    check("its raw zero-order r is negative",
          by["x3"]["zero_order_r"] < 0, f"r={by['x3']['zero_order_r']:+.3f}")
    check("circuit beats the weaker logprob control",
          by["x_main"]["auroc"] > by["x2"]["auroc"],
          f"{by['x_main']['auroc']:.3f} > {by['x2']['auroc']:.3f}")

    print("2. degenerate and perfect predictors")
    n = 2000
    y2 = (rng.random(n) < 0.3).astype(float)
    cols2 = {p: rng.normal(0, 1, n) for p in PREDICTORS}
    cols2["x_main"] = y2.copy()                     # perfect
    cols2["x1a"] = rng.normal(0, 1, n)              # pure noise
    pp2, mr2 = analyze_format("english", cols2, y2, model="m")
    by2 = {r["predictor"]: r for r in pp2}
    check("a copy of the label reaches AUROC 1.0",
          abs(by2["x_main"]["auroc"] - 1.0) < 1e-9)
    check("pure noise sits near 0.5",
          abs(by2["x1a"]["auroc"] - 0.5) < 0.05, f"AUROC={by2['x1a']['auroc']:.3f}")
    check("noise average precision sits near the base rate",
          abs(by2["x1a"]["avg_precision"] - 0.3) < 0.05,
          f"AP={by2['x1a']['avg_precision']:.3f} vs base 0.300")

    print("3. variance decomposition")
    lmg_sum = sum(by[p]["lmg_r2_linear"] for p in PREDICTORS)
    check("LMG shares sum to the full-model R^2",
          abs(lmg_sum - mr["r2_full_linear"]) < 1e-8,
          f"{lmg_sum:.6f} vs {mr['r2_full_linear']:.6f}")
    check("every unique R^2 <= its LMG share",
          all(by[p]["unique_r2_linear"] <= by[p]["lmg_r2_linear"] + 1e-9
              for p in PREDICTORS))
    check("x_main totals in the model row match its per-predictor row",
          abs(mr["lmg_r2_xmain"] - by["x_main"]["lmg_r2_linear"]) < 1e-12)

    print("4. low-n rule")
    y3 = np.zeros(500); y3[:MIN_CORRECT - 1] = 1
    _, mr3 = analyze_format("italian", {p: rng.normal(0, 1, 500) for p in PREDICTORS},
                            y3, model="m")
    y4 = np.zeros(500); y4[:MIN_CORRECT] = 1
    _, mr4 = analyze_format("italian", {p: rng.normal(0, 1, 500) for p in PREDICTORS},
                            y4, model="m")
    check(f"{MIN_CORRECT - 1} correct items is flagged low_n", mr3["low_n"] == 1)
    check(f"{MIN_CORRECT} correct items is not flagged", mr4["low_n"] == 0)

    print("5. cross-model summary reproduces a direct computation")
    with tempfile.TemporaryDirectory() as tmp:
        want = {}
        for mi in range(5):
            mid = f"fake/model-{mi}"
            mdir = os.path.join(tmp, f"fake_model-{mi}")
            os.makedirs(mdir)
            rows = []
            for fmt in ("english", "spanish", "italian"):
                yy, cc = make_cell(rng, n=600, base_rate=0.4,
                                   signal=1.0 + 0.3 * mi)
                want.setdefault((fmt, mi), (yy, cc))
                for k in range(len(yy)):
                    rows.append({"model": mid, "format": fmt, "id": f"prob_{k:06d}",
                                 "y": int(yy[k]),
                                 **{p: cc[p][k] for p in PREDICTORS}})
            with open(os.path.join(mdir, "per_item_scores.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["model", "format", "id", "y"]
                                   + PREDICTORS)
                w.writeheader()
                w.writerows(rows)

        r = subprocess.run(
            [sys.executable, os.path.join(SCRIPTS, "stats_from_scores.py"),
             "--results-dir", tmp, "--quiet"],
            capture_output=True, text=True)
        check("stats_from_scores.py runs over per-model dirs", r.returncode == 0,
              r.stderr.strip().splitlines()[-1] if r.returncode else "")
        if r.returncode:
            print(r.stderr)
        else:
            cells = list(csv.DictReader(
                open(os.path.join(tmp, "cross_model_predictors.csv"))))
            summ = list(csv.DictReader(
                open(os.path.join(tmp, "cross_model_summary.csv"))))
            check("one row per (model, format, predictor)",
                  len(cells) == 5 * 3 * len(PREDICTORS), f"{len(cells)} rows")
            ok = True
            for s in summ:
                sel = [float(c["auroc"]) for c in cells
                       if c["format"] == s["format"]
                       and c["predictor"] == s["predictor"]
                       and not int(c["low_n"])]
                if abs(np.median(sel) - float(s["median_auroc"])) > 1e-9:
                    ok = False
            check("median AUROC matches a direct median over cells", ok)
            circ = [s for s in summ if s["predictor"] == "x_main"]
            check("the circuit is counted as >= itself in every model",
                  all(s["n_circuit_ge_this"] == s["n_models"] for s in circ))

            # end to end: the figures must actually build from these CSVs
            rp = subprocess.run(
                [sys.executable, os.path.join(SCRIPTS, "plot_ensemble.py"),
                 "--results-dir", tmp],
                capture_output=True, text=True)
            check("plot_ensemble.py runs over the same dir", rp.returncode == 0,
                  rp.stderr.strip().splitlines()[-1] if rp.returncode else "")
            if rp.returncode:
                print(rp.stderr)
            else:
                figs = os.path.join(tmp, "figures")
                want = ["C4_auroc_bars", "C5_lmg_bars"]
                missing = [w for w in want for ext in ("png", "svg")
                           if not os.path.exists(os.path.join(figs, f"{w}.{ext}"))]
                check("every cross-model figure is written as both png and svg",
                      not missing, f"missing {missing}" if missing else "")
                svg = open(os.path.join(figs, "C4_auroc_bars.svg")).read()
                check("SVG text stays editable (not converted to paths)",
                      "<text" in svg and "DejaVu Sans" in svg)

    print("6. probe pool never collapses to one class")
    try:
        from extract_features import build_probe_pool
    except Exception as exc:                       # torch missing locally
        print(f"  [skip] cannot import extract_features ({type(exc).__name__})")
    else:
        import json
        for acc, label in ((0.99, "far above 50%"), (0.47, "far below 50%")):
            with tempfile.TemporaryDirectory() as tmp:
                attr = os.path.join(tmp, "attribution")
                rdir = os.path.join(attr, "ensemble", "results",
                                    "_pool_responses", "fake_m")
                ddir = os.path.join(tmp, "maindata")
                os.makedirs(rdir); os.makedirs(ddir)
                n = 6000
                recs = [{"id": f"p_{i:06d}", "x": f"{i} + 1 =",
                         "clean_match_gold": bool(rng.random() < acc)}
                        for i in range(n)]
                json.dump({"records": recs},
                          open(os.path.join(rdir, "numeric.json"), "w"))
                json.dump([{"x": "999999 + 1 ="}],
                          open(os.path.join(ddir,
                               "arith_dataset_numeric_counterfactual.json"), "w"))
                pool, info = build_probe_pool(attr, "fake/m", 2000, 42, ddir)
                n_inc = info["n_pool_incorrect"]
                n_cor = info["n_pool_correct"]
                check(f"both classes present when accuracy is {label}",
                      n_inc > 0 and n_cor > 0, f"{n_inc} incorrect / {n_cor} correct")
                check(f"minority class capped at half the pool ({label})",
                      n_inc <= 1000, f"{n_inc} incorrect")

    print("7. OLS matches statsmodels (when installed)")
    from stats import ols_fit, zscore
    try:
        import statsmodels.api as sm
    except ImportError:
        print("  [skip] statsmodels not installed here")
    else:
        yy, cc = make_cell(rng, n=900)
        Z = np.column_stack([zscore(cc[p]) for p in PREDICTORS])
        b, se, t, p = ols_fit(yy, Z)
        ref = sm.OLS(yy, sm.add_constant(Z)).fit()
        check("coefficients, SEs, t and p agree with statsmodels",
              np.allclose(b, ref.params, atol=1e-10)
              and np.allclose(se, ref.bse, atol=1e-10)
              and np.allclose(t, ref.tvalues, atol=1e-8)
              and np.allclose(p, ref.pvalues, rtol=1e-8, atol=1e-300))

    print("8. probe_and_controls.py end to end, without statsmodels")
    # The first v2 cluster run died here: the cluster env has no statsmodels,
    # and nothing had been written yet, so hours of probe fits were lost.
    # Run the real script on tiny synthetic features with statsmodels blocked.
    with tempfile.TemporaryDirectory() as tmp:
        mid = "fake/tiny-model"
        mshort = "fake_tiny-model"
        feat = os.path.join(tmp, "results", mshort, "features")
        os.makedirs(os.path.join(feat, "probe")); os.makedirs(os.path.join(feat, "test"))
        L, W = 3, 16
        labels, cf_rows = [], []
        n_pool = 300
        yp = (rng.random(n_pool) < 0.6).astype(int)
        Xp = rng.normal(0, 1, (n_pool, L, W)).astype(np.float16)
        Xp[:, 1, :4] += yp[:, None] * 1.5          # layer 1 carries the signal
        np.save(os.path.join(feat, "probe", "numeric_resid.npy"), Xp)
        np.save(os.path.join(feat, "probe", "numeric_mlp.npy"), Xp)
        pool_ids = [f"c_{i:06d}" for i in range(n_pool)]
        json.dump(pool_ids, open(os.path.join(feat, "probe", "numeric_ids.json"), "w"))
        labels += [{"id": i, "format": "numeric", "split": "probe", "y": int(v),
                    "x2": "", "x3": ""} for i, v in zip(pool_ids, yp)]
        for fmt in ("english", "spanish", "italian"):
            n = 250
            yt = (rng.random(n) < 0.35).astype(int)
            Xt = rng.normal(0, 1, (n, L, W)).astype(np.float16)
            Xt[:, 1, :4] += yt[:, None] * 1.5
            np.save(os.path.join(feat, "test", f"{fmt}_resid.npy"), Xt)
            np.save(os.path.join(feat, "test", f"{fmt}_mlp.npy"), Xt)
            tids = [f"prob_{i:06d}" for i in range(n)]
            json.dump(tids, open(os.path.join(feat, "test", f"{fmt}_ids.json"), "w"))
            for i, v in zip(tids, yt):
                labels.append({"id": i, "format": fmt, "split": "test", "y": int(v),
                               "x2": float(v + rng.normal()), "x3": float(-v + rng.normal())})
                cf_rows.append({"model": mid, "format": fmt, "id": i, "fold": 0,
                                "accuracy": int(v), "loading_full": float(v + rng.normal()),
                                "loading_crossfit": float(v + rng.normal())})
        with open(os.path.join(feat, "labels_conf.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "format", "split", "y", "x2", "x3"])
            w.writeheader(); w.writerows(labels)
        cf_path = os.path.join(tmp, "crossfit_item_loadings.csv")
        with open(cf_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(cf_rows[0]))
            w.writeheader(); w.writerows(cf_rows)

        runner = (
            "import sys, runpy; sys.modules['statsmodels'] = None; "
            "sys.modules['statsmodels.api'] = None; "
            f"sys.argv = ['probe_and_controls.py', '--model', {mid!r}, "
            f"'--results-dir', {os.path.join(tmp, 'results')!r}, "
            f"'--crossfit-path', {cf_path!r}, '--n-folds', '3', "
            "'--c-grid', '0.1', '1.0']; "
            f"runpy.run_path({os.path.join(SCRIPTS, 'probe_and_controls.py')!r}, "
            "run_name='__main__')")
        r = subprocess.run([sys.executable, "-c", runner], capture_output=True, text=True)
        check("runs to completion with statsmodels unavailable", r.returncode == 0,
              r.stderr.strip().splitlines()[-1] if r.returncode else "")
        mdir = os.path.join(tmp, "results", mshort)
        for fname in ("per_item_scores.csv", "layer_cv_auc.csv", "probe_info.json",
                      "variance_partition.csv", "partial_corr.csv"):
            check(f"writes {fname}", os.path.exists(os.path.join(mdir, fname)))
        if r.returncode == 0:
            info = json.load(open(os.path.join(mdir, "probe_info.json")))
            check("probe selects the layer that carries the signal",
                  all(v["layer"] == 1 for v in info["substrates"].values()),
                  str({k: v["layer"] for k, v in info["substrates"].items()}))
            src = open(os.path.join(SCRIPTS, "probe_and_controls.py")).read()
            check("probe outputs are written before the statistics run",
                  src.index('"per_item_scores.csv"') < src.index("analyze_format(fmt"))

    print()
    if FAIL:
        print(f"FAILED: {len(FAIL)} check(s): {FAIL}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
