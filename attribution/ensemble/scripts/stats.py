"""Shared per-item statistics for the predictor comparison (ensemble) analysis.

All statistics operate on a per-format design matrix (y and the five
predictors), so they can be computed either on the cluster right after the
probes (probe_and_controls.py) or locally from per_item_scores.csv
(stats_from_scores.py) -- a single source of truth.

For each (model, test format) we report, per predictor:
  - direct predictive metrics of the predictor alone as a score of
    correctness on the held-out verbal items: AUROC and average precision
    (entropy is sign-flipped so that higher always means "more likely
    correct"), plus the zero-order point-biserial r;
  - its contribution under a linear probability model (OLS, y in {0,1}) with
    all five predictors: standardized coef + t + p, unique R^2 ("last":
    R2_full - R2_leave-one-out), and LMG relative importance (Shapley /
    relaimpo default);
and the x_main headline (partial correlation | controls) plus model totals.
"""

import numpy as np

PREDICTORS = ["x_main", "x1a", "x1b", "x2", "x3"]
# sign so that a larger oriented score means "more likely correct"
ORIENT = {"x_main": 1.0, "x1a": 1.0, "x1b": 1.0, "x2": 1.0, "x3": -1.0}
MIN_CORRECT = 30   # cells below this are flagged low_n (main-text rule)


def zscore(a):
    a = np.asarray(a, float)
    sd = a.std()
    return (a - a.mean()) / sd if sd > 0 else a - a.mean()


def compute_auc(scores, labels):
    from sklearn.metrics import roc_auc_score
    s = np.clip(np.asarray(scores, float), -1e9, 1e9)
    y = np.asarray(labels, int)
    if len(set(y.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def compute_ap(scores, labels):
    from sklearn.metrics import average_precision_score
    s = np.clip(np.asarray(scores, float), -1e9, 1e9)
    y = np.asarray(labels, int)
    if len(set(y.tolist())) < 2:
        return float("nan")
    return float(average_precision_score(y, s))


# ---------------------------------------------------------------------------
# LMG / Shapley relative importance (relaimpo's default metric)
# ---------------------------------------------------------------------------

def _lmg_decomposition(Zcols, y, subset_fit):
    """Exact LMG relative importance: predictor j's share = its marginal fit
    improvement averaged over all orderings in which it can enter the model
    (Shapley value over the 2^p predictor subsets). Shares sum to the
    full-model fit. relaimpo::calc.relimp's default 'lmg' for the linear R^2.

    subset_fit(Z) -> scalar fit of the model on columns Z (empty set = 0)."""
    from itertools import combinations
    from math import factorial

    p = len(PREDICTORS)
    cache = {frozenset(): 0.0}

    def fit(subset):
        if subset not in cache:
            Z = np.column_stack([Zcols[c] for c in PREDICTORS if c in subset])
            cache[subset] = float(subset_fit(Z))
        return cache[subset]

    lmg = {}
    for j in PREDICTORS:
        others = [c for c in PREDICTORS if c != j]
        total = 0.0
        for k in range(p):
            w = factorial(k) * factorial(p - k - 1) / factorial(p)
            for S in combinations(others, k):
                fs = frozenset(S)
                total += w * (fit(fs | {j}) - fit(fs))
        lmg[j] = float(total)
    return lmg


# ---------------------------------------------------------------------------
# linear probability model (OLS)
# ---------------------------------------------------------------------------

def _ols_r2(Z, y):
    Xd = np.column_stack([np.ones(len(y)), Z]) if Z.size else np.ones((len(y), 1))
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def ols_fit(y, Z):
    """Classical OLS with an intercept: coefficients, standard errors, t and
    two-sided p values (nonrobust covariance, residual df = n - k - 1).

    Written with numpy/scipy so the cluster environment needs no statsmodels
    (its absence made every v2 analysis job fail after the probe fits). It is
    the same estimator as statsmodels.OLS(y, add_constant(Z)).fit(); the self
    test checks the two agree when statsmodels is installed.
    Returns arrays aligned with [intercept, Z columns...]."""
    from scipy.stats import t as t_dist
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y)), Z])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    df = X.shape[0] - X.shape[1]
    sigma2 = float(resid @ resid) / df
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    tvals = beta / se
    pvals = 2 * t_dist.sf(np.abs(tvals), df)
    return beta, se, tvals, pvals


def linear_model(Zcols, y):
    """Zcols: dict predictor -> standardized 1D array. Returns (per_pred, meta)."""
    Z = np.column_stack([Zcols[p] for p in PREDICTORS])
    r2_full = _ols_r2(Z, y)
    params, bse, tvalues, pvalues = ols_fit(y, Z)
    lmg = _lmg_decomposition(Zcols, y, lambda Zs: _ols_r2(Zs, y))
    per_pred = {}
    for j, p in enumerate(PREDICTORS):
        others = np.column_stack([Zcols[o] for o in PREDICTORS if o != p])
        r2_wo = _ols_r2(others, y)
        per_pred[p] = {
            "lin_coef": float(params[j + 1]),
            "lin_se": float(bse[j + 1]),
            "lin_t": float(tvalues[j + 1]),
            "lin_p": float(pvalues[j + 1]),
            "unique_r2_linear": float(r2_full - r2_wo),
            "lmg_r2_linear": lmg[p],
        }
    return per_pred, {"r2_full_linear": float(r2_full)}


# ---------------------------------------------------------------------------
# x_main headline: partial correlation | controls
# ---------------------------------------------------------------------------

def partial_corr_xmain(Zcols, y):
    from scipy.stats import pearsonr
    controls = np.column_stack([Zcols[p] for p in PREDICTORS if p != "x_main"])
    C = np.column_stack([np.ones(len(y)), controls])

    def resid(v):
        beta, *_ = np.linalg.lstsq(C, v, rcond=None)
        return v - C @ beta
    rx, ry = resid(Zcols["x_main"]), resid(np.asarray(y, float))
    if rx.std() == 0 or ry.std() == 0:
        return float("nan"), float("nan"), (float("nan"), float("nan"))
    r, p = pearsonr(rx, ry)
    k = controls.shape[1]
    ci = _fisher_ci(float(r), len(y), k)
    return float(r), float(p), ci


def _fisher_ci(r, n, k, alpha=0.05):
    from scipy.stats import norm
    # |r| == 1 is finite but arctanh diverges; it only arises for a degenerate
    # predictor (an exact copy of the label), where an interval is meaningless.
    if not np.isfinite(r) or abs(r) >= 1.0 or n - k - 3 <= 0:
        return (float("nan"), float("nan"))
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - k - 3)
    lo, hi = z - norm.ppf(1 - alpha / 2) * se, z + norm.ppf(1 - alpha / 2) * se
    return (float(np.tanh(lo)), float(np.tanh(hi)))


# ---------------------------------------------------------------------------
# top-level: everything for one (model, format)
# ---------------------------------------------------------------------------

def analyze_format(fmt, cols, y, model=""):
    """cols: dict predictor -> raw 1D array; y: 1D 0/1. Returns
    (per_pred_rows, model_row)."""
    from scipy.stats import pointbiserialr
    y = np.asarray(y, float)
    n_correct = int(y.sum())
    Zcols = {p: zscore(cols[p]) for p in PREDICTORS}

    lin_pp, lin_meta = linear_model(Zcols, y)
    pr, pp_p, (ci_lo, ci_hi) = partial_corr_xmain(Zcols, y)

    per_pred_rows = []
    for p in PREDICTORS:
        zo = pointbiserialr(y, cols[p])
        oriented = ORIENT[p] * np.asarray(cols[p], float)
        per_pred_rows.append({
            "model": model, "format": fmt, "predictor": p,
            "n": int(len(y)), "n_correct": n_correct,
            "low_n": int(n_correct < MIN_CORRECT),
            "auroc": compute_auc(oriented, y),
            "avg_precision": compute_ap(oriented, y),
            "zero_order_r": float(zo[0]), "zero_order_p": float(zo[1]),
            **lin_pp[p],
        })

    model_row = {
        "model": model, "format": fmt, "n": int(len(y)),
        "n_correct": n_correct, "acc_rate": float(y.mean()),
        "low_n": int(n_correct < MIN_CORRECT),
        "partial_r_xmain": pr, "partial_ci_lo": ci_lo, "partial_ci_hi": ci_hi,
        "partial_p_xmain": pp_p,
        "zero_order_r_xmain": float(pointbiserialr(y, cols["x_main"])[0]),
        "auroc_xmain": compute_auc(cols["x_main"], y),
        "r2_full_linear": lin_meta["r2_full_linear"],
        "unique_r2_xmain": lin_pp["x_main"]["unique_r2_linear"],
        "lmg_r2_xmain": lin_pp["x_main"]["lmg_r2_linear"],
    }
    return per_pred_rows, model_row


PER_PRED_FIELDS = [
    "model", "format", "predictor", "n", "n_correct", "low_n",
    "auroc", "avg_precision", "zero_order_r", "zero_order_p",
    "lin_coef", "lin_se", "lin_t", "lin_p",
    "unique_r2_linear", "lmg_r2_linear",
]

MODEL_FIELDS = [
    "model", "format", "n", "n_correct", "acc_rate", "low_n",
    "partial_r_xmain", "partial_ci_lo", "partial_ci_hi", "partial_p_xmain",
    "zero_order_r_xmain", "auroc_xmain", "r2_full_linear",
    "unique_r2_xmain", "lmg_r2_xmain",
]
