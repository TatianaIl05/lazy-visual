"""ROC, AUC, DeLong CI, bootstrap CI (pure numpy + scipy.rankdata)."""
from __future__ import annotations

import numpy as np

from .compare import StatResult


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(fn(y, x))


def _roc_curve(y_true: np.ndarray, y_score: np.ndarray):
    order = np.argsort(-y_score, kind="mergesort")
    y = y_true[order]
    s = y_score[order]
    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)
    distinct = np.where(np.diff(s))[0]
    idx = np.r_[distinct, len(s) - 1]
    tpr = np.r_[0.0, tps[idx] / max(tps[-1], 1)]
    fpr = np.r_[0.0, fps[idx] / max(fps[-1], 1)]
    return fpr, tpr, np.r_[np.inf, s[idx]]


def _auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    return _trapz(tpr, fpr)


def _delong_fast(pos: np.ndarray, neg: np.ndarray) -> tuple[float, float]:
    """DeLong (Sun & Xu 2014). Returns (auc, se).

    Optimized: single combined sort + single combined rankdata; ranks
    within each class computed directly from the combined ordering.
    """
    from scipy.stats import rankdata

    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        return float("nan"), float("nan")

    combined = np.concatenate([pos, neg])
    # one sort, one rankdata for combined
    r_combined = rankdata(combined, method="average")
    r_pos_comb = r_combined[:m]
    r_neg_comb = r_combined[m:]

    # ranks within each class via the same rankdata but on class-only arrays
    # (this is the only place we still pay rankdata twice more, but on
    #  half the data each -> cheaper than before)
    r_pos_within = rankdata(pos, method="average")
    r_neg_within = rankdata(neg, method="average")

    auc = (r_pos_comb.sum() - m * (m + 1) / 2.0) / (m * n)

    v10 = (r_pos_comb - r_pos_within) / n
    v01 = 1.0 - (r_neg_comb - r_neg_within) / m

    s10 = float(v10.var(ddof=1)) if m > 1 else 0.0
    s01 = float(v01.var(ddof=1)) if n > 1 else 0.0
    se = float(np.sqrt(s10 / m + s01 / n))
    return float(auc), se


def delong_ci(y_true: np.ndarray, y_score: np.ndarray, level: float = 0.95):
    from scipy import stats as sps
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    auc, se = _delong_fast(pos, neg)
    if not np.isfinite(se) or se == 0:
        return auc, auc, auc
    z = sps.norm.ppf(0.5 + level / 2)
    return auc, max(0.0, auc - z * se), min(1.0, auc + z * se)


def bootstrap_ci(y_true: np.ndarray, y_score: np.ndarray,
                 n_boot: int = 1000, level: float = 0.95,
                 random_state: int = 0):
    rng = np.random.default_rng(random_state)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    aucs = np.empty(n_boot, float)
    for k in range(n_boot):
        pi = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        ni = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([pi, ni])
        f, t, _ = _roc_curve(y_true[idx], y_score[idx])
        aucs[k] = _auc(f, t)
    a = (1 - level) / 2
    point = _auc(*_roc_curve(y_true, y_score)[:2])
    return point, float(np.quantile(aucs, a)), float(np.quantile(aucs, 1 - a))


def compute_roc(y_true: np.ndarray, y_score: np.ndarray, *,
                ci: bool = True, ci_method: str = "delong",
                ci_level: float = 0.95, n_boot: int = 1000,
                random_state: int = 0):
    fpr, tpr, _ = _roc_curve(y_true, y_score)
    auc_point = _auc(fpr, tpr)
    lo = hi = None
    if ci:
        if ci_method == "delong":
            auc_point, lo, hi = delong_ci(y_true, y_score, ci_level)
        else:
            auc_point, lo, hi = bootstrap_ci(
                y_true, y_score, n_boot, ci_level, random_state)
    stat = StatResult(method="ROC AUC", estimate=auc_point,
                      ci_low=lo, ci_high=hi,
                      ci_level=ci_level if ci else None,
                      n=int(len(y_true)),
                      random_state=random_state)
    return fpr, tpr, stat
