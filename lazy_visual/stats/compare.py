from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .._errors import LvDataWarning


@dataclass(frozen=True)
class StatResult:
    method: str
    statistic: float | None = None
    p_value: float | None = None
    p_adjusted: float | None = None
    effect_size: float | None = None
    estimate: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    ci_level: float | None = None
    n: int | None = None
    groups: tuple = ()
    random_state: int | None = None

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan")
    s1, s2 = a.var(ddof=1), b.var(ddof=1)
    sp = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if sp == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / sp)


def holm(pvals: list) -> list:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    prev = 0.0
    for rank, idx in enumerate(order, start=1):
        val = min(1.0, (m - rank + 1) * pvals[idx])
        val = max(val, prev)
        adj[idx] = val
        prev = val
    return adj


def _sample_warn(groups: dict) -> None:
    for name, arr in groups.items():
        if len(arr) < 3:
            warnings.warn(
                f"[LV-W504] Group '{name}' has <3 observations; stats skipped.",
                LvDataWarning, stacklevel=3)
        elif np.all(arr == arr[0]):
            warnings.warn(
                f"[LV-W504] Group '{name}' is constant; stats skipped.",
                LvDataWarning, stacklevel=3)


def compare_groups(df: pd.DataFrame, *, y: str, x: str | None,
                   test: str = "auto",
                   pairs: list | None = None,
                   random_state: int | None = None) -> list:
    from scipy import stats as sps
    if x is None:
        return []
    groups = {str(k): v[y].to_numpy() for k, v in df.groupby(x, observed=True)}
    _sample_warn(groups)
    names = list(groups.keys())
    results: list = []
    if test == "none" or len(names) < 2:
        return results
    if test == "auto":
        test = "mannwhitney" if len(names) == 2 else "kruskal"
    if test == "kruskal":
        samples = [groups[g] for g in names]
        try:
            stat, p = sps.kruskal(*samples)
        except ValueError:
            return results
        results.append(StatResult(
            method="Kruskal-Wallis", statistic=float(stat),
            p_value=float(p), n=int(sum(len(s) for s in samples)),
            groups=tuple(names), random_state=random_state))
        if pairs:
            raw = []
            pair_stats = []
            for g1, g2 in pairs:
                if g1 not in groups or g2 not in groups:
                    continue
                try:
                    u, p2 = sps.mannwhitneyu(
                        groups[g1], groups[g2], alternative="two-sided")
                except ValueError:
                    continue
                raw.append(float(p2))
                pair_stats.append(
                    (g1, g2, float(u), float(p2),
                     cohens_d(groups[g1], groups[g2])))
            if raw:
                adj = holm(raw)
                for (g1, g2, u, p2, d), pa in zip(pair_stats, adj):
                    results.append(StatResult(
                        method="Mann-Whitney U", statistic=u,
                        p_value=p2, p_adjusted=pa, effect_size=d,
                        n=int(len(groups[g1]) + len(groups[g2])),
                        groups=(g1, g2), random_state=random_state))
        return results
    if test == "mannwhitney" and len(names) == 2:
        g1, g2 = names
        try:
            u, p = sps.mannwhitneyu(
                groups[g1], groups[g2], alternative="two-sided")
        except ValueError:
            return results
        results.append(StatResult(
            method="Mann-Whitney U", statistic=float(u),
            p_value=float(p),
            effect_size=cohens_d(groups[g1], groups[g2]),
            n=int(len(groups[g1]) + len(groups[g2])),
            groups=(g1, g2), random_state=random_state))
        return results
    if test == "welch" and len(names) == 2:
        g1, g2 = names
        try:
            t, p = sps.ttest_ind(
                groups[g1], groups[g2], equal_var=False)
        except ValueError:
            return results
        results.append(StatResult(
            method="Welch t-test", statistic=float(t), p_value=float(p),
            effect_size=cohens_d(groups[g1], groups[g2]),
            n=int(len(groups[g1]) + len(groups[g2])),
            groups=(g1, g2), random_state=random_state))
        return results
    return results


def format_p(p: float | None) -> str:
    if p is None:
        return ""
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def format_d(d: float | None) -> str:
    if d is None or (isinstance(d, float) and np.isnan(d)):
        return ""
    return f"d = {d:.2f}"
