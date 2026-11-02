from __future__ import annotations

import pandas as pd

from .._errors import err_upset_sets_count


def _to_membership(data, *, sets=None, item=None, set_col=None):
    if isinstance(data, dict):
        names = list(data.keys())
        universe = sorted(set().union(*data.values()))
        return pd.DataFrame({n: [u in data[n] for u in universe] for n in names},
                            index=universe)
    if isinstance(data, pd.DataFrame):
        if all(pd.api.types.is_bool_dtype(data[c]) for c in data.columns):
            return data.astype(bool)
        if item and set_col:
            return pd.crosstab(data[item], data[set_col]) > 0
        if all(pd.api.types.is_numeric_dtype(data[c]) for c in data.columns):
            return (data > 0).astype(bool)
        raise ValueError("Cannot infer membership; provide item/set_col.")
    raise ValueError("Unsupported input for upset.")

def compute_intersections(data, *, sets=None, item=None, set_col=None,
                          top_n=30, min_size=1, sort_by="size"):
    m = _to_membership(data, sets=sets, item=item, set_col=set_col)
    n_sets = m.shape[1]
    if not (2 <= n_sets <= 20):
        raise err_upset_sets_count(n_sets)
    cols = list(m.columns)
    patterns = m.apply(lambda r: tuple(c for c in cols if r[c]), axis=1)
    counts = {}
    for p in patterns:
        if not p:
            continue
        counts[p] = counts.get(p, 0) + 1
    rows = []
    for pattern, size in counts.items():
        if size < min_size:
            continue
        rows.append({"pattern": pattern, "size": size, "degree": len(pattern)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if sort_by == "size":
        df = df.sort_values(["size", "degree"], ascending=[False, False])
    else:
        df = df.sort_values(["degree", "size"], ascending=[False, False])
    return df.head(top_n).reset_index(drop=True)
