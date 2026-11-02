from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .._errors import (
    LvDataError,
    LvDataWarning,
    err_column_not_found,
    err_duplicate_columns,
    err_empty_after_nan,
    err_wrong_dtype,
)


def to_dataframe(data):
    if hasattr(data, "to_pandas") and type(data).__module__.startswith("polars"):
        return data.to_pandas()
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, pd.Series):
        return data.to_frame()
    if isinstance(data, dict):
        return pd.DataFrame(data)
    if isinstance(data, np.ndarray):
        arr = data if data.ndim == 2 else data.reshape(-1, 1)
        cols = [f"x{i}" for i in range(arr.shape[1])]
        return pd.DataFrame(arr, columns=cols)
    if isinstance(data, (list, tuple)):
        arr = np.asarray(data)
        if arr.ndim == 1:
            return pd.DataFrame({"x0": arr})
        if arr.ndim == 2:
            cols = [f"x{i}" for i in range(arr.shape[1])]
            return pd.DataFrame(arr, columns=cols)
    raise LvDataError(f"Unsupported data type: {type(data).__name__}.",
        code="LV-E102", hint="pass DataFrame, dict, numpy array or list")

def _check_unique_columns(df):
    dupes = df.columns[df.columns.duplicated()].tolist()
    if dupes:
        raise err_duplicate_columns(list({str(c) for c in dupes}))

def check_columns(df, needed):
    available = [str(c) for c in df.columns]
    for name in needed:
        if name not in df.columns:
            raise err_column_not_found(name, available)

def prepare(data, needed, *, numeric=None, categorical=None, dropna=True):
    df = to_dataframe(data)
    _check_unique_columns(df)
    needed = list(needed)
    check_columns(df, needed)
    for col in (numeric or []):
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise err_wrong_dtype(col, "numeric", str(df[col].dtype))
    out = df[needed].copy()
    for col in (categorical or []):
        if not isinstance(out[col].dtype, pd.CategoricalDtype):
            out[col] = pd.Categorical(out[col])
    n_before = len(out)
    if dropna:
        out = out.dropna()
    n_dropped = n_before - len(out)
    if n_dropped > 0:
        warnings.warn(
            f"[LV-W501] Dropped {n_dropped} rows with NaN in required columns.",
            LvDataWarning, stacklevel=3)
    if len(out) == 0:
        raise err_empty_after_nan(n_before, n_dropped)
    return out, n_dropped

def ordered_categories(s):
    if isinstance(s.dtype, pd.CategoricalDtype):
        return [c for c in s.cat.categories if (s == c).any()]
    return list(pd.unique(s.dropna()))
