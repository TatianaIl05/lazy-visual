from __future__ import annotations

import io as _io
import urllib.request
from pathlib import Path

import pandas as pd

from .. import defaults
from .._errors import (
    LvPerformanceWarning,
    err_file_not_found,
    err_missing_dependency,
    err_unsupported_format,
    err_url_unreachable,
)


def _ext(path):
    return Path(path).suffix.lower().lstrip(".")

def _read_bytes(payload, fmt, read_kwargs):
    buf = _io.BytesIO(payload)
    if fmt == "csv":
        return pd.read_csv(buf, **read_kwargs)
    if fmt in ("tsv", "tab"):
        return pd.read_csv(buf, sep="\t", **read_kwargs)
    if fmt in ("parquet", "pq"):
        try:
            return pd.read_parquet(buf, **read_kwargs)
        except ImportError as e:
            raise err_missing_dependency("pyarrow", "parquet") from e
    if fmt in ("xlsx", "xls"):
        try:
            return pd.read_excel(buf, **read_kwargs)
        except ImportError as e:
            raise err_missing_dependency("openpyxl", "excel") from e
    if fmt == "json":
        return pd.read_json(buf, **read_kwargs)
    raise err_unsupported_format(fmt)

def load(source, *, fmt=None, **read_kwargs):
    if isinstance(source, pd.DataFrame):
        return source
    s = str(source)
    if s.startswith(("http://", "https://")):
        if fmt is None:
            fmt = _ext(s.split("?")[0]) or "csv"
        try:
            with urllib.request.urlopen(s, timeout=defaults.URL_TIMEOUT_S) as r:
                payload = r.read()
        except Exception as e:
            raise err_url_unreachable(s, str(e)) from e
        return _read_bytes(payload, fmt, read_kwargs)
    path = Path(s)
    if not path.exists() or not path.is_file():
        raise err_file_not_found(s)
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > 500:
        import warnings
        warnings.warn(f"[LV-W503] File is {size_mb:.0f} MB.",
            LvPerformanceWarning, stacklevel=2)
    fmt = fmt or _ext(s)
    if fmt == "csv":
        return pd.read_csv(path, **read_kwargs)
    if fmt in ("tsv", "tab"):
        return pd.read_csv(path, sep="\t", **read_kwargs)
    if fmt in ("parquet", "pq"):
        try:
            return pd.read_parquet(path, **read_kwargs)
        except ImportError as e:
            raise err_missing_dependency("pyarrow", "parquet") from e
    if fmt in ("xlsx", "xls"):
        try:
            return pd.read_excel(path, **read_kwargs)
        except ImportError as e:
            raise err_missing_dependency("openpyxl", "excel") from e
    if fmt == "json":
        return pd.read_json(path, **read_kwargs)
    raise err_unsupported_format(fmt)
