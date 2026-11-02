from __future__ import annotations

import warnings

from .. import defaults
from .._errors import LvPerformanceWarning
from ..core.result import LvResult
from ..core.style import StyleArgs
from ..export import save_result


def build_style(kwargs):
    allowed = set(StyleArgs.__dataclass_fields__)
    st = StyleArgs(**{k: v for k, v in kwargs.items() if k in allowed})
    st.validate()
    return st

def check_rows(df):
    if len(df) > defaults.PERF_WARN_ROWS:
        warnings.warn(
            f"[LV-W503] Large input ({len(df)} rows); rendering may take >3 s.",
            LvPerformanceWarning, stacklevel=3)

def subsample(df, n, seed):
    if n is None or len(df) <= n:
        return df
    warnings.warn(f"[LV-W502] Subsampled {len(df)} rows to {n}.",
        UserWarning, stacklevel=3)
    return df.sample(n=n, random_state=seed)

def finalize_result(fig, ax, spec, style, stats=None, tables=None):
    files = []
    if style.save is not None:
        files = save_result(fig, style.save, fmt=style.fmt,
                            dpi=style.dpi or defaults.DPI_SAVE)
    return LvResult(fig=fig, ax=ax, spec=spec, stats=list(stats or []),
                    tables=dict(tables or {}), files=files)

def apply_labels(ax, spec, style):
    if style.title: ax.set_title(style.title)
    if style.xlabel: ax.set_xlabel(style.xlabel)
    if style.ylabel: ax.set_ylabel(style.ylabel)
    if style.show_grid is not None:
        ax.grid(style.show_grid)
