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
    # skip tight_layout for multi-panel plots (upset) and colorbar (heatmap)
    if spec.kind not in ("upset", "heatmap"):
        try:
            fig.tight_layout()
        except Exception:
            pass
    return LvResult(fig=fig, ax=ax, spec=spec, stats=list(stats or []),
                    tables=dict(tables or {}), files=files)

def apply_labels(ax, spec, style):
    from ..core.style import get_theme
    from .._options import get_options

    if style.title:
        ax.set_title(style.title, pad=10)
    if style.xlabel:
        ax.set_xlabel(style.xlabel)
    if style.ylabel:
        ax.set_ylabel(style.ylabel)
    # grid: only when user explicitly passes show_grid=True
    # (avoid grid over scatter/line/heatmap data)
    if style.show_grid is True:
        ax.grid(True, which="major", color="#dddddd", linewidth=0.6)
        ax.set_axisbelow(True)
    else:
        ax.grid(False)
    # disable minor ticks (no "small cells")
    ax.minorticks_off()
    # disable ticks on top and right
    ax.tick_params(which="both", top=False, right=False)

    # respect theme spines (mpl rcParams only apply to new axes)
    theme = style.theme if hasattr(style, "theme") else None
    if isinstance(theme, str):
        t = get_theme(theme)
    else:
        t = get_theme(get_options().active_theme)
    for side in ("left", "bottom", "top", "right"):
        ax.spines[side].set_visible(side in t.spines)

    # no top/right tick marks
    ax.tick_params(top=False, right=False)
