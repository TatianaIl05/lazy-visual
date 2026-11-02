from __future__ import annotations

import numpy as np

from .. import defaults
from .._errors import err_too_many_groups
from .._options import get_options
from ..core.result import LvResult
from ..core.spec import AxesSpec, Layer, PlotSpec
from ..core.style import apply_mpl_theme, resolve
from ..io.normalize import ordered_categories, prepare
from ..render.mpl import new_figure
from ..stats.compare import compare_groups, format_d, format_p
from ._common import apply_labels, build_style, check_rows, finalize_result


def _boxplot_kwargs(labels):
    """mpl<3.9 uses `labels`, mpl>=3.9 uses `tick_labels`."""
    import matplotlib as _mpl
    try:
        ver = tuple(int(x) for x in _mpl.__version__.split(".")[:2])
    except ValueError:
        ver = (3, 9)
    key = "tick_labels" if ver >= (3, 9) else "labels"
    return {key: labels}


def box(data, *, y, x=None, test="auto", pairs=None, **style) -> LvResult:
    return _box_or_violin(data, y=y, x=x, test=test, pairs=pairs,
                          kind="box", **style)


def _box_or_violin(data, *, y, x, test, pairs, kind, **style) -> LvResult:
    st = build_style(style)
    t, _ = resolve(st)
    apply_mpl_theme(t)
    needed = [y] + ([x] if x else [])
    df, _ = prepare(data, needed, numeric=[y],
                    categorical=[x] if x else [])
    check_rows(df)
    fig, ax = new_figure(st.figsize or t.figsize, st.dpi or t.dpi, st.ax)
    palette = t.palette_colors()
    alpha = st.alpha if st.alpha is not None else t.alpha
    lw = st.linewidth if st.linewidth is not None else t.linewidth
    stats = []
    if x is not None:
        cats = ordered_categories(df[x])
        if len(cats) > defaults.MAX_GROUPS:
            raise err_too_many_groups(len(cats), defaults.MAX_GROUPS)
        groups = [df[df[x] == c][y].to_numpy() for c in cats]
        labels = [str(c) for c in cats]
        if kind == "box":
            bp = ax.boxplot(groups, **_boxplot_kwargs(labels),
                            patch_artist=True, widths=0.6,
                            medianprops=dict(color="black"))
            for i, patch in enumerate(bp["boxes"]):
                patch.set_facecolor(palette[i % len(palette)])
                patch.set_alpha(alpha)
                patch.set_linewidth(lw)
        else:
            parts = ax.violinplot(groups, showmedians=True, widths=0.8)
            for i, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(palette[i % len(palette)])
                pc.set_alpha(alpha)
                pc.set_edgecolor("black")
                pc.set_linewidth(lw)
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels)
        stats = compare_groups(df, y=y, x=x, test=test, pairs=pairs,
                               random_state=get_options().random_state)
        if st.annotate and stats:
            _draw_brackets(ax, cats, stats)
    else:
        ax.boxplot([df[y].to_numpy()], **_boxplot_kwargs([y]),
                   patch_artist=True)
    spec = PlotSpec(kind=kind, layers=[Layer(kind, {"y": y, "x": x})],
                    axes=AxesSpec(xlabel=st.xlabel or (x or ""),
                                  ylabel=st.ylabel or y, title=st.title))
    apply_labels(ax, spec, st)
    return finalize_result(fig, ax, spec, st, stats=stats)


def _draw_brackets(ax, cats, stats):
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    step = span * 0.06
    base = ymax
    idx_map = {c: i + 1 for i, c in enumerate(cats)}
    for i, s in enumerate(stats):
        if not s.groups or len(s.groups) != 2:
            continue
        g1, g2 = s.groups
        if g1 not in idx_map or g2 not in idx_map:
            continue
        a, b = idx_map[g1], idx_map[g2]
        lo, hi = min(a, b), max(a, b)
        yb = base + step * (i + 1)
        ax.plot([lo, lo, hi, hi],
                [yb, yb + step * 0.3, yb + step * 0.3, yb],
                color="black", lw=1)
        label = format_p(
            s.p_adjusted if s.p_adjusted is not None else s.p_value)
        d = format_d(s.effect_size)
        text = f"{label}\n{d}" if d else label
        ax.text((lo + hi) / 2, yb + step * 0.35, text,
                ha="center", va="bottom", fontsize=8)
    ax.set_ylim(top=base + step * (len(stats) + 1))
