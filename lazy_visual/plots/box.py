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
    if x is not None:
        _cats_tmp = ordered_categories(df[x])
        if len(_cats_tmp) > len(palette):
            import matplotlib as _mpl
            palette = [tuple(c) for c in _mpl.colormaps['tab20'].colors]
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
                            medianprops=dict(color="black", linewidth=1.2))
            for i, patch in enumerate(bp["boxes"]):
                color = palette[i % len(palette)]
                patch.set_facecolor(color)
                patch.set_edgecolor(color)
                patch.set_alpha(alpha)
                patch.set_linewidth(lw)
                patch.set_zorder(2)
            for i, w in enumerate(bp["whiskers"]):
                w.set_color(palette[(i // 2) % len(palette)])
                w.set_linewidth(lw)
                w.set_zorder(2)
            for i, c in enumerate(bp["caps"]):
                c.set_color(palette[(i // 2) % len(palette)])
                c.set_linewidth(lw)
                c.set_zorder(2)
            for i, m in enumerate(bp["medians"]):
                m.set_color("black")
                m.set_linewidth(lw * 1.2)
                m.set_zorder(3)
            for i, f in enumerate(bp["fliers"]):
                f.set_markeredgecolor(palette[i % len(palette)])
                f.set_markerfacecolor(palette[i % len(palette)])
                f.set_markersize(3)
                f.set_alpha(0.7)
                f.set_zorder(2)
        else:
            parts = ax.violinplot(groups, showmedians=False,
                                  showextrema=False, widths=0.8)
            for i, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(palette[i % len(palette)])
                pc.set_alpha(alpha)
                pc.set_edgecolor("black")
                pc.set_linewidth(lw)
            # draw box-plot inner structure (whiskers, IQR, median) in black
            for i, g in enumerate(groups):
                if len(g) < 2:
                    continue
                g = np.asarray(g, float)
                q1, med, q3 = np.percentile(g, [25, 50, 75])
                iqr = q3 - q1
                lo = max(g.min(), q1 - 1.5 * iqr)
                hi = min(g.max(), q3 + 1.5 * iqr)
                xpos = i + 1
                # thin whisker lines
                ax.vlines(xpos, lo, q1, color="black", lw=lw * 0.8, zorder=3)
                ax.vlines(xpos, q3, hi, color="black", lw=lw * 0.8, zorder=3)
                # wide caps at whisker ends
                ax.hlines(lo, xpos - 0.10, xpos + 0.10,
                          color="black", lw=lw * 1.2, zorder=4)
                ax.hlines(hi, xpos - 0.10, xpos + 0.10,
                          color="black", lw=lw * 1.2, zorder=4)
                # thick IQR bar (clearly thicker than whiskers)
                ax.vlines(xpos, q1, q3, color="black", lw=lw * 3.0, zorder=3)
                # thick median, wider than the IQR bar
                ax.hlines(med, xpos - 0.13, xpos + 0.13,
                          color="black", lw=lw * 2.0, zorder=4)
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels)
        stats = compare_groups(df, y=y, x=x, test=test, pairs=pairs,
                               random_state=get_options().random_state)
        if st.annotate and stats:
            _draw_brackets(ax, cats, stats, offset=1.0)
    else:
        ax.boxplot([df[y].to_numpy()], **_boxplot_kwargs([y]),
                   patch_artist=True)
    spec = PlotSpec(kind=kind, layers=[Layer(kind, {"y": y, "x": x})],
                    axes=AxesSpec(xlabel=st.xlabel or (x or ""),
                                  ylabel=st.ylabel or y, title=st.title))
    apply_labels(ax, spec, st)
    return finalize_result(fig, ax, spec, st, stats=stats)


def _draw_brackets(ax, cats, stats, offset=0.0):
    pairs = [s for s in stats
             if s.groups and len(s.groups) == 2
             and s.groups[0] in cats and s.groups[1] in cats]
    if not pairs:
        return
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    n = len(pairs)
    # generous spacing: each bracket gets >= 10% of span, or 0.5/n + 0.05
    step = span * (0.14 + 0.04 / max(n, 1))
    base = ymax
    idx_map = {c: i for i, c in enumerate(cats)}
    for i, s in enumerate(pairs):
        g1, g2 = s.groups
        a, b = idx_map[g1] + offset, idx_map[g2] + offset
        lo, hi = min(a, b), max(a, b)
        yb = base + step * (i + 1)
        ax.plot([lo, lo, hi, hi],
                [yb, yb + step * 0.25, yb + step * 0.25, yb],
                color="black", lw=1)
        label = format_p(
            s.p_adjusted if s.p_adjusted is not None else s.p_value)
        d = format_d(s.effect_size)
        text = f"{label}\n{d}" if d else label
        ax.text((lo + hi) / 2, yb + step * 0.25, text,
                ha="center", va="bottom", fontsize=6,
                linespacing=1.3)
    ax.set_ylim(top=base + step * (n + 1.0))
