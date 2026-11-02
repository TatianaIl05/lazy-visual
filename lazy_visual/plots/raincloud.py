from __future__ import annotations

import warnings

import numpy as np
from matplotlib.patches import Rectangle

from .. import defaults
from .._errors import LvDataWarning, err_too_many_groups
from .._options import get_options
from ..core.spec import AxesSpec, Layer, PlotSpec
from ..core.style import apply_mpl_theme, resolve
from ..io.normalize import ordered_categories, prepare
from ..render.mpl import new_figure
from ..stats.compare import compare_groups
from ..stats.density import kde_curve
from ._common import apply_labels, build_style, check_rows, finalize_result
from .box import _draw_brackets


def _draw_raincloud(ax, vals, pos, color, alpha, lw, orient, rng):
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return
    half, box_w, pts_shift, jitter = 0.4, 0.15, 0.25, 0.08
    res = kde_curve(vals) if len(vals) >= 2 and not np.all(vals == vals[0]) else None
    if orient == "v":
        if res is not None:
            grid, dens = res
            dens = dens / dens.max() * half
            ax.fill_betweenx(grid, pos - dens, pos, color=color,
                             alpha=alpha, edgecolor="black", linewidth=lw*0.5)
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        lo = max(vals.min(), q1 - 1.5*iqr)
        hi = min(vals.max(), q3 + 1.5*iqr)
        bc = pos + half * 0.5
        ax.add_patch(Rectangle((bc - box_w/2, q1), box_w, q3 - q1,
            facecolor=color, edgecolor="black", alpha=alpha, linewidth=lw))
        ax.plot([bc, bc], [lo, q1], color="black", lw=lw)
        ax.plot([bc, bc], [q3, hi], color="black", lw=lw)
        xj = pos + half + pts_shift + rng.uniform(-jitter, jitter, size=len(vals))
        ax.scatter(xj, vals, s=8, color=color, alpha=alpha,
                   edgecolors="none", rasterized=True)
    else:
        if res is not None:
            grid, dens = res
            dens = dens / dens.max() * half
            ax.fill_between(grid, pos - dens, pos, color=color,
                            alpha=alpha, edgecolor="black", linewidth=lw*0.5)
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        lo = max(vals.min(), q1 - 1.5*iqr)
        hi = min(vals.max(), q3 + 1.5*iqr)
        bc = pos + half * 0.5
        ax.add_patch(Rectangle((q1, bc - box_w/2), q3 - q1, box_w,
            facecolor=color, edgecolor="black", alpha=alpha, linewidth=lw))
        ax.plot([lo, q1], [bc, bc], color="black", lw=lw)
        ax.plot([q3, hi], [bc, bc], color="black", lw=lw)
        yj = pos + half + pts_shift + rng.uniform(-jitter, jitter, size=len(vals))
        ax.scatter(vals, yj, s=8, color=color, alpha=alpha,
                   edgecolors="none", rasterized=True)

def raincloud(data, *, y, x=None, test="auto", pairs=None, orient="v",
              max_points=None, **style):
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
    seed = get_options().random_state
    limit = max_points or get_options().max_points
    rng = np.random.default_rng(seed)
    if x is not None:
        cats = ordered_categories(df[x])
        if len(cats) > defaults.MAX_GROUPS:
            raise err_too_many_groups(len(cats), defaults.MAX_GROUPS)
        groups = {str(c): df[df[x] == c][y].to_numpy(float) for c in cats}
    else:
        cats = [y]
        groups = {y: df[y].to_numpy(float)}
    for i, name in enumerate(cats):
        vals = groups[str(name)]
        if len(vals) > limit:
            warnings.warn(
                f"[LV-W502] Subsampled {len(vals)} points to {limit}.",
                LvDataWarning, stacklevel=3)
            vals = rng.choice(vals, size=limit, replace=False)
        _draw_raincloud(ax, vals, i, palette[i % len(palette)],
                        alpha, lw, orient, rng)
    if x is not None:
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels([str(c) for c in cats])
    stats = []
    if x is not None:
        stats = compare_groups(df, y=y, x=x, test=test, pairs=pairs,
                               random_state=seed)
        if st.annotate and stats:
            _draw_brackets(ax, cats, stats)
    spec = PlotSpec(kind="raincloud",
        layers=[Layer("half_violin", {"y": y, "x": x}),
                Layer("box", {"y": y, "x": x}),
                Layer("point", {"y": y, "x": x})],
        axes=AxesSpec(xlabel=st.xlabel or (x or ""), ylabel=st.ylabel or y,
                      title=st.title))
    apply_labels(ax, spec, st)
    return finalize_result(fig, ax, spec, st, stats=stats)
