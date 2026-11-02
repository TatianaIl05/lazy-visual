from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np

from .. import defaults
from .._errors import LvDataWarning, err_incompatible_args
from ..core.spec import AxesSpec, Layer, PlotSpec
from ..core.style import apply_mpl_theme, resolve
from ..stats.sets import _to_membership, compute_intersections
from ._common import build_style, finalize_result


def upset(data, *, sets=None, item=None, set_col=None,
          top_n=defaults.UPSET_TOP_N, min_size=1, sort_by="size", **style):
    st = build_style(style)
    if st.ax is not None:
        raise err_incompatible_args("ax cannot be used with upset.")
    t, _ = resolve(st)
    apply_mpl_theme(t)
    inter = compute_intersections(data, sets=sets, item=item,
        set_col=set_col, top_n=top_n, min_size=min_size, sort_by=sort_by)
    membership = _to_membership(data, sets=sets, item=item, set_col=set_col)
    set_names = list(membership.columns)
    fig = plt.figure(figsize=st.figsize or (6.5, 4.5), dpi=st.dpi or t.dpi,
                     constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 4], height_ratios=[2, 2],
                          hspace=0.05, wspace=0.05)
    ax_sets = fig.add_subplot(gs[0, 0])
    ax_bars = fig.add_subplot(gs[0, 1])
    ax_matrix = fig.add_subplot(gs[1, 1])
    axes = {"bars": ax_bars, "matrix": ax_matrix, "sets": ax_sets}
    if inter.empty:
        for a in axes.values():
            a.set_xticks([]); a.set_yticks([])
        ax_bars.text(0.5, 0.5, f"No intersections with size >= {min_size}",
                     ha="center", va="center", transform=ax_bars.transAxes)
        warnings.warn(f"[LV-W504] No intersections with size >= {min_size}.",
            LvDataWarning, stacklevel=3)
        spec = PlotSpec(kind="upset", panels=list(axes.keys()),
                        axes=AxesSpec(title=st.title))
        return finalize_result(fig, axes, spec, st,
                               tables={"intersections": inter})
    palette = t.palette_colors()
    n = len(inter)
    xs = np.arange(n)
    ax_bars.bar(xs, inter["size"], color=palette[0], alpha=st.alpha or t.alpha)
    ax_bars.set_xticks([])
    ax_bars.set_ylabel(st.ylabel or "Intersection size")
    if st.title:
        ax_bars.set_title(st.title)
    for x, s in zip(xs, inter["size"]):
        ax_bars.text(x, s, str(int(s)), ha="center", va="bottom", fontsize=8)
    n_sets = len(set_names)
    ypos = {name: i for i, name in enumerate(set_names)}
    ax_matrix.set_xlim(-0.5, n - 0.5)
    ax_matrix.set_ylim(n_sets - 0.5, -0.5)
    for j, pattern in enumerate(inter["pattern"]):
        pat = set(pattern)
        for name in set_names:
            y = ypos[name]
            present = name in pat
            color = palette[0] if present else "#dddddd"
            ax_matrix.scatter(j, y, s=80, color=color, zorder=2)
        pys = [ypos[name] for name in pattern if name in ypos]
        if len(pys) > 1:
            ax_matrix.plot([j, j], [min(pys), max(pys)], color="black",
                           lw=1.5, zorder=1)
    ax_matrix.set_xticks([])
    ax_matrix.set_yticks(range(n_sets))
    ax_matrix.set_yticklabels(set_names)
    ax_matrix.invert_yaxis()
    sizes = membership.sum(axis=0)
    y = np.arange(n_sets)
    ax_sets.barh(y, sizes.values, color=palette[0], alpha=st.alpha or t.alpha)
    ax_sets.set_yticks(y)
    ax_sets.set_yticklabels([])
    ax_sets.invert_yaxis()
    ax_sets.set_xlabel(st.xlabel or "Set size")
    spec = PlotSpec(kind="upset", panels=list(axes.keys()),
        layers=[Layer("bar", {"size": "size"}),
                Layer("dot_matrix", {"pattern": "pattern"})],
        axes=AxesSpec(title=st.title))
    return finalize_result(fig, axes, spec, st, tables={"intersections": inter})
