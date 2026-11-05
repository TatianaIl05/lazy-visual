from __future__ import annotations

import warnings

import numpy as np
import matplotlib.pyplot as plt

from .. import defaults
from .._errors import LvDataWarning, err_incompatible_args
from ..core.result import LvResult
from ..core.spec import AxesSpec, Layer, PlotSpec
from ..core.style import apply_mpl_theme, resolve
from ..stats.sets import compute_intersections, _to_membership
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

    _w, _h = st.figsize or (8.5, 5.5)
    if _w < 7.5:
        _w = 7.5
    if _h < 5.0:
        _h = 5.0
    fig = plt.figure(figsize=(_w, _h), dpi=st.dpi or t.dpi,
                     constrained_layout=False)
    # generous wspace so axes labels don't collide
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 4.0],
                          height_ratios=[2, 2],
                          hspace=0.05, wspace=0.25,
                          left=0.14, right=0.97, top=0.88, bottom=0.10)
    ax_sets = fig.add_subplot(gs[0, 0])
    ax_bars = fig.add_subplot(gs[0, 1])
    ax_matrix = fig.add_subplot(gs[1, 1])
    axes = {"bars": ax_bars, "matrix": ax_matrix, "sets": ax_sets}

    if st.title:
        fig.suptitle(st.title, fontsize=11, y=0.95)

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
    main_color = palette[1] if len(palette) > 1 else palette[0]
    inactive_color = "#cccccc"

    n = len(inter)
    xs = np.arange(n)

    n_sets = len(set_names)
    ypos = {name: n_sets - 1 - i for i, name in enumerate(set_names)}
    ylim = (-0.5, n_sets - 0.5)

    # ---------- BARS ----------
    ax_bars.bar(xs, inter["size"], color=main_color,
                alpha=st.alpha or t.alpha, edgecolor="black", linewidth=0.4)
    ax_bars.set_xticks([])
    ax_bars.set_ylabel(st.ylabel or "Intersection size", fontsize=8, labelpad=6)
    ax_bars.set_xlim(-0.5, n - 0.5)
    for x, s in zip(xs, inter["size"]):
        ax_bars.text(x, s, str(int(s)), ha="center", va="bottom", fontsize=8)
    ax_bars.tick_params(axis="y", labelsize=8)
    ax_bars.spines["bottom"].set_visible(False)
    ax_bars.tick_params(bottom=False)

    # ---------- MATRIX ----------
    ax_matrix.set_xlim(-0.5, n - 0.5)
    ax_matrix.set_ylim(ylim)
    for j, pattern in enumerate(inter["pattern"]):
        pat = set(pattern)
        for name in set_names:
            y = ypos[name]
            present = name in pat
            color = main_color if present else inactive_color
            ax_matrix.scatter(j, y, s=55, color=color, zorder=2,
                              edgecolors="black", linewidths=0.4)
        pys = [ypos[name] for name in pattern if name in ypos]
        if len(pys) > 1:
            ax_matrix.plot([j, j], [min(pys), max(pys)], color="black",
                           lw=1.3, zorder=1)
    ax_matrix.set_xticks([])
    ax_matrix.set_yticks([ypos[name] for name in set_names])
    ax_matrix.set_yticklabels(set_names, fontsize=8)
    for side in ("top", "right", "left"):
        ax_matrix.spines[side].set_visible(False)
    ax_matrix.tick_params(left=False)

    # ---------- SETS (left) ----------
    # Bars grow LEFT (x inverted) with POSITIVE values: 0..max
    sizes = membership.sum(axis=0)
    y_positions = [ypos[name] for name in set_names]
    ax_sets.barh(y_positions, sizes.values, color=main_color,
                 alpha=st.alpha or t.alpha, edgecolor="black",
                 linewidth=0.4, height=0.6)
    ax_sets.set_yticks([])
    ax_sets.set_ylim(ylim)
    smax = float(sizes.max()) if len(sizes) else 1.0
    ax_sets.set_xlim(0, smax * 1.05)
    ax_sets.invert_xaxis()   # x=0 on the RIGHT, positive values grow LEFT
    ax_sets.set_xlabel(st.xlabel or "Set size", fontsize=8, labelpad=4)

    # Y-axis on the RIGHT side (between sets and bars)
    ax_sets.yaxis.set_ticks_position("right")
    ax_sets.yaxis.set_label_position("right")
    for side in ("left", "top", "bottom"):
        ax_sets.spines[side].set_visible(False)
    ax_sets.spines["right"].set_visible(True)

    # labels centered inside each bar: "A (5)"
    for name, size in zip(set_names, sizes.values):
        ax_sets.text(size / 2.0, ypos[name], f"{name} ({int(size)})",
                     va="center", ha="center", fontsize=8, color="black")

    spec = PlotSpec(kind="upset", panels=list(axes.keys()),
        layers=[Layer("bar", {"size": "size"}),
                Layer("dot_matrix", {"pattern": "pattern"})],
        axes=AxesSpec(title=st.title))
    return finalize_result(fig, axes, spec, st,
                           tables={"intersections": inter})
