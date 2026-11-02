from __future__ import annotations

import pandas as pd

from ..core.spec import AxesSpec, Layer, PlotSpec
from ..core.style import apply_mpl_theme, resolve
from ..io.normalize import prepare
from ..render.mpl import new_figure
from ..stats.density import kde_curve
from ._common import apply_labels, build_style, check_rows, finalize_result


def _draw(ax, vals, kind, bins, color, alpha, lw, label):
    if len(vals) == 0:
        return
    if kind in ("hist", "both"):
        ax.hist(vals, bins=bins, color=color, alpha=alpha, label=label,
                edgecolor="white", linewidth=0.5, density=(kind == "both"))
    if kind in ("kde", "both"):
        res = kde_curve(vals)
        if res is not None:
            grid, dens = res
            ax.plot(grid, dens, color=color, lw=lw,
                    label=label if kind == "kde" else None)

def distribution(data, *, x, hue=None, kind="hist", bins="auto", **style):
    st = build_style(style)
    t, _ = resolve(st)
    apply_mpl_theme(t)
    needed = [x] + ([hue] if hue else [])
    df, _ = prepare(data, needed, numeric=[x])
    check_rows(df)
    fig, ax = new_figure(st.figsize or t.figsize, st.dpi or t.dpi, st.ax)
    palette = t.palette_colors()
    alpha = st.alpha if st.alpha is not None else t.alpha
    lw = st.linewidth if st.linewidth is not None else t.linewidth
    if hue is not None:
        for i, cat in enumerate(pd.unique(df[hue])):
            sub = df[df[hue] == cat][x].dropna().to_numpy(float)
            _draw(ax, sub, kind, bins, palette[i % len(palette)], alpha, lw,
                  label=str(cat))
        if st.show_legend:
            ax.legend(frameon=False)
    else:
        _draw(ax, df[x].to_numpy(float), kind, bins,
              st.color or palette[0], alpha, lw, None)
    spec = PlotSpec(kind="distribution", layers=[Layer("bar", {"x": x})],
                    axes=AxesSpec(xlabel=st.xlabel or x,
                                  ylabel=st.ylabel or "density",
                                  title=st.title))
    apply_labels(ax, spec, st)
    return finalize_result(fig, ax, spec, st)
