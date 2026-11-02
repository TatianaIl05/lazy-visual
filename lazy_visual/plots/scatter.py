from __future__ import annotations

import numpy as np
import pandas as pd

from .._options import get_options
from ..core.spec import AxesSpec, Layer, PlotSpec
from ..core.style import apply_mpl_theme, resolve
from ..io.normalize import prepare
from ..render.mpl import new_figure
from ..stats.compare import StatResult
from ._common import apply_labels, build_style, check_rows, finalize_result


def scatterplot(data, *, x, y, hue=None, size_by=None, fit="none", **style):
    st = build_style(style)
    t, _ = resolve(st)
    apply_mpl_theme(t)
    needed = [x, y] + ([hue] if hue else []) + ([size_by] if size_by else [])
    df, _ = prepare(data, needed, numeric=[x, y])
    check_rows(df)
    fig, ax = new_figure(st.figsize or t.figsize, st.dpi or t.dpi, st.ax)
    palette = t.palette_colors()
    alpha = st.alpha if st.alpha is not None else t.alpha
    lw = st.linewidth if st.linewidth is not None else t.linewidth
    size = st.size or 20.0
    layers = []
    if hue is not None:
        if pd.api.types.is_numeric_dtype(df[hue]):
            sc = ax.scatter(df[x], df[y], c=df[hue], alpha=alpha, s=size,
                            cmap=st.palette or "viridis", linewidths=0)
            fig.colorbar(sc, ax=ax, label=hue)
            layers.append(Layer("point", {"x": x, "y": y, "color_by": hue}))
        else:
            cats = list(pd.unique(df[hue]))
            for i, cat in enumerate(cats):
                sub = df[df[hue] == cat]
                ax.scatter(sub[x], sub[y], color=palette[i % len(palette)],
                           alpha=alpha, s=size, linewidths=lw, label=str(cat))
            if st.show_legend:
                ax.legend(
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                    frameon=False,
                    fontsize=7,
                )
                fig.subplots_adjust(right=0.78)
            layers.append(Layer("point", {"x": x, "y": y, "hue": hue}))
    else:
        color = st.color or palette[0]
        ax.scatter(df[x], df[y], color=color, alpha=alpha, s=size, linewidths=lw)
        layers.append(Layer("point", {"x": x, "y": y}))
    stats = []
    if fit == "linear":
        xv = df[x].to_numpy(float); yv = df[y].to_numpy(float)
        n = len(xv)
        coef = np.polyfit(xv, yv, 1)
        xs = np.linspace(xv.min(), xv.max(), 100)
        ys = np.polyval(coef, xs)
        ax.plot(xs, ys, color="black", lw=lw, linestyle="--")
        # 95% CI band (t-distribution, df = n - 2)
        if n > 2:
            from scipy import stats as sps
            yhat = np.polyval(coef, xv)
            resid = yv - yhat
            ss_res = np.sum(resid ** 2)
            dof = n - 2
            s_err = np.sqrt(ss_res / dof) if dof > 0 else 0.0
            x_mean = xv.mean()
            sxx = np.sum((xv - x_mean) ** 2)
            if sxx > 0:
                t_crit = sps.t.ppf(0.975, dof)
                se = s_err * np.sqrt(1.0 / n + (xs - x_mean) ** 2 / sxx)
                ax.fill_between(xs, ys - t_crit * se, ys + t_crit * se,
                                color="black", alpha=0.12, linewidth=0,
                                label="_nolegend_")
        from scipy import stats as sps
        r, p = sps.pearsonr(xv, yv)
        stats.append(StatResult(method="Pearson r", statistic=float(r),
            p_value=float(p), estimate=float(r), n=n,
            random_state=get_options().random_state))
        ax.text(0.02, 0.98, f"r = {r:.2f}, p = {p:.3f}",
                transform=ax.transAxes, va="top", fontsize=8)
    spec = PlotSpec(kind="scatterplot", layers=layers,
                    axes=AxesSpec(xlabel=st.xlabel or x, ylabel=st.ylabel or y,
                                  title=st.title))
    apply_labels(ax, spec, st)
    return finalize_result(fig, ax, spec, st, stats=stats)
