from __future__ import annotations

import numpy as np
import pandas as pd

from .. import defaults
from .._errors import err_incompatible_args
from ..core.result import LvResult
from ..core.spec import AxesSpec, Layer, PlotSpec
from ..core.style import apply_mpl_theme, resolve
from ..io.normalize import prepare
from ..render.mpl import new_figure
from ._common import apply_labels, build_style, check_rows, finalize_result


def _annot_color(val, vmax):
    return "white" if abs(val) > 0.6 * vmax else "black"


def heatmap(data, *, x=None, y=None, value=None, corr=None, annot=None,
            cmap=None, **style):
    st = build_style(style)
    t, _ = resolve(st)
    apply_mpl_theme(t)
    if corr is not None and value is not None:
        raise err_incompatible_args("Cannot use 'corr' with 'value'.")
    if corr is not None:
        df = data if isinstance(data, pd.DataFrame) else prepare(data, [], dropna=False)[0]
        matrix = df.select_dtypes(include=[np.number]).corr(method=corr)
        xlab = ylab = ""
        diverging = True
    elif x and y and value:
        df, _ = prepare(data, [x, y, value], numeric=[value],
                        categorical=[x, y])
        matrix = df.pivot_table(index=y, columns=x, values=value,
                                aggfunc="mean", observed=True)
        xlab, ylab = x, y
        diverging = False
    elif isinstance(data, pd.DataFrame):
        matrix = data.select_dtypes(include=[np.number])
        xlab = ylab = ""
        diverging = False
    else:
        raise err_incompatible_args(
            "heatmap needs corr=..., (x,y,value)=..., or a numeric DataFrame.")

    check_rows(matrix.reset_index())
    fig, ax = new_figure(st.figsize or t.figsize, st.dpi or t.dpi, st.ax)

    # diverging: red (positive) -> white -> blue (negative), seaborn-style
    cmap_name = cmap or ("RdBu_r" if diverging else "viridis")
    arr = matrix.to_numpy(float)

    if diverging:
        finite = arr[np.isfinite(arr)]
        if finite.size:
            vmax = float(np.percentile(np.abs(finite), 95))
            vmax = max(0.3, min(1.0, vmax))
        else:
            vmax = 1.0
        im = ax.imshow(arr, cmap=cmap_name, vmin=-vmax, vmax=vmax,
                       aspect="auto", interpolation="nearest")
        threshold = vmax
    else:
        im = ax.imshow(arr, cmap=cmap_name, aspect="auto", interpolation="nearest")
        threshold = float(np.nanmax(np.abs(arr))) if arr.size else 1.0

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels([str(c) for c in matrix.columns], rotation=45,
                       ha="right", fontsize=8)
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels([str(i) for i in matrix.index], fontsize=8)

    show_annot = annot if annot is not None else (
        arr.size and arr.shape[0] <= defaults.HEATMAP_ANNOT_LIMIT
        and arr.shape[1] <= defaults.HEATMAP_ANNOT_LIMIT)
    if show_annot:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                v = arr[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=8, color=_annot_color(v, threshold))


    spec = PlotSpec(kind="heatmap",
                    layers=[Layer("rect", {"matrix_shape": arr.shape})],
                    axes=AxesSpec(xlabel=st.xlabel or xlab,
                                  ylabel=st.ylabel or ylab,
                                  title=st.title))
    ax.minorticks_off()
    # heatmap: grid and spines just cover the image, turn them off
    ax.grid(False)
    for _side in ('top', 'right'):
        ax.spines[_side].set_visible(False)
    apply_labels(ax, spec, st)
    try:
        fig.tight_layout()
    except Exception:
        pass
    return finalize_result(fig, ax, spec, st, tables={"matrix": matrix})

