from __future__ import annotations

import pandas as pd

from .. import defaults
from .._errors import err_out_of_range, err_y_true_not_binary
from .._options import get_options
from ..core.spec import AxesSpec, Layer, PlotSpec
from ..core.style import apply_mpl_theme, resolve
from ..io.normalize import prepare
from ..render.mpl import new_figure
from ..stats.roc import compute_roc
from ._common import apply_labels, build_style, check_rows, finalize_result


def _label(name, stat):
    if stat.ci_low is None or stat.ci_high is None:
        return f"{name}: AUC {stat.estimate:.3f}"
    return f"{name}: AUC {stat.estimate:.3f} [{stat.ci_low:.3f}, {stat.ci_high:.3f}]"

def roc(data, *, y_true, y_score, ci=True, ci_method="delong", ci_level=0.95,
        n_boot=None, show_diagonal=True, **style):
    st = build_style(style)
    t, _ = resolve(st)
    apply_mpl_theme(t)
    scores = [y_score] if isinstance(y_score, str) else list(y_score)
    if not (1 <= len(scores) <= defaults.MAX_MODELS):
        raise err_out_of_range("y_score", len(scores), f"1-{defaults.MAX_MODELS}")
    needed = [y_true] + scores
    df, _ = prepare(data, needed, numeric=scores)
    check_rows(df)
    classes = pd.unique(df[y_true].dropna())
    if len(classes) != 2:
        raise err_y_true_not_binary(list(classes))
    y_bin = (df[y_true] == classes[1]).to_numpy(int)
    fig, ax = new_figure(st.figsize or t.figsize, st.dpi or t.dpi, st.ax)
    palette = t.palette_colors()
    lw = st.linewidth if st.linewidth is not None else t.linewidth
    nb = n_boot if n_boot is not None else get_options().n_boot
    seed = get_options().random_state
    stats = []
    for i, col in enumerate(scores):
        fpr, tpr, stat = compute_roc(y_bin, df[col].to_numpy(float),
            ci=ci, ci_method=ci_method, ci_level=ci_level,
            n_boot=nb, random_state=seed)
        ax.plot(fpr, tpr, color=palette[i % len(palette)], lw=lw,
                label=_label(col, stat))
        stats.append(stat)
    if show_diagonal:
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
    ax.set_xlabel(st.xlabel or "False positive rate")
    ax.set_ylabel(st.ylabel or "True positive rate")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_aspect("equal")
    if st.show_legend:
        ax.legend(loc="lower right", frameon=False, fontsize=8)
    spec = PlotSpec(kind="roc", layers=[Layer("line", {"models": scores})],
                    axes=AxesSpec(xlabel="FPR", ylabel="TPR", title=st.title))
    apply_labels(ax, spec, st)
    return finalize_result(fig, ax, spec, st, stats=stats)
