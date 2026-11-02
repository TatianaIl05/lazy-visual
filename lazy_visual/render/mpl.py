from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


def new_figure(figsize, dpi, ax=None):
    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=True)
    return fig, ax
