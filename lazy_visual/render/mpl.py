from __future__ import annotations

from typing import Any, Optional

import matplotlib
try:
    matplotlib.get_backend()
except Exception:
    matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt


def new_figure(figsize, dpi, ax: Optional[Any] = None):
    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    return fig, ax
