from __future__ import annotations

import warnings

import numpy as np

from .._errors import LvDataWarning


def kde_curve(x: np.ndarray, n_points: int = 200
              ) -> tuple[np.ndarray, np.ndarray] | None:
    from scipy import stats as sps
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2 or np.all(x == x[0]):
        warnings.warn("[LV-W504] KDE skipped: <2 points or constant.",
                      LvDataWarning, stacklevel=3)
        return None
    try:
        kde = sps.gaussian_kde(x, bw_method="scott")
    except (ValueError, np.linalg.LinAlgError):
        return None
    grid = np.linspace(x.min() - 0.5 * x.std(),
                       x.max() + 0.5 * x.std(), n_points)
    return grid, kde(grid)
