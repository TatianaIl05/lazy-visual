"""lazy-visual: one-line statistical plots with built-in stats."""
from __future__ import annotations

from ._errors import (
    LvArgumentError,
    LvDataError,
    LvDataWarning,
    LvDependencyError,
    LvError,
    LvExportError,
    LvPerformanceWarning,
)
from ._options import Options, get_options, set_options

__version__ = "0.1.0"

__all__ = [
    "LvArgumentError",
    "LvDataError",
    "LvDataWarning",
    "LvDependencyError",
    "LvError",
    "LvExportError",
    "LvPerformanceWarning",
    "LvResult",
    "Options",
    "Theme",
    "__version__",
    "box",
    "distribution",
    "get_options",
    "heatmap",
    "load",
    "raincloud",
    "roc",
    "scatterplot",
    "set_options",
    "theme",
    "upset",
    "violin",
]


def __getattr__(name: str):
    """Lazy imports so `import lazy_visual` stays fast (NFR-1)."""
    if name == "load":
        from .io.loader import load
        return load
    if name == "theme":
        from .core.style import theme
        return theme
    if name == "Theme":
        from .core.style import Theme
        return Theme
    if name == "LvResult":
        from .core.result import LvResult
        return LvResult
    if name == "scatterplot":
        from .plots.scatter import scatterplot
        return scatterplot
    if name == "distribution":
        from .plots.distribution import distribution
        return distribution
    if name == "box":
        from .plots.box import box
        return box
    if name == "violin":
        from .plots.violin import violin
        return violin
    if name == "raincloud":
        from .plots.raincloud import raincloud
        return raincloud
    if name == "heatmap":
        from .plots.heatmap import heatmap
        return heatmap
    if name == "upset":
        from .plots.upset import upset
        return upset
    if name == "roc":
        from .plots.roc import roc
        return roc
    raise AttributeError(f"module 'lazy_visual' has no attribute {name!r}")
