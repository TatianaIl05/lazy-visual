from __future__ import annotations

import json
from dataclasses import dataclass, replace
from importlib import resources
from typing import Any

import matplotlib as mpl

from .. import defaults
from .._errors import (
    err_out_of_range,
    err_unknown_theme,
    err_wrong_arg_type,
)

PALETTES: dict[str, list[Any]] = {
    "okabe": ["#000000", "#E69F00", "#56B4E9", "#009E73",
              "#F0E442", "#0072B2", "#D55E00", "#CC79A7"],
    "tab10": [mpl.colormaps["tab10"](i / 10)
              for i in range(10)],
    "viridis": [mpl.colormaps["viridis"](x)
                for x in (0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9)],
    "gray": ["#111111", "#444444", "#777777", "#999999", "#bbbbbb"],
}
KNOWN_THEMES: tuple[str, ...] = ("default", "nature", "apa", "dark")


@dataclass(frozen=True)
class Theme:
    name: str = "default"
    font_family: tuple[str, ...] = ("DejaVu Sans",)
    font_size: float = 10.0
    palette: str = "tab10"
    linewidth: float = 1.0
    alpha: float = 0.85
    grid: bool = True
    spines: tuple[str, ...] = ("left", "bottom", "top", "right")
    background: str = "white"
    figsize: tuple[float, float] = (6.0, 4.0)
    dpi: int = defaults.DPI_DEFAULT

    def palette_colors(self) -> list[Any]:
        p = self.palette
        if isinstance(p, str) and p in PALETTES:
            return list(PALETTES[p])
        if isinstance(p, str):
            try:
                cmap = mpl.colormaps[p]
            except KeyError:
                return list(PALETTES["tab10"])
            return [cmap(x) for x in (0.0, 0.2, 0.4, 0.6, 0.8)]
        return list(PALETTES["tab10"])


@dataclass(frozen=True)
class StyleArgs:
    color: str | None = None
    palette: Any | None = None
    alpha: float | None = None
    linewidth: float | None = None
    size: float | None = None
    title: str | None = None
    xlabel: str | None = None
    ylabel: str | None = None
    show_legend: bool = True
    show_grid: bool | None = None
    annotate: bool = True
    figsize: tuple[float, float] | None = None
    dpi: int | None = None
    save: Any | None = None
    fmt: Any | None = None
    theme: Any | None = None
    ax: Any | None = None
    backend: str = "matplotlib"

    def validate(self) -> None:
        if self.alpha is not None and not (0.0 <= self.alpha <= 1.0):
            raise err_out_of_range("alpha", self.alpha, "0-1")
        if self.linewidth is not None and not (0.0 < self.linewidth <= 10.0):
            raise err_out_of_range(
                "linewidth", self.linewidth, "0 < lw <= 10")
        if self.size is not None and not (self.size > 0):
            raise err_out_of_range("size", self.size, "> 0")
        if self.figsize is not None:
            if len(self.figsize) != 2:
                raise err_wrong_arg_type(
                    "figsize", "tuple of 2 floats", str(self.figsize))
            for v in self.figsize:
                if not (1 <= v <= 30):
                    raise err_out_of_range(
                        "figsize", self.figsize, "1-30 per side")
        if self.dpi is not None and not (
            defaults.DPI_MIN <= self.dpi <= defaults.DPI_MAX
        ):
            raise err_out_of_range(
                "dpi", self.dpi,
                f"{defaults.DPI_MIN}-{defaults.DPI_MAX}")
        if self.backend != "matplotlib":
            from .._errors import err_backend_unavailable
            raise err_backend_unavailable(self.backend)


def _theme_from_dict(d: dict[str, Any]) -> Theme:
    known = set(Theme.__dataclass_fields__)
    kwargs: dict[str, Any] = {k: v for k, v in d.items() if k in known}
    if "font_family" in kwargs:
        kwargs["font_family"] = tuple(kwargs["font_family"])
    if "spines" in kwargs:
        kwargs["spines"] = tuple(kwargs["spines"])
    if "figsize" in kwargs:
        kwargs["figsize"] = tuple(kwargs["figsize"])
    return Theme(**kwargs)


def _load_builtin(name: str) -> Theme:
    if name not in KNOWN_THEMES:
        raise err_unknown_theme(name, list(KNOWN_THEMES))
    try:
        data = resources.files("lazy_visual.themes").joinpath(
            f"{name}.json").read_text("utf-8")
        return _theme_from_dict(json.loads(data))
    except (FileNotFoundError, ModuleNotFoundError):
        return Theme(name=name)


_BUILTIN_CACHE: dict[str, Theme] = {}


def get_theme(name: str) -> Theme:
    if name not in _BUILTIN_CACHE:
        _BUILTIN_CACHE[name] = _load_builtin(name)
    return _BUILTIN_CACHE[name]


def theme(name: str = "default", **overrides: Any) -> Theme:
    base = get_theme(name)
    if overrides:
        base = replace(base, **{
            k: v for k, v in overrides.items()
            if k in Theme.__dataclass_fields__
        })
    return base


def resolve(style: StyleArgs) -> tuple[Theme, StyleArgs]:
    from .._options import get_options
    if style.theme is None:
        t = get_theme(get_options().active_theme)
    elif isinstance(style.theme, str):
        t = get_theme(style.theme)
    elif isinstance(style.theme, dict):
        base_name = style.theme.get("name", get_options().active_theme)
        base = get_theme(base_name)
        merged = {**base.__dict__, **style.theme}
        t = _theme_from_dict(merged)
    elif isinstance(style.theme, Theme):
        t = style.theme
    else:
        raise err_wrong_arg_type(
            "theme", "str, dict or Theme",
            type(style.theme).__name__)
    return t, style


def apply_mpl_theme(t: Theme) -> None:
    mpl.rcParams["font.family"] = list(t.font_family)
    mpl.rcParams["font.size"] = t.font_size
    mpl.rcParams["axes.grid"] = t.grid
    mpl.rcParams["axes.facecolor"] = t.background
    mpl.rcParams["figure.facecolor"] = t.background
    for side in ("left", "bottom", "top", "right"):
        mpl.rcParams[f"axes.spines.{side}"] = side in t.spines
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["pdf.fonttype"] = 42
