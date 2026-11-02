from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Layer:
    geom: str
    columns: dict[str, Any] = field(default_factory=dict)
    style_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class Annotation:
    kind: str
    text: str = ""
    position: tuple[float, float] = (0.0, 0.0)
    groups: tuple[str, ...] = ()
    stat_ref: int | None = None


@dataclass
class AxesSpec:
    xlabel: str | None = None
    ylabel: str | None = None
    title: str | None = None
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None


@dataclass
class PlotSpec:
    kind: str
    panels: list[str] = field(default_factory=list)
    layers: list[Layer] = field(default_factory=list)
    annotations: list[Annotation] = field(default_factory=list)
    axes: AxesSpec = field(default_factory=AxesSpec)
    legend: bool = True
    style: dict[str, Any] = field(default_factory=dict)
