from typing import Protocol

from ..core.spec import PlotSpec


class Renderer(Protocol):
    def render(self, spec: PlotSpec): ...
