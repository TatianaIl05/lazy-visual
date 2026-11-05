from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .spec import PlotSpec


@dataclass
class LvResult:
    fig: Any
    ax: Any
    spec: PlotSpec
    stats: list[Any] = field(default_factory=list)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    files: list[Path] = field(default_factory=list)

    def save(self, path: Any, *, fmt: Any = None,
             dpi: Any = None) -> list[Path]:
        from ..export import save_result
        self.files = save_result(self.fig, path, fmt=fmt, dpi=dpi)
        return self.files

    def show(self) -> None:
        """Display the figure inline (Jupyter/Colab) or via plt.show()."""
        # Jupyter/Colab: display the figure directly
        try:
            from IPython.display import display
            display(self.fig)
            return
        except ImportError:
            pass
        # fallback: interactive backend
        import matplotlib.pyplot as plt
        plt.show()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.spec.kind,
            "stats": [
                s.to_dict() if hasattr(s, "to_dict") else s.__dict__
                for s in self.stats
            ],
            "files": [str(p) for p in self.files],
        }

    def __repr__(self) -> str:
        head = f"<LvResult {self.spec.kind}"
        if self.stats:
            s = self.stats[0]
            p = getattr(s, "p_value", None)
            if p is not None:
                ps = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
                head += f", {getattr(s, 'method', '')} {ps}"
        if self.files:
            head += f", saved: {', '.join(p.name for p in self.files)}"
        return head + ">"
