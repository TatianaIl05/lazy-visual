from __future__ import annotations

from dataclasses import dataclass

from . import defaults
from ._errors import err_out_of_range


@dataclass
class Options:
    active_theme: str = "default"
    random_state: int | None = defaults.RANDOM_STATE
    max_points: int = defaults.MAX_POINTS
    n_boot: int = defaults.N_BOOT

_OPTIONS = Options()

def get_options():
    return _OPTIONS

def set_options(*, random_state=defaults.RANDOM_STATE,
                max_points=defaults.MAX_POINTS,
                n_boot=defaults.N_BOOT, active_theme=None):
    if max_points < 1:
        raise err_out_of_range("max_points", max_points, ">= 1")
    if not (defaults.N_BOOT_MIN <= n_boot <= defaults.N_BOOT_MAX):
        raise err_out_of_range("n_boot", n_boot,
            f"{defaults.N_BOOT_MIN}-{defaults.N_BOOT_MAX}")
    _OPTIONS.random_state = random_state
    _OPTIONS.max_points = int(max_points)
    _OPTIONS.n_boot = int(n_boot)
    if active_theme is not None:
        _OPTIONS.active_theme = active_theme
