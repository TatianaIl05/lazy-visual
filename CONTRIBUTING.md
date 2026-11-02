# Contributing

1. `pip install -e .[dev]`
2. `pytest -q`
3. `ruff check lazy_visual/`
4. `mypy --config-file mypy.ini lazy_visual/`

Keep `lazy_visual/plots/` and `lazy_visual/stats/` free of matplotlib imports; `stats/` must not know about plots.
