# Changelog

## 0.1.3

- `heatmap`: `interpolation="nearest"` for deterministic cell rendering across matplotlib versions
- `README`: full rewrite — install, 3 examples (raincloud, ROC, UpSet), badges, Colab badge
- `examples/demo.ipynb`: outputs regenerated for `lazy-visual==0.1.3`
- bump to 0.1.3

## 0.1.2

- `heatmap`: RdBu_r default, robust `vmax`, grid off over data, larger annotation fonts
- `upset`: `A (n)` labels inside bars, Y-axis on the right, grid off, fixed indentation
- `raincloud`: geometric layout (half-violin + box + jittered dots) without overlap
- `box`: adaptive p-value bracket spacing, wider caps, smaller fliers
- `violin`: black quartiles/median (no red default), no duplicate bracket text
- `scatter`: 95% CI band for linear fit, legend outside axes
- `distribution`: transparent hist, colored KDE lines, external legend
- `roc`: DeLong via `scipy.rankdata`, thinned plot for fast rendering, legend outside
- `core/style.py`: disabled minor ticks, `grid.which = "major"` only
- `_common.py`: grid only when `show_grid=True`, no `tight_layout` for multi-panel (`upset`) or colorbar (`heatmap`)
- bump to 0.1.2

## 0.1.1

- `LvResult.show()`: no-op in Jupyter inline backends (avoid double figure)
- `render/mpl.py`: removed forced `Agg` backend so Jupyter/Colab auto-displays
- `scatter` / `roc` / `distribution`: legend outside axes without `tight_layout` warnings
- bump to 0.1.1

## 0.1.0

First public release.

- 8 plot functions: `scatterplot`, `distribution`, `box`, `violin`, `raincloud`, `heatmap`, `upset`, `roc`
- built-in stats: Mann-Whitney U, Welch t-test, Kruskal-Wallis, Holm correction, Cohen's d, DeLong & bootstrap AUC CI
- 4 themes: `default`, `nature`, `apa`, `dark`
- export: PNG / SVG / PDF (atomic writes)
- `lv.load()` for CSV, TSV, Parquet, Excel, JSON, HTTP(S) URL
- `LvResult` with `fig`, `ax`, `spec`, `stats`, `tables`, `files`
- 22 error / warning codes (`LV-E101..E310`, `LV-W501..W504`)