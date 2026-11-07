# lazy-visual

[![CI](https://github.com/TatianaIl05/lazy-visual/actions/workflows/ci.yml/badge.svg)](https://github.com/TatianaIl05/lazy-visual/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/lazy-visual.svg)](https://pypi.org/project/lazy-visual/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.10–3.13](https://img.shields.io/badge/python-3.10--3.13-blue.svg)](pyproject.toml)

**One-line statistical plots with built-in stats.**

```python
import lazy_visual as lv
lv.raincloud(df, x="group", y="score", save="fig.svg")
```

Returns a figure **and** statistics (p-value, Cohen's d, AUC with CI) in a single `LvResult`.

## Install

```bash
pip install lazy-visual
```

Python 3.10–3.13. Optional extras: `pip install "lazy-visual[parquet]"` or `"lazy-visual[excel]"`.

## Example 1 - raincloud with p-value and Cohen's d

```python
import numpy as np
import pandas as pd
import lazy_visual as lv

rng = np.random.default_rng(0)
df = pd.DataFrame({
    "group": rng.choice(["control", "drug_a", "drug_b"], 300),
    "score": rng.normal(0, 1, 300) + rng.choice([0, 1, 2], 300),
})

lv.raincloud(
    df, x="group", y="score",
    pairs=[("control", "drug_a"), ("control", "drug_b")],
    save="raincloud.svg",
)
```

Result: half-violin + box + jittered points, plus p-value and Cohen's d brackets above the groups.

## Example 2 - ROC curves with DeLong CI

```python
rng = np.random.default_rng(1)
n = 200
y = rng.choice([0, 1], n)
df_roc = pd.DataFrame({
    "y_true": y,
    "model_a": y * 1.5 + rng.normal(0, 1.0, n),
    "model_b": y * 1.0 + rng.normal(0, 1.2, n),
})

lv.roc(
    df_roc, y_true="y_true", y_score=["model_a", "model_b"],
    ci_method="delong", save="roc.svg",
)
```

Result: ROC curves with AUC and 95% DeLong CI shown in the legend.

## Example 3 - UpSet for set intersections

```python
lv.upset(
    {"A": {1, 2, 3, 4, 5},
     "B": {3, 4, 5, 6},
     "C": {4, 5, 6, 7}},
    save="upset.svg",
)
```

Result: three-panel UpSet plot (bars, matrix, sets) plus `LvResult.tables["intersections"]` with exclusive intersection sizes.

## What's in the box

- **8 plots**: `scatterplot`, `distribution`, `box`, `violin`, `raincloud`, `heatmap`, `upset`, `roc`
- **Built-in stats**: Mann-Whitney U, Welch t-test, Kruskal-Wallis, Holm correction, Cohen's d, DeLong & bootstrap AUC CI
- **4 themes**: `default`, `nature`, `apa`, `dark`
- **Export**: PNG / SVG / PDF (atomic writes)
- **`lv.load()`**: CSV, TSV, Parquet, Excel, JSON, HTTP(S) URL
- **Only 4 dependencies**: numpy, scipy, pandas, matplotlib. MIT licensed.

## Demo notebook

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TatianaIl05/lazy-visual/blob/main/examples/demo.ipynb)

## Documentation

- Full user guide: [USER_GUIDE.md](USER_GUIDE.md)
- Online docs: <https://tatianail05.github.io/lazy-visual/>

## Development

```bash
git clone https://github.com/TatianaIl05/lazy-visual.git
cd lazy-visual
pip install -e ".[dev]"
pytest -q
```