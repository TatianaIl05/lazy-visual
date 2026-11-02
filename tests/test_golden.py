import matplotlib
matplotlib.use("Agg", force=True)

import numpy as np
import pandas as pd
import pytest

import lazy_visual as lv


@pytest.fixture
def df():
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame({
        "group": np.tile(["a", "b", "c"], n // 3 + 1)[:n],
        "score": rng.normal(0, 1, n) + np.repeat([0, 1, 2], n // 3 + 1)[:n],
        "x": rng.normal(0, 1, n),
        "y": rng.normal(0, 1, n),
        "class": np.tile([0, 1], n // 2),
        "s1": rng.uniform(0, 1, n),
        "s2": rng.uniform(0, 1, n),
    })


@pytest.mark.mpl_image_compare(baseline_dir="baseline", tolerance=15)
def test_golden_scatter(df):
    return lv.scatterplot(df, x="x", y="y", hue="group", dpi=80).fig


@pytest.mark.mpl_image_compare(baseline_dir="baseline", tolerance=15)
def test_golden_distribution(df):
    return lv.distribution(df, x="score", hue="group", kind="both",
                           dpi=80).fig


@pytest.mark.mpl_image_compare(baseline_dir="baseline", tolerance=15)
def test_golden_box(df):
    return lv.box(df, y="score", x="group", dpi=80).fig


@pytest.mark.mpl_image_compare(baseline_dir="baseline", tolerance=15)
def test_golden_violin(df):
    return lv.violin(df, y="score", x="group", dpi=80).fig


@pytest.mark.mpl_image_compare(baseline_dir="baseline", tolerance=15)
def test_golden_raincloud(df):
    return lv.raincloud(df, y="score", x="group", dpi=80).fig


@pytest.mark.mpl_image_compare(baseline_dir="baseline", tolerance=15)
def test_golden_heatmap(df):
    return lv.heatmap(df, corr="pearson", dpi=80).fig


@pytest.mark.mpl_image_compare(baseline_dir="baseline", tolerance=15)
def test_golden_upset():
    data = {"A": {1, 2, 3, 4, 5}, "B": {3, 4, 5, 6}, "C": {4, 5, 6, 7}}
    return lv.upset(data, dpi=80).fig


@pytest.mark.mpl_image_compare(baseline_dir="baseline", tolerance=15)
def test_golden_roc(df):
    return lv.roc(df, y_true="class", y_score=["s1", "s2"], dpi=80).fig
