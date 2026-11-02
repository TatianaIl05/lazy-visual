"""NFR-2: plot <= 2s @ 100k rows.
NFR-3: AUC+CI DeLong <= 1s @ 100k rows, 3 models (compute_roc per ТЗ).
"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pytest_benchmark")
import lazy_visual as lv
from lazy_visual.stats.roc import compute_roc


@pytest.fixture
def big_df():
    rng = np.random.default_rng(0)
    n = 100_000
    return pd.DataFrame({
        "group": rng.choice(["a", "b", "c"], n),
        "score": rng.normal(0, 1, n),
        "class": rng.choice([0, 1], n),
        "s1": rng.uniform(0, 1, n),
        "s2": rng.uniform(0, 1, n),
        "s3": rng.uniform(0, 1, n),
    })


def test_nfr2_raincloud_100k(benchmark, big_df):
    r = benchmark(lambda: lv.raincloud(
        big_df, y="score", x="group", max_points=5000))
    assert r.fig is not None


def test_nfr3_roc_delong_100k_3models(benchmark, big_df):
    y = big_df["class"].to_numpy(int)
    s1 = big_df["s1"].to_numpy(float)
    s2 = big_df["s2"].to_numpy(float)
    s3 = big_df["s3"].to_numpy(float)

    def run():
        for s in (s1, s2, s3):
            compute_roc(y, s, ci=True, ci_method="delong")

    benchmark(run)
