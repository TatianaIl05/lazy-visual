import numpy as np, pandas as pd, pytest, lazy_visual as lv

@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame({
        "group": rng.choice(["a", "b", "c"], n),
        "score": rng.normal(0, 1, n),
        "x": rng.normal(0, 1, n), "y": rng.normal(0, 1, n),
        "class": rng.choice([0, 1], n),
        "s1": rng.uniform(0, 1, n), "s2": rng.uniform(0, 1, n),
    })

def test_scatter(df):
    r = lv.scatterplot(df, x="x", y="y")
    assert r.fig is not None
    r = lv.scatterplot(df, x="x", y="y", hue="group", fit="linear")
    assert r.stats

def test_distribution(df):
    assert lv.distribution(df, x="score", kind="both").fig is not None

def test_box(df):
    assert lv.box(df, y="score", x="group").stats

def test_violin(df):
    assert lv.violin(df, y="score", x="group").fig is not None

def test_raincloud(df):
    assert lv.raincloud(df, y="score", x="group").stats

def test_heatmap(df):
    assert lv.heatmap(df, corr="pearson").fig is not None

def test_upset():
    data = {"A": {1,2,3,4}, "B": {3,4,5}, "C": {4,5,6}}
    r = lv.upset(data)
    assert r.fig is not None and "intersections" in r.tables

def test_roc(df):
    assert lv.roc(df, y_true="class", y_score=["s1", "s2"]).stats
