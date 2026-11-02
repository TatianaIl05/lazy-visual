import lazy_visual as lv

def test_scatter(df):
    r = lv.scatterplot(df, x="x", y="y", hue="group", fit="linear")
    assert r.fig is not None and r.stats

def test_distribution(df):
    assert lv.distribution(df, x="score", hue="group", kind="both").fig

def test_box(df):
    assert lv.box(df, y="score", x="group").stats

def test_violin(df):
    assert lv.violin(df, y="score", x="group").fig

def test_raincloud_v(df):
    assert lv.raincloud(df, y="score", x="group").stats

def test_raincloud_h(df):
    assert lv.raincloud(df, y="score", x="group", orient="h").fig

def test_heatmap_corr(df):
    assert lv.heatmap(df, corr="pearson").fig

def test_heatmap_long(df):
    assert lv.heatmap(df, x="group", y="class", value="score").fig

def test_upset():
    r = lv.upset({"A": {1,2,3,4}, "B": {3,4,5}, "C": {4,5,6}})
    assert r.fig and "intersections" in r.tables

def test_roc_delong(df):
    assert lv.roc(df, y_true="class", y_score=["s1","s2"]).stats

def test_roc_bootstrap(df):
    assert lv.roc(df, y_true="class", y_score="s1",
                  ci_method="bootstrap", n_boot=100).stats
