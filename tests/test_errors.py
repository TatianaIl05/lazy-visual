import numpy as np
import pandas as pd
import pytest
import lazy_visual as lv


def test_e101(df):
    with pytest.raises(lv.LvDataError) as e:
        lv.scatterplot(df, x="nope", y="y")
    assert "LV-E101" in str(e.value)

def test_e102():
    with pytest.raises(lv.LvDataError) as e:
        lv.scatterplot(pd.DataFrame({"x": ["a"], "y": ["b"]}), x="x", y="y")
    assert "LV-E102" in str(e.value)

def test_e103():
    with pytest.warns(lv.LvDataWarning):
        with pytest.raises(lv.LvDataError) as e:
            lv.scatterplot(pd.DataFrame({"x": [np.nan], "y": [np.nan]}),
                           x="x", y="y")
    assert "LV-E103" in str(e.value)

def test_e201(df):
    with pytest.raises(lv.LvArgumentError) as e:
        lv.scatterplot(df, x="x", y="y", dpi=1)
    assert "LV-E201" in str(e.value)

def test_e203():
    with pytest.raises(lv.LvArgumentError) as e:
        lv.theme("nope")
    assert "LV-E203" in str(e.value)

def test_e301(df):
    with pytest.raises(lv.LvExportError) as e:
        lv.scatterplot(df, x="x", y="y", save="f.bmp")
    assert "LV-E301" in str(e.value)

def test_e106(df):
    with pytest.raises(lv.LvDataError) as e:
        lv.roc(df, y_true="group", y_score="s1")
    assert "LV-E106" in str(e.value)
