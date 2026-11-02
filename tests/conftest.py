import numpy as np
import pandas as pd
import pytest

@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame({
        "group": rng.choice(["a", "b", "c"], n),
        "score": rng.normal(0, 1, n) + rng.choice([0, 1, 2], n),
        "x": rng.normal(0, 1, n),
        "y": rng.normal(0, 1, n),
        "class": rng.choice([0, 1], n),
        "s1": rng.uniform(0, 1, n),
        "s2": rng.uniform(0, 1, n),
    })


import matplotlib.pyplot as plt

@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")
