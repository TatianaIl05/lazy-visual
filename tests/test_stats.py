import numpy as np
from lazy_visual.stats.compare import cohens_d, holm
from lazy_visual.stats.roc import compute_roc


def test_holm_monotone():
    p = [0.01, 0.02, 0.03, 0.04]
    adj = holm(p)
    assert all(adj[i] <= adj[i+1] + 1e-12 for i in range(len(adj)-1))
    assert all(0 <= x <= 1 for x in adj)

def test_cohens_d_sign():
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([6, 7, 8, 9, 10])
    assert cohens_d(a, b) < 0

def test_roc_perfect():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    _, _, stat = compute_roc(y, s, ci=False)
    assert stat.estimate == 1.0
