# Quickstart

```python
import numpy as np, pandas as pd, lazy_visual as lv
rng = np.random.default_rng(0); n = 300
df = pd.DataFrame({
    "group": rng.choice(["a","b","c"], n),
    "score": rng.normal(0,1,n) + rng.choice([0,1,2], n),
    "x": rng.normal(0,1,n), "y": rng.normal(0,1,n),
    "class": rng.choice([0,1], n),
    "s1": rng.uniform(0,1,n), "s2": rng.uniform(0,1,n),
})
r = lv.raincloud(df, x="group", y="score", save="fig.svg")
print(r)
lv.set_options(random_state=42)
lv.theme("nature")
```
