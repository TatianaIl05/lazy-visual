# lazy-visual

One-line statistical plots with built-in stats.

```python
import lazy_visual as lv
lv.raincloud(df, x="group", y="score", save="fig.svg")
```

- 8 plots: scatterplot, distribution, box, violin, raincloud, heatmap, upset, roc
- Built-in stats: Mann-Whitney U, Welch, Kruskal-Wallis, Holm, Cohen's d, DeLong/bootstrap AUC CI
- 4 themes: default, nature, apa, dark
- Export PNG/SVG/PDF (atomic)

MIT. Deps: numpy, scipy, pandas, matplotlib.
