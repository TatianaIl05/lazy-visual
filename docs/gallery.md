# Gallery

Same df as Quickstart.

```python
lv.scatterplot(df, x="x", y="y", hue="group", fit="linear", save="scatter.png")
lv.distribution(df, x="score", hue="group", kind="both", save="dist.png")
lv.box(df, y="score", x="group", pairs=[("a","b"),("a","c")], save="box.png")
lv.violin(df, y="score", x="group", save="violin.png")
lv.raincloud(df, y="score", x="group", save="raincloud.png")
lv.heatmap(df, corr="pearson", save="heatmap.png")
lv.upset({"A":{1,2,3,4,5},"B":{3,4,5,6},"C":{4,5,6,7}}, save="upset.png")
lv.roc(df, y_true="class", y_score=["s1","s2"], save="roc.png")
```
