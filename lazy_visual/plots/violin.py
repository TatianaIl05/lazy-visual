from .box import _box_or_violin


def violin(data, *, y, x=None, test="auto", pairs=None, **style):
    return _box_or_violin(data, y=y, x=x, test=test, pairs=pairs,
                          kind="violin", **style)
