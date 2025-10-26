import pandas as pd
from datamat import utils


def test_dummies_shapes():
    idx = pd.MultiIndex.from_tuples([(i,) for i in range(4)], names=["i"])
    foo = pd.DataFrame({"cat": ["a", "b", "b", "c"]}, index=idx)

    assert utils.dummies(foo, ["i"]).shape == (4, 4)
    assert utils.dummies(foo, ["cat"]).shape == (4, 3)
