import numpy as np
import pandas as pd
from datamat import utils


def test_use_indices():
    idx = pd.MultiIndex.from_tuples([(i,) for i in range(4)], names=["i"])
    foo = pd.DataFrame({"cat": ["a", "b", "b", "c"]}, index=idx)

    result_i = utils.use_indices(foo, ["i"])
    assert result_i.shape[0] == foo.shape[0]
    assert np.all(result_i.index == idx)

    result_cat = utils.use_indices(foo, ["cat"])
    assert result_cat.size == 0

    result_both = utils.use_indices(foo, ["cat", "i"])
    assert result_both.shape[0] == foo.shape[0]
