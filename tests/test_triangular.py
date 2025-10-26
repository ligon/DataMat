import datamat as dm
import numpy as np
import pandas as pd


def build_matrix():
    idx = pd.MultiIndex.from_tuples(
        [("i0", "a"), ("i1", "b"), ("i2", "c")], names=["obs", "grp"]
    )
    cols = pd.MultiIndex.from_tuples(
        [
            ("x", 0),
            ("y", 1),
            ("z", 2),
        ],
        names=["feat", "slot"],
    )
    data = np.arange(9).reshape(3, 3)
    return dm.DataMat(data, index=idx, columns=cols)


def test_triu_preserves_labels():
    A = build_matrix()
    upper = A.triu()

    assert isinstance(upper, dm.DataMat)
    assert upper.index.equals(A.index)
    assert upper.columns.equals(A.columns)
    np.testing.assert_array_equal(upper.values, np.triu(A.values))


def test_tril_preserves_labels():
    A = build_matrix()
    lower = A.tril(k=-1)

    assert isinstance(lower, dm.DataMat)
    assert lower.index.equals(A.index)
    assert lower.columns.equals(A.columns)
    np.testing.assert_array_equal(lower.values, np.tril(A.values, k=-1))
