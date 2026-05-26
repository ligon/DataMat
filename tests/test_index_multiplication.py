import datamat as dm
import pandas as pd


def test_index_multiplication():
    idx = pd.MultiIndex.from_tuples(
        [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1)],
        names=["i", "j", "k"],
    )
    X = dm.DataMat([[1, 2, 3, 4]], columns=idx, idxnames=["l"])
    Y = dm.DataMat(
        [[1, 2, 3, 0]],
        columns=idx.droplevel("j"),
        idxnames="m",
    ).T

    result = X @ Y
    assert result.index.names == ["l"]

    # align=True verifies MultiIndex reconciliation by dropping vestigial 'j'.
    X.matmul(Y, align=True)


def test_matmul_strict_raises_on_label_mismatch():
    import pytest

    idx = pd.MultiIndex.from_tuples(
        [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1)],
        names=["i", "j", "k"],
    )
    X = dm.DataMat([[1, 2, 3, 4]], columns=idx, idxnames=["l"])
    Y = dm.DataMat(
        [[1, 2, 3, 0]],
        columns=idx.droplevel("j"),
        idxnames="m",
    ).T

    with pytest.raises(ValueError, match="strict=True"):
        X.matmul(Y, strict=True)


def test_matmul_align_does_not_mutate_caller_labels():
    """``align=True`` must not change the caller's ``X.columns`` or ``Y.index``."""
    idx = pd.MultiIndex.from_tuples(
        [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1)],
        names=["i", "j", "k"],
    )
    X = dm.DataMat([[1, 2, 3, 4]], columns=idx, idxnames=["l"])
    Y = dm.DataMat(
        [[1, 2, 3, 0]],
        columns=idx.droplevel("j"),
        idxnames="m",
    ).T

    before_x = list(X.columns.names)
    before_y = list(Y.index.names)
    X.matmul(Y, align=True)
    assert list(X.columns.names) == before_x
    assert list(Y.index.names) == before_y
