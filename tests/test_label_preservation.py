import datamat as dm
import numpy as np
import pandas as pd
import pytest


def test_kron_preserves_multiindex_labels():
    idx_a = pd.MultiIndex.from_tuples([(0, "A"), (1, "B")], names=["entity", "panel"])
    col_a = pd.MultiIndex.from_tuples([("x", 0), ("y", 1)], names=["var", "slot"])
    A = dm.DataMat([[1, 2], [3, 4]], index=idx_a, columns=col_a)

    idx_b = pd.MultiIndex.from_tuples([("lo", 0), ("hi", 1)], names=["level", "state"])
    col_b = pd.MultiIndex.from_tuples([("z", "left")], names=["coef", "side"])
    B = dm.DataMat([[5], [6]], index=idx_b, columns=col_b)

    kron_ab = A.kron(B)

    assert kron_ab.index.names == ("entity", "panel", "level", "state")
    assert kron_ab.columns.names == ("var", "slot", "coef", "side")

    expected_index = [(*ia, *ib) for ia in idx_a.tolist() for ib in idx_b.tolist()]
    expected_columns = [(*ca, *cb) for ca in col_a.tolist() for cb in col_b.tolist()]
    assert list(kron_ab.index) == expected_index
    assert list(kron_ab.columns) == expected_columns


@pytest.mark.xfail(reason="Matmul drops column names on 1-column outputs", strict=True)
def test_matmul_preserves_labels():
    A = dm.DataMat([[1, 2]], idxnames=["obs"], colnames=["feature"])
    B = dm.DataMat([[3], [4]], idxnames=["feature"])
    B.columns = pd.MultiIndex.from_tuples([("target",)], names=["measurement"])

    product = A @ B

    assert product.index.names == ("obs",)
    assert product.columns.names == ("measurement",)
    assert list(product.columns) == [("target",)]


def test_diag_preserves_index_names():
    idx = pd.MultiIndex.from_tuples([(0, "A"), (1, "B")], names=["entity", "panel"])
    vec = dm.DataVec([1, 2], index=idx)

    diag = vec.dg(sparse=False)

    assert diag.index.names == idx.names
    assert diag.columns.names == idx.names

    np.testing.assert_array_equal(diag.values, np.diag(vec.values))
