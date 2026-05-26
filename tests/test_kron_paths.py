"""Coverage tests for ``utils.kron`` across operand-type combinations.

``kron`` has four input shapes and one orthogonal ``sparse`` flag.
This file pins the supported combinations:

  - DataFrame × DataFrame (label preservation on both axes)
  - DataFrame × 1-D array  (column-vector shorthand)
  - 1-D array × DataFrame
  - array × array
  - sparse=True on DataFrame × DataFrame

For each: shape sanity, value equality with ``np.kron``, and (where
applicable) label preservation.

The mixed-type code path (DataFrame × non-DataFrame) only carries
enough metadata to handle a vector — a 2-D non-DataFrame operand
with more than one column is rejected explicitly. The previous
implementation silently flattened it and either produced a wrong
result or raised a confusing pandas length-mismatch.
"""

import datamat as dm
import numpy as np
import pandas as pd
import pytest
from datamat import utils


def _df(values, idxnames="i", colnames="j"):
    return pd.DataFrame(
        values,
        index=pd.MultiIndex.from_tuples(
            [(c,) for c in range(values.shape[0])], names=[idxnames]
        ),
        columns=pd.MultiIndex.from_tuples(
            [(c,) for c in range(values.shape[1])], names=[colnames]
        ),
    )


# ---------------------------------------------------------------------------
# DataFrame × DataFrame (the main, well-supported path)
# ---------------------------------------------------------------------------


def test_kron_dataframe_dataframe_matches_numpy_kron():
    A = _df(np.arange(4).reshape(2, 2).astype(float), "ai", "aj")
    B = _df(np.arange(6).reshape(2, 3).astype(float), "bi", "bj")

    out = utils.kron(A, B)

    np.testing.assert_array_equal(out.values, np.kron(A.values, B.values))


def test_kron_dataframe_dataframe_preserves_level_names():
    A = _df(np.eye(2), "ai", "aj")
    B = _df(np.eye(3), "bi", "bj")

    out = utils.kron(A, B)

    assert list(out.index.names) == ["ai", "bi"]
    assert list(out.columns.names) == ["aj", "bj"]
    assert out.shape == (6, 6)


# ---------------------------------------------------------------------------
# DataFrame × 1-D array  (column-vector shorthand)
# ---------------------------------------------------------------------------


def test_kron_dataframe_vector_replicates_each_row():
    A = _df(np.arange(4).reshape(2, 2).astype(float), "ai", "aj")
    b = np.array([10.0, 20.0])

    out = utils.kron(A, b)

    expected = np.kron(A.values, b.reshape(-1, 1))
    np.testing.assert_array_equal(out.values, expected)
    # Columns inherit from A; index gets one extra level for the
    # vector position.
    assert list(out.columns.names) == ["aj"]
    assert out.index.names == pd.core.indexes.frozen.FrozenList(["ai", None])
    assert out.shape == (4, 2)


# ---------------------------------------------------------------------------
# 1-D array × DataFrame
# ---------------------------------------------------------------------------


def test_kron_vector_dataframe_replicates_blockwise():
    a = np.array([10.0, 20.0])
    B = _df(np.arange(4).reshape(2, 2).astype(float), "bi", "bj")

    out = utils.kron(a, B)

    expected = np.kron(a.reshape(-1, 1), B.values)
    np.testing.assert_array_equal(out.values, expected)
    assert list(out.columns.names) == ["bj"]
    assert out.index.names == pd.core.indexes.frozen.FrozenList([None, "bi"])
    assert out.shape == (4, 2)


# ---------------------------------------------------------------------------
# array × array
# ---------------------------------------------------------------------------


def test_kron_array_array_matches_numpy_and_drops_labels():
    a = np.arange(4).reshape(2, 2)
    b = np.arange(6).reshape(2, 3)

    out = utils.kron(a, b)

    np.testing.assert_array_equal(out.values, np.kron(a, b))
    assert out.index.names == pd.core.indexes.frozen.FrozenList([None])
    assert out.columns.names == pd.core.indexes.frozen.FrozenList([None])


# ---------------------------------------------------------------------------
# sparse=True
# ---------------------------------------------------------------------------


def test_kron_sparse_dataframe_dataframe_matches_dense():
    A = _df(np.eye(3), "ai", "aj")
    B = _df(np.eye(2), "bi", "bj")

    sparse_out = utils.kron(A, B, sparse=True)
    dense_out = utils.kron(A, B, sparse=False)

    # ``sparse.to_dense()`` uses NaN as the fill marker for unstored
    # zeros; treat NaNs and explicit zeros as equivalent here.
    sparse_arr = np.nan_to_num(np.asarray(sparse_out.sparse.to_dense()), nan=0.0)
    np.testing.assert_array_equal(sparse_arr, dense_out.values)
    assert list(sparse_out.index.names) == ["ai", "bi"]
    assert list(sparse_out.columns.names) == ["aj", "bj"]


# ---------------------------------------------------------------------------
# Identity property
# ---------------------------------------------------------------------------


def test_kron_with_1x1_identity_is_identity():
    """kron(A, [[1]]) == A (up to the extra labelled level)."""
    A = _df(np.arange(6).reshape(2, 3).astype(float), "ai", "aj")
    eye1 = _df(np.array([[1.0]]), "bi", "bj")

    out = utils.kron(A, eye1)

    np.testing.assert_array_equal(out.values, A.values)


# ---------------------------------------------------------------------------
# Reject 2-D non-DataFrame operands (#25)
# ---------------------------------------------------------------------------


def test_kron_rejects_2d_array_on_B_side():
    A = dm.DataMat(np.eye(2), idxnames="i", colnames="j")
    B_2d = np.array([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)

    with pytest.raises(ValueError, match="non-DataFrame B operand"):
        A.kron(B_2d)


def test_kron_rejects_2d_array_on_A_side():
    A_2d = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = dm.DataMat(np.eye(2), idxnames="i", colnames="j")

    with pytest.raises(ValueError, match="non-DataFrame A operand"):
        utils.kron(A_2d, B)


def test_kron_accepts_column_vector_2d_array():
    """A 2-D array with a single column is the documented column-vector
    shorthand and must still work."""
    A = _df(np.arange(4).reshape(2, 2).astype(float), "ai", "aj")
    b = np.array([[10.0], [20.0]])  # (2, 1)

    out = utils.kron(A, b)

    expected = np.kron(A.values, b)
    np.testing.assert_array_equal(out.values, expected)
