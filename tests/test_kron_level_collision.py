"""Regression tests for level-name collisions in ``utils.kron``.

Previously kron silently produced a MultiIndex with duplicate level
names whenever the operands shared a name on the same axis. Pandas
accepts duplicates but operations like ``get_level_values(name)`` and
``droplevel(name)`` then raise ``ValueError: The name <n> occurs
multiple times`` from deep inside pandas — far from the kron call.
kron now raises with a clear message at the call site.
"""

import datamat as dm
import numpy as np
import pandas as pd
import pytest


def _square(name_idx, name_col, value=1):
    return dm.DataMat(
        [[value]],
        index=pd.MultiIndex.from_tuples([("p",)], names=[name_idx]),
        columns=pd.MultiIndex.from_tuples([("x",)], names=[name_col]),
    )


def test_kron_raises_when_column_names_collide():
    A = _square("a_idx", "shared", 1)
    B = _square("b_idx", "shared", 2)

    with pytest.raises(ValueError, match="kron: operand columns"):
        A.kron(B)


def test_kron_raises_when_index_names_collide():
    A = _square("shared", "a_col", 1)
    B = _square("shared", "b_col", 2)

    with pytest.raises(ValueError, match="kron: operand index"):
        A.kron(B)


def test_kron_accepts_distinct_level_names():
    A = _square("a_idx", "a_col", 1)
    B = _square("b_idx", "b_col", 2)

    K = A.kron(B)

    assert list(K.index.names) == ["a_idx", "b_idx"]
    assert list(K.columns.names) == ["a_col", "b_col"]
    np.testing.assert_array_equal(K.values, np.array([[2]]))
