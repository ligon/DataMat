"""Cover the small behavioural fixes batched under Low priority.

Focused checks for:
  - L2: ``DataMat.set_index(levels='name')`` accepts a bare string.
  - L4: ``DataMat.dg()`` raises ``ValueError`` (not ``AssertionError``)
    with a clear message on non-square inputs or label mismatches.
"""

import pandas as pd
import pytest
from datamat.core import DataMat


def test_set_index_accepts_bare_string_levels():
    X = DataMat(
        [[1, 2, 3], [4, 5, 6]],
        columns=["a", "b", "c"],
        colnames="cols",
        idxnames="rows",
    )

    Y = X.set_index("a", levels="myname")

    assert list(Y.index.names) == ["myname"]


def test_dg_raises_value_error_on_non_square():
    M = DataMat([[1, 2, 3], [4, 5, 6]], idxnames="i", colnames="j")

    with pytest.raises(ValueError, match="square matrix"):
        M.dg()


def test_dg_raises_value_error_on_nlevels_mismatch():
    M = DataMat(
        [[1, 2], [3, 4]],
        index=pd.MultiIndex.from_tuples([("a", "x"), ("b", "y")], names=["i", "j"]),
        columns=pd.MultiIndex.from_tuples([("p",), ("q",)], names=["k"]),
    )

    with pytest.raises(ValueError, match="index and columns to be identical"):
        M.dg()
