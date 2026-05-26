"""Regression tests for ``DataVec`` construction from 2-D inputs.

A ``(1, n)`` pandas DataFrame is ambiguous to squeeze: ``.squeeze()`` keys
the result by *columns*, silently dropping the row label. Previously the
constructor accepted this and produced a vector with surprising labels;
it now refuses with a clear error.
"""

import numpy as np
import pandas as pd
import pytest
from datamat.core import DataVec


def test_datavec_from_column_dataframe_keeps_row_labels():
    df = pd.DataFrame([[1.0], [2.0], [3.0]], index=["a", "b", "c"], columns=["x"])

    v = DataVec(df)

    assert list(v.index.get_level_values(0)) == ["a", "b", "c"]
    assert list(v.values) == [1.0, 2.0, 3.0]


def test_datavec_from_1x1_dataframe_collapses_to_length_one():
    """A ``(1, 1)`` DataFrame is unambiguous: squeeze to a 1-element vector."""
    df = pd.DataFrame([[42.0]], index=["row"], columns=["col"])

    v = DataVec(df)

    assert len(v) == 1
    assert float(v.iloc[0]) == 42.0


def test_datavec_rejects_row_dataframe_with_multiple_columns():
    df = pd.DataFrame([[1.0, 2.0, 3.0]], index=["row"], columns=["x", "y", "z"])

    with pytest.raises(ValueError, match="Cannot infer DataVec orientation"):
        DataVec(df)


def test_datavec_from_row_numpy_array_squeezes_silently():
    """Unlike a DataFrame, a (1, n) numpy array carries no axis labels — the
    only sensible interpretation is to squeeze, so we allow it."""
    v = DataVec(np.array([[1.0, 2.0, 3.0]]))

    assert len(v) == 3
    assert list(v.values) == [1.0, 2.0, 3.0]
