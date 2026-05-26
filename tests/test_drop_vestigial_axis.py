"""Tests for ``DataMat.drop_vestigial_levels``'s axis-targeted branches.

H4 unified the underlying "vestigial" definition and indirectly
covered axis=0 via reconcile / concat. The public method (core.py)
also accepts axis=None (drop on both axes) and axis=1 / "columns"
(drop on columns only). Both branches were uncovered until now.
"""

import datamat as dm
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def matrix_with_vestigial_on_both_axes():
    """Build a DataMat whose row MultiIndex has a vestigial outer level
    and whose column MultiIndex has a vestigial inner level."""
    idx = pd.MultiIndex.from_tuples(
        [("v", "p"), ("v", "q"), ("v", "r")],
        names=["row_vest", "row_real"],
    )
    cols = pd.MultiIndex.from_tuples(
        [("x", "k"), ("y", "k")],
        names=["col_real", "col_vest"],
    )
    values = np.arange(6).reshape(3, 2).astype(float)
    return dm.DataMat(values, index=idx, columns=cols)


def test_drop_vestigial_axis_none_drops_both_axes(
    matrix_with_vestigial_on_both_axes,
):
    M = matrix_with_vestigial_on_both_axes
    M.drop_vestigial_levels()  # axis=None default

    assert list(M.index.names) == ["row_real"]
    assert list(M.columns.names) == ["col_real"]


def test_drop_vestigial_axis_columns_int_only_drops_columns(
    matrix_with_vestigial_on_both_axes,
):
    M = matrix_with_vestigial_on_both_axes
    M.drop_vestigial_levels(axis=1)

    # Columns lose the vestigial level; rows are untouched.
    assert list(M.columns.names) == ["col_real"]
    assert list(M.index.names) == ["row_vest", "row_real"]


def test_drop_vestigial_axis_columns_string_alias(
    matrix_with_vestigial_on_both_axes,
):
    """Pandas-style ``axis='columns'`` should work identically to axis=1."""
    M = matrix_with_vestigial_on_both_axes
    M.drop_vestigial_levels(axis="columns")

    assert list(M.columns.names) == ["col_real"]
    assert list(M.index.names) == ["row_vest", "row_real"]


def test_drop_vestigial_axis_index_string_alias(
    matrix_with_vestigial_on_both_axes,
):
    """``axis='index'`` should work identically to axis=0."""
    M = matrix_with_vestigial_on_both_axes
    M.drop_vestigial_levels(axis="index")

    assert list(M.index.names) == ["row_real"]
    assert list(M.columns.names) == ["col_real", "col_vest"]


def test_drop_vestigial_returns_self_for_chaining(
    matrix_with_vestigial_on_both_axes,
):
    M = matrix_with_vestigial_on_both_axes
    returned = M.drop_vestigial_levels()
    assert returned is M


def test_drop_vestigial_noop_on_already_clean_matrix():
    """A matrix with no vestigial levels should pass through unchanged."""
    idx = pd.MultiIndex.from_tuples([("a", 0), ("b", 1)], names=["g", "h"])
    cols = pd.MultiIndex.from_tuples([("x", 0), ("y", 1)], names=["f", "k"])
    M = dm.DataMat(np.eye(2), index=idx, columns=cols)

    M.drop_vestigial_levels()  # axis=None

    assert list(M.index.names) == ["g", "h"]
    assert list(M.columns.names) == ["f", "k"]
