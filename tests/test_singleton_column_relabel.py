"""Pin the contract of ``_apply_names_to_singleton_columns``.

This internal helper rewrites the *value* at level 0 of any column index
that contains exactly one entry, leaving lower-level values and all
level *names* intact. Multi-entry column indices pass through.
"""

import pandas as pd
import pytest
from datamat.core import _apply_names_to_singleton_columns


def test_single_entry_single_level_replaces_value():
    col = pd.MultiIndex.from_tuples([("placeholder",)], names=["who"])

    (out,) = _apply_names_to_singleton_columns([col], ["renamed"])

    assert out.tolist() == [("renamed",)]
    assert out.names == ["who"]


def test_single_entry_multi_level_keeps_lower_levels_and_names():
    col = pd.MultiIndex.from_tuples(
        [("placeholder", "inner_val")], names=["who", "inner"]
    )

    (out,) = _apply_names_to_singleton_columns([col], ["renamed"])

    assert out.tolist() == [("renamed", "inner_val")]
    assert out.names == ["who", "inner"]


def test_multi_entry_column_passes_through_unchanged():
    col = pd.MultiIndex.from_tuples([("a", 1), ("b", 2)], names=["x", "y"])

    (out,) = _apply_names_to_singleton_columns([col], ["ignored"])

    assert out.equals(col)


def test_mismatched_length_inputs_raise():
    cols = [pd.MultiIndex.from_tuples([("p",)], names=["x"])]
    keys = ["k1", "k2"]

    with pytest.raises(ValueError):
        _apply_names_to_singleton_columns(cols, keys)
