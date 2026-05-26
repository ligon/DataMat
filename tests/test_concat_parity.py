"""Pin parity between :func:`datamat.concat` and :meth:`DataMat.concat`.

Both entry points share a single body (:func:`_finish_concat`). These
tests verify they produce equivalent results across the supported
input shapes — guarding against the kind of drift that previously
produced H2 (drop_vestigial_levels column-parity bug) and M4 (dead
allnames assignment).
"""

import datamat as dm
import numpy as np
import pandas as pd
import pytest


def _equal(a, b):
    """Compare two DataMat/DataVec for structural equality.

    Uses ``np.array_equal(..., equal_nan=True)`` on the value arrays so
    NaNs introduced by the union of differing column sets compare equal
    when they occupy the same positions.
    """
    assert type(a) is type(b), f"types differ: {type(a).__name__} vs {type(b).__name__}"
    assert list(a.index.names) == list(b.index.names)
    if hasattr(a, "columns"):
        assert list(a.columns.names) == list(b.columns.names)
        assert a.columns.tolist() == b.columns.tolist()
    assert a.index.tolist() == b.index.tolist()
    assert np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)


@pytest.fixture
def trio():
    a = dm.DataVec([1, 2], name="a", idxnames="i")
    b = dm.DataMat([[3, 4], [5, 6]], name="b", idxnames="i", colnames="j")
    c = dm.DataMat([[7, 8], [9, 10]], name="c", idxnames="i", colnames="j")
    return a, b, c


@pytest.mark.parametrize("axis", [0, 1])
def test_list_input_parity(trio, axis):
    a, b, c = trio
    module = dm.concat([b, a, c], axis=axis)
    method = b.concat([a, c], axis=axis)
    _equal(module, method)


@pytest.mark.parametrize("axis", [0, 1])
def test_dict_input_parity(trio, axis):
    a, b, c = trio
    module = dm.concat({"b": b, "a": a, "c": c}, axis=axis)
    method = b.concat({"a": a, "c": c}, axis=axis)
    _equal(module, method)


@pytest.mark.parametrize("axis", [0, 1])
def test_list_input_with_levelnames_parity(trio, axis):
    a, b, c = trio
    module = dm.concat([b, a, c], axis=axis, levelnames=True)
    method = b.concat([a, c], axis=axis, levelnames=True)
    _equal(module, method)


@pytest.mark.parametrize("axis", [0, 1])
def test_drop_vestigial_levels_parity(trio, axis):
    """Already covered for axis=0 elsewhere; sweep both axes here."""
    a, b, c = trio
    # Add a vestigial level to the indices.
    for x in (a, b, c):
        x.index = pd.MultiIndex.from_arrays(
            [["v"] * len(x), x.index.get_level_values(0)],
            names=["vest", "i"],
        )
    module = dm.concat([b, a, c], axis=axis, drop_vestigial_levels=True)
    method = b.concat([a, c], axis=axis, drop_vestigial_levels=True)
    _equal(module, method)


def test_single_object_module_and_method_agree(trio):
    """Edge case: passing one object on each side should be a no-op merge."""
    _, b, _ = trio
    module = dm.concat([b])
    method = b.concat([])  # other = [] → effectively just self
    _equal(module, method)
