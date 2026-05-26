"""Regression tests for ``datamat.core.reconcile_indices``.

The previous implementation mutated ``x`` while iterating ``x.levels`` and
indexed ``x.names`` with a stale enumeration index, which raised
``IndexError`` once two or more vestigial levels needed to be dropped.
"""

import pandas as pd
import pytest
from datamat.core import reconcile_indices


def test_drops_multiple_trailing_vestigial_levels():
    idx = pd.MultiIndex.from_tuples(
        [("x", "A", "P"), ("y", "A", "P")],
        names=["varying", "vest1", "vest2"],
    )

    (out,) = reconcile_indices([idx], drop_vestigial_levels=True)

    assert list(out.names) == ["varying"]
    assert out.tolist() == [("x",), ("y",)]


def test_drops_only_vestigial_level_with_varying_neighbours():
    idx = pd.MultiIndex.from_tuples(
        [("x", "A", 0), ("y", "A", 1)],
        names=["a", "b", "c"],
    )

    (out,) = reconcile_indices([idx], drop_vestigial_levels=True)

    assert list(out.names) == ["a", "c"]
    assert out.tolist() == [("x", 0), ("y", 1)]


def test_all_vestigial_keeps_one_level():
    """Never collapse an index to zero levels."""
    idx = pd.MultiIndex.from_tuples([("a", "b", "c")], names=["p", "q", "r"])

    (out,) = reconcile_indices([idx], drop_vestigial_levels=True)

    assert out.nlevels == 1
    assert len(out) == 1


def test_reconcile_returns_same_index_for_already_uniform_inputs():
    a = pd.MultiIndex.from_tuples([(0, "x"), (1, "y")], names=["i", "j"])
    b = pd.MultiIndex.from_tuples([(2, "z"), (3, "w")], names=["i", "j"])

    out_a, out_b = reconcile_indices([a, b])

    assert list(out_a.names) == ["i", "j"]
    assert list(out_b.names) == ["i", "j"]
    assert out_a.tolist() == a.tolist()
    assert out_b.tolist() == b.tolist()


@pytest.mark.parametrize(
    "names, tuples",
    [
        (["a", "b", "c", "d"], [("x", "A", 0, "P"), ("y", "A", 1, "P")]),
        (["a", "b", "c", "d"], [("x", "A", "P", "Q"), ("y", "A", "P", "Q")]),
    ],
)
def test_four_level_vestigial_dropping_does_not_crash(names, tuples):
    idx = pd.MultiIndex.from_tuples(tuples, names=names)

    (out,) = reconcile_indices([idx], drop_vestigial_levels=True)

    # Each remaining level must vary across rows (or be the single survivor).
    if out.nlevels > 1:
        for level in range(out.nlevels):
            assert out.get_level_values(level).nunique() > 1
