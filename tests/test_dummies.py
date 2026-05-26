import pandas as pd
from datamat import utils


def test_dummies_shapes():
    idx = pd.MultiIndex.from_tuples([(i,) for i in range(4)], names=["i"])
    foo = pd.DataFrame({"cat": ["a", "b", "b", "c"]}, index=idx)

    assert utils.dummies(foo, ["i"]).shape == (4, 4)
    assert utils.dummies(foo, ["cat"]).shape == (4, 3)


def test_dummies_preserves_input_column_order():
    """Column-level names must follow the order of ``cols`` regardless of
    whether the names are index levels or data columns."""
    idx = pd.MultiIndex.from_tuples(
        [("A", 1), ("A", 2), ("B", 1), ("B", 2)],
        names=["g", "h"],
    )
    foo = pd.DataFrame({"cat": ["p", "q", "p", "q"]}, index=idx)

    # Mix of index levels and data columns — order should match input.
    result = utils.dummies(foo, ["g", "cat", "h"])
    assert list(result.columns.names) == ["g", "h", "cat"]
    # idx levels come first (in input order: g then h), then data cols (cat).

    # All-index case.
    result2 = utils.dummies(foo, ["h", "g"])
    assert list(result2.columns.names) == ["h", "g"]
