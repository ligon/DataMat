"""Pin a single definition of "vestigial level" across the package.

Previously :func:`datamat.utils.drop_vestigial_levels` used
``len(set(idx.codes[level])) <= 1`` (one *used* code) while
:func:`datamat.core.reconcile_indices` used ``len(level) == 1`` (one
*declared* category). The two disagreed on indices with unused
categorical levels — for example, a MultiIndex that has been boolean-
filtered without calling ``remove_unused_levels()``.

Both functions now ``remove_unused_levels()`` at entry, after which the
declared-categories test is the canonical rule.
"""

import pandas as pd
from datamat import utils
from datamat.core import reconcile_indices


def _filtered_with_unused_level():
    """MultiIndex with a level whose declared categories overshoot its rows."""
    full = pd.MultiIndex.from_product(
        [["x", "y"], ["a", "b"]], names=["outer", "inner"]
    )
    return full[full.get_level_values("inner") == "a"]


def test_drop_and_reconcile_agree_on_unused_levels():
    """Both helpers must treat a level with only unused declared categories
    as vestigial."""
    idx = _filtered_with_unused_level()
    # Declared inner: ['a', 'b']; used inner codes: only 0. Vestigial.
    assert idx.levels[1].tolist() == ["a", "b"]

    util_out = utils.drop_vestigial_levels(idx, axis=0, multiindex=False)
    (core_out,) = reconcile_indices([idx], drop_vestigial_levels=True)

    assert list(util_out.names) == list(core_out.names) == ["outer"]


def test_drop_vestigial_keeps_genuinely_varying_levels():
    idx = pd.MultiIndex.from_tuples(
        [("x", "a"), ("y", "a"), ("x", "b"), ("y", "b")],
        names=["outer", "inner"],
    )

    util_out = utils.drop_vestigial_levels(idx, axis=0, multiindex=False)
    (core_out,) = reconcile_indices([idx], drop_vestigial_levels=True)

    assert list(util_out.names) == ["outer", "inner"]
    assert list(core_out.names) == ["outer", "inner"]


def test_drop_vestigial_on_dataframe_with_unused_levels():
    """The DataFrame entrypoint of utils.drop_vestigial_levels should also
    use the unified definition.
    """
    idx = _filtered_with_unused_level()
    df = pd.DataFrame({"v": [1.0, 2.0]}, index=idx)

    out = utils.drop_vestigial_levels(df, axis=0, multiindex=False)

    assert list(out.index.names) == ["outer"]
    assert out["v"].tolist() == [1.0, 2.0]
