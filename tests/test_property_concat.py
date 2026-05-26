"""Property-based parity tests for ``concat`` / ``DataMat.concat``.

After the consolidation that routed both entry points through
``_finish_concat``, the two should produce equivalent output for every
supported input shape. Hand-written tests cover the common cases;
Hypothesis explores the combinatorial space.
"""

from itertools import product

import datamat as dm
import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

_LEVEL_NAMES = ["a", "b", "c"]


@st.composite
def small_datamats(draw, max_levels=2, max_rows=4, max_cols=3):
    """Strategy: small DataMats with named MultiIndex on both axes."""
    n_idx_levels = draw(st.integers(1, max_levels))
    idx_names = draw(
        st.lists(
            st.sampled_from(_LEVEL_NAMES),
            min_size=n_idx_levels,
            max_size=n_idx_levels,
            unique=True,
        )
    )
    n_col_levels = draw(st.integers(1, max_levels))
    col_names = draw(
        st.lists(
            st.sampled_from(_LEVEL_NAMES),
            min_size=n_col_levels,
            max_size=n_col_levels,
            unique=True,
        )
    )

    def axis_tuples(n_levels, max_n):
        level_values = []
        for _ in range(n_levels):
            n_vals = draw(st.integers(1, 2))
            vs = draw(
                st.lists(
                    st.integers(0, 5),
                    min_size=n_vals,
                    max_size=n_vals,
                    unique=True,
                )
            )
            level_values.append(vs)
        all_rows = list(product(*level_values))
        n = draw(st.integers(1, min(max_n, len(all_rows))))
        chosen = draw(
            st.lists(
                st.integers(0, len(all_rows) - 1),
                min_size=n,
                max_size=n,
                unique=True,
            ).map(sorted)
        )
        return [all_rows[i] for i in chosen]

    idx_tuples = axis_tuples(n_idx_levels, max_rows)
    col_tuples = axis_tuples(n_col_levels, max_cols)

    idx = pd.MultiIndex.from_tuples(idx_tuples, names=idx_names)
    cols = pd.MultiIndex.from_tuples(col_tuples, names=col_names)
    nrows, ncols = len(idx_tuples), len(col_tuples)
    values = np.arange(nrows * ncols, dtype=float).reshape(nrows, ncols)
    name = draw(st.sampled_from(["X", "Y", "Z", None]))
    return dm.DataMat(values, index=idx, columns=cols, name=name)


def _structurally_equal(a, b):
    if type(a) is not type(b):
        return False
    if list(a.index.names) != list(b.index.names):
        return False
    if hasattr(a, "columns") and list(a.columns.names) != list(b.columns.names):
        return False
    if a.index.tolist() != b.index.tolist():
        return False
    if hasattr(a, "columns") and a.columns.tolist() != b.columns.tolist():
        return False
    return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)


@settings(max_examples=40, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(small_datamats(), min_size=2, max_size=3))
def test_module_method_axis0_parity(mats):
    head, *tail = mats
    via_module = dm.concat(mats, axis=0)
    via_method = head.concat(tail, axis=0)
    assert _structurally_equal(via_module, via_method)


@settings(max_examples=40, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(small_datamats(), min_size=2, max_size=3))
def test_module_method_axis1_parity(mats):
    head, *tail = mats
    via_module = dm.concat(mats, axis=1)
    via_method = head.concat(tail, axis=1)
    assert _structurally_equal(via_module, via_method)


@settings(max_examples=40, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(small_datamats(), min_size=2, max_size=3))
def test_module_method_axis0_levelnames_parity(mats):
    head, *tail = mats
    via_module = dm.concat(mats, axis=0, levelnames=True)
    via_method = head.concat(tail, axis=0, levelnames=True)
    assert _structurally_equal(via_module, via_method)
