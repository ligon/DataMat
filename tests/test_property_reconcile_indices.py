"""Property-based tests for ``reconcile_indices``.

The failure modes we have fixed in reconcile_indices (C1 IndexError on
multi-vestigial drop, M7 KeyError on differently-named-level reconcile)
were combinatorial — they only show up at specific level counts /
orderings / vestigial structures. Concrete tests can pin one example
each; Hypothesis explores the space.

Properties exercised:

  - Idempotence: ``reconcile([reconcile([X])[0]]) == reconcile([X])``.
  - Uniform names across outputs: every reconciled output has the same
    level names, in the same order.
  - Vestigial drop never collapses to zero levels.
  - With ``drop_vestigial_levels=True``, surviving multi-level outputs
    have no remaining vestigial levels.
"""

from itertools import product

import pandas as pd
from datamat.core import reconcile_indices
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

# Keep generated indices small so the test stays fast and the failure-
# minimisation runs are still readable.
_LEVEL_NAMES = ["a", "b", "c", "d", "e"]


@st.composite
def multiindexes(draw, min_levels=1, max_levels=3, min_rows=1, max_rows=5):
    """Strategy: small MultiIndexes with letter-named levels and integer values."""
    n_levels = draw(st.integers(min_levels, max_levels))
    names = draw(
        st.lists(
            st.sampled_from(_LEVEL_NAMES),
            min_size=n_levels,
            max_size=n_levels,
            unique=True,
        )
    )
    level_values = []
    for _ in range(n_levels):
        n_vals = draw(st.integers(1, 3))
        values = draw(
            st.lists(
                st.integers(0, 9),
                min_size=n_vals,
                max_size=n_vals,
                unique=True,
            )
        )
        level_values.append(values)
    rows = list(product(*level_values))
    n_rows = draw(st.integers(min_rows, min(max_rows, len(rows))))
    chosen = draw(
        st.lists(
            st.integers(0, len(rows) - 1),
            min_size=n_rows,
            max_size=n_rows,
            unique=True,
        ).map(sorted)
    )
    selected = [rows[i] for i in chosen]
    return pd.MultiIndex.from_tuples(selected, names=names)


@settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
@given(multiindexes())
def test_reconcile_idempotent_on_single_input(idx):
    (once,) = reconcile_indices([idx])
    (twice,) = reconcile_indices([once])
    assert list(once.names) == list(twice.names)
    assert once.tolist() == twice.tolist()


@settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(multiindexes(), min_size=1, max_size=4))
def test_reconcile_outputs_share_names(idxs):
    outs = reconcile_indices(idxs)
    canonical = list(outs[0].names)
    for o in outs:
        assert list(o.names) == canonical


@settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
@given(multiindexes())
def test_reconcile_drop_vestigial_never_empty(idx):
    (out,) = reconcile_indices([idx], drop_vestigial_levels=True)
    assert out.nlevels >= 1
    assert len(out) >= 1


@settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
@given(multiindexes(min_levels=2, max_levels=3))
def test_reconcile_drop_vestigial_leaves_no_vestigial_unless_forced(idx):
    """When the output retains more than one level, none of those levels
    should still be vestigial (the survivor-protection rule only applies
    when every level was vestigial)."""
    (out,) = reconcile_indices([idx], drop_vestigial_levels=True)
    if out.nlevels > 1:
        for n, level in zip(out.names, out.levels, strict=True):
            assert (
                len(level) > 1
            ), f"level {n!r} is still vestigial after drop: {level.tolist()}"


@settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(multiindexes(), min_size=1, max_size=4))
def test_reconcile_output_name_set_is_union_of_inputs(idxs):
    outs = reconcile_indices(idxs)
    expected_names = set()
    for i in idxs:
        expected_names.update(i.names)
    assert set(outs[0].names) == expected_names
