"""Pin the narrowed ``__getitem__`` retry semantics on DataMat / DataVec.

The previous implementation caught every ``KeyError`` and re-tried the
lookup with ``(key,)``, which (a) re-raised genuine misses as the more
confusing ``KeyError: ('missing',)`` and (b) silently returned a
*different* column if a coincidentally-existing tuple key matched.

Both methods now retry only when the relevant axis is a MultiIndex
*and* the user-supplied key is not already a tuple. Any other KeyError
re-raises with its original payload.
"""

import pandas as pd
import pytest
from datamat.core import DataMat, DataVec


def test_datamat_missing_column_reraises_original_key():
    X = DataMat([[1, 2], [3, 4]], colnames="cols", idxnames="rows")

    with pytest.raises(KeyError) as exc_info:
        _ = X["definitely_not_a_column"]

    # The KeyError's args should be the original scalar key, not a tuple.
    assert exc_info.value.args[0] == "definitely_not_a_column"


def test_datavec_missing_label_reraises_original_key():
    v = DataVec([1, 2, 3], index=["a", "b", "c"], idxnames="i")

    with pytest.raises(KeyError) as exc_info:
        _ = v["definitely_not_a_label"]

    assert exc_info.value.args[0] == "definitely_not_a_label"


def test_datamat_scalar_key_still_resolves_via_tuple_retry():
    """The legitimate retry path — bare scalar against a MultiIndex
    column — must keep working.

    Note: with a 1-level MultiIndex, partial-match returns a 1-column
    DataMat (not a DataVec), so we verify via ``.values``.
    """
    X = DataMat([[1, 2, 3], [4, 5, 6]], colnames="cols", idxnames="rows")

    col0 = X[0]  # columns wrapped to MultiIndex as [(0,), (1,), (2,)]
    assert col0.values.flatten().tolist() == [1, 4]


def test_datamat_does_not_retry_for_tuple_keys():
    """If the user already passed a tuple and it misses, do NOT re-wrap."""
    cols = pd.MultiIndex.from_tuples([("a", 1), ("a", 2)], names=["x", "y"])
    X = DataMat([[1, 2], [3, 4]], columns=cols, idxnames="rows")

    with pytest.raises(KeyError):
        _ = X[("b", 3)]
