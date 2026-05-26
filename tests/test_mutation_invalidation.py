"""Regression tests for matrix derivations on a mutable DataMat.

Previously ``inv``, ``det``, ``trace``, ``pinv`` and ``leverage`` were
``@cached_property`` attributes, so mutating the underlying matrix would
silently return the stale derivation. They are now plain methods that
recompute on every call.
"""

import datamat as dm
import numpy as np


def test_inv_recomputes_after_mutation():
    M = dm.DataMat([[2.0, 0.0], [0.0, 2.0]], idxnames="i", colnames="j")

    np.testing.assert_allclose(np.diag(M.inv().values), [0.5, 0.5])

    M.iloc[0, 0] = 4.0
    np.testing.assert_allclose(np.diag(M.inv().values), [0.25, 0.5])


def test_det_recomputes_after_mutation():
    M = dm.DataMat([[2.0, 0.0], [0.0, 2.0]], idxnames="i", colnames="j")
    assert np.isclose(M.det(), 4.0)
    M.iloc[0, 0] = 4.0
    assert np.isclose(M.det(), 8.0)


def test_trace_recomputes_after_mutation():
    M = dm.DataMat([[2.0, 0.0], [0.0, 2.0]], idxnames="i", colnames="j")
    assert np.isclose(M.trace(), 4.0)
    M.iloc[0, 0] = 4.0
    assert np.isclose(M.trace(), 6.0)


def test_pinv_recomputes_after_mutation():
    M = dm.DataMat([[2.0, 0.0], [0.0, 2.0]], idxnames="i", colnames="j")
    np.testing.assert_allclose(np.diag(M.pinv().values), [0.5, 0.5])
    M.iloc[0, 0] = 4.0
    np.testing.assert_allclose(np.diag(M.pinv().values), [0.25, 0.5])


def test_leverage_recomputes_after_mutation():
    X = dm.DataMat([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], idxnames="i")
    first = np.asarray(X.leverage())
    X.iloc[2, 1] = 5.0
    second = np.asarray(X.leverage())
    assert not np.allclose(first, second)
