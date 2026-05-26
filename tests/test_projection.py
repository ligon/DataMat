"""Regression tests for projection and lstsq across DataMat/DataVec.

Previously ``DataVec.lstsq`` referenced ``self.columns`` on a Series and
``DataVec.proj`` / ``DataMat.proj`` raised when given a DataVec ``other``;
both are now upcast through a single-column DataMat.
"""

import datamat as dm
import numpy as np


def test_datavec_lstsq_single_regressor():
    """``v.lstsq(u)`` should return the OLS coefficient of u on v."""
    rng = np.random.default_rng(0)
    n = 50
    v = dm.DataVec(rng.normal(size=n), name="v")
    u = 2.5 * v + 0.1 * rng.normal(size=n)

    b = v.lstsq(u)

    # Single coefficient near 2.5.
    coef = float(np.asarray(b).squeeze())
    assert abs(coef - 2.5) < 0.1


def test_datavec_proj_on_datavec_returns_scaled_vector():
    rng = np.random.default_rng(1)
    n = 30
    u = dm.DataVec(rng.normal(size=n), name="u")
    v = dm.DataVec(rng.normal(size=n), name="v")

    p = u.proj(v)  # projection of u onto span(v)

    # p must be parallel to v
    p_arr = np.asarray(p).reshape(-1)
    v_arr = np.asarray(v)
    # outer-product-of-ratios test: p / v should be constant where v != 0
    ratio = p_arr / v_arr
    assert np.allclose(ratio - ratio[0], 0, atol=1e-10)


def test_datavec_proj_on_datamat_idempotent():
    rng = np.random.default_rng(2)
    n, k = 40, 3
    X = dm.DataMat(rng.normal(size=(n, k)), colnames="feat")
    v = dm.DataVec(rng.normal(size=n), name="v")

    p1 = v.proj(X)
    p2 = dm.DataVec(p1).proj(X)
    np.testing.assert_allclose(np.asarray(p1), np.asarray(p2), atol=1e-10)


def test_datamat_proj_on_datavec_returns_rank_one_projection():
    """``M.proj(v)`` projects each column of M onto span(v)."""
    rng = np.random.default_rng(3)
    n, p = 25, 4
    M = dm.DataMat(rng.normal(size=(n, p)), colnames="feat")
    v = dm.DataVec(rng.normal(size=n), name="v")

    proj = M.proj(v)

    assert proj.shape == M.shape
    # The (n × p) projection should have rank 1: every column ∝ v.
    proj_arr = np.asarray(proj)
    assert np.linalg.matrix_rank(proj_arr) == 1
    # And projecting again should be a no-op.
    proj2 = dm.DataMat(proj).proj(v)
    np.testing.assert_allclose(np.asarray(proj), np.asarray(proj2), atol=1e-10)
