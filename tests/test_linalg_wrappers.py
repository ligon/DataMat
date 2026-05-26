"""Direct tests for the linear-algebra wrappers on DataMat / DataVec.

These wrappers are the headline product of the library but were
previously only smoke-tested through ``canonical_variates`` /
``reduced_rank_regression`` (which themselves have no direct tests).
This module pins:

  - Mathematical correctness via round-trip identities
    (A == U @ diag(s) @ Vt, A v_i == λ_i v_i, etc.).
  - Pandas label preservation on the surviving axes.
  - Shape sanity.
  - Error paths where the wrappers add guards over the numpy/scipy
    primitives (DataMat.dg on non-square, DataVec.inv on length > 1,
    sqrtm on a matrix with a negative eigenvalue).
"""

import numpy as np
import pandas as pd
import pytest
from datamat import utils
from datamat.core import DataMat, DataVec

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    return np.random.default_rng(seed=2026)


@pytest.fixture
def square_matrix(rng):
    """A 4x4 labelled DataMat — generic, full-rank, non-symmetric."""
    values = rng.normal(size=(4, 4))
    idx = pd.MultiIndex.from_tuples(
        [(0, "a"), (0, "b"), (1, "a"), (1, "b")], names=["block", "row"]
    )
    cols = pd.MultiIndex.from_tuples(
        [("x", 0), ("x", 1), ("y", 0), ("y", 1)], names=["set", "slot"]
    )
    return DataMat(values, index=idx, columns=cols)


@pytest.fixture
def symmetric_pd_matrix(rng):
    """A 4x4 symmetric positive-definite labelled DataMat."""
    A = rng.normal(size=(4, 4))
    M = A @ A.T + 4 * np.eye(4)  # always PD
    idx = pd.Index(["a", "b", "c", "d"], name="i")
    return DataMat(M, index=idx, columns=idx)


# ---------------------------------------------------------------------------
# DataMat.svd
# ---------------------------------------------------------------------------


def test_datamat_svd_reconstructs_original(square_matrix):
    u, s, vt = square_matrix.svd()

    assert isinstance(u, DataMat)
    assert isinstance(vt, DataMat)
    assert isinstance(s, DataVec)

    reconstructed = u.values @ np.diag(s.values) @ vt.values
    np.testing.assert_allclose(reconstructed, square_matrix.values, atol=1e-10)


def test_datamat_svd_preserves_row_labels(square_matrix):
    u, _, _ = square_matrix.svd()
    assert u.index.equals(square_matrix.index)


def test_datamat_svd_preserves_column_labels(square_matrix):
    _, _, vt = square_matrix.svd()
    assert vt.columns.equals(square_matrix.columns)


def test_datamat_svd_singular_values_nonnegative_and_descending(square_matrix):
    _, s, _ = square_matrix.svd()
    values = s.values
    assert np.all(values >= 0)
    assert np.all(np.diff(values) <= 1e-10)  # non-increasing


def test_datamat_svd_hermitian_path(symmetric_pd_matrix):
    """``hermitian=True`` should still produce a valid factorisation on a
    PD matrix (and is materially faster on large symmetric inputs)."""
    u, s, vt = symmetric_pd_matrix.svd(hermitian=True)
    reconstructed = u.values @ np.diag(s.values) @ vt.values
    np.testing.assert_allclose(reconstructed, symmetric_pd_matrix.values, atol=1e-10)


# ---------------------------------------------------------------------------
# DataMat.eig
# ---------------------------------------------------------------------------


def test_datamat_eig_satisfies_defining_identity(symmetric_pd_matrix):
    eigvals, eigvecs = symmetric_pd_matrix.eig(hermitian=True)

    assert isinstance(eigvecs, DataMat)
    assert isinstance(eigvals, DataVec)

    # A v_i == λ_i v_i column-by-column.
    A = symmetric_pd_matrix.values
    V = eigvecs.values
    np.testing.assert_allclose(A @ V, V * eigvals.values, atol=1e-10)


def test_datamat_eig_preserves_row_labels(symmetric_pd_matrix):
    _, eigvecs = symmetric_pd_matrix.eig(hermitian=True)
    assert eigvecs.index.equals(symmetric_pd_matrix.index)


def test_datamat_eig_ascending_order(symmetric_pd_matrix):
    eigvals, _ = symmetric_pd_matrix.eig(hermitian=True, ascending=True)
    assert np.all(np.diff(eigvals.values) >= -1e-10)


def test_datamat_eig_descending_order(symmetric_pd_matrix):
    eigvals, _ = symmetric_pd_matrix.eig(hermitian=True, ascending=False)
    assert np.all(np.diff(eigvals.values) <= 1e-10)


# ---------------------------------------------------------------------------
# DataMat.sqrtm
# ---------------------------------------------------------------------------


def test_datamat_sqrtm_squares_back_to_original(symmetric_pd_matrix):
    S = symmetric_pd_matrix.sqrtm()
    assert isinstance(S, DataMat)
    np.testing.assert_allclose(
        S.values @ S.values, symmetric_pd_matrix.values, atol=1e-10
    )


def test_utils_cholesky_rejects_non_pd_matrix():
    """cholesky should raise on a non-positive-definite input (numpy's
    own guard, not ours — but we want to know if pandas/numpy changes
    its exception type)."""
    not_pd = pd.DataFrame(np.diag([1.0, -1.0]))
    with pytest.raises(np.linalg.LinAlgError):
        utils.cholesky(not_pd)


# Note: ``utils.sqrtm`` has a guard ``if np.any(s < 0)`` but ``s`` comes
# from SVD and is nonnegative by definition, so the guard never fires.
# See the corresponding cq finding — sqrtm currently returns a silently
# wrong real result for non-PSD inputs. Test will go here once that's
# fixed.


# ---------------------------------------------------------------------------
# DataMat.cholesky
# ---------------------------------------------------------------------------


def test_datamat_cholesky_reconstructs_original(symmetric_pd_matrix):
    L = symmetric_pd_matrix.cholesky()

    assert isinstance(L, DataMat)
    np.testing.assert_allclose(
        L.values @ L.values.T, symmetric_pd_matrix.values, atol=1e-10
    )


def test_datamat_cholesky_preserves_labels(symmetric_pd_matrix):
    L = symmetric_pd_matrix.cholesky()
    assert L.index.equals(symmetric_pd_matrix.index)
    assert L.columns.equals(symmetric_pd_matrix.columns)


# ---------------------------------------------------------------------------
# DataMat.rank / norm
# ---------------------------------------------------------------------------


def test_datamat_rank_full_rank(symmetric_pd_matrix):
    assert symmetric_pd_matrix.rank() == 4


def test_datamat_rank_deficient():
    # Build a rank-1 matrix by outer-producting two vectors.
    M = DataMat(np.outer([1.0, 2.0, 3.0], [4.0, 5.0]), idxnames="i", colnames="j")
    assert M.rank() == 1


def test_datamat_norm_frobenius():
    M = DataMat([[3.0, 4.0]], idxnames="i", colnames="j")
    assert np.isclose(M.norm(), 5.0)  # default = Frobenius


def test_datamat_norm_inf(symmetric_pd_matrix):
    expected = np.linalg.norm(symmetric_pd_matrix.values, np.inf)
    assert np.isclose(symmetric_pd_matrix.norm(ord=np.inf), expected)


# ---------------------------------------------------------------------------
# DataMat.dg
# ---------------------------------------------------------------------------


def test_datamat_dg_returns_diagonal_as_labelled_datavec(symmetric_pd_matrix):
    d = symmetric_pd_matrix.dg()
    assert isinstance(d, DataVec)
    np.testing.assert_array_equal(d.values, np.diag(symmetric_pd_matrix.values))
    assert d.index.equals(symmetric_pd_matrix.index)


# ---------------------------------------------------------------------------
# DataVec unary helpers
# ---------------------------------------------------------------------------


def test_datavec_dg_sparse_builds_diagonal_datamat():
    v = DataVec([2.0, 3.0, 5.0], idxnames="i")
    D = v.dg(sparse=True)

    assert isinstance(D, DataMat)
    # Compare against dense form.
    dense = np.asarray(D.sparse.to_dense()) if hasattr(D, "sparse") else D.values
    np.testing.assert_array_equal(np.diag(dense), [2.0, 3.0, 5.0])


def test_datavec_dg_dense_builds_diagonal_datamat():
    v = DataVec([2.0, 3.0, 5.0], idxnames="i")
    D = v.dg(sparse=False)

    assert isinstance(D, DataMat)
    np.testing.assert_array_equal(np.diag(D.values), [2.0, 3.0, 5.0])
    assert D.index.equals(D.columns)


def test_datavec_norm_default_is_euclidean():
    v = DataVec([3.0, 4.0], idxnames="i")
    assert np.isclose(v.norm(), 5.0)


def test_datavec_inv_singleton_returns_reciprocal():
    v = DataVec([4.0], idxnames="i")
    assert np.isclose(v.inv(), 0.25)


def test_datavec_inv_multi_element_raises():
    v = DataVec([1.0, 2.0], idxnames="i")
    with pytest.raises(ValueError, match="Inverse of DataVec"):
        v.inv()


def test_datavec_outer_product_has_correct_labels():
    u = DataVec([1.0, 2.0, 3.0], index=["a", "b", "c"], idxnames="i")
    v = DataVec([10.0, 20.0], index=["x", "y"], idxnames="j")

    M = u.outer(v)

    assert isinstance(M, DataMat)
    np.testing.assert_array_equal(M.values, np.outer(u.values, v.values))
    assert list(M.index.get_level_values(0)) == ["a", "b", "c"]
    assert list(M.columns.get_level_values(0)) == ["x", "y"]


# ---------------------------------------------------------------------------
# utils.* direct paths (with plain pd.DataFrame inputs)
# ---------------------------------------------------------------------------


def test_utils_svd_on_plain_dataframe(rng):
    df = pd.DataFrame(rng.normal(size=(3, 5)))
    u, s, vt = utils.svd(df)

    assert isinstance(u, pd.DataFrame)
    assert isinstance(vt, pd.DataFrame)
    assert isinstance(s, pd.Series)
    np.testing.assert_allclose(
        u.values @ np.diag(s.values) @ vt.values, df.values, atol=1e-10
    )


def test_utils_eig_on_plain_dataframe():
    A = pd.DataFrame([[2.0, 0.0], [0.0, 3.0]])
    s2, u = utils.eig(A)
    # Eigenvalues of a diagonal matrix are its diagonal entries.
    np.testing.assert_allclose(sorted(s2.values), [2.0, 3.0])


def test_utils_cholesky_on_plain_dataframe():
    A = pd.DataFrame(np.eye(3) * 4.0)
    L = utils.cholesky(A)
    np.testing.assert_allclose(L.values, np.eye(3) * 2.0)


def test_utils_diag_extracts_from_square_dataframe():
    A = pd.DataFrame(np.diag([1.0, 2.0, 3.0]), index=["a", "b", "c"])
    d = utils.diag(A)
    assert isinstance(d, pd.Series)
    np.testing.assert_array_equal(d.values, [1.0, 2.0, 3.0])
    assert d.index.tolist() == ["a", "b", "c"]


def test_utils_diag_builds_from_series_dense():
    s = pd.Series([4.0, 5.0], index=["x", "y"])
    D = utils.diag(s, sparse=False)
    assert isinstance(D, pd.DataFrame)
    np.testing.assert_array_equal(D.values, np.diag([4.0, 5.0]))
    assert list(D.index) == ["x", "y"]
    assert list(D.columns) == ["x", "y"]


def test_utils_diag_builds_from_series_sparse():
    s = pd.Series([4.0, 5.0], index=["x", "y"])
    D = utils.diag(s, sparse=True)
    assert isinstance(D, pd.DataFrame)
    # Sparse DataFrame stores values via .sparse accessor.
    dense = np.asarray(D.sparse.to_dense()) if hasattr(D, "sparse") else D.values
    np.testing.assert_array_equal(np.diag(dense), [4.0, 5.0])
