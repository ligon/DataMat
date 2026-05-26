"""Direct tests for ``generalized_eig`` / ``canonical_variates`` /
``reduced_rank_regression``.

These are the most likely-to-be-used-in-research methods in the
package and previously had zero direct coverage — they were only
imported, not exercised, by the test suite. Each test pins the
defining identity of the method rather than a numerical snapshot,
so a change in scipy / numpy alignment behaviour will surface here.
"""

import numpy as np
import pandas as pd
import pytest
from datamat.core import (
    DataMat,
    DataVec,
    canonical_variates,
    generalized_eig,
    reduced_rank_regression,
)

# ---------------------------------------------------------------------------
# generalized_eig
# ---------------------------------------------------------------------------


def test_generalized_eig_known_answer_with_B_identity():
    """When B = I, the generalized eigenvalue problem reduces to the
    ordinary one on A. With A = diag(4, 1) we should recover eigenvalues
    {4, 1} ordered biggest first."""
    A = DataMat(np.diag([4.0, 1.0]), idxnames="i", colnames="i")
    B = DataMat(np.eye(2), idxnames="i", colnames="i")

    eigvals, eigvecs = generalized_eig(A, B)

    np.testing.assert_allclose(eigvals.values, [4.0, 1.0], atol=1e-10)
    # eigvecs are unit-norm; their absolute values should match identity.
    np.testing.assert_allclose(np.abs(eigvecs.values), np.eye(2), atol=1e-10)


def test_generalized_eig_satisfies_defining_identity():
    """For arbitrary symmetric A and PD B, every column should satisfy
    A v_i = λ_i B v_i."""
    rng = np.random.default_rng(seed=2026)
    raw_A = rng.normal(size=(4, 4))
    A_arr = raw_A + raw_A.T  # symmetric
    raw_B = rng.normal(size=(4, 4))
    B_arr = raw_B @ raw_B.T + 4 * np.eye(4)  # symmetric PD

    A = DataMat(A_arr, idxnames="i", colnames="i")
    B = DataMat(B_arr, idxnames="i", colnames="i")

    eigvals, eigvecs = generalized_eig(A, B)

    AV = A_arr @ eigvecs.values
    BV_lambda = (B_arr @ eigvecs.values) * eigvals.values
    np.testing.assert_allclose(AV, BV_lambda, atol=1e-10)


def test_generalized_eig_returns_eigenvalues_descending():
    rng = np.random.default_rng(seed=7)
    raw_A = rng.normal(size=(3, 3))
    A_arr = raw_A + raw_A.T
    B_arr = np.eye(3)

    A = DataMat(A_arr, idxnames="i", colnames="i")
    B = DataMat(B_arr, idxnames="i", colnames="i")

    eigvals, _ = generalized_eig(A, B)

    assert np.all(np.diff(eigvals.values) <= 1e-10)


def test_generalized_eig_preserves_index_labels():
    A = DataMat(
        np.eye(3) * 2,
        index=pd.Index(["a", "b", "c"], name="row"),
        columns=pd.Index(["a", "b", "c"], name="col"),
    )
    B = DataMat(
        np.eye(3),
        index=pd.Index(["a", "b", "c"], name="row"),
        columns=pd.Index(["a", "b", "c"], name="col"),
    )

    _, eigvecs = generalized_eig(A, B)

    assert list(eigvecs.index.get_level_values(0)) == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# canonical_variates
# ---------------------------------------------------------------------------


def _make_correlated_xy(rng, n=200, k=3):
    """Build X (n, k) and Y (n, k) where each Y[:, i] is a noisy version
    of X[:, i]. Canonical correlations on (X, Y) should each be close
    to 1, ordered by signal-to-noise ratio."""
    X_raw = rng.normal(size=(n, k))
    noise = rng.normal(size=(n, k))
    # Per-column noise levels: column 0 cleanest, column k-1 noisiest.
    sigmas = np.linspace(0.05, 0.5, k)
    Y_raw = X_raw + sigmas * noise
    X = DataMat(X_raw, idxnames="i", colnames="x")
    Y = DataMat(Y_raw, idxnames="i", colnames="y")
    return X, Y


def test_canonical_variates_returns_three_components():
    rng = np.random.default_rng(seed=2026)
    X, Y = _make_correlated_xy(rng)

    rho, L, M = canonical_variates(X, Y)

    assert isinstance(rho, DataVec)
    assert isinstance(L, DataMat)
    assert isinstance(M, DataMat)


def test_canonical_variates_correlations_are_in_unit_interval():
    rng = np.random.default_rng(seed=2026)
    X, Y = _make_correlated_xy(rng)

    rho, _, _ = canonical_variates(X, Y)

    assert np.all(rho.values >= 0.0 - 1e-10)
    assert np.all(rho.values <= 1.0 + 1e-10)


def test_canonical_variates_first_correlation_high_for_noisy_clone():
    """With Y = X + small noise, the first canonical correlation must
    be close to 1 — that's the whole point of CCA on a near-identical
    pair."""
    rng = np.random.default_rng(seed=2026)
    X, Y = _make_correlated_xy(rng, n=400, k=3)

    rho, _, _ = canonical_variates(X, Y)

    assert rho.values[0] > 0.95


def test_canonical_variates_correlations_match_empirical():
    """For each canonical pair, corr(X L[:, i], Y M[:, i]) should equal
    the reported canonical correlation rho[i] (up to sign of the
    canonical vectors)."""
    rng = np.random.default_rng(seed=2026)
    X, Y = _make_correlated_xy(rng, n=300, k=3)

    rho, L, M = canonical_variates(X, Y)

    XL = (X - X.mean()).values @ L.values
    YM = (Y - Y.mean()).values @ M.values

    for i in range(len(rho)):
        empirical = np.corrcoef(XL[:, i], YM[:, i])[0, 1]
        assert np.isclose(abs(empirical), rho.values[i], atol=0.05)


# ---------------------------------------------------------------------------
# reduced_rank_regression
# ---------------------------------------------------------------------------


def _make_low_rank_yx(rng, n=200, p=4, q=3, true_rank=1):
    """Build X (n, p) and Y (n, q) from Y = X B + small_noise with B of
    true_rank ``true_rank``."""
    X_raw = rng.normal(size=(n, p))
    # Build a rank-r B by outer-producting two random matrices.
    L = rng.normal(size=(p, true_rank))
    R = rng.normal(size=(true_rank, q))
    B_true = L @ R
    Y_raw = X_raw @ B_true + 0.01 * rng.normal(size=(n, q))
    X = DataMat(X_raw, idxnames="i", colnames="x")
    Y = DataMat(Y_raw, idxnames="i", colnames="y")
    return X, Y, B_true


def test_rrr_returns_datamat_with_rank_le_r():
    rng = np.random.default_rng(seed=2026)
    X, Y, _ = _make_low_rank_yx(rng, true_rank=2)

    Br = reduced_rank_regression(X, Y, r=2)

    assert isinstance(Br, DataMat)
    assert Br.rank() <= 2


def test_rrr_recovers_known_rank_one_solution():
    """When the true B has rank 1, RRR with r=1 should recover B
    up to small numerical error."""
    rng = np.random.default_rng(seed=2026)
    X, Y, B_true = _make_low_rank_yx(rng, n=500, p=4, q=3, true_rank=1)

    Br = reduced_rank_regression(X, Y, r=1)

    # The recovered coefficients should be close to the truth — both
    # have rank 1, so their column spaces should align.
    Y_pred_truth = X.values @ B_true
    Y_pred_est = X.values @ Br.values
    np.testing.assert_allclose(Y_pred_est, Y_pred_truth, atol=0.1)


def test_rrr_at_full_rank_approximates_ols():
    """With r == min(p, q), RRR is the OLS solution (no rank constraint)."""
    rng = np.random.default_rng(seed=2026)
    X, Y, _ = _make_low_rank_yx(rng, n=500, p=4, q=3, true_rank=3)

    full_r = min(X.shape[1], Y.shape[1])
    Br = reduced_rank_regression(X, Y, r=full_r)

    # Compare to plain OLS on centred data.
    Xc = X - X.mean()
    Yc = Y - Y.mean()
    B_ols = Xc.lstsq(Yc)

    np.testing.assert_allclose(Br.values, B_ols.values, atol=1e-8)


@pytest.mark.parametrize("r", [1, 2, 3])
def test_rrr_rank_constraint_holds_for_various_r(r):
    rng = np.random.default_rng(seed=2026)
    X, Y, _ = _make_low_rank_yx(rng, n=300, p=4, q=3, true_rank=3)

    Br = reduced_rank_regression(X, Y, r=r)
    assert Br.rank() <= r
