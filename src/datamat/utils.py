"""Utility functions supporting DataMat/DataVec classes."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import MultiIndex, concat, get_dummies
from pandas.errors import InvalidIndexError
from scipy import sparse as scipy_sparse


__all__ = [
    "concat",
    "inv",
    "pinv",
    "matrix_product",
    "diag",
    "drop_vestigial_levels",
    "leverage",
    "svd",
    "eig",
    "sqrtm",
    "cholesky",
    "kron",
    "dummies",
    "use_indices",
]


def concat(objs, axis=0, names=None, **kwargs):
    """Wrapper around pandas.concat that accepts dicts or iterables."""
    if isinstance(objs, dict):
        return pd.concat(objs, axis=axis, names=names, **kwargs)

    if names is not None:
        kwargs.setdefault("keys", names)
    return pd.concat(objs, axis=axis, **kwargs)


def inv(A):
    """Matrix inverse preserving pandas structure."""
    if np.isscalar(A):
        A = pd.DataFrame(np.array([[A]]))

    B = np.linalg.inv(A)
    return pd.DataFrame(B, columns=A.columns, index=A.index)


def pinv(A):
    """Moore-Penrose pseudo-inverse preserving pandas structure."""
    if np.isscalar(A):
        A = pd.DataFrame(np.array([[A]]))

    B = np.linalg.pinv(A)
    return pd.DataFrame(B, columns=A.index, index=A.columns)


def matrix_product(X, Y, strict=False, fillmiss=True):
    """Matrix product X @ Y allowing for missing data."""
    if strict and not all(X.columns == Y.index):
        X.columns = drop_vestigial_levels(X.columns)
        Y.index = drop_vestigial_levels(Y.index)

    if fillmiss:
        X = X.fillna(0)
        Y = Y.fillna(0)

    prod = np.dot(X, Y)

    if len(prod.shape) == 1 or prod.shape[1] == 1:
        return pd.Series(prod.squeeze(), index=X.index)

    try:
        cols = Y.columns
    except AttributeError:
        cols = None
    return pd.DataFrame(prod, index=X.index, columns=cols)


def diag(X, sparse=True):
    """Return diagonal of matrix or build diagonal matrix from Series."""
    try:
        assert X.shape[0] == X.shape[1]
        d = pd.Series(np.diag(X), index=X.index)
    except IndexError:
        if sparse:
            d = scipy_sparse.diags(X.values)
            d = pd.DataFrame.sparse.from_spmatrix(
                d, index=X.index, columns=X.index
            )
        else:
            d = pd.DataFrame(np.diag(X.values), index=X.index, columns=X.index)
    except AttributeError:
        d = np.diag(X)

    return d


def use_indices(df, idxnames):
    """Return DataFrame whose columns are the requested index levels."""
    if len(set(idxnames).intersection(df.index.names)) == 0:
        return pd.DataFrame(index=df.index)

    try:
        idx = df.index
        df = df.reset_index()[idxnames]
        df.index = idx
        return df
    except InvalidIndexError:
        return df


def drop_vestigial_levels(idx, axis=0, both=False, multiindex=True):
    """Drop index/column levels that do not vary."""
    if both:
        return drop_vestigial_levels(
            drop_vestigial_levels(idx, axis=1), multiindex=multiindex
        )

    transpose = axis == 1
    if transpose:
        idx = idx.T

    if isinstance(idx, (pd.DataFrame, pd.Series)):
        df = idx
        idx = df.index
        humpty_dumpty = True
    else:
        humpty_dumpty = False

    try:
        level = 0
        n_levels = len(idx.levels)
        while level < n_levels:
            if len(set(idx.codes[level])) <= 1:
                idx = idx.droplevel(level)
                n_levels -= 1
            else:
                level += 1
                if level >= n_levels:
                    break
    except AttributeError:
        pass

    if multiindex and not isinstance(idx, pd.MultiIndex):
        idx = pd.MultiIndex.from_tuples(idx.str.split("|").tolist(), names=[idx.name])

    if humpty_dumpty:
        df.index = idx
        idx = df
        if transpose:
            idx = idx.T

    return idx


def qr(X):
    """Pandas-friendly QR decomposition."""
    Q, R = np.linalg.qr(X)
    Q = pd.DataFrame(Q, index=X.index, columns=X.columns)
    R = pd.DataFrame(R, index=X.columns, columns=X.columns)
    return Q, R


def leverage(X):
    """Return leverage (diagonal of the hat matrix) for design matrix X."""
    Q, _ = qr(X)
    return (Q ** 2).sum(axis=1)


def svd(A, hermitian=False):
    """Singular value decomposition preserving pandas structures."""
    idx = A.index
    cols = A.columns
    u, s, vt = np.linalg.svd(
        A, compute_uv=True, full_matrices=False, hermitian=hermitian
    )
    u = pd.DataFrame(u, index=idx)
    vt = pd.DataFrame(vt, columns=cols)
    s = pd.Series(s)
    return u, s, vt


def eig(A, hermitian=False, ascending=True):
    """Eigenvalue decomposition preserving pandas index/columns."""
    idx = A.index
    if hermitian:
        s2, u = np.linalg.eigh(A)
    else:
        s2, u = np.linalg.eig(A)

    order = np.argsort(s2 if ascending else -s2)
    s2 = s2[order]
    u = u[:, order]

    u = pd.DataFrame(u, index=idx)
    s2 = pd.Series(s2.squeeze())
    return s2, u


def sqrtm(A, hermitian=False):
    """Return symmetric square root of positive semi-definite matrix."""
    u, s, vt = svd(A, hermitian=hermitian)
    if np.any(s < 0):
        raise ValueError("Matrix must be positive semi-definite.")
    return u @ np.diag(np.sqrt(s)) @ vt


def cholesky(A):
    """Cholesky factor of positive definite matrix."""
    L = np.linalg.cholesky(A)
    return pd.DataFrame(L, index=A.index, columns=A.columns)


def kron(A, B, sparse=False):
    """Kronecker product that preserves pandas index/column metadata."""
    if sparse:
        from scipy.sparse import kron as sparse_kron

    if isinstance(A, pd.DataFrame):
        a = A.values
        if isinstance(B, pd.DataFrame):
            columns = MultiIndex.from_tuples(
                [(*i, *j) for i in A.columns for j in B.columns],
                names=A.columns.names + B.columns.names,
            )
            b = B.values
        else:
            columns = (
                A.columns.remove_unused_levels()
                if hasattr(A.columns, "remove_unused_levels")
                else A.columns
            )
            b = np.asarray(B).reshape((-1, 1))
    elif isinstance(B, pd.DataFrame):
        columns = (
            B.columns.remove_unused_levels()
            if hasattr(B.columns, "remove_unused_levels")
            else B.columns
        )
        a = np.asarray(A).reshape((-1, 1))
        b = B.values
    else:
        a = np.asarray(A)
        b = np.asarray(B)
        columns = None

    if isinstance(A, pd.DataFrame) and isinstance(B, pd.DataFrame):
        index = MultiIndex.from_tuples(
            [(*i, *j) for i in A.index for j in B.index],
            names=A.index.names + B.index.names,
        )
    elif isinstance(A, pd.DataFrame):
        index = MultiIndex.from_tuples(
            [(*i, j) for i in A.index for j in range(np.asarray(B).shape[0])],
            names=A.index.names + [None],
        )
    elif isinstance(B, pd.DataFrame):
        index = MultiIndex.from_tuples(
            [(i, *j) for i in range(np.asarray(A).shape[0]) for j in B.index],
            names=[None] + B.index.names,
        )
    else:
        index = None

    if sparse:
        result = sparse_kron(a, b)
        return pd.DataFrame.sparse.from_spmatrix(result, index=index, columns=columns)

    result = np.kron(a, b)
    return pd.DataFrame(result, index=index, columns=columns)


def dummies(df, cols, suffix=False):
    """Construct indicator variables for combinations of ``cols``."""
    idxcols = list(set(df.index.names).intersection(cols))
    colcols = list(set(cols).difference(idxcols))

    if len(idxcols):
        idx = use_indices(df, idxcols)
        v = concat([idx, df[colcols]], axis=1)
    else:
        v = df[colcols]

    usecols = [v[s].squeeze() for s in idxcols + colcols]
    tuples = pd.Series(list(zip(*usecols)), index=v.index)

    v = get_dummies(tuples).astype(int)

    if suffix is True:
        suffix = "_d"

    if suffix not in (False, ""):
        columns = [tuple(f"{c}{suffix}" for c in t) for t in v.columns]
    else:
        columns = v.columns

    v.columns = MultiIndex.from_tuples(columns, names=idxcols + colcols)
    return v
