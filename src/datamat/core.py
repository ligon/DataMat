from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import count
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import numpy as np
import pandas as pd
from scipy import sparse as scipy_sparse

try:  # Optional JAX dependency for device array conversions.
    import jax
    import jax.numpy as jnp
    from jax import tree_util as jax_tree_util

    _JAX_AVAILABLE = True
except ImportError:  # pragma: no cover - JAX not installed.
    jax = cast(Any, None)
    jnp = cast(Any, None)
    jax_tree_util = cast(Any, None)
    _JAX_AVAILABLE = False

if TYPE_CHECKING:  # pragma: no cover - typing aid
    import jax.numpy as _jax_typing

    JaxArray: TypeAlias = _jax_typing.ndarray
else:
    JaxArray: TypeAlias = Any

from . import utils
from .utils import inv as matrix_inv
from .utils import matrix_product
from .utils import pinv as matrix_pinv

try:  # Optional dependency retained for robust Stata ingestion.
    from lsms.tools import from_dta
except ImportError:  # pragma: no cover - optional path
    from_dta = None

__all__ = [
    "DataVec",
    "DataMat",
    "DataMatJax",
    "DataVecJax",
    "concat",
    "get_names",
    "reconcile_indices",
    "read_parquet",
    "read_pickle",
    "read_stata",
    "generalized_eig",
    "canonical_variates",
    "reduced_rank_regression",
]


_vec_name_counter = count()
RandomSpec = str | tuple[Any, ...] | Callable[..., Any]
RNGInput = np.random.Generator | int | None


def _normalize_rng(rng: RNGInput) -> np.random.Generator:
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _coerce_distribution_params(
    name: str, extra_args: Sequence[Any], params: dict[str, Any]
) -> dict[str, Any]:
    if not extra_args:
        return params

    if len(extra_args) == 1 and isinstance(extra_args[0], dict):
        params.update(extra_args[0])
        return params

    primary = extra_args[0]
    if name in {"chi2", "chisquare"}:
        params.setdefault("df", primary)
    elif name in {"bernoulli"}:
        params.setdefault("p", primary)
    elif name in {"binomial"}:
        params.setdefault("n", primary)
        if len(extra_args) > 1:
            params.setdefault("p", extra_args[1])
    elif name in {"poisson"}:
        params.setdefault("lam", primary)
    elif name in {"exponential"}:
        params.setdefault("scale", primary)
    elif name in {"uniform"}:
        if len(extra_args) == 1:
            params.setdefault("low", 0.0)
            params.setdefault("high", primary)
        else:
            params.setdefault("low", primary)
            params.setdefault("high", extra_args[1])
    elif name in {"normal", "gaussian"}:
        if len(extra_args) == 1:
            params.setdefault("scale", primary)
        else:
            params.setdefault("loc", primary)
            params.setdefault("scale", extra_args[1])
    else:
        raise ValueError(
            f"Positional parameters not supported for distribution '{name}'. "
            "Pass a dict for explicit keywords."
        )
    return params


def _draw_random_array(
    shape: tuple[int, ...],
    distribution: RandomSpec,
    rng: RNGInput = None,
    **dist_kwargs: Any,
) -> np.ndarray:
    generator = _normalize_rng(rng)

    if callable(distribution):
        try:
            sample = distribution(size=shape, rng=generator, **dist_kwargs)
        except TypeError:
            try:
                sample = distribution(size=shape, random_state=generator, **dist_kwargs)
            except TypeError:
                sample = distribution(size=shape, **dist_kwargs)
        return np.asarray(sample)

    if isinstance(distribution, tuple):
        if not distribution:
            raise ValueError("Distribution tuple must contain a name.")
        name = str(distribution[0]).lower()
        extra_args = distribution[1:]
    else:
        name = str(distribution).lower()
        extra_args = ()

    params = dict(dist_kwargs)
    params = _coerce_distribution_params(name, extra_args, params)

    if name in {"normal", "gaussian"}:
        loc = params.pop("loc", 0.0)
        scale = params.pop("scale", 1.0)
        return generator.normal(loc=loc, scale=scale, size=shape)
    if name in {"standard_normal"}:
        return generator.standard_normal(size=shape)
    if name == "uniform":
        low = params.pop("low", 0.0)
        high = params.pop("high", 1.0)
        return generator.uniform(low=low, high=high, size=shape)
    if name in {"chi2", "chisquare"}:
        df = params.pop("df", None)
        if df is None:
            raise ValueError("Chi-square distribution requires 'df'.")
        return generator.chisquare(df=df, size=shape)
    if name == "exponential":
        scale = params.pop("scale", 1.0)
        return generator.exponential(scale=scale, size=shape)
    if name == "bernoulli":
        p = params.pop("p", 0.5)
        return generator.binomial(n=1, p=p, size=shape)
    if name == "binomial":
        n = params.pop("n", 1)
        p = params.pop("p", 0.5)
        return generator.binomial(n=n, p=p, size=shape)
    if name == "poisson":
        lam = params.pop("lam", params.pop("lambda", 1.0))
        return generator.poisson(lam=lam, size=shape)

    raise ValueError(f"Unknown distribution '{name}'.")


def _fresh_vec_name(prefix: str = "vec") -> str:
    suffix = next(_vec_name_counter)
    return prefix if suffix == 0 else f"{prefix}_{suffix}"


def _unwrap_scalar_name(name: Any) -> Any:
    """Unwrap a single-element tuple name to its bare scalar.

    DataMat columns are always wrapped to a MultiIndex, so a 1-level
    column index yields tuple keys like ``('c',)``. When such a column
    is the only one in a single-column collapse (matmul on a (n, 1)
    operand, axis=0 concat that collapses to one column, etc.), the
    surrounding scalar is what users expect on ``DataVec.name``. Leave
    multi-level tuples alone — those carry real structure.
    """
    if isinstance(name, tuple) and len(name) == 1:
        return name[0]
    return name


class DataVec(pd.Series):
    """Column vector with labeled index for linear-algebra operations."""

    __pandas_priority__ = 5000

    def __init__(self, data=None, **kwargs):
        """Create a DataVec.

        Inherit from :meth: `pd.Series.__init__`.

        Additional Parameters
        ---------------------
        idxnames
                (List of) name(s) for levels of index.
        """
        if "idxnames" in kwargs.keys():
            idxnames = kwargs.pop("idxnames")
        else:
            idxnames = None

        if data is not None:
            try:
                if len(data.shape) == 2 and 1 in data.shape:
                    # A (1, n) pandas DataFrame is ambiguous: ``.squeeze()``
                    # would silently re-key the resulting vector by columns
                    # and drop the row label. Refuse rather than guess.
                    if (
                        isinstance(data, pd.DataFrame)
                        and data.shape[0] == 1
                        and data.shape[1] > 1
                    ):
                        raise ValueError(
                            "Cannot infer DataVec orientation from a (1, n) "
                            "DataFrame with n > 1: squeezing would key by "
                            "columns and drop the row label "
                            f"{data.index.tolist()!r}. Pass ``data.iloc[0]`` "
                            "(to take the row as a Series indexed by columns) "
                            "or ``data.T.iloc[:, 0]`` (to transpose first) to "
                            "make the intent explicit."
                        )
                    data = data.squeeze()
            except AttributeError:
                pass

        if kwargs.get("name") is None and getattr(data, "name", None) is None:
            kwargs["name"] = _fresh_vec_name()

        # Pre-wrap an explicit ``index=`` argument as a MultiIndex so the
        # finished Series is already MultiIndexed and we don't need to
        # rebind ``self.index`` after super().__init__. The rebind path
        # still exists below for the cases we can't anticipate up front
        # (list/scalar input → pandas-assigned RangeIndex; Series input
        # carrying its own non-MultiIndex), but skipping it on the common
        # ``DataVec(data, index=[...])`` path avoids a redundant index
        # copy and is friendlier to pandas Copy-on-Write semantics.
        explicit_index = kwargs.get("index")
        if explicit_index is not None and not isinstance(explicit_index, pd.MultiIndex):
            level_names = (
                list(explicit_index.names)
                if hasattr(explicit_index, "names")
                else [None]
            )
            kwargs["index"] = pd.MultiIndex.from_arrays(
                [explicit_index], names=level_names
            )

        super().__init__(data=data, **kwargs)

        # Residual case: ``data`` brought its own non-MultiIndex (e.g. a
        # Series with a flat Index) or pandas synthesised a RangeIndex.
        # Wrap to MultiIndex here. Operates on a freshly-constructed
        # Series so there's no aliasing risk with caller-held views.
        if not isinstance(self.index, pd.MultiIndex):
            self.index = pd.MultiIndex.from_arrays([self.index], names=self.index.names)

        if idxnames is None:
            idxnames = list(self.index.names)
            it = 0
            for i, lvl_name in enumerate(idxnames):
                if lvl_name is None:
                    idxnames[i] = f"_{it:d}"
                    it += 1
        elif isinstance(idxnames, str):
            idxnames = [idxnames]

        self.index.names = idxnames

    def __getitem__(self, key):
        """v.__getitem__(k) == v[k]

        For a DataVec whose index is a MultiIndex, a bare scalar key
        (e.g. ``v['a']``) is automatically wrapped to ``('a',)`` on
        retry. A genuine miss re-raises the *original* scalar-keyed
        KeyError so the message is not the more confusing
        ``KeyError: ('missing',)``.

        >>> v = DataVec({'a':1,'b':2})
        >>> v['a']
        1
        """
        try:
            return super().__getitem__(key)
        except KeyError as original:
            if isinstance(self.index, pd.MultiIndex) and not isinstance(key, tuple):
                try:
                    return super().__getitem__((key,))
                except KeyError:
                    pass
            raise original from None

    @property
    def _constructor(self):
        return DataVec

    @property
    def _constructor_expanddim(self):
        return DataMat

    # Unary operations
    def dg(self, sparse=True):
        """Return the diagonal matrix diag(v)."""
        if sparse:
            # We can wind up blowing ram if not careful...
            d = scipy_sparse.diags(self.values)
            return DataMat(
                pd.DataFrame.sparse.from_spmatrix(
                    d, index=self.index, columns=self.index
                )
            )
        else:
            return DataMat(np.diag(self.values), index=self.index, columns=self.index)

    def norm(self, ord=None, **kwargs):
        """Vector norm ‖v‖_ord (defaults to the Euclidean norm)."""

        return np.linalg.norm(self, ord, **kwargs)

    def inv(self):
        """Inverse of a vector defined for 1-vector case."""
        if self.shape[0] == 1:
            return 1 / self.iloc[0]
        else:
            raise ValueError("Inverse of DataVec not defined.")

    # Binary operations
    def outer(self, other):
        """Outer product of two series (vectors)."""
        return DataMat(np.outer(self, other), index=self.index, columns=other.index)

    def proj(self, other):
        """Projection of self onto the column space of ``other``.

        Accepts either a DataMat (multi-dimensional basis) or a DataVec
        (single regressor). The DataVec path previously raised
        AttributeError because the helper referenced ``.columns`` on a
        Series.
        """
        if isinstance(other, DataVec):
            other = DataMat(other.to_frame())
        b = other.lstsq(self)
        return other @ b

    def lstsq(self, other):
        """Least-squares fit of ``other`` on ``self`` as a single regressor.

        Delegates to :meth:`DataMat.lstsq` after upcasting ``self`` to a
        single-column matrix. The previous implementation referenced
        ``self.columns`` (which does not exist on a Series); reaching that
        line was masked by an upstream LinAlgError, so the bug was dead
        code, but the contract was broken.
        """
        return DataMat(self.to_frame()).lstsq(other)

    def resid(self, other):
        """Residual from projection of self on other."""
        return self.squeeze() - self.proj(other)

    # Other manipulations
    def concat(
        self,
        other,
        axis=0,
        levelnames=False,
        toplevelname="v",
        suffixer="_",
        drop_vestigial_levels=False,
        **kwargs,
    ):
        p = DataMat(self)
        out = p.concat(
            other,
            axis=axis,
            levelnames=levelnames,
            toplevelname=toplevelname,
            suffixer=suffixer,
            drop_vestigial_levels=drop_vestigial_levels,
            **kwargs,
        )

        if axis == 0:
            out = out.squeeze()

        return out

    def dummies(self, cols, suffix=""):
        return DataMat(utils.dummies(pd.DataFrame(self), cols, suffix=suffix))

    def drop_vestigial_levels(self):
        """Drop index levels that don't vary."""
        self.index = utils.drop_vestigial_levels(self.index, axis=0)
        return self

    def to_jax(self, *, dtype: Any | None = float) -> "DataVecJax":
        """Convert the vector into a JAX-compatible wrapper retaining labels.

        Mirrors :meth:`DataMat.to_jax`. The returned :class:`DataVecJax`
        carries the JAX values, the pandas index (rebuilt through a
        hashable tuple representation inside the pytree machinery — see
        :class:`DataMatJax`), and the vector's ``name`` so it round-trips
        through ``to_datavec()`` without losing identity.
        """
        if not _JAX_AVAILABLE:  # pragma: no cover - depends on optional JAX
            raise RuntimeError(
                "DataVec.to_jax() requires JAX. Install the 'jax' extra or add "
                "JAX to your environment."
            )

        # ``to_numpy(dtype=...)`` already produces an array of the requested
        # dtype; passing the same dtype to ``jnp.asarray`` is redundant and
        # triggers a UserWarning when the requested dtype isn't JAX's
        # currently-configured precision (the common ``dtype=float`` →
        # ``float64`` case under default x32 JAX). Drop the redundant
        # ``dtype=`` kwarg: JAX will downcast silently to match its
        # configured precision, identical to the previous behaviour but
        # without the noise.
        if dtype is None:
            numpy_array = self.to_numpy()
        else:
            numpy_array = self.to_numpy(dtype=dtype)
        values = jnp.asarray(numpy_array)
        return DataVecJax(
            values=values,
            index=self.index.copy(),
            name=self.name,
        )

    @classmethod
    def from_jax(cls, wrapper: "DataVecJax") -> "DataVec":
        """Rebuild a :class:`DataVec` from a :class:`DataVecJax` wrapper."""
        return wrapper.to_datavec()

    @classmethod
    def random(
        cls,
        size: int,
        distribution: RandomSpec = "normal",
        *,
        index: Sequence[Any] | pd.Index | None = None,
        idxnames: str | Sequence[str] | None = None,
        name: str | None = None,
        rng: RNGInput = None,
        **dist_kwargs: Any,
    ) -> "DataVec":
        """Draw a random vector with optional labelled index.

        Parameters
        ----------
        size : int
            Length of the vector.
        distribution : random-spec, default "normal"
            Name/tuple/callable describing the distribution. Built-in shorthands:
              - ``"normal"`` / ``"gaussian"`` (loc, scale)
              - ``"uniform"`` (low, high)
              - ``"chi2"`` / ``"chisquare"`` (df)
              - ``"exponential"`` (scale)
              - ``"bernoulli"`` (p)
              - ``"binomial"`` (n, p)
              - ``"poisson"`` (lam)
              - ``"standard_normal"``
            A tuple like ``("chi2", 3)`` maps to the appropriate parameters.
            Custom callables must accept ``size=`` and return an array.
        index : Sequence or pandas.Index, optional
            Values for the index; defaults to 0..size-1.
        idxnames : str or sequence of str, optional
            Name(s) applied to the index levels.
        name : str, optional
            Vector name; defaults to an auto-generated ``vec_*`` identifier.
        rng : numpy.random.Generator | int | None, optional
            RNG or seed used by the draw (falls back to ``default_rng()``).
        **dist_kwargs
            Additional keyword arguments forwarded to the distribution.

        Returns
        -------
        DataVec
            Labelled random vector of shape ``(size,)``.
        """

        values = _draw_random_array((size,), distribution, rng, **dist_kwargs)

        if index is None:
            index_obj = pd.RangeIndex(size)
        elif isinstance(index, pd.Index):
            index_obj = index.copy()
        else:
            index_obj = pd.Index(index)

        if idxnames is not None:
            index_obj = index_obj.set_names(idxnames)

        series = pd.Series(values, index=index_obj, name=name)
        return cls(series, idxnames=idxnames, name=name)


class DataMat(pd.DataFrame):
    """Matrix with labeled rows and columns supporting linear algebra semantics."""

    __pandas_priority__ = 6000

    def __init__(self, *args, **kwargs):
        """Create a DataMat.

        Inherit from :meth: `pd.DataFrame.__init__`.

        Additional Parameters
        ---------------------
        idxnames
                (List of) name(s) for levels of index.
        colnames
                (List of) name(s) for levels of columns.
        name
                String naming DataMat object.
        """
        if "idxnames" in kwargs.keys():
            idxnames = kwargs.pop("idxnames")
        else:
            idxnames = None

        if "colnames" in kwargs.keys():
            colnames = kwargs.pop("colnames")
        else:
            colnames = None

        if "name" in kwargs.keys():
            name = kwargs.pop("name")
        else:
            name = None

        super().__init__(*args, **kwargs)

        self.name = name

        # Always work with MultiIndex on both axes
        if not isinstance(self.index, pd.MultiIndex):
            self.index = pd.MultiIndex.from_arrays([self.index], names=self.index.names)

        if not isinstance(self.columns, pd.MultiIndex):
            self.columns = pd.MultiIndex.from_arrays(
                [self.columns], names=self.columns.names
            )

        if idxnames is None:
            idxnames = list(self.index.names)
            it = 0
            for i, lvl_name in enumerate(idxnames):
                if lvl_name is None:
                    idxnames[i] = f"_{it:d}"
                    it += 1
        elif isinstance(idxnames, str):
            idxnames = [idxnames]

        self.index.names = idxnames

        if colnames is None:
            colnames = list(self.columns.names)
            it = 0
            for i, lvl_name in enumerate(colnames):
                if lvl_name is None:
                    colnames[i] = f"_{it:d}"
                    it += 1
        elif isinstance(colnames, str):
            colnames = [colnames]

        self.columns.names = colnames

    def __getitem__(self, key):
        """X.__getitem__(k) == X[k]

        For a DataMat whose columns are a MultiIndex, a bare scalar key
        (e.g. ``X[0]``) is automatically wrapped to ``(0,)`` on retry.
        A genuine miss re-raises the *original* scalar-keyed KeyError
        rather than the more confusing ``KeyError: ('missing',)`` from
        the retry attempt.

        >>> X = DataMat([[1,2],[3,4]], columns=['a','b'], colnames='c')
        >>> X['a'].values.flatten().tolist()
        [1, 3]
        """
        try:
            return pd.DataFrame.__getitem__(self, key)
        except KeyError as original:
            if isinstance(self.columns, pd.MultiIndex) and not isinstance(key, tuple):
                try:
                    return pd.DataFrame.__getitem__(self, (key,))
                except KeyError:
                    pass
            raise original from None

    def set_index(self, columns, levels=None, inplace=False):
        """Set the DataMat index using existing columns.

        ``levels`` is the name(s) to apply to the resulting index levels
        (passed through to ``index.names``). It defaults to ``columns``;
        a bare string is wrapped to a 1-tuple either way.

        >>> X = DataMat([[1,2,3],[4,5,6]],columns=['a','b','c'],colnames='cols',idxnames='rows')
        >>> X.set_index(['a','b'])
        """
        if inplace:
            frame = self
        else:
            # GH 49473 Use "lazy copy" with Copy-on-Write
            frame = self.copy(deep=None)

        if levels is None:
            levels = columns
        # Wrap a bare string regardless of branch: ``frame.index.names = "foo"``
        # raises, but the previous version only wrapped when levels defaulted
        # from columns.
        if isinstance(levels, str):
            levels = (levels,)

        try:
            frame.index = pd.MultiIndex.from_frame(
                pd.DataFrame(frame.reset_index()[columns])
            )
        except ValueError:  # Issue with index vs. multiindex?
            columns = [(i,) for i in columns]
            frame.index = pd.MultiIndex.from_frame(
                pd.DataFrame(frame.reset_index()[columns])
            )

        frame.drop(columns, inplace=True, axis=1)
        frame.index.names = levels

        if not inplace:
            return frame

    @property
    def _constructor(self):
        return DataMat

    @property
    def _constructor_sliced(self):
        return DataVec

    def stack(self, **kwargs):
        if "future_stack" in kwargs.keys():
            return pd.DataFrame.stack(self, **kwargs)
        else:
            return pd.DataFrame.stack(self, future_stack=True, **kwargs)

    def drop_vestigial_levels(self, axis=None):
        """Drop index & column levels that don't vary.

        Takes a single optional parameter:
        - axis (default None): If axis=0, operate on index;
          if 1, on columns; if None, on both.
        """
        if axis is None:
            self.drop_vestigial_levels(axis=0)
            self.drop_vestigial_levels(axis=1)
        elif axis in (0, "index"):
            self.index = utils.drop_vestigial_levels(self.index, axis=0)
        elif axis in (1, "columns"):
            self.columns = utils.drop_vestigial_levels(self.columns, axis=1)
        return self

    def to_jax(self, *, dtype: Any | None = float) -> "DataMatJax":
        """Convert the matrix into a JAX-compatible wrapper retaining labels."""

        if not _JAX_AVAILABLE:  # pragma: no cover - depends on optional JAX
            raise RuntimeError(
                "DataMat.to_jax() requires JAX. Install the 'jax' extra or add "
                "JAX to your environment."
            )

        # See :meth:`DataVec.to_jax` for the rationale: ``to_numpy(dtype=...)``
        # already returns an array of the requested dtype, so passing
        # ``dtype=`` to ``jnp.asarray`` is redundant and the redundancy
        # triggers a UserWarning under default x32 JAX. JAX still downcasts
        # silently if the input dtype doesn't fit its configured precision.
        if dtype is None:
            numpy_array = self.to_numpy()
        else:
            numpy_array = self.to_numpy(dtype=dtype)
        values = jnp.asarray(numpy_array)
        return DataMatJax(
            values=values,
            index=self.index.copy(),
            columns=self.columns.copy(),
        )

    @classmethod
    def from_jax(cls, wrapper: "DataMatJax") -> "DataMat":
        """Rebuild a :class:`DataMat` from a :class:`DataMatJax` wrapper."""

        return wrapper.to_datamat()

    # Unary operations
    def inv(self):
        """Matrix inverse :math:`M^{-1}`.

        Computed fresh on each call: caching is unsafe because DataMat is a
        mutable pandas subclass and a cached value would not be invalidated
        by in-place edits to ``self``.
        """
        return DataMat(matrix_inv(self))

    def norm(self, ord=None, **kwargs):
        """Matrix norm ‖M‖_ord mirroring :func:`numpy.linalg.norm`."""

        return np.linalg.norm(self, ord, **kwargs)

    def det(self):
        """Matrix determinant ``det(M)`` (fresh each call; see :meth:`inv`)."""
        return np.linalg.det(self)

    def trace(self):
        """Sum of diagonal entries ``tr(M)`` (fresh each call; see :meth:`inv`)."""
        return np.trace(self)

    def dg(self):
        """Return the diagonal vector diag(M) of a square matrix.

        Raises ``ValueError`` if ``self.index`` and ``self.columns`` do not
        match (shape or labels), with a clearer message than the
        ``ValueError`` that would otherwise come out of an elementwise
        comparison with mismatched ``nlevels``.
        """
        if self.shape[0] != self.shape[1]:
            raise ValueError(
                f"DataMat.dg() requires a square matrix; got shape {self.shape}."
            )
        if self.index.nlevels != self.columns.nlevels or not np.all(
            self.index == self.columns
        ):
            raise ValueError(
                "DataMat.dg() requires index and columns to be identical "
                "(same nlevels and same labels)."
            )
        return DataVec(np.diag(self.values), index=self.index)

    def leverage(self):
        """Diagonal of the projection (hat) matrix for design ``self``.

        Fresh each call; see :meth:`inv` for the reason caching is unsafe.

        >>> DataMat([[1,2],[3,4],[5,6]],idxnames='i').leverage()  # doctest: +ELLIPSIS
        0  0    ...
        """
        return utils.leverage(self)

    def rank(self, **kwargs):
        """Matrix rank"""
        return np.linalg.matrix_rank(self, **kwargs)

    def svd(self, hermitian=False):
        """Singular value composition into U@S.dg@V.T."""

        u, s, vt = utils.svd(self, hermitian=hermitian)
        u = DataMat(u)
        vt = DataMat(vt)
        s = DataVec(s)

        return u, s, vt

    def eig(self, hermitian=False, ascending=True):
        """Eigendecomposition.  Returns eigenvalues & corresponding eigenvectors."""
        s2, u = utils.eig(self, hermitian=hermitian, ascending=ascending)
        u = DataMat(u)
        s2 = DataVec(s2)

        return s2, u

    def sqrtm(self, hermitian=False):
        return DataMat(utils.sqrtm(self))

    def cholesky(self):
        return DataMat(utils.cholesky(self))

    def pinv(self):
        """Moore-Penrose pseudo-inverse :math:`M^{+}`.

        Computed fresh on each call; see :meth:`inv` for the reason caching
        is unsafe on a mutable pandas subclass.
        """
        return DataMat(matrix_pinv(self))

    def triu(self, k: int = 0):
        """Return upper triangular part as DataMat, preserving labels."""
        data = np.triu(self.values, k=k)
        return DataMat(data, index=self.index, columns=self.columns)

    def tril(self, k: int = 0):
        """Return lower triangular part as DataMat, preserving labels."""
        data = np.tril(self.values, k=k)
        return DataMat(data, index=self.index, columns=self.columns)

    @property
    def vec(self):
        """Column-stacked vectorisation vec(M)."""

        levels = list(range(self.columns.nlevels))
        stacked = self.stack(level=levels, future_stack=True)
        return DataVec(stacked, idxnames=stacked.index.names)

    @classmethod
    def random(
        cls,
        shape: tuple[int, int],
        distribution: RandomSpec = "normal",
        *,
        index: Sequence[Any] | pd.Index | None = None,
        columns: Sequence[Any] | pd.Index | None = None,
        idxnames: str | Sequence[str] | None = None,
        colnames: str | Sequence[str] | None = None,
        rng: RNGInput = None,
        name: str | None = None,
        **dist_kwargs: Any,
    ) -> "DataMat":
        """Draw a random matrix with optional labelled axes.

        Parameters
        ----------
        shape : tuple[int, int]
            Number of rows and columns.
        distribution : random-spec, default "normal"
            Name/tuple/callable describing the distribution. Supports the
            same shorthands as :meth:`DataVec.random` (normal, uniform, chi-square,
            exponential, Bernoulli, binomial, Poisson, etc.).
        index, columns : sequence or pandas.Index, optional
            Labels for rows/columns; defaults to simple RangeIndex.
        idxnames, colnames : str or sequence of str, optional
            Names applied to the index/column levels.
        rng : numpy.random.Generator | int | None, optional
            RNG or seed used by the draw.
        name : str, optional
            Matrix name.
        **dist_kwargs
            Additional keyword arguments forwarded to the distribution.

        Returns
        -------
        DataMat
            Labelled matrix of shape ``shape``.
        """

        if len(shape) != 2:
            raise ValueError("Shape must be a tuple (rows, cols).")

        nrows, ncols = shape
        values = _draw_random_array((nrows, ncols), distribution, rng, **dist_kwargs)

        if index is None:
            index_obj = pd.RangeIndex(nrows)
        elif isinstance(index, pd.Index):
            index_obj = index.copy()
        else:
            index_obj = pd.Index(index)

        if idxnames is not None:
            index_obj = index_obj.set_names(idxnames)

        if columns is None:
            columns_obj = pd.RangeIndex(ncols)
        elif isinstance(columns, pd.Index):
            columns_obj = columns.copy()
        else:
            columns_obj = pd.Index(columns)

        if colnames is not None:
            columns_obj = columns_obj.set_names(colnames)

        frame = pd.DataFrame(values, index=index_obj, columns=columns_obj)
        return cls(frame, idxnames=idxnames, colnames=colnames, name=name)

    # Binary operations
    def matmul(self, other, strict=False, fillmiss=False, align=False):
        """Matrix product preserving labels on the surviving axes.

        See :func:`datamat.utils.matrix_product` for the meaning of
        ``strict``, ``fillmiss``, and ``align``. Briefly: ``strict=True``
        raises on a column/index mismatch, ``align=True`` reconciles by
        dropping vestigial levels (on local copies, leaving the caller's
        labels untouched).
        """
        Y = matrix_product(self, other, strict=strict, fillmiss=fillmiss, align=align)

        if len(other.shape) <= 1:
            if isinstance(other, DataVec):
                series = (
                    Y if isinstance(Y, pd.Series) else pd.Series(Y, index=self.index)
                )
                name = _unwrap_scalar_name(other.name)
                if name is None:
                    name = _fresh_vec_name()
                series.name = name
                return DataVec(series, idxnames=self.index.names, name=name)

            return DataVec(Y)

        if isinstance(other, DataMat) and other.shape[1] == 1:
            if isinstance(Y, pd.DataFrame):
                column_series = Y.iloc[:, 0]
            else:
                column_series = Y
            column_series = column_series.copy()
            name = _unwrap_scalar_name(other.columns[0])
            column_series.name = name
            return DataVec(column_series, idxnames=self.index.names, name=name)

        return DataMat(Y)

    __matmul__ = matmul

    def kron(self, other, sparse=False):
        return DataMat(utils.kron(self, other, sparse=sparse))

    def lstsq(self, other):
        rslt = np.linalg.lstsq(self, other, rcond=None)

        if len(rslt[0].shape) < 2 or rslt[0].shape[1] == 1:
            b = DataVec(rslt[0], index=self.columns)
        else:
            b = DataMat(rslt[0], index=self.columns, columns=other.columns)

        return b

    def proj(self, other):
        """Linear projection of self onto the column space of ``other``.

        Accepts either a DataMat (multi-dimensional basis) or a DataVec
        (single regressor); the DataVec is upcast to a single-column
        matrix so the standard lstsq path can be reused.
        """
        if isinstance(other, DataVec):
            other = DataMat(other.to_frame())
        b = other.lstsq(self)
        return other @ b

    def resid(self, other):
        """Residual from projection of self on other."""
        return self.squeeze() - self.proj(other)

    # Other transformations
    def dummies(self, cols, suffix=""):
        return DataMat(utils.dummies(pd.DataFrame(self), cols, suffix=suffix))

    def concat(
        self,
        other,
        axis=0,
        levelnames=False,
        toplevelname="v",
        suffixer="_",
        drop_vestigial_levels=False,
        **kwargs,
    ):
        """Concatenate ``self`` with ``other``.

        See the module-level :func:`concat` for the full semantics. This
        method is a thin shim over the same machinery
        (:func:`_finish_concat`); it differs only in that ``self`` is
        always prepended as the first object, using ``self.name`` as its
        name in the resulting MultiIndex.

        USAGE
        -----
        >>> a = DataVec([1,2],name='a',idxnames='i')
        >>> b = DataMat([[1,2],[3,4]],name='b',idxnames='i',colnames='j')
        >>> b.concat([a,b],axis=1,levelnames=True).columns.levels[0].tolist()
        ['b', 'a', 'b_0']
        """
        assign_missing, levelnames = _normalize_concat_levelnames(levelnames)

        if isinstance(other, dict):
            allobjs: list[Any] = [self] + list(other.values())
            allnames: list[Any] = [self.name] + list(other.keys())
        else:
            if isinstance(other, tuple):
                allobjs = [self] + list(other)
            elif isinstance(other, DataMat | DataVec):
                allobjs = [self, other]
            elif isinstance(other, list):
                allobjs = [self] + other
            else:
                raise ValueError(f"Unexpected type: {type(other).__name__}")
            allnames = get_names(allobjs, assign_missing=assign_missing)

        return _finish_concat(
            allobjs,
            allnames,
            axis=axis,
            levelnames=levelnames,
            toplevelname=toplevelname,
            suffixer=suffixer,
            drop_vestigial_levels=drop_vestigial_levels,
            **kwargs,
        )


# Type alias for the hashable label payload that lives in JAX pytree aux
# data: a tuple of (label-tuples, level-names). See ``_index_to_static`` /
# ``_static_to_index`` below.
_IndexStatic = tuple[tuple[Any, ...], tuple[Any, ...]]


def _index_to_static(idx: pd.Index) -> _IndexStatic:
    """Convert a pandas Index/MultiIndex to a hashable, ==-friendly
    representation suitable for use as JAX pytree aux metadata.

    pd.Index itself is *not* a valid metadata payload — it isn't
    hashable and ``Index.__eq__`` returns an element-wise array
    rather than a bool. Both are required by JAX's pytree-cache
    machinery; the symptom is

      ValueError: Exception raised while checking equality of metadata
                  fields of pytree.

    raised the moment two DataMatJax instances participate in the same
    traced computation. We collapse the labels to ``(tuple_of_tuples,
    tuple_of_level_names)`` — both fully hashable — and rebuild the
    pd.MultiIndex on the way out.
    """
    if isinstance(idx, pd.MultiIndex):
        return tuple(map(tuple, idx.tolist())), tuple(idx.names)
    return tuple((v,) for v in idx.tolist()), (idx.name,)


def _static_to_index(meta: _IndexStatic) -> pd.MultiIndex:
    """Inverse of :func:`_index_to_static` — rebuild a MultiIndex from the
    hashable payload."""
    tuples, names = meta
    return pd.MultiIndex.from_tuples(list(tuples), names=list(names))


def _check_axis_alignment(left: pd.Index, right: pd.Index, op_label: str) -> None:
    """Raise if two axes don't have identical values *and* identical level names.

    Used by the JAX-side wrappers' operators (``DataMatJax @ DataVecJax``,
    etc.) to enforce strict label safety on contracted axes. Unlike the
    pandas-side ``DataMat.matmul`` default — which aligns purely by
    position — the labelled JAX wrappers refuse misaligned axes by
    default. The rationale: the whole point of going through
    ``DataMatJax`` / ``DataVecJax`` rather than raw ``jnp`` arrays is to
    pin down which axis means what; silent positional alignment defeats
    that purpose.

    The check is host-side: ``index.equals`` returns a bool (unlike
    ``index ==`` which returns an array) so it works fine even when
    ``self.values`` is a JAX tracer.
    """
    same_names = list(left.names) == list(right.names)
    same_content = left.equals(right)
    if not (same_names and same_content):
        raise ValueError(
            f"{op_label}: contracted-axis labels do not match.\n"
            f"  left:  names={list(left.names)}, len={len(left)}\n"
            f"  right: names={list(right.names)}, len={len(right)}\n"
            "Reconcile the labels at the boundary (e.g. via "
            "``DataMat.matmul(..., align=True)`` on the pandas side, "
            "or by re-indexing the DataVec / DataMat before "
            "``.to_jax()``) — the JAX wrappers do not perform silent "
            "positional alignment."
        )


@dataclass(frozen=True)
class DataMatJax:
    """PyTree wrapper coupling JAX arrays with DataMat metadata.

    The ``values`` field is the single PyTree leaf (so ``jax.grad``
    differentiates with respect to it, ``jax.vmap`` maps over it, etc.).
    ``index`` and ``columns`` are stored alongside but live in the
    pytree's static aux data — see :func:`_index_to_static` for the
    JIT-safety dance required to put pandas labels in aux without
    breaking JAX's cache equality check.
    """

    values: JaxArray
    index: pd.Index
    columns: pd.Index

    def tree_flatten(
        self,
    ) -> tuple[tuple[JaxArray], tuple[_IndexStatic, _IndexStatic]]:
        return (
            (self.values,),
            (_index_to_static(self.index), _index_to_static(self.columns)),
        )

    @classmethod
    def tree_unflatten(
        cls,
        metadata: tuple[_IndexStatic, _IndexStatic],
        children: tuple[JaxArray],
    ) -> "DataMatJax":
        idx_meta, col_meta = metadata
        (values,) = children
        return cls(
            values=values,
            index=_static_to_index(idx_meta),
            columns=_static_to_index(col_meta),
        )

    def to_datamat(self) -> DataMat:
        return DataMat(
            self.values,
            index=self.index.copy(),
            columns=self.columns.copy(),
        )

    # ----- Labelled operators -----------------------------------------
    # All operators check contracted-axis label alignment strictly (host
    # side), unlike the pandas-side default which aligns positionally.
    # See :func:`_check_axis_alignment` for the rationale.

    @property
    def T(self) -> "DataMatJax":
        """Transpose: swap ``values``, ``index``, and ``columns``."""
        return DataMatJax(
            values=self.values.T,
            index=self.columns,
            columns=self.index,
        )

    def __matmul__(self, other: Any) -> "DataMatJax | DataVecJax":
        if isinstance(other, DataVecJax):
            _check_axis_alignment(self.columns, other.index, "DataMatJax @ DataVecJax")
            return DataVecJax(
                values=self.values @ other.values,
                index=self.index,
                name=other.name,
            )
        if isinstance(other, DataMatJax):
            _check_axis_alignment(self.columns, other.index, "DataMatJax @ DataMatJax")
            return DataMatJax(
                values=self.values @ other.values,
                index=self.index,
                columns=other.columns,
            )
        return NotImplemented


if _JAX_AVAILABLE:  # pragma: no cover - depends on optional JAX
    jax_tree_util.register_pytree_node(
        DataMatJax,
        lambda wrapper: wrapper.tree_flatten(),
        DataMatJax.tree_unflatten,
    )


@dataclass(frozen=True)
class DataVecJax:
    """PyTree wrapper coupling a 1-D JAX array with DataVec metadata.

    Mirrors :class:`DataMatJax` for the labelled-vector case. The
    ``values`` field is the single PyTree leaf — so ``jax.grad`` w.r.t.
    a ``DataVecJax`` returns a fresh ``DataVecJax`` with the same labels
    — while ``index`` and ``name`` live in the pytree's static aux data.
    Like :class:`DataMatJax`, the pandas index is tupled at flatten time
    to keep JAX's cache-equality machinery happy across multiple
    instances.

    ``__post_init__`` validates that ``values`` is 1-D and that its
    length matches the index. The check is skipped when ``values``
    doesn't expose a concrete ``.shape`` (e.g. during a trace where
    only an abstract value is available) — JAX guarantees length /
    rank consistency in that path.
    """

    values: JaxArray
    index: pd.Index
    name: Any = None

    def __post_init__(self) -> None:
        shape = getattr(self.values, "shape", None)
        if shape is None:
            return
        if len(shape) != 1:
            raise ValueError(f"DataVecJax.values must be 1-D; got shape {shape}.")
        if shape[0] != len(self.index):
            raise ValueError(
                f"DataVecJax: values length {shape[0]} != "
                f"index length {len(self.index)}."
            )

    def tree_flatten(
        self,
    ) -> tuple[tuple[JaxArray], tuple[_IndexStatic, Any]]:
        return (self.values,), (_index_to_static(self.index), self.name)

    @classmethod
    def tree_unflatten(
        cls,
        metadata: tuple[_IndexStatic, Any],
        children: tuple[JaxArray],
    ) -> "DataVecJax":
        idx_meta, name = metadata
        (values,) = children
        return cls(values=values, index=_static_to_index(idx_meta), name=name)

    def to_datavec(self) -> "DataVec":
        return DataVec(self.values, index=self.index.copy(), name=self.name)

    # ----- Labelled operators -----------------------------------------
    # Strict-by-default contracted-axis label check; see
    # :func:`_check_axis_alignment`.

    def __matmul__(self, other: Any) -> Any:
        if isinstance(other, DataVecJax):
            # Dot product. Contracted axis is each vector's own index.
            _check_axis_alignment(self.index, other.index, "DataVecJax @ DataVecJax")
            return self.values @ other.values  # scalar (0-D JaxArray)
        return NotImplemented


if _JAX_AVAILABLE:  # pragma: no cover - depends on optional JAX
    jax_tree_util.register_pytree_node(
        DataVecJax,
        lambda wrapper: wrapper.tree_flatten(),
        DataVecJax.tree_unflatten,
    )


def get_names(dms, assign_missing=False):
    """
    Given an iterable of DataMats or DataVecs, return a list of names.

    If an item does not have a name, give "None" unless assign_missing,
    in which case:

       assign_missing==True: use a sequence "_0", "_1", etc.
       assign_missing is a list: Use this list to assign names.

    >>> a = DataVec([1,2],name='a')
    >>> b = DataMat([[1,2]],name='b')
    >>> c = DataMat([[1,2]])

    >>> get_names([a,b,c])
    ['a', 'b', None]

    >>> get_names([a,b,c],assign_missing=True)
    ['a', 'b', '_0']
    """
    names = []
    for item in dms:
        try:
            names += [item.name]
        except AttributeError:
            names += [None]

    if not assign_missing:
        return names

    if assign_missing is True:
        missnames = (f"_{i:d}" for i in range(len(names)))
    else:
        missnames = (name for name in assign_missing)

    for i, item in enumerate(names):
        if item is None:
            names[i] = next(missnames)
    return names


def reconcile_indices(idxs, fillvalue="", drop_vestigial_levels=False):
    """
    Given a list of indices, give them all the same levels.

    >>> idx0 = pd.MultiIndex
    """
    # Get union of index level names, preserving order
    names = []
    dropped_level_values = []
    newidxs = []
    for x in idxs:
        # Identify vestigial levels & drop. Collect names first so we don't
        # mutate the index while iterating its levels (which previously caused
        # ``x.names[i]`` to dereference a shrunk FrozenList and raise
        # IndexError once two or more vestigial levels were dropped).
        droppednames: dict[Any, Any] = {}
        if drop_vestigial_levels and isinstance(x, pd.MultiIndex):
            # Normalise stale declared categories so "vestigial" is decided
            # by used-value count alone — same definition as
            # :func:`datamat.utils.drop_vestigial_levels`.
            x = x.remove_unused_levels()
            vestigial = [
                (n, level[0])
                for n, level in zip(x.names, x.levels, strict=True)
                if len(level) == 1
            ]
            # Keep at least one level so the index never collapses to empty.
            if len(vestigial) == x.nlevels and vestigial:
                vestigial = vestigial[:-1]
            for dropname, value in vestigial:
                x = x.droplevel(dropname)
                droppednames[dropname] = value
        dropped_level_values.append(droppednames)
        newidxs.append(x)
        for newname in x.names:
            if newname not in names:
                names += [newname]

    # Add levels to indices where necessary
    out = []
    for i, idx in enumerate(newidxs):
        for levelname in names:
            if levelname not in idx.names:
                droppednames = dropped_level_values[i]
                try:
                    fillvalue = droppednames[levelname]
                except KeyError:
                    fillvalue = ""
                idx = utils.concat(
                    [DataMat(index=idx)], keys=[fillvalue], names=[levelname]
                ).index
        if not isinstance(idx, pd.MultiIndex):
            idx = pd.MultiIndex.from_arrays([idx], names=idx.names)

        out.append(idx.reorder_levels(names))

    return out


def _normalize_concat_objects(objs: list[Any]) -> list[DataMat | DataVec]:
    """Convert pandas structures into DataMat/DataVec for consistent handling."""
    normalized: list[DataMat | DataVec] = []
    for obj in objs:
        if isinstance(obj, DataMat | DataVec):
            normalized.append(obj)
            continue
        if isinstance(obj, pd.Series):
            normalized.append(DataVec(obj))
            continue
        if isinstance(obj, pd.DataFrame):
            normalized.append(DataMat(obj))
            continue
        normalized.append(obj)
    return normalized


def _normalize_axis_arg(axis: Any) -> int:
    """Return normalized axis as 0 (index) or 1 (columns)."""
    if axis is None:
        return 0
    if isinstance(axis, str):
        axis_lower = axis.lower()
        if axis_lower in {"columns", "column", "col"}:
            return 1
        return 0
    return int(axis)


def _normalize_concat_levelnames(
    levelnames: bool | Sequence[Any],
) -> tuple[bool | Sequence[Any], bool]:
    """Return ``(assign_missing, levelnames_bool)`` for concat entry points.

    ``levelnames=False`` is the "no level header" mode (assign_missing=True
    so unnamed objects get an ``_i`` placeholder); any truthy value means
    "add a level header, use the supplied value as the assign_missing
    iterable".
    """
    if levelnames is False:
        return True, False
    return levelnames, True


def _finish_concat(
    allobjs: list[Any],
    allnames: list[Any],
    *,
    axis: Any,
    levelnames: bool,
    toplevelname: str,
    suffixer: str,
    drop_vestigial_levels: bool,
    **kwargs: Any,
) -> "DataMat | DataVec":
    """Shared body of :func:`concat` and :meth:`DataMat.concat`.

    Both entry points are responsible for unpacking the user's input into
    a parallel ``(allobjs, allnames)`` pair; everything from there —
    object normalisation, unique-name suffixing, index/column
    reconciliation, the underlying :func:`pandas.concat` call, and the
    final DataMat/DataVec return-type coercion — happens here in a
    single canonical pipeline. Keeping the two entry points fed by one
    body is how we avoid the kind of drift that produced H2 (the
    drop_vestigial_levels parity bug) and M4 (the dead allnames
    assignment).
    """
    allobjs = _normalize_concat_objects(list(allobjs))

    # Build a list of unique display names by suffixing duplicates.
    suffix = (f"{suffixer}{i:d}" for i in range(len(allnames)))
    unique_names: list[Any] = []
    for name in allnames:
        if name is None:
            name = next(suffix)
        if name not in unique_names:
            unique_names.append(name)
        else:
            unique_names.append(name + next(suffix))

    axis_number = _normalize_axis_arg(axis)

    # Reconcile row indices to a common level set.
    idxs = reconcile_indices(
        [obj.index for obj in allobjs],
        drop_vestigial_levels=drop_vestigial_levels,
    )
    for i in range(len(idxs)):
        allobjs[i].index = idxs[i]

    # Reconcile columns, lifting DataVec → DataMat as we go.
    allcols = []
    for i, obj in enumerate(allobjs):
        try:
            allcols.append(obj.columns)
        except AttributeError:  # No columns attribute (raw pd.Series, etc.)
            obj = DataMat(obj)
            allobjs[i] = obj
            allcols.append(obj.columns)
    cols = reconcile_indices(allcols, drop_vestigial_levels=drop_vestigial_levels)
    if axis_number == 1 and not levelnames:
        cols = _apply_names_to_singleton_columns(cols, unique_names)
    for i in range(len(idxs)):
        allobjs[i].columns = cols[i]

    # Dispatch to pandas. ``strict=True`` because unique_names is built
    # one-per-object above; any mismatch indicates a bug in that loop.
    d = dict(zip(unique_names, allobjs, strict=True))
    if levelnames:
        result = utils.concat(d, axis=axis, names=toplevelname, **kwargs)
    else:
        result = utils.concat(allobjs, axis=axis, **kwargs)

    # Coerce return type — DataMat on horizontal stacks, DataVec when an
    # axis=0 stack collapses to a single column.
    if axis_number == 1:
        if isinstance(result, DataMat):
            return result
        if isinstance(result, pd.DataFrame):
            return DataMat(result)
        if isinstance(result, pd.Series):
            return DataMat(result.to_frame())
        return DataMat(pd.DataFrame(result))

    if isinstance(result, DataVec):
        return result
    if isinstance(result, pd.Series):
        return DataVec(result)
    if isinstance(result, pd.DataFrame):
        if result.shape[1] == 1:
            series = result.iloc[:, 0].copy()
            series.name = _unwrap_scalar_name(result.columns[0])
            return DataVec(series)
        return DataMat(result)
    return result


def _apply_names_to_singleton_columns(
    columns: list[pd.MultiIndex], keys: Sequence[str]
) -> list[pd.MultiIndex]:
    """Rebrand the top-level *value* on each single-entry column index.

    "Singleton" here means *one column entry*, not "one level": for each
    pair ``(key, column_index)`` where ``len(column_index) == 1``, replace
    the value at level 0 with ``key``. Lower-level values and *all* level
    names are preserved.

    Multi-entry column indices (``len(column_index) > 1``) pass through
    unchanged. This is used by ``concat`` (horizontal, without a new
    level header) to give a converted-from-DataVec single column its
    owner's name instead of the synthetic placeholder produced by
    :func:`_normalize_concat_objects`.

    Callers must supply one key per column index; ``strict=True`` makes a
    length mismatch fail loudly rather than silently dropping the tail.
    """
    adjusted = []
    for key, column_index in zip(keys, columns, strict=True):
        if len(column_index) == 1:
            new_arrays = [np.repeat(key, len(column_index))]
            for level in range(1, column_index.nlevels):
                new_arrays.append(column_index.get_level_values(level))
            adjusted.append(
                pd.MultiIndex.from_arrays(new_arrays, names=column_index.names)
            )
        else:
            adjusted.append(column_index)
    return adjusted


def concat(
    dms,
    axis=0,
    levelnames=False,
    toplevelname="v",
    suffixer="_",
    drop_vestigial_levels=False,
    **kwargs,
):
    """Concatenate DataMat / DataVec / pandas objects.

    Uses :func:`pandas.concat` underneath, but ensures that when objects
    have MultiIndices with differently-named levels new levels are added
    so the result still has a MultiIndex on both axes. If ``dms`` is a
    dict, the keys become the new top-level on the output MultiIndex.

    This is the function form; :meth:`DataMat.concat` is the method form
    and shares the same machinery via :func:`_finish_concat`.

    USAGE
    -----
    >>> a = DataVec([1,2],name='a',idxnames='i')
    >>> b = DataMat([[1,2],[3,4]],name='b',idxnames='i',colnames='j')
    >>> concat([a,b],axis=1,levelnames=True).columns.levels[0].tolist()
    ['b', 'a', 'b_0']
    """
    assign_missing, levelnames = _normalize_concat_levelnames(levelnames)

    if isinstance(dms, dict):
        allobjs: list[Any] = list(dms.values())
        allnames: list[Any] = list(dms.keys())
    else:
        if isinstance(dms, tuple):
            allobjs = list(dms)
        elif isinstance(dms, DataMat | DataVec):
            allobjs = [dms]
        elif isinstance(dms, list):
            allobjs = list(dms)
        else:
            raise ValueError(f"Unexpected type: {type(dms).__name__}")
        allnames = get_names(allobjs, assign_missing=assign_missing)

    return _finish_concat(
        allobjs,
        allnames,
        axis=axis,
        levelnames=levelnames,
        toplevelname=toplevelname,
        suffixer=suffixer,
        drop_vestigial_levels=drop_vestigial_levels,
        **kwargs,
    )


def read_parquet(fn, **kwargs):
    return DataMat(pd.read_parquet(fn, **kwargs))


def read_pickle(fn, **kwargs):
    return DataMat(pd.read_pickle(fn, **kwargs))


def read_stata(fn, **kwargs):
    if from_dta is None:
        raise ImportError(
            "lsms.tools.from_dta is required for read_stata; install the lsms package."
        )
    return DataMat(from_dta(fn, **kwargs))


if __name__ == "__main__":
    a = DataVec([1, 2], name="a", idxnames="i")
    b = DataMat([[1, 2]], name="b", idxnames="i", colnames="j")
    c = DataMat([[1, 2]], colnames="k")
    d = c.concat([a, b], levelnames=True, axis=1)

    import doctest

    doctest.testmod()


def generalized_eig(A, B):
    """
    Generalized eigenvalue problem for symmetric matrices A & B, B positive definite.

    Roots λ solve A@v = λ * B@v.

    Returns list of roots λ and corresponding eigenvectors V.
    """
    from scipy.linalg import eigh

    eigvals, eigvecs = eigh(A, B)
    eigvals = eigvals[::-1]  # Biggest eigenvalues first
    eigvecs = eigvecs[:, ::-1]

    # Sanity check the leading eigenpair via the defining identity
    # ``A v = λ B v``. Use ``np.allclose`` (relative + absolute tol) instead
    # of a fixed 1e-10 absolute bound so the check scales with ‖A‖ and ‖B‖.
    residual = (A - eigvals[0] * B) @ eigvecs[:, 0]
    assert np.allclose(
        residual, 0.0, atol=1e-10, rtol=1e-8
    ), "generalized_eig: leading eigenpair fails A v = lambda B v."

    eigvecs = DataMat(eigvecs, index=A.index)
    eigvals = DataVec(eigvals)

    return eigvals, eigvecs


def canonical_variates(X, Y):
    """
    Canonical variates from Canonical Correlation Analysis.

    Returns u,v such that corr^2(Yu[m],Xv[m]) is maximized for m=1,...

    See Hastie-Tibshirani-Friedman (2009) Exercise 3.20 or Rao (1965) 8f.
    """

    m = min(X.shape[1], Y.shape[1])
    U1 = X - X.mean()
    U2 = Y - Y.mean()

    T = U1.shape[0]

    S11 = U1.T @ U1 / T
    S22 = U2.T @ U2 / T

    S12 = U1.T @ U2 / T
    S21 = S12.T

    eigvals, M = generalized_eig(S21 @ S11.inv() @ S12, S22)
    eigvals_alt, L = generalized_eig(S12 @ S22.inv() @ S21, S11)

    assert np.allclose(eigvals[:m], eigvals_alt[:m])

    # Flip signs if necessary to have positive correlations
    sign = np.sign(((S12 @ M) / (S11 @ L * np.sqrt(eigvals))).mean())  # cf. Rao 8f.1.2

    # Interpret as a correlation coefficient
    eigvals = np.sqrt(eigvals)

    return eigvals, L * sign, M


def reduced_rank_regression(X, Y, r):
    """
    Reduced rank multivariate regression Y = XB + e.

    Minimizes sum of squared errors subject to requirement that B.rank()==r.

    See Hastie et al (2009) S. 4.2 or She-Chen (2017).
    """

    muX = X.mean()
    muY = Y.mean()

    X = X - muX
    Y = Y - muY

    # ``utils.sqrtm`` returns a plain ``pd.DataFrame``, so going through
    # the DataMat method wraps the result back into a DataMat — needed
    # so the matmul chain below produces a DataMat all the way through
    # and the final ``.svd()`` resolves to our wrapper, not pandas'
    # missing attribute.
    C = Y.cov().sqrtm()

    U, rho, Vt = ((C @ Y.T @ (Y.proj(X))) @ C).svd()
    V = Vt.T

    # ``Y @ V.iloc[:, :r]`` and ``X.lstsq(...)`` will both collapse to a
    # DataVec when r == 1 (matmul / lstsq auto-collapse single-column
    # results). The back-transformation needs them as 2-D matrices, so
    # promote any DataVec back to a single-column DataMat.
    Vr = V.iloc[:, :r]
    Z = Y @ Vr
    if isinstance(Z, DataVec):
        Z = DataMat(Z.to_frame())
    b = X.lstsq(Z)
    if isinstance(b, DataVec):
        b = DataMat(b.to_frame())

    return b @ Vr.pinv()
