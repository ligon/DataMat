from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cached_property
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
                    data = data.squeeze()
            except AttributeError:
                pass

        if kwargs.get("name") is None and getattr(data, "name", None) is None:
            kwargs["name"] = _fresh_vec_name()

        super().__init__(data=data, **kwargs)

        # Always work with a MultiIndex
        if not isinstance(self.index, pd.MultiIndex):
            self.index = pd.MultiIndex.from_arrays([self.index], names=self.index.names)

        if idxnames is None:
            idxnames = list(self.index.names)
            it = 0
            for i, name in enumerate(idxnames):
                if name is None:
                    idxnames[i] = f"_{it:d}"
                    it += 1
        elif isinstance(idxnames, str):
            idxnames = [idxnames]

        self.index.names = idxnames

    def __getitem__(self, key):
        """v.__getitem__(k) == v[k]

        >>> v = DataVec({'a':1,'b':2})
        >>> v['a']
        1
        """
        try:
            return super().__getitem__(key)
        except KeyError:  # Perhaps key was for an index?
            return super().__getitem__((key,))

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
        """Projection of self on other."""
        b = other.lstsq(self)
        return other @ b

    def lstsq(self, other):
        rslt = np.linalg.lstsq(self, other, rcond=None)

        if len(rslt[0].shape) < 2 or rslt[0].shape[1] == 1:
            b = DataVec(rslt[0], index=self.columns)
        else:
            b = DataMat(rslt[0], index=self.columns, columns=other.columns)

        return b

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
            for i, name in enumerate(idxnames):
                if name is None:
                    idxnames[i] = f"_{it:d}"
                    it += 1
        elif isinstance(idxnames, str):
            idxnames = [idxnames]

        self.index.names = idxnames

        if colnames is None:
            colnames = list(self.columns.names)
            it = 0
            for i, name in enumerate(colnames):
                if name is None:
                    colnames[i] = f"_{it:d}"
                    it += 1
        elif isinstance(colnames, str):
            colnames = [colnames]

        self.columns.names = colnames

    def __getitem__(self, key):
        """X.__getitem__(k) == X[k]

        >>> X = DataMat([[1,2,3],[4,5,6]],colnames='cols',idxnames='rows')
        >>> X[0].sum().squeeze()==5
        True
        """
        try:
            return pd.DataFrame.__getitem__(self, key)
        except KeyError:  # Perhaps key was for an index?
            return pd.DataFrame.__getitem__(self, (key,))

    def set_index(self, columns, levels=None, inplace=False):
        """Set the DataMat index using existing columns.

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

        if dtype is None:
            numpy_array = self.to_numpy()
            values = jnp.asarray(numpy_array, dtype=numpy_array.dtype)
        else:
            numpy_array = self.to_numpy(dtype=dtype)
            values = jnp.asarray(numpy_array, dtype=numpy_array.dtype)
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
    @cached_property
    def inv(self):
        return DataMat(matrix_inv(self))

    def norm(self, ord=None, **kwargs):
        """Matrix norm ‖M‖_ord mirroring :func:`numpy.linalg.norm`."""

        return np.linalg.norm(self, ord, **kwargs)

    @cached_property
    def det(self):
        return np.linalg.det(self)

    @cached_property
    def trace(self):
        return np.trace(self)

    def dg(self):
        """Return the diagonal vector diag(M) of a square matrix."""

        assert np.all(self.index == self.columns), "Should have columns same as index."
        return DataVec(np.diag(self.values), index=self.index)

    @cached_property
    def leverage(self):
        """Return leverage of matrix (diagonal of projection matrix).

        >>> DataMat([[1,2],[3,4],[5,6]],idxnames='i').leverage()
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

    @cached_property
    def pinv(self):
        """Moore-Penrose pseudo-inverse."""
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
    def matmul(self, other, strict=False, fillmiss=False):
        """Matrix product preserving labels on the surviving axes."""
        Y = matrix_product(self, other, strict=strict, fillmiss=fillmiss)

        if len(other.shape) <= 1:
            if isinstance(other, DataVec):
                series = (
                    Y if isinstance(Y, pd.Series) else pd.Series(Y, index=self.index)
                )
                name = other.name
                if isinstance(name, tuple) and len(name) == 1:
                    name = name[0]
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
            name = other.columns[0]
            if isinstance(name, tuple) and len(name) == 1:
                name = name[0]
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
        """Linear projection of self on other."""
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
        """Concatenate self and other.

        This uses the machinery of pandas.concat, but ensures that when two
        DataMats having multiindices with different number of levels are
        concatenated that new levels are added so as to preserve a result with a
        multiindex.

        if other is a dictionary and levelnames is not False, then a new level in the multiindex is created naming the columns belonging to the original DataMats.

        USAGE
        -----
        >>> a = DataVec([1,2],name='a',idxnames='i')
        >>> b = DataMat([[1,2],[3,4]],name='b',idxnames='i',colnames='j')
        >>> b.concat([a,b],axis=1,levelnames=True).columns.levels[0].tolist()
        ['b', 'a', 'b_0']
        """
        # Make other a list, unless it's a dict, and get allnames.
        if levelnames is False:
            assign_missing = True
        else:
            assign_missing = levelnames
            levelnames = True

        dict_input = isinstance(other, dict)

        allobjs = []
        if dict_input:
            allobjs = [self] + list(other.values())
            allnames = [self.name] + list(other.keys())
        else:
            if isinstance(other, tuple):
                allobjs = [self] + list(other)
            elif isinstance(other, DataMat | DataVec):
                allobjs = [self, other]
                allnames = [self.name] + get_names(
                    [other], assign_missing=assign_missing
                )
            elif isinstance(other, list):
                allobjs = [self] + other
            else:
                raise ValueError("Unexpected type")

            allnames = get_names(allobjs, assign_missing=assign_missing)

        allobjs = _normalize_concat_objects(list(allobjs))

        # Have list of all names, but may not be unique.

        suffix = (f"{suffixer}{i:d}" for i in range(len(allnames)))
        unique_names = []
        for _, name in enumerate(allnames):
            if name is None:
                name = next(suffix)
            if name not in unique_names:
                unique_names.append(name)
            else:
                unique_names.append(name + next(suffix))

        # Reconcile indices so they all have common named levels.
        idxs = reconcile_indices(
            [obj.index for obj in allobjs], drop_vestigial_levels=drop_vestigial_levels
        )
        for i in range(len(idxs)):
            allobjs[i].index = idxs[i]

        # Get list of columns, allowing for DataVec
        allcols = []
        for i, obj in enumerate(allobjs):
            try:
                allcols += [obj.columns]
            except AttributeError:  # No columns attribute?
                obj = DataMat(obj)
                allobjs[i] = obj
                allcols += [obj.columns]
        cols = reconcile_indices(allcols, drop_vestigial_levels=drop_vestigial_levels)
        if dict_input and axis == 1 and not levelnames:
            cols = _apply_dict_keys_to_singleton_columns(cols, unique_names)
        for i in range(len(idxs)):
            allobjs[i].columns = cols[i]

        # Now have a list of unique names, build a dictionary
        d = dict(zip(unique_names, allobjs, strict=False))

        if levelnames:
            return utils.concat(d, axis=axis, names=toplevelname, **kwargs)
        else:
            return utils.concat(allobjs, axis=axis, **kwargs)


@dataclass(frozen=True)
class DataMatJax:
    """PyTree wrapper coupling JAX arrays with DataMat metadata."""

    values: JaxArray
    index: pd.Index
    columns: pd.Index

    def tree_flatten(self) -> tuple[tuple[JaxArray], tuple[pd.Index, pd.Index]]:
        return (self.values,), (self.index, self.columns)

    @classmethod
    def tree_unflatten(
        cls, metadata: tuple[pd.Index, pd.Index], children: tuple[JaxArray]
    ) -> "DataMatJax":
        index, columns = metadata
        (values,) = children
        return cls(values=values, index=index, columns=columns)

    def to_datamat(self) -> DataMat:
        return DataMat(
            self.values,
            index=self.index.copy(),
            columns=self.columns.copy(),
        )


if _JAX_AVAILABLE:  # pragma: no cover - depends on optional JAX
    jax_tree_util.register_pytree_node(
        DataMatJax,
        lambda wrapper: wrapper.tree_flatten(),
        DataMatJax.tree_unflatten,
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
        # Identify vestigial levels & drop
        droppednames = {}
        for i, level in enumerate(x.levels):
            if drop_vestigial_levels and len(level) == 1:  # Vestigial level
                try:
                    if len(x.levels) > 1:
                        dropname = x.names[i]
                        x = x.droplevel(dropname)
                        droppednames[dropname] = level[0]
                except AttributeError:  # May be an index
                    pass
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


def _apply_dict_keys_to_singleton_columns(
    columns: list[pd.MultiIndex], keys: Sequence[str]
) -> list[pd.MultiIndex]:
    """Override singleton column entries with dictionary keys."""
    adjusted = []
    for key, column_index in zip(keys, columns, strict=False):
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
    """Concatenate self and other.

    This uses the machinery of pandas.concat, but ensures that when two
    DataMats having multiindices with different number of levels are
    concatenated that new levels are added so as to preserve a result with a
    multiindex.

    if other is a dictionary and levelnames is not False, then a new level in the multiindex is created naming the columns belonging to the original DataMats.

    USAGE
    -----
    >>> a = DataVec([1,2],name='a',idxnames='i')
    >>> b = DataMat([[1,2],[3,4]],name='b',idxnames='i',colnames='j')
    >>> concat([a,b],axis=1,levelnames=True).columns.levels[0].tolist()
    ['b', 'a', 'b_0']
    """

    # Make dms a list, unless it's a dict, and get allnames.
    if levelnames is False:
        assign_missing = True
    else:
        assign_missing = levelnames
        levelnames = True

    dict_input = isinstance(dms, dict)

    allobjs = []
    if dict_input:
        allobjs = list(dms.values())
        allnames = list(dms.keys())
    else:
        if isinstance(dms, tuple):
            allobjs = list(dms)
        elif isinstance(dms, DataMat | DataVec):
            allobjs = [dms]
            allnames = get_names([dms], assign_missing=assign_missing)
        elif isinstance(dms, list):
            allobjs = dms
        else:
            raise ValueError("Unexpected type")

        allnames = get_names(allobjs, assign_missing=assign_missing)

    allobjs = _normalize_concat_objects(list(allobjs))

    # Have list of all names, but may not be unique.

    suffix = (f"{suffixer}{i:d}" for i in range(len(allnames)))
    unique_names = []
    for name in allnames:
        if name is None:
            name = next(suffix)
        if name not in unique_names:
            unique_names.append(name)
        else:
            unique_names.append(name + next(suffix))

    # Reconcile indices so they all have common named levels.
    idxs = reconcile_indices(
        [obj.index for obj in allobjs], drop_vestigial_levels=drop_vestigial_levels
    )
    for i in range(len(idxs)):
        allobjs[i].index = idxs[i]

    # Get list of columns, allowing for DataVec
    allcols = []
    for i, obj in enumerate(allobjs):
        try:
            allcols += [obj.columns]
        except AttributeError:  # No columns attribute?
            obj = DataMat(obj)
            allobjs[i] = obj
            allcols += [obj.columns]
    cols = reconcile_indices(allcols)
    if dict_input and axis == 1 and not levelnames:
        cols = _apply_dict_keys_to_singleton_columns(cols, unique_names)
    for i in range(len(idxs)):
        allobjs[i].columns = cols[i]

    # Now have a list of unique names, build a dictionary
    d = dict(zip(unique_names, allobjs, strict=False))

    if axis is None:
        axis_number = 0
    elif isinstance(axis, str):
        axis_lower = axis.lower()
        if axis_lower in {"columns", "column", "col"}:
            axis_number = 1
        else:
            axis_number = 0
    else:
        axis_number = int(axis)

    if levelnames:
        result = utils.concat(d, axis=axis, names=toplevelname, **kwargs)
    else:
        result = utils.concat(allobjs, axis=axis, **kwargs)

    if axis_number == 1:
        if isinstance(result, DataMat):
            return result
        if isinstance(result, pd.DataFrame):
            return DataMat(result)
        if isinstance(result, pd.Series):
            return DataMat(result.to_frame())
        return DataMat(pd.DataFrame(result))

    # axis==0: favour returning a DataVec when result collapses to a single column
    if isinstance(result, DataVec):
        return result
    if isinstance(result, pd.Series):
        return DataVec(result)
    if isinstance(result, pd.DataFrame):
        if result.shape[1] == 1:
            series = result.iloc[:, 0].copy()
            series.name = result.columns[0]
            return DataVec(series)
        return DataMat(result)
    return result


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

    assert np.all(np.abs((A - eigvals[0] * B) @ eigvecs[:, 0]) < 1e-10)

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

    eigvals, M = generalized_eig(S21 @ S11.inv @ S12, S22)
    eigvals_alt, L = generalized_eig(S12 @ S22.inv @ S21, S11)

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

    C = utils.sqrtm(Y.cov())

    U, rho, Vt = ((C @ Y.T @ (Y.proj(X))) @ C).svd()
    V = Vt.T

    Br = X.lstsq(Y @ V.iloc[:, :r]) @ V.iloc[:, :r].pinv

    return Br
