# DataVec API

`DataVec` extends `pandas.Series` with linear-algebra friendly behaviour.

## Construction

```python
DataVec(data=None, *, idxnames=None, name=None)
```

- **data**: array-like or Series/Vector.
- **idxnames**: optional name(s) for the index levels.
- **name**: optional vector name. If omitted a unique `vec_*` identifier is generated.

## Key Methods

- `dg(sparse=True)`: diagonal matrix `diag(v)`.
- `norm(ord=None, **kwargs)`: vector norm \(\|v\|_p\).
- `outer(other)`: outer product `v ⊗ other`.
- `proj(other)`: projection of `self` on `other`.
- `concat(...)`: concatenate with other vectors/matrices, preserving labels.
- `random(...)`: classmethod returning a labelled random vector; supports normal, uniform, chi-square, Bernoulli, Poisson and custom callables.

Refer to the Python docstrings for detailed parameter descriptions and examples. All operations preserve index metadata and return `DataVec`/`DataMat` objects.
