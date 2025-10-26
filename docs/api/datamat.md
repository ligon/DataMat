# DataMat API

`DataMat` extends `pandas.DataFrame` with matrix-oriented operations that retain labels.

## Construction

```python
DataMat(data, *, idxnames=None, colnames=None, name=None)
```

- **data**: array-like or DataFrame.
- **idxnames** / **colnames**: optional index/column level names.
- **name**: optional matrix name.

## Selected Methods

- `matmul(other, strict=False, fillmiss=False)`: labelled matrix product (`@`).
- `kron(other, sparse=False)`: Kronecker product.
- `triu(k=0)` / `tril(k=0)`: triangular parts with metadata.
- `vec`: property returning column-stacked `DataVec` (vec operator).
- `random(shape, distribution="normal", ...)`: classmethod for random matrices.
- `inv`, `pinv`, `sqrtm`, `cholesky`: familiar matrix decompositions.
- `svd(hermitian=False)` / `eig(...)`: singular/eigen decompositions.
- `concat(...)`: labelled concatenation of vectors/matrices.

Full details are provided in the docstrings; the class honours pandas broadcasting rules while keeping multi-index metadata aligned.
