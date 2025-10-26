# Norm Conventions

`DataVec.norm` and `DataMat.norm` follow NumPy's conventions while preserving labels.

## Vector Norms (`DataVec.norm`)

- `ord=2` (default): Euclidean norm \(\|v\|_2 = \sqrt{\sum_i v_i^2}\).
- `ord=1`: Manhattan norm \(\sum_i |v_i|\).
- `ord=\infty`: maximum absolute entry.
- `ord=0`: count of non-zero elements.
- Negative orders and other values behave exactly like `numpy.linalg.norm`.

## Matrix Norms (`DataMat.norm`)

- `ord=None` or `'fro'`: Frobenius norm \(\|A\|_F = \sqrt{\sum_{i,j} a_{ij}^2}\).
- `ord=1`: induced column-sum norm \(\max_j \sum_i |a_{ij}|\).
- `ord=\infty`: induced row-sum norm \(\max_i \sum_j |a_{ij}|\).
- `ord=2`: spectral norm (largest singular value).
- `ord='nuc'`: nuclear norm (sum of singular values).
- Any other options are forwarded to `numpy.linalg.norm`.

Remember that calling these methods returns scalars; the underlying labels remain available on the original objects.
