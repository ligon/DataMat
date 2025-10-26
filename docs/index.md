# DataMat Documentation

DataMat provides `DataVec` and `DataMat` objects—thin wrappers around pandas Series/DataFrames—that preserve labels while supporting linear-algebra operations. This documentation covers installation, quick-start examples, API reference, and design notes collected during development.

```python
import datamat as dm

A = dm.DataMat([[1, 2], [3, 4]], idxnames=["i"], colnames=["j"])
b = dm.DataVec([1, 1], idxnames=["j"], name="weights")

result = A @ b
print(result)
```

Use the navigation on the left to explore guides, API details, and background notes.
