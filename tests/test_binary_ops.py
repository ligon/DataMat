import datamat as dm
import numpy as np
import pandas as pd


def test_matmul_matrix():
    numpy_A = np.array([[1, 2], [3, 4]])
    numpy_B = np.array([[1], [1]])

    pandas_A = pd.DataFrame([[1, 2], [3, 4]])
    pandas_B = pd.DataFrame([[1], [1]])

    datamat_A = dm.DataMat([[1, 2], [3, 4]])
    datamat_B = dm.DataMat([[1], [1]])

    for A, B in [
        (numpy_A, numpy_B),
        (pandas_A, pandas_B),
        (datamat_A, datamat_B),
    ]:
        result = A @ B
        assert isinstance(result, type(A))


def test_matmul_matvec():
    datamat_A = dm.DataMat([[1, 2], [3, 4]])
    datamat_b = dm.DataVec([1, 1])

    pandas_b = pd.Series([1, 1])

    for A, b in [(datamat_A, datamat_b), (datamat_A, pandas_b)]:
        result = A @ b
        assert isinstance(result, type(b))
