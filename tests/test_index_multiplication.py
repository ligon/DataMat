import numpy as np
import pandas as pd

import datamat as dm


def test_index_multiplication():
    idx = pd.MultiIndex.from_tuples(
        [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1)],
        names=["i", "j", "k"],
    )
    X = dm.DataMat([[1, 2, 3, 4]], columns=idx, idxnames=["l"])
    Y = dm.DataMat(
        [[1, 2, 3, 0]],
        columns=idx.droplevel("j"),
        idxnames="m",
    ).T

    result = X @ Y
    assert result.index.names == ["l"]

    # strict multiplication verifies MultiIndex reconciliation path
    X.matmul(Y, strict=True)
