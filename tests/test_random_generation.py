import datamat as dm
import numpy as np
import pandas as pd


def test_datavec_random_defaults():
    v1 = dm.DataVec.random(4, rng=0, idxnames="i")
    v2 = dm.DataVec.random(4, rng=0, idxnames="i")

    assert isinstance(v1, dm.DataVec)
    assert v1.shape == (4,)
    assert v1.index.names == ["i"]
    np.testing.assert_array_equal(v1.values, v2.values)


def test_datamat_random_uniform_with_labels():
    rows = pd.Index(["r1", "r2"], name="row")
    cols = pd.Index(["c1", "c2", "c3"], name="col")
    A = dm.DataMat.random(
        (2, 3), distribution="uniform", index=rows, columns=cols, rng=1
    )

    assert isinstance(A, dm.DataMat)
    assert A.shape == (2, 3)
    assert list(A.index.get_level_values(0)) == list(rows)
    assert A.index.names == [rows.name]
    assert list(A.columns.get_level_values(0)) == list(cols)
    assert A.columns.names == [cols.name]
    assert np.all((A.values >= 0.0) & (A.values <= 1.0))


def test_custom_callable_distribution():
    def constant(size, **kwargs):
        return np.full(size, 7.0)

    mat = dm.DataMat.random((3, 2), distribution=constant)
    assert np.all(mat.values == 7.0)


def test_chi_square_tuple_spec():
    rng = 42
    vec = dm.DataVec.random(5, distribution=("chi2", 3), rng=rng)
    assert np.all(vec.values >= 0.0)


def test_bernoulli_parameter():
    vec = dm.DataVec.random(10, distribution=("bernoulli", 0.8), rng=10)
    assert set(vec.values).issubset({0, 1})
