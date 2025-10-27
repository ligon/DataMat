from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from datamat import DataMat, DataMatJax

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


def _make_sample_datamat(seed: int = 0) -> DataMat:
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(4, 3))
    index = pd.MultiIndex.from_product(
        [pd.Index([0, 1], name="i"), pd.Index(["a", "b"], name="obs")]
    )
    columns = pd.MultiIndex.from_product([pd.Index(["x", "y", "z"], name="var")])
    return DataMat(data, index=index, columns=columns)


def test_to_jax_round_trip_preserves_structure():
    matrix = _make_sample_datamat()

    wrapper = matrix.to_jax()
    assert isinstance(wrapper, DataMatJax)
    assert wrapper.index.equals(matrix.index)
    assert wrapper.columns.equals(matrix.columns)

    np.testing.assert_allclose(np.asarray(wrapper.values), matrix.to_numpy())

    leaves, _ = jax.tree_util.tree_flatten(wrapper)
    assert len(leaves) == 1
    np.testing.assert_allclose(np.asarray(leaves[0]), matrix.to_numpy())

    round_trip = wrapper.to_datamat()
    np.testing.assert_allclose(
        round_trip.to_numpy(dtype=float), matrix.to_numpy(dtype=float)
    )

    round_trip_classmethod = DataMat.from_jax(wrapper)
    np.testing.assert_allclose(
        round_trip_classmethod.to_numpy(dtype=float), matrix.to_numpy(dtype=float)
    )


def test_to_jax_allows_dtype_override():
    matrix = _make_sample_datamat()

    wrapper = matrix.to_jax(dtype=None)
    expected = matrix.to_numpy().dtype
    if wrapper.values.dtype != expected:
        assert not jax.config.x64_enabled
        assert wrapper.values.dtype == jnp.float32
    else:
        assert wrapper.values.dtype == expected

    float_wrapper = matrix.to_jax(dtype=np.float32)
    assert float_wrapper.values.dtype == jnp.float32
