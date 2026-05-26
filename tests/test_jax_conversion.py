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


def test_jit_caches_across_two_datamatjax_instances_with_same_labels():
    """Previously DataMatJax stored pd.Index objects directly in the
    pytree aux data. pd.Index is unhashable and ``Index.__eq__`` returns
    an array, so JAX raised

      ValueError: Exception raised while checking equality of metadata
                  fields of pytree.

    as soon as a JIT'd function ran on two DataMatJax instances. The
    labels are now collapsed to (tuple-of-tuples, tuple-of-names) at
    flatten time and rebuilt on the way out, keeping JAX's cache
    comparison machinery happy.
    """
    from datamat import DataMat

    A_1 = DataMat(np.eye(2), idxnames="i", colnames="j").to_jax()
    A_2 = DataMat(np.eye(2) * 2.0, idxnames="i", colnames="j").to_jax()

    @jax.jit
    def total(A):
        return jnp.sum(A.values)

    # The previous implementation would raise here on the second call.
    assert float(total(A_1)) == 2.0
    assert float(total(A_2)) == 4.0


def test_jit_works_across_two_instances_with_different_label_shapes():
    """Recompilation under JIT should kick in (different aux → different
    cache key) but must *not* error."""
    from datamat import DataMat

    A_2x2 = DataMat(np.eye(2), idxnames="i", colnames="j").to_jax()
    A_3x3 = DataMat(np.eye(3), idxnames="i", colnames="j").to_jax()

    @jax.jit
    def total(A):
        return jnp.sum(A.values)

    assert float(total(A_2x2)) == 2.0
    assert float(total(A_3x3)) == 3.0


def test_grad_wrt_datamatjax_returns_labelled_wrapper():
    """``jax.grad`` differentiating w.r.t. a DataMatJax must yield a
    DataMatJax with the same labels as the input."""
    from datamat import DataMat, DataMatJax

    A = DataMat(
        np.array([[3.0, 1.0], [1.0, 2.0]]),
        index=pd.Index(["r0", "r1"], name="row"),
        columns=pd.Index(["c0", "c1"], name="col"),
    )
    A_jax = A.to_jax()

    def f(A_w):
        # ∂/∂A_w of  trace(A_w @ A_w)  =  2 * A_w^T
        return jnp.trace(A_w.values @ A_w.values)

    g = jax.grad(f)(A_jax)

    assert isinstance(g, DataMatJax)
    assert list(g.index.names) == ["row"]
    assert list(g.columns.names) == ["col"]
    # The trace gradient should be 2 * A^T = 2 * A (for symmetric A).
    np.testing.assert_allclose(np.asarray(g.values), 2 * A.values, atol=1e-5)


def test_jit_step_in_optimisation_loop_across_multiple_datamatjax():
    """Realistic scenario: a jit-compiled optimisation step called with
    two different DataMatJax instances (e.g. swapping out the operator
    between phases of an algorithm)."""
    from datamat import DataMat

    A_1 = DataMat(
        np.array([[3.0, 1.0], [1.0, 2.0]]), idxnames="i", colnames="j"
    ).to_jax()
    A_2 = DataMat(np.eye(2) * 4.0, idxnames="i", colnames="j").to_jax()

    @jax.jit
    def step(x, A):
        grad = jax.grad(lambda x: -jnp.sum(x * (A.values @ x)))(x)
        return x - 0.05 * grad

    x = jnp.array([0.5, 0.5])
    x = step(x, A_1)
    x = step(x, A_2)
    assert x.shape == (2,)
