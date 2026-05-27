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


# ---------------------------------------------------------------------------
# DataVecJax
# ---------------------------------------------------------------------------


def test_datavec_to_jax_round_trip_preserves_labels_and_name():
    from datamat import DataVec, DataVecJax

    v = DataVec([1.0, 2.0, 3.0], index=["a", "b", "c"], idxnames="i", name="x")

    wrapper = v.to_jax()
    assert isinstance(wrapper, DataVecJax)
    np.testing.assert_allclose(np.asarray(wrapper.values), v.values)
    assert list(wrapper.index.names) == ["i"]
    assert wrapper.name == "x"

    v_back = DataVec.from_jax(wrapper)
    assert isinstance(v_back, DataVec)
    assert v_back.name == "x"
    assert list(v_back.index.names) == ["i"]
    np.testing.assert_allclose(np.asarray(v_back.values), v.values)


def test_grad_wrt_datavecjax_returns_labelled_wrapper():
    from datamat import DataVec, DataVecJax

    v = DataVec([1.0, 2.0, 3.0], idxnames="i", name="x").to_jax()

    def loss(v_jax):
        return jnp.sum(v_jax.values**2)

    g = jax.grad(loss)(v)

    assert isinstance(g, DataVecJax)
    np.testing.assert_allclose(np.asarray(g.values), [2.0, 4.0, 6.0])
    assert list(g.index.names) == ["i"]
    assert g.name == "x"


def test_jit_caches_across_datavecjax_instances_with_same_labels():
    from datamat import DataVec

    v1 = DataVec([1.0, 2.0], idxnames="i", name="a").to_jax()
    v2 = DataVec([3.0, 4.0], idxnames="i", name="b").to_jax()

    @jax.jit
    def total(v):
        return jnp.sum(v.values)

    # Both calls must succeed even though the wrappers are distinct
    # objects with different ``.name`` aux entries.
    assert float(total(v1)) == 3.0
    assert float(total(v2)) == 7.0


def test_datavecjax_rejects_length_mismatch():
    from datamat import DataVecJax

    with pytest.raises(ValueError, match="values length 2"):
        DataVecJax(
            values=jnp.array([1.0, 2.0]),
            index=pd.Index(["a", "b", "c"], name="i"),
        )


def test_datavecjax_rejects_non_1d_values():
    from datamat import DataVecJax

    with pytest.raises(ValueError, match="must be 1-D"):
        DataVecJax(
            values=jnp.eye(3),
            index=pd.Index(["a", "b", "c"], name="i"),
        )


# ---------------------------------------------------------------------------
# Labelled operators on the JAX wrappers
# ---------------------------------------------------------------------------


def _labelled_setup():
    from datamat import DataMat, DataVec

    A = DataMat(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        index=pd.Index(["r0", "r1"], name="i"),
        columns=pd.Index(["c0", "c1", "c2"], name="j"),
    ).to_jax()
    v = DataVec(
        [10.0, 20.0, 30.0],
        index=pd.Index(["c0", "c1", "c2"], name="j"),
        name="v",
    ).to_jax()
    return A, v


def test_datamatjax_matmul_datavecjax_propagates_row_axis_and_vector_name():
    from datamat import DataVecJax

    A, v = _labelled_setup()
    r = A @ v

    assert isinstance(r, DataVecJax)
    np.testing.assert_allclose(np.asarray(r.values), [140.0, 320.0])
    assert list(r.index.names) == ["i"]
    assert [t[0] for t in r.index.tolist()] == ["r0", "r1"]
    assert r.name == "v"


def test_datamatjax_matmul_datamatjax_propagates_left_rows_and_right_columns():
    from datamat import DataMat, DataMatJax

    A, _ = _labelled_setup()
    B = DataMat(
        [[1.0, 0.0], [0.0, 1.0], [2.0, 3.0]],
        index=pd.Index(["c0", "c1", "c2"], name="j"),
        columns=pd.Index(["k0", "k1"], name="k"),
    ).to_jax()
    r = A @ B

    assert isinstance(r, DataMatJax)
    assert list(r.index.names) == ["i"]
    assert list(r.columns.names) == ["k"]
    assert r.values.shape == (2, 2)


def test_datavecjax_matmul_datavecjax_returns_scalar():
    from datamat import DataVec

    a, b = (
        DataVec(
            [1.0, 2.0, 3.0],
            index=pd.Index(["c0", "c1", "c2"], name="j"),
            name="a",
        ).to_jax(),
        DataVec(
            [10.0, 20.0, 30.0],
            index=pd.Index(["c0", "c1", "c2"], name="j"),
            name="b",
        ).to_jax(),
    )

    s = a @ b
    # Dot product is a scalar (0-D JaxArray), not a DataVecJax.
    assert getattr(s, "shape", ()) == ()
    assert float(s) == 140.0


def test_datavecjax_matmul_datamatjax_keeps_columns_and_vector_name():
    """``v @ M`` is the transpose of ``M @ v``: contracted axis is the
    vector's index against the matrix's index (rows); the matrix's
    columns survive as the result's index; the vector's name propagates."""
    from datamat import DataMat, DataVec, DataVecJax

    A = DataMat(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        index=pd.Index(["r0", "r1", "r2"], name="i"),
        columns=pd.Index(["c0", "c1"], name="j"),
    ).to_jax()
    v = DataVec(
        [10.0, 20.0, 30.0],
        index=pd.Index(["r0", "r1", "r2"], name="i"),
        name="v",
    ).to_jax()

    r = v @ A

    assert isinstance(r, DataVecJax)
    assert list(r.index.names) == ["j"]
    assert [t[0] for t in r.index.tolist()] == ["c0", "c1"]
    assert r.name == "v"

    # Algebraic identity: v @ A == A.T @ v.
    transposed = A.T @ v
    np.testing.assert_allclose(np.asarray(r.values), np.asarray(transposed.values))


def test_datavecjax_matmul_datamatjax_rejects_mismatched_axis():
    """``v @ M`` requires ``v.index`` to match ``M.index`` (not ``M.columns``);
    a vector keyed to the columns is a common confusion that should raise."""
    from datamat import DataMat, DataVec

    A = DataMat(
        [[1.0, 2.0], [3.0, 4.0]],
        index=pd.Index(["r0", "r1"], name="i"),
        columns=pd.Index(["c0", "c1"], name="j"),
    ).to_jax()
    # v has axis "j" but v @ A wants v's axis to match A.index ("i").
    v_col_keyed = DataVec(
        [1.0, 2.0],
        index=pd.Index(["c0", "c1"], name="j"),
        name="v",
    ).to_jax()

    with pytest.raises(ValueError, match="DataVecJax @ DataMatJax"):
        v_col_keyed @ A


# ---------------------------------------------------------------------------
# Scalar arithmetic + elementwise +/-
# ---------------------------------------------------------------------------


def test_datamatjax_scalar_mul_preserves_labels():
    from datamat import DataMat

    A = DataMat(
        [[1.0, 2.0], [3.0, 4.0]],
        index=pd.Index(["r0", "r1"], name="i"),
        columns=pd.Index(["c0", "c1"], name="j"),
    ).to_jax()

    half_A = 0.5 * A
    A_half = A * 0.5

    np.testing.assert_allclose(np.asarray(half_A.values), 0.5 * np.asarray(A.values))
    np.testing.assert_allclose(np.asarray(A_half.values), 0.5 * np.asarray(A.values))
    assert list(half_A.index.names) == ["i"]
    assert list(A_half.columns.names) == ["j"]


def test_datamatjax_scalar_div_and_neg_preserve_labels():
    from datamat import DataMat

    A = DataMat([[2.0, 4.0]], idxnames="i", colnames="j").to_jax()

    halved = A / 2.0
    neg = -A
    reverse_div = 8.0 / A

    np.testing.assert_array_equal(np.asarray(halved.values), [[1.0, 2.0]])
    np.testing.assert_array_equal(np.asarray(neg.values), [[-2.0, -4.0]])
    np.testing.assert_array_equal(np.asarray(reverse_div.values), [[4.0, 2.0]])


def test_datamatjax_scalar_add_and_sub_work_in_both_directions():
    from datamat import DataMat

    A = DataMat([[1.0, 2.0]], idxnames="i", colnames="j").to_jax()

    np.testing.assert_array_equal(np.asarray((A + 1.0).values), [[2.0, 3.0]])
    np.testing.assert_array_equal(np.asarray((1.0 + A).values), [[2.0, 3.0]])
    np.testing.assert_array_equal(np.asarray((A - 0.5).values), [[0.5, 1.5]])
    np.testing.assert_array_equal(np.asarray((10.0 - A).values), [[9.0, 8.0]])


def test_datamatjax_elementwise_add_sub_with_matching_labels():
    from datamat import DataMat, DataMatJax

    A = DataMat(
        [[1.0, 2.0], [3.0, 4.0]],
        index=pd.Index(["r0", "r1"], name="i"),
        columns=pd.Index(["c0", "c1"], name="j"),
    ).to_jax()
    B = DataMat(
        [[10.0, 20.0], [30.0, 40.0]],
        index=pd.Index(["r0", "r1"], name="i"),
        columns=pd.Index(["c0", "c1"], name="j"),
    ).to_jax()

    s = A + B
    d = A - B

    assert isinstance(s, DataMatJax) and isinstance(d, DataMatJax)
    np.testing.assert_array_equal(np.asarray(s.values), [[11.0, 22.0], [33.0, 44.0]])
    np.testing.assert_array_equal(np.asarray(d.values), [[-9.0, -18.0], [-27.0, -36.0]])
    assert list(s.index.names) == ["i"]
    assert list(s.columns.names) == ["j"]


def test_datamatjax_elementwise_add_rejects_mismatched_rows():
    from datamat import DataMat

    A = DataMat(
        [[1.0, 2.0]],
        index=pd.Index(["r0"], name="i"),
        columns=pd.Index(["c0", "c1"], name="j"),
    ).to_jax()
    B = DataMat(
        [[10.0, 20.0]],
        index=pd.Index(["different"], name="i"),
        columns=pd.Index(["c0", "c1"], name="j"),
    ).to_jax()

    with pytest.raises(ValueError, match="DataMatJax \\+ DataMatJax \\(row axis\\)"):
        A + B


def test_datamatjax_elementwise_add_rejects_mismatched_columns():
    from datamat import DataMat

    A = DataMat(
        [[1.0, 2.0]],
        index=pd.Index(["r0"], name="i"),
        columns=pd.Index(["c0", "c1"], name="j"),
    ).to_jax()
    B = DataMat(
        [[10.0, 20.0]],
        index=pd.Index(["r0"], name="i"),
        columns=pd.Index(["c0", "c1"], name="OTHER"),
    ).to_jax()

    with pytest.raises(ValueError, match="DataMatJax \\+ DataMatJax \\(column axis\\)"):
        A + B


def test_datamatjax_elementwise_multiplication_is_forbidden():
    from datamat import DataMat

    A = DataMat([[1.0]], idxnames="i", colnames="j").to_jax()
    B = DataMat([[2.0]], idxnames="i", colnames="j").to_jax()

    with pytest.raises(TypeError, match="Element-wise multiplication"):
        A * B


# Same battery for DataVecJax
def test_datavecjax_scalar_arithmetic_preserves_labels():
    from datamat import DataVec

    v = DataVec(
        [1.0, 2.0, 3.0],
        index=pd.Index(["a", "b", "c"], name="i"),
        name="v",
    ).to_jax()

    for w in (2.0 * v, v * 2.0, v / 0.5):
        np.testing.assert_allclose(np.asarray(w.values), [2.0, 4.0, 6.0])
        assert w.name == "v"
        assert list(w.index.names) == ["i"]

    np.testing.assert_array_equal(np.asarray((-v).values), [-1.0, -2.0, -3.0])
    np.testing.assert_array_equal(np.asarray((v + 1.0).values), [2.0, 3.0, 4.0])
    np.testing.assert_array_equal(np.asarray((10.0 - v).values), [9.0, 8.0, 7.0])


def test_datavecjax_elementwise_add_sub_with_matching_index():
    from datamat import DataVec, DataVecJax

    u = DataVec([1.0, 2.0], idxnames="i", name="u").to_jax()
    w = DataVec([10.0, 20.0], idxnames="i", name="w").to_jax()

    s = u + w
    d = u - w

    assert isinstance(s, DataVecJax) and isinstance(d, DataVecJax)
    np.testing.assert_array_equal(np.asarray(s.values), [11.0, 22.0])
    np.testing.assert_array_equal(np.asarray(d.values), [-9.0, -18.0])
    # LHS name propagates.
    assert s.name == "u" and d.name == "u"


def test_datavecjax_elementwise_add_rejects_mismatched_index():
    from datamat import DataVec

    u = DataVec([1.0, 2.0], idxnames="i", name="u").to_jax()
    w = DataVec([10.0, 20.0], idxnames="k", name="w").to_jax()

    with pytest.raises(ValueError, match="DataVecJax \\+ DataVecJax"):
        u + w


def test_arithmetic_works_inside_jit_and_grad():
    """End-to-end: a label-checked gradient step for ‖A x - b‖² written
    entirely in labelled-operator form (matmul, +, -, * by lr)."""
    from datamat import DataMat, DataVec, DataVecJax

    A = DataMat(
        [[1.0, 2.0], [3.0, 4.0]],
        index=pd.Index(["r0", "r1"], name="i"),
        columns=pd.Index(["c0", "c1"], name="j"),
    ).to_jax()
    b = DataVec(
        [5.0, 11.0],
        index=pd.Index(["r0", "r1"], name="i"),
        name="b",
    ).to_jax()

    @jax.jit
    def step(x, A, b, lr):
        # Closed form for the gradient of ‖A x - b‖² is 2 A^T (A x - b).
        # Build it with labelled operators.
        residual = A @ x - b  # DataVecJax (index = i, name = x)
        grad = A.T @ residual  # DataVecJax (index = j, name = x)
        return x - lr * grad  # DataVecJax (index = j, name = x)

    x = DataVec(
        [0.0, 0.0],
        index=pd.Index(["c0", "c1"], name="j"),
        name="x",
    ).to_jax()
    for _ in range(2000):
        x = step(x, A, b, 0.02)

    assert isinstance(x, DataVecJax)
    assert x.name == "x"
    assert list(x.index.names) == ["j"]
    closed = np.linalg.solve(np.asarray(A.values), np.asarray(b.values))
    # Convergence is slow for this ill-conditioned 2x2 system under naive
    # gradient descent; allow a relatively loose tolerance.
    np.testing.assert_allclose(np.asarray(x.values), closed, atol=0.1)


def test_datamatjax_transpose_swaps_axes_and_labels():
    from datamat import DataMatJax

    A, _ = _labelled_setup()
    At = A.T

    assert isinstance(At, DataMatJax)
    assert At.values.shape == (3, 2)
    assert list(At.index.names) == ["j"]
    assert list(At.columns.names) == ["i"]
    np.testing.assert_array_equal(np.asarray(At.values), np.asarray(A.values).T)


def test_datamatjax_matmul_rejects_mismatched_axis_labels():
    from datamat import DataMat, DataVec

    A = DataMat(
        [[1.0, 2.0]],
        index=pd.Index(["r"], name="i"),
        columns=pd.Index(["c0", "c1"], name="j"),
    ).to_jax()
    # Same length, different level name and values on the contracted axis.
    v_wrong = DataVec(
        [10.0, 20.0],
        index=pd.Index(["X", "Y"], name="k"),
        name="v",
    ).to_jax()

    with pytest.raises(ValueError, match="contracted-axis labels do not match"):
        A @ v_wrong


def test_datamatjax_matmul_rejects_same_values_different_level_names():
    """Strict means *names* must also match — not just values."""
    from datamat import DataMat, DataVec

    A = DataMat(
        [[1.0, 2.0]],
        index=pd.Index(["r"], name="i"),
        columns=pd.Index(["c0", "c1"], name="j"),
    ).to_jax()
    v_renamed = DataVec(
        [10.0, 20.0],
        index=pd.Index(["c0", "c1"], name="other_name"),
        name="v",
    ).to_jax()

    with pytest.raises(ValueError, match="contracted-axis labels do not match"):
        A @ v_renamed


def test_datavecjax_matmul_rejects_mismatched_indices():
    from datamat import DataVec

    a = DataVec(
        [1.0, 2.0],
        index=pd.Index(["x", "y"], name="i"),
        name="a",
    ).to_jax()
    b = DataVec(
        [3.0, 4.0],
        index=pd.Index(["p", "q"], name="i"),
        name="b",
    ).to_jax()

    with pytest.raises(ValueError, match="DataVecJax @ DataVecJax"):
        a @ b


def test_labelled_operators_work_under_jit_and_grad():
    """The whole point of the wrappers: optimise ``‖A x - b‖²`` written
    in labelled form, jit-compiled and differentiated end-to-end."""
    from datamat import DataMat, DataVec, DataVecJax

    A = DataMat(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        index=pd.Index(["r0", "r1", "r2"], name="i"),
        columns=pd.Index(["c0", "c1"], name="j"),
    ).to_jax()
    b = DataVec(
        [5.0, 11.0, 17.0],
        index=pd.Index(["r0", "r1", "r2"], name="i"),
        name="b",
    ).to_jax()

    # x has the same labels as A.columns (j); b has the same as A.index (i).
    # Quadratic loss: ‖A x - b‖² via subtraction-free form
    # = (A x) · (A x) - 2 (A x) · b + b · b.
    @jax.jit
    def loss(x, A, b):
        Ax = A @ x
        return (Ax @ Ax) - 2.0 * (Ax @ b) + (b @ b)

    @jax.jit
    def step(x, A, b):
        grad_vals = jax.grad(
            lambda xv: loss(DataVecJax(values=xv, index=x.index, name=x.name), A, b)
        )(x.values)
        return DataVecJax(
            values=x.values - 0.01 * grad_vals,
            index=x.index,
            name=x.name,
        )

    x = DataVec(
        [0.0, 0.0],
        index=pd.Index(["c0", "c1"], name="j"),
        name="x",
    ).to_jax()
    for _ in range(500):
        x = step(x, A, b)

    # Closed-form LS solution.
    expected = np.linalg.lstsq(np.asarray(A.values), np.asarray(b.values), rcond=None)[
        0
    ]
    np.testing.assert_allclose(np.asarray(x.values), expected, atol=0.05)


def test_optimisation_loop_with_labelled_variable_and_operator():
    """End-to-end: jit-compile a gradient step that takes a labelled
    parameter vector and a labelled operator. This is the use case the
    DataVecJax + JIT-safe DataMatJax combination is designed for."""
    from datamat import DataMat, DataVec, DataVecJax

    A = DataMat(
        np.array([[3.0, 1.0], [1.0, 2.0]]),
        idxnames="i",
        colnames="i",
    ).to_jax()
    a = DataVec([1.5, -0.5], idxnames="i", name="a").to_jax()

    @jax.jit
    def step(x, A, a):
        # Quadratic: ‖A x - a‖²  →  ∂/∂x = 2 A^T (A x - a)
        grad = jax.grad(lambda x_arr: jnp.sum((A.values @ x_arr - a.values) ** 2))(
            x.values
        )
        return DataVecJax(
            values=x.values - 0.05 * grad,
            index=x.index,
            name=x.name,
        )

    x = DataVec([0.0, 0.0], idxnames="i", name="x").to_jax()
    for _ in range(100):
        x = step(x, A, a)

    # Optimum: x* = A⁻¹ a ≈ [0.8, -0.65]
    np.testing.assert_allclose(
        np.asarray(x.values),
        np.linalg.solve(np.asarray(A.values), np.asarray(a.values)),
        atol=0.01,
    )
    assert x.name == "x"
    assert list(x.index.names) == ["i"]
