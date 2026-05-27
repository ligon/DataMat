---
name: jax-pandas-bridge
description: Use this skill when wrapping a pandas-labelled tensor type (DataFrame, Series, or a subclass) for JAX autodiff and JIT — covers PyTree registration, hashable label representations, abstract-tracer-safe validation, and strict-by-default labelled operators with bool-safe equality.
license: Apache-2.0
---

# JAX–pandas bridge: labelled tensors for autodiff

## When to Use

Apply when:
- A user wants `jax.grad`, `jax.jit`, `jax.vmap`, or similar to flow through a labelled-tensor type whose labels are pandas Index / MultiIndex.
- Adding operators (matmul, transpose, scalar arithmetic, elementwise +/−) that preserve labels through traced computations.
- Diagnosing `ValueError: Exception raised while checking equality of metadata fields of pytree`, or a `jax.jit` that recompiles unexpectedly each call.
- Auditing an existing wrapper that holds `pd.Index` in its PyTree aux data — likely subtly broken for JIT even if `jax.grad` looks OK.

## Philosophy

The point of a labelled wrapper around a JAX array is to *pin down which axis means what*. Anything that defeats that — silent positional alignment, unhashable labels that break the JIT cache, operations that drop to raw `jnp` and lose the names — should be visible at design time, not at a debugging session three weeks later.

This skill captures a recipe for getting that right.

## The Six-Step Recipe

### 1. Use `@dataclass(frozen=True)` and register as a PyTree

```python
from dataclasses import dataclass
from jax import tree_util as jax_tree_util

@dataclass(frozen=True)
class LabelledArray:
    values: jnp.ndarray   # the PyTree LEAF (tracers OK)
    index: pd.Index       # static AUX (label metadata)
    # ...maybe more aux fields (columns, name, etc.)

    def tree_flatten(self): ...
    @classmethod
    def tree_unflatten(cls, metadata, children): ...

jax_tree_util.register_pytree_node(
    LabelledArray,
    lambda w: w.tree_flatten(),
    LabelledArray.tree_unflatten,
)
```

`frozen=True` makes the dataclass hashable on its own — but that's not what matters here. What matters is the **aux data** returned by `tree_flatten`.

### 2. Make aux data hashable with bool ==

JAX's pytree-cache machinery compares aux objects using Python `hash()` and `==`. Both must work cleanly, returning a hash and a bool respectively. **`pd.Index` fails both**: it's unhashable (raises `TypeError`) and its `__eq__` returns an element-wise array.

Symptom of getting this wrong:

```
ValueError: Exception raised while checking equality of metadata fields of pytree.
Make sure that metadata fields are hashable and have simple equality semantics.
```

Fires the moment a JIT'd function runs on two distinct wrapper instances — even if their labels are structurally identical. `jax.grad` may still appear to work (it doesn't hit the cache path the same way), masking the bug.

**Fix**: store labels as nested tuples in aux, rebuild the `pd.Index` on the way out.

```python
_IndexStatic = tuple[tuple[Any, ...], tuple[Any, ...]]

def _index_to_static(idx: pd.Index) -> _IndexStatic:
    if isinstance(idx, pd.MultiIndex):
        return tuple(map(tuple, idx.tolist())), tuple(idx.names)
    return tuple((v,) for v in idx.tolist()), (idx.name,)

def _static_to_index(meta: _IndexStatic) -> pd.MultiIndex:
    tuples, names = meta
    return pd.MultiIndex.from_tuples(list(tuples), names=list(names))


class LabelledArray:
    def tree_flatten(self):
        return (self.values,), (_index_to_static(self.index),)

    @classmethod
    def tree_unflatten(cls, metadata, children):
        (idx_meta,) = metadata
        (values,) = children
        return cls(values=values, index=_static_to_index(idx_meta))
```

`tuple_of_tuples` and `tuple_of_names` are both fully hashable. The `MultiIndex` reconstruction cost is paid only when JAX flattens/unflattens — at trace boundaries, not in every traced operation.

### 3. Validate in `__post_init__` but skip for abstract tracers

```python
def __post_init__(self):
    shape = getattr(self.values, "shape", None)
    if shape is None:
        return  # not array-like; let downstream code complain
    if len(shape) != 1:                        # for a vector type
        raise ValueError(f"values must be 1-D; got shape {shape}.")
    if shape[0] != len(self.index):
        raise ValueError(
            f"values length {shape[0]} != index length {len(self.index)}."
        )
```

Two subtleties:
- **`frozen=True` does not block `__post_init__`** — it runs during construction, before the freeze takes effect.
- **Don't validate when `.shape` is unavailable**. Abstract tracers under JIT may pass through `tree_unflatten` without a concrete shape; the validation should be a no-op in that case. `getattr(values, "shape", None)` is the safe probe.

### 4. Use `Index.equals(other)` for boolean equality, never `==`

The cardinal mistake. For host-side label-alignment checks inside operators:

```python
def _check_axis_alignment(left: pd.Index, right: pd.Index, op_label: str):
    # `left == right` returns an array — bool() raises "truth value ambiguous".
    # `left.equals(right)` returns a bool and works regardless of nlevels.
    if list(left.names) != list(right.names) or not left.equals(right):
        raise ValueError(f"{op_label}: contracted-axis labels do not match.")
```

`.equals()` checks *values*; level *names* must be compared separately if name match is also required (it usually is for strict labelled semantics).

The same trap fires inside tuple equality: `(idx1,) == (idx2,)` invokes elementwise `__eq__` on `idx1`/`idx2` and breaks. Be careful if pd.Index objects ever end up inside container types compared with `==`.

### 5. Strict-by-default operators

The pandas convention for `DataMat @ DataVec` is **positional by default** — labels don't need to align; only positions do. The JAX-side wrapper should usually go the *other* way: **strict by default**.

Rationale: a user reaches for raw `jnp` when they just want positional math. They reach for a labelled wrapper specifically to pin down axis meaning. Silently aligning by position defeats the type.

The host-side label check costs nothing under JIT (the labels live in aux, never in tracers), so there's no performance argument for laxity. Mirror the pandas surface but make `strict=True` the default — provide an `align=` or `strict=False` opt-out only if a use case demands it.

### 6. Refuse operations whose semantics are ambiguous

For elementwise multiplication / division between two labelled wrappers, the meaning is ambiguous:
- Hadamard product (requires identical shapes and labels)?
- Broadcasting (along which axis)?
- Outer product?

Don't pick. Raise `TypeError` and tell the user to drop to `wrapper.values * other.values` for raw `jnp` semantics if they need that path. Be explicit in the error.

Scalar arithmetic, by contrast, is unambiguous and should be supported. Detect scalars by `shape == ()` (covers Python scalars, numpy scalars, 0-D JAX arrays, and abstract 0-D tracers) rather than by `isinstance` checks:

```python
def _is_jax_scalar(value) -> bool:
    return getattr(value, "shape", ()) == ()
```

## Conventions to Mirror From Pandas

These are the choices the pandas world has converged on; carrying them through to the labelled-JAX wrapper makes user code read consistently in both worlds.

### Matmul label propagation

| Operation | Surviving axes | Propagated name |
|---|---|---|
| `M @ v` (matrix-vector) | `M.index` becomes result's `index` | `v.name` becomes result's `name` |
| `v @ M` (vector-matrix) | `M.columns` becomes result's `index` | `v.name` becomes result's `name` |
| `M1 @ M2` (matrix-matrix) | `M1.index` + `M2.columns` | (no name) |
| `v1 @ v2` (dot product) | (none — scalar result) | (no name) |
| `M.T` | swap `index` ⇄ `columns`, transpose values | unchanged |

In all cases, the **contracted axis** must match strictly (values + names) — that's where the label check lives.

### Single-element-tuple name unwrap

`DataMat`'s columns are always wrapped to a `MultiIndex`, so a single-level 1-column matrix yields tuple keys like `('c',)`. When a matmul collapses to a vector type, unwrap the 1-tuple:

```python
def _unwrap_scalar_name(name):
    if isinstance(name, tuple) and len(name) == 1:
        return name[0]
    return name
```

Leave 2-tuples (genuine multi-level keys) alone.

## Anti-patterns to Catch in Review

1. **`pd.Index` directly in tree_flatten's aux output** — see Step 2.
2. **`if idx == other:` or `bool(idx == other)`** anywhere in host-side code — see Step 4.
3. **`isinstance(x, (jnp.ndarray, np.ndarray, float, int))` to detect scalars** — misses 0-D tracers and numpy scalars; use the `shape == ()` test in Step 6.
4. **`jnp.asarray(arr, dtype=arr.dtype)`** — redundant dtype kwarg triggers JAX's truncation warning under default x32. Drop the kwarg.
5. **Operator returning raw `jnp.ndarray` instead of the wrapper type** — silently loses labels mid-pipeline.
6. **Mutating `self.values` in an operator** — incompatible with `@dataclass(frozen=True)`; the operator must return a *new* wrapper instance via the constructor.
7. **Calling `wrapper.to_pandas()` (or equivalent) inside a traced region** — pandas conversion materialises the tracer to numpy, breaking autodiff. Restrict round-trips to the boundary.

## Quick-Start Checklist

When asked to JAX-wrap a labelled type, run through:

1. **PyTree registered?** `@dataclass(frozen=True)` + `register_pytree_node`.
2. **Aux hashable?** `pd.Index` → tuple representation in `tree_flatten`.
3. **`__post_init__` validates concrete shapes** but no-ops for abstract tracers.
4. **`Index.equals` (not `==`)** used in all host-side label checks.
5. **Strict label alignment** by default on all operators that mix two wrapper instances.
6. **Wrapper × wrapper Hadamard explicitly rejected** with a clear TypeError.
7. **Scalar arithmetic** preserves labels; detection via `shape == ()`.
8. **`jnp.asarray` calls** carry no redundant `dtype=` kwarg.
9. **Round-trip tests**: construction → flatten → unflatten → operator → grad.
10. **JIT cache test**: same function called on two structurally-identical wrappers must not error.

## Worked Example

The `DataMat` project at `https://github.com/ligon/DataMat` (post-2026-05 refactor) ships a complete worked example:

- `src/datamat/core.py` `DataMatJax` and `DataVecJax` classes (lookup `class DataMatJax` and `class DataVecJax`).
- Helpers `_index_to_static` / `_static_to_index` / `_check_axis_alignment` / `_is_jax_scalar` / `_unwrap_scalar_name`.
- `tests/test_jax_conversion.py` covers the full operator surface, label-mismatch raises, JIT cache across instances, and an end-to-end gradient-descent loop on `‖A x - b‖²` written entirely in labelled-operator form.

## Pitfalls

- **`jax.grad` may appear to work even with a broken wrapper.** It hits the pytree-cache equality path less often than `jax.jit`. Always test the JIT cache too.
- **`pd.Index.equals` ignores level names**. `pd.Index([1,2,3], name='a').equals(pd.Index([1,2,3], name='b'))` returns True. If level-name match matters (it usually does for strict semantics), compare `list(idx.names)` separately.
- **`from typing import Any` is not auto-imported by `from __future__ import annotations`.** `ruff F821` will catch this — make sure type annotations that mention `Any` have the explicit import even when annotations are stringified.
- **JAX's `to_pandas()`-style round trips lose tracers.** Document that the wrapper-to-pandas conversion is only safe outside a traced region.

## Related Skills

- `code-reviewer` for the broader review patterns; the anti-patterns list above is the JAX-specific addendum to the general checklist.

## Changelog

- **<2026-05-26 Mon>** — Initial version distilled from the DataMat refactor.
