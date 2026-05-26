import datamat as dm
import pandas as pd
from datamat.core import DataMat, DataVec


def test_concat_dict_uses_keys_for_column_labels():
    a = DataVec([1, 2], name="a")
    b = a + 1

    result = dm.concat({"a": a, "b": b}, axis=1)

    assert list(result.columns.get_level_values(0)) == ["a", "b"]
    assert isinstance(result, DataMat)


def test_concat_dict_preserves_keys_when_levelnames_requested():
    a = DataVec([1, 2], name="a")
    b = a + 1

    result = dm.concat({"a": a, "b": b}, axis=1, levelnames=True)

    assert list(result.columns.get_level_values(0)) == ["a", "b"]
    assert list(result.columns.get_level_values(1)) == ["a", "a"]
    assert isinstance(result, DataMat)


def test_concat_accepts_pandas_series_horizontal():
    c = pd.Series([1, 2], name="c")
    d = c + 1

    result = dm.concat({"c": c, "d": d}, axis=1)

    assert isinstance(result, DataMat)
    assert list(result.columns.get_level_values(0)) == ["c", "d"]
    assert list(result.iloc[:, 0]) == [1, 2]
    assert list(result.iloc[:, 1]) == [2, 3]


def test_concat_list_assigns_suffix_for_duplicate_names():
    a = DataVec([1, 2], name="a")
    b = a + 1

    result = dm.concat([a, b], axis=1)

    assert isinstance(result, DataMat)
    assert list(result.columns.get_level_values(0)) == ["a", "a_0"]
    assert list(result.iloc[:, 0]) == [1, 2]
    assert list(result.iloc[:, 1]) == [2, 3]


def test_concat_accepts_pandas_series_vertical():
    c = pd.Series([1, 2], name="c")
    d = c + 1

    result = dm.concat({"c": c, "d": d}, axis=0)

    assert isinstance(result, DataVec)
    assert list(result.index.get_level_values(0)) == [0, 1, 0, 1]
    assert list(result.values) == [1, 2, 2, 3]
    assert result.name == ("c",)


def test_concat_drop_vestigial_levels_parity_module_vs_method():
    """``dm.concat(...)`` and ``A.concat(B, ...)`` must drop vestigial column
    levels identically when ``drop_vestigial_levels=True``.

    Previously the module-level wrapper omitted the flag from its column
    reconcile call (core.py:1164), so vestigial column levels survived the
    module-level path while being dropped by the method-level one.
    """
    idx = pd.MultiIndex.from_tuples([("a", 0), ("a", 1)], names=["ves", "i"])
    col = pd.MultiIndex.from_tuples([("ves_c", "x"), ("ves_c", "y")], names=["vc", "j"])
    A = dm.DataMat([[1, 2], [3, 4]], index=idx, columns=col)
    B = dm.DataMat([[5, 6], [7, 8]], index=idx, columns=col)

    out_mod = dm.concat([A, B], axis=0, drop_vestigial_levels=True)
    out_meth = A.concat(B, axis=0, drop_vestigial_levels=True)

    assert list(out_mod.columns.names) == list(out_meth.columns.names)
    assert out_mod.columns.tolist() == out_meth.columns.tolist()
    # And the vestigial column level should genuinely be gone.
    assert "vc" not in out_mod.columns.names
