"""Matrix Operations on Data Arrays."""

from . import utils
from .core import (
    DataMat,
    DataMatJax,
    DataVec,
    canonical_variates,
    concat,
    generalized_eig,
    get_names,
    read_parquet,
    read_pickle,
    read_stata,
    reconcile_indices,
    reduced_rank_regression,
)

__all__ = [
    "DataMat",
    "DataVec",
    "DataMatJax",
    "canonical_variates",
    "concat",
    "generalized_eig",
    "get_names",
    "read_parquet",
    "read_pickle",
    "read_stata",
    "reconcile_indices",
    "reduced_rank_regression",
    "utils",
]

__version__ = "0.2.0a1"
