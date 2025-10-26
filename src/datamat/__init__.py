"""Matrix Operations on Data Arrays."""

from .core import (
    DataMat,
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
from . import utils

__all__ = [
    "DataMat",
    "DataVec",
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

__version__ = "0.1.0"
