# -*- coding: utf-8 -*-
"""
stamps.general
==============
General-purpose spatial utility functions used across the stamps package.

Public API
----------
coord2dist         Pairwise Minkowski distance matrix.
neighbours         Nearest-neighbour search (original MATLAB port).
neighbours_kd      KD-tree-accelerated nearest-neighbour search.
neighbours_cat     Unified categorical-aware neighbour search with separable
                   axes and TV/MI ranking *(new in stamps v3)*.
local_marginal     Kernel-weighted local class marginal with Kish
                   regularization *(new in stamps v3)*.
read_geoeas        Parse a GeoEAS flat-file into (title, col_names, data).
"""
from .coord2dist import coord2dist  # noqa: F401
from .neighbours import (           # noqa: F401
    neighbours,
    neighbours_kd,
    neighbours_index_kd,
    neighbours_cat,
)
from .local_marginal import local_marginal  # noqa: F401
from .read_geoeas import read_geoeas  # noqa: F401

__all__ = [
    "coord2dist",
    "neighbours",
    "neighbours_kd",
    "neighbours_index_kd",
    "neighbours_cat",
    "local_marginal",
    "read_geoeas",
]
