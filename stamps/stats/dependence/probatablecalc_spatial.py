# -*- coding: utf-8 -*-
"""Backward-compatibility shim — moved to stamps.stats.dependence.ptable.spatial."""
import warnings as _warnings
_warnings.warn(
    "Importing from 'stamps.stats.dependence.probatablecalc_spatial' is deprecated. Use 'stamps.stats.dependence.ptable.spatial' instead.",
    DeprecationWarning, stacklevel=2,
)
from .ptable.spatial import (  # noqa: F401
    probatablecalc_spatial,
    identify_homogeneous_regions,
)
