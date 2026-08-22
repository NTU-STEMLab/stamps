# -*- coding: utf-8 -*-
"""Backward-compatibility shim — moved to stamps.stats.dependence.ptable.empirical."""
import warnings as _warnings
_warnings.warn(
    "Importing from 'stamps.stats.dependence.probatablecalc' is deprecated. Use 'stamps.stats.dependence.ptable.empirical' instead.",
    DeprecationWarning, stacklevel=2,
)
from .ptable.empirical import (  # noqa: F401
    normalize_indicators,
    probatablecalc,
    adaptive_bins,
    pair_distance_diagnostic,
    probatablecalc_directional,
    probatablecalc_optimized,
)
