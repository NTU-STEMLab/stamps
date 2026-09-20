# -*- coding: utf-8 -*-
"""Backward-compatibility shim — moved to stamps.stats.dependence.ptable.empirical."""
import warnings as _warnings
_warnings.warn(
    "Importing from 'stamps.stats.dependence.probatablecalc_optimized' is deprecated. Use 'stamps.stats.dependence.ptable.empirical' instead.",
    DeprecationWarning, stacklevel=2,
)
from .ptable.empirical import probatablecalc_optimized  # noqa: F401
