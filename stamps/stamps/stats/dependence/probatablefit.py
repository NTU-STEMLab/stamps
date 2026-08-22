# -*- coding: utf-8 -*-
"""Backward-compatibility shim — moved to stamps.stats.dependence.ptable.fit."""
import warnings as _warnings
_warnings.warn(
    "Importing from 'stamps.stats.dependence.probatablefit' is deprecated. Use 'stamps.stats.dependence.ptable.fit' instead.",
    DeprecationWarning, stacklevel=2,
)
from .ptable.fit import *  # noqa: F401,F403
