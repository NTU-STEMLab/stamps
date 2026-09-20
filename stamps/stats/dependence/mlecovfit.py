# -*- coding: utf-8 -*-
"""Backward-compatibility shim — moved to stamps.stats.dependence.covariance.mlecovfit."""
import warnings as _warnings
_warnings.warn(
    "Importing from 'stamps.stats.dependence.mlecovfit' is deprecated. Use 'stamps.stats.dependence.covariance.mlecovfit' instead.",
    DeprecationWarning, stacklevel=2,
)
from .covariance.mlecovfit import *  # noqa: F401,F403
