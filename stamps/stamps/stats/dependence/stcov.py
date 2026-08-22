# -*- coding: utf-8 -*-
"""Backward-compatibility shim — moved to stamps.stats.dependence.covariance.stcov."""
import warnings as _warnings
_warnings.warn(
    "Importing from 'stamps.stats.dependence.stcov' is deprecated. Use 'stamps.stats.dependence.covariance.stcov' instead.",
    DeprecationWarning, stacklevel=2,
)
from .covariance.stcov import *  # noqa: F401,F403
