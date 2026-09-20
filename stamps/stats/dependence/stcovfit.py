# -*- coding: utf-8 -*-
"""Backward-compatibility shim — moved to stamps.stats.dependence.covariance.stcovfit."""
import warnings as _warnings
_warnings.warn(
    "Importing from 'stamps.stats.dependence.stcovfit' is deprecated. Use 'stamps.stats.dependence.covariance.stcovfit' instead.",
    DeprecationWarning, stacklevel=2,
)
from .covariance.stcovfit import *  # noqa: F401,F403
