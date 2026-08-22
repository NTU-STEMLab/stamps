# -*- coding: utf-8 -*-
"""
Backward-compatibility shim — moved to :mod:`stamps.stamps.categorical.prior`.
"""
import warnings as _warnings

_warnings.warn(
    "Importing from 'stamps.stamps.stest.BMEcatPrior' is deprecated. "
    "Use 'stamps.stamps.categorical.prior' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from ..categorical.prior import *  # noqa: F401,F403
from ..categorical.prior import (  # noqa: F401 – explicit names for IDEs
    BMEcatPrior,
    BMEcatPriorLOOCV,
    estimate_prior_xgboostlss,
    estimate_prior_xgboost,
    estimate_prior_uniform,
    estimate_prior,
    SpotpyHyperparameterSetup,
)
