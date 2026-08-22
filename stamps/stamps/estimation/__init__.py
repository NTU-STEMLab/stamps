# -*- coding: utf-8 -*-
"""
stamps.estimation
==================
Spatial estimation methods (non-categorical): kriging, IDW, kernel smoothing,
regression, local-mean BME, and spatio-temporal mean.

Formerly ``stamps.stest``.
"""
from .kriging import (  # noqa: F401
    kriging,
    krigingFac,
    cokriging,
    cokrigingT,
)

from . import idw  # noqa: F401
from . import stmean  # noqa: F401
from . import designmatrix  # noqa: F401
from . import kernelsmoothing  # noqa: F401
from . import kernelsmoothing_truncate  # noqa: F401
from . import regression  # noqa: F401
from . import localmeanBME  # noqa: F401

__all__ = [
    "kriging",
    "krigingFac",
    "cokriging",
    "cokrigingT",
    "idw",
    "stmean",
    "designmatrix",
    "kernelsmoothing",
    "kernelsmoothing_truncate",
    "regression",
    "localmeanBME",
]
