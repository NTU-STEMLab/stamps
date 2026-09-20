# -*- coding: utf-8 -*-
"""
Backward-compatibility shim — moved to
:mod:`stamps.stats.dependence.ptable.nonstationary`.
"""
import warnings as _warnings

_warnings.warn(
    "Importing from 'stamps.stats.dependence.gw_pmodel' is deprecated. "
    "Use 'stamps.stats.dependence.ptable.nonstationary' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .ptable.nonstationary.gw_pmodel import (  # noqa: F401
    fit_regional_pmodels,
    GWPmodel,
    fit_covariate_pmodel,
)
from .ptable.nonstationary.gw_estimation import (  # noqa: F401
    BMEcatPdf_GW,
    MCPcatPdf_GW,
    loocv_gw,
)
