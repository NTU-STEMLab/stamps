# -*- coding: utf-8 -*-
"""
stamps.stats.dependence.ptable.nonstationary
==============================================
Nonstationary probability-table models: homogeneous regions, GW-Pmodel.
"""
from .regions import identify_homogeneous_regions  # noqa: F401
from .gw_pmodel import (  # noqa: F401
    fit_regional_pmodels,
    GWPmodel,
    fit_covariate_pmodel,
    fit_covariate_interaction_pmodel,
)
from .gw_estimation import (  # noqa: F401
    BMEcatPdf_GW,
    MCPcatPdf_GW,
    loocv_gw,
)
