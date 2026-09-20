# -*- coding: utf-8 -*-
"""
stamps.graph
============
Plotting utilities for stamps spatial estimation workflows.

Sub-modules
-----------
dataplot    General data scatter / histogram plots.
modelplot   Covariance model diagnostic plots.
pdfplot     PDF and probability density plots.
categorical Categorical spatial estimation figures (Pmodel, maps, LOOCV, …).
"""
from .dataplot import (  # noqa: F401
    histplot,
    colorplot,
    plot_ecdf,
    plot_grid_map,
    plot_exceedance_map,
    plot_iqr_map,
    plot_pdf_ribbon_transect,
)
from .modelplot import (  # noqa: F401
    modelplot,
    covmodel_plot,
    semivario_plot,
    lmcplot,
)
from .categorical import (  # noqa: F401
    plot_cross_section,
    plot_borehole_locations,
    plot_depth_slices,
    plot_pmodel_matrix,
    plot_pmodel_self_transitions,
    plot_xgb_prior,
    plot_estimation_maps,
    plot_posterior_maps,
    plot_loocv,
    plot_loocv_bar,
)

__all__ = [
    "histplot",
    "colorplot",
    "plot_ecdf",
    "plot_grid_map",
    "plot_exceedance_map",
    "plot_iqr_map",
    "plot_pdf_ribbon_transect",
    "modelplot",
    "covmodel_plot",
    "semivario_plot",
    "lmcplot",
    "plot_cross_section",
    "plot_borehole_locations",
    "plot_depth_slices",
    "plot_pmodel_matrix",
    "plot_pmodel_self_transitions",
    "plot_xgb_prior",
    "plot_estimation_maps",
    "plot_posterior_maps",
    "plot_loocv",
    "plot_loocv_bar",
]
