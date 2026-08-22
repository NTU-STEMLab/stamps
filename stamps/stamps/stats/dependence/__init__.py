# -*- coding: utf-8 -*-
"""
stamps.stats.dependence
========================
Spatial dependence modeling: covariance and probability-table methods.

Subpackages
-----------
covariance/     Continuous covariance estimation and fitting
ptable/         Categorical probability-table computation and fitting

This ``__init__`` re-exports all symbols from the new subpackages so that
existing ``from stamps.stamps.stats.dependence import ...`` statements
continue to work unchanged.
"""
from __future__ import division

# ------------------------------------------------------------------
# Covariance subpackage (continuous spatial dependence)
# ------------------------------------------------------------------
from .covariance import (  # noqa: F401
    stcov,
    stcov_kdtree,
    cov_avg_nd,
    stcov_split,
    stcov_dask,
    stcov_dask_half,
    covmodelwls,
    covmodelfit,
    covmodeldef,
    covmodelest,
    anisocovmodelest,
    covdownscale,
    coregfit,
    mlecovfitv,
    mlecovfitg,
    mlecovfit_sub,
    polarstcovmap,
    iso2aniso,
    aniso2iso,
    covariancemap,
    estimate_anisotropy_params,
    isotropy_test,
)

# ------------------------------------------------------------------
# Probability-table subpackage (categorical dependence)
# ------------------------------------------------------------------
from .ptable import (  # noqa: F401
    normalize_indicators,
    probatablecalc,
    probatablecalc_directional,
    probatablecalc_optimized,
    adaptive_bins,
    pair_distance_diagnostic,
    probatablecalc_spatial,
    probamodel2bitable,
    probamodel2bitable_separable,
    probatablefit,
    build_pmodel,
    spline_fit_pmodel,
    regularize_pmodel,
    alpha_empirical_bayes,
    loocv_tune_pmodel,
    polar_ptable_map,
    estimate_anisotropy_from_ptables,
    isotropy_test_categorical,
    ipf_recenter,
    recenter_pmodel,
)

# ------------------------------------------------------------------
# Nonstationary subpackage (regions, GW-Pmodel)
# ------------------------------------------------------------------
from .ptable.nonstationary import (  # noqa: F401
    identify_homogeneous_regions,
    fit_regional_pmodels,
    fit_covariate_pmodel,
    fit_covariate_interaction_pmodel,
    GWPmodel,
    BMEcatPdf_GW,
    MCPcatPdf_GW,
    loocv_gw,
)

# ------------------------------------------------------------------
# Legacy: keep old direct-module imports working
# ------------------------------------------------------------------
from .ptable.nonstationary.regions import (  # noqa: F401
    identify_homogeneous_regions as _identify_homogeneous_regions_pmodel,
)

# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------
__all__ = [
    # covariance
    "stcov", "stcov_kdtree", "cov_avg_nd", "stcov_split",
    "stcov_dask", "stcov_dask_half",
    "covmodelwls", "covmodelfit", "covmodeldef", "covmodelest",
    "anisocovmodelest", "covdownscale", "coregfit",
    "mlecovfitv", "mlecovfitg", "mlecovfit_sub",
    "polarstcovmap", "iso2aniso", "aniso2iso", "covariancemap",
    "estimate_anisotropy_params", "isotropy_test",
    # ptable
    "normalize_indicators", "probatablecalc", "probatablecalc_directional",
    "probatablecalc_optimized", "adaptive_bins", "pair_distance_diagnostic",
    "probatablecalc_spatial",
    "probamodel2bitable", "probamodel2bitable_separable",
    "probatablefit", "build_pmodel", "spline_fit_pmodel",
    "regularize_pmodel", "alpha_empirical_bayes", "loocv_tune_pmodel",
    "polar_ptable_map", "estimate_anisotropy_from_ptables",
    "isotropy_test_categorical",
    "ipf_recenter", "recenter_pmodel",
    # nonstationary
    "identify_homogeneous_regions",
    "fit_regional_pmodels", "fit_covariate_pmodel",
    "fit_covariate_interaction_pmodel",
    "GWPmodel", "BMEcatPdf_GW", "MCPcatPdf_GW", "loocv_gw",
]
