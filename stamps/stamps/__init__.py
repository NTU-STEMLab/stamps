# -*- coding: utf-8 -*-
"""
stamps
======
A spatiotemporal data analysis package.

The ``stats`` sub-package is organised into three focused groups:

    stamps.stats.dependence   Covariance / dependence modeling
    stamps.stats.analysis     Spatiotemporal analysis methods
    stamps.stats.simulation   Spatial random field generation

All primary symbols are re-exported here for direct access:
``import stamps; stamps.stcov(...)``

Additional specialised sub-packages (bme, stest, general, models, mvn, graph)
remain accessible via their own namespaces.
"""
from __future__ import division
import warnings

# Suppress rpy2 DeprecationWarning ("activate/deactivate are deprecated") that
# fires during module initialisation regardless of stacklevel.  rpy2 is an
# *optional* dependency used only by gam / dlnm / stl; its internal
# deprecations should not pollute user notebooks.
# The filter is applied at this outermost level because rpy2 uses a large
# stacklevel value that walks all the way up to this frame.
warnings.filterwarnings(
    "ignore",
    message=".*activate.*deprecated.*|.*deactivate.*deprecated.*",
    category=DeprecationWarning,
)

# ------------------------------------------------------------------
# dependence  (via stats.dependence)
# ------------------------------------------------------------------
from .stats.dependence import (  # noqa: F401
    # empirical covariance
    stcov,
    stcov_kdtree,
    cov_avg_nd,
    stcov_split,
    stcov_dask,
    stcov_dask_half,
    # model fitting – WLS
    covmodelwls,
    covmodelfit,
    covmodeldef,
    covmodelest,
    anisocovmodelest,
    covdownscale,
    # model fitting – MLE
    mlecovfitv,
    mlecovfitg,
    mlecovfit_sub,
    # anisotropy
    polarstcovmap,
    iso2aniso,
    aniso2iso,
    # probability tables
    probatablecalc,
    probatablecalc_optimized,
    probatablecalc_spatial,
    identify_homogeneous_regions,
    probamodel2bitable,
    probatablefit,
)

# ------------------------------------------------------------------
# analysis  (via stats.analysis)
# ------------------------------------------------------------------
from .stats.analysis import (  # noqa: F401
    # EOF / decomposition
    eof,
    eeof,
    sreof,
    varimax,
    meof,
    srmeof,
    HOSVD,
    unfold,
    fold,
    # PCA
    pca,
    srpca,
    stica,
    srica,
    # MCA / CCA
    mca,
    smca,
    cca,
    # entropy
    entropyD,
    entropyC,
    condentropy,
    # MaxEnt PDF
    maxentpdf_gh,
    maxentpdf_gcp,
    maxentpdf_gkhk,
    maxentcondpdf_gc,
    maxentpdf_gc,
    gkhk,
    # EKF
    getK,
    getXu,
    getPu,
)

# ------------------------------------------------------------------
# simulation  (via stats.simulation)
# ------------------------------------------------------------------
from .stats.simulation import (  # noqa: F401
    stationary_gaussian_process,
    simuchol,
    simucholcond,
    simucholcondME,
    anisosimuchol,
    simuseq,
    simuseqcond,
    simuseqcondME,
    simuseqcondInt,
    simuprobabilistic,
    simuinterval,
)

# ------------------------------------------------------------------
# Public API — what IDE autocomplete exposes under `import stamps`
# ------------------------------------------------------------------
__all__ = [
    # ── dependence ──────────────────────────────────────────────
    "stcov", "stcov_kdtree", "cov_avg_nd", "stcov_split",
    "stcov_dask", "stcov_dask_half",
    "covmodelwls", "covmodelfit", "covmodeldef", "covmodelest",
    "anisocovmodelest", "covdownscale",
    "mlecovfitv", "mlecovfitg", "mlecovfit_sub",
    "polarstcovmap", "iso2aniso", "aniso2iso",
    "probatablecalc", "probatablecalc_optimized",
    "probatablecalc_spatial", "identify_homogeneous_regions",
    "probamodel2bitable", "probatablefit",
    # ── analysis ────────────────────────────────────────────────
    "eof", "eeof", "sreof", "varimax", "meof", "srmeof",
    "HOSVD", "unfold", "fold",
    "pca", "srpca", "stica", "srica",
    "mca", "smca", "cca",
    "entropyD", "entropyC", "condentropy",
    "maxentpdf_gh", "maxentpdf_gcp", "maxentpdf_gkhk",
    "maxentcondpdf_gc", "maxentpdf_gc", "gkhk",
    "getK", "getXu", "getPu",
    # "gam", "construct_gam_formula", "construct_gam_model", "predict_gam",  # rpy2 optional
    # "crossbasis",   # rpy2 optional
    # "stl", "stl_plot",  # rpy2 optional
    # ── simulation ──────────────────────────────────────────────
    "stationary_gaussian_process",
    "simuchol", "simucholcond", "simucholcondME", "anisosimuchol",
    "simuseq", "simuseqcond", "simuseqcondME", "simuseqcondInt",
    "simuprobabilistic", "simuinterval",
]
