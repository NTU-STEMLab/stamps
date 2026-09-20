# -*- coding: utf-8 -*-
"""
stamps.stats
============
Statistical and spatiotemporal analysis functions, organised into three
focused sub-packages:

    stamps.stats.dependence   Covariance & spatial dependence modeling
    stamps.stats.analysis     Spatiotemporal analysis methods
    stamps.stats.simulation   Spatial random field generation

All public symbols are re-exported here so that existing code using
``from stamps.stats import …`` continues to work unchanged.
"""
from __future__ import division
import warnings

# ------------------------------------------------------------------
# dependence
# ------------------------------------------------------------------
from .dependence import (  # noqa: F401
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
# analysis (rpy2-based sub-modules may raise DeprecationWarning on import)
# ------------------------------------------------------------------
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from .analysis import (  # noqa: F401
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
# simulation
# ------------------------------------------------------------------
from .simulation import (  # noqa: F401
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
# evaluate — categorical estimation metrics
# ------------------------------------------------------------------
from .evaluate import (  # noqa: F401
    confusion_matrix,
    plot_confusion_matrix,
    classification_report,
    balanced_accuracy,
    brier_score,
    mean_log_likelihood,
    loocv_categorical,
)

# ------------------------------------------------------------------
# Public API
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
    # ── simulation ──────────────────────────────────────────────
    "stationary_gaussian_process",
    "simuchol", "simucholcond", "simucholcondME", "anisosimuchol",
    "simuseq", "simuseqcond", "simuseqcondME", "simuseqcondInt",
    "simuprobabilistic", "simuinterval",
    # ── evaluate ────────────────────────────────────────────────
    "confusion_matrix",
    "plot_confusion_matrix",
    "classification_report",
    "balanced_accuracy",
    "brier_score",
    "mean_log_likelihood",
    "loocv_categorical",
]
