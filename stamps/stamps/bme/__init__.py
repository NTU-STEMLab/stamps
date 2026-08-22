# -*- coding: utf-8 -*-
"""
stamps.bme — Bayesian Maximum Entropy estimation package
=========================================================

This subpackage provides a unified API for BME spatial/temporal estimation,
probabilistic soft-data handling, and Normal Score Transform utilities.

Modules
-------
BMEoptions
    Algorithm options container.
BMEprobaEstimations
    Core BME posterior computation: moments, mode, PDF, credible intervals.
bme_transform
    Normal Score Transform (NST) primitives and T-variant BME wrappers.
pystks_variable
    Type constants for soft PDF representations.
softconverter
    Soft-data construction, manipulation, and conversion utilities.

Quick start
-----------
>>> from stamps.bme import (
...     BMEoptions,
...     BMEPosteriorMoments,
...     BMEPosteriorMode,
...     BMEPosteriorPDF,
...     BMEPosteriorCI,
...     BMEPosteriorQtl,
...     probaGaussian,
...     probaUniform,
...     NormalScoreTransform,
... )
"""
from __future__ import annotations

# ------------------------------------------------------------------
# Options
# ------------------------------------------------------------------
from .BMEoptions import BMEoptions

# ------------------------------------------------------------------
# Core BME posterior estimators
# ------------------------------------------------------------------
from .BMEprobaEstimations import (
    BMEPosteriorMoments,     # posterior moments (mean, variance, …)
    BMEPosteriorPDF,         # posterior PDF evaluated on an automatic z-grid
    BMEPosteriorPDF_grid,    # deprecated alias → BMEPosteriorPDF(..., z_grid=z_grid)
    BMEPosteriorMode,        # posterior mode (MAP estimate)
    BMEPosteriorCI,          # posterior credible intervals
    BMEPosteriorQtl,         # posterior quantiles at arbitrary probability levels
    BMEprobaGaussian,        # analytical Gaussian-only BME
    BMEpriorMean,            # unified prior mean helper (all methods)
    compute_bme_prior,       # backward-compat alias for BMEpriorMean
    compute_bme_local_prior, # backward-compat alias (kernel + soft data)
)

# ------------------------------------------------------------------
# T-variant estimators (with Normal Score Transform)
# ------------------------------------------------------------------
from .bme_transform import (
    BMEprobaTMode,           # mode with NST
    BMEprobaTPdf,            # PDF with NST
    BMEprobaTCI,             # CI with NST
)

# ------------------------------------------------------------------
# Normal Score Transform class & primitives
# ------------------------------------------------------------------
from .bme_transform import (
    NormalScoreTransform,
    other2gauss,
    pdfgauss2other,
    probaother2gauss,
    gausspdf,
    gaussinv,
    transformderiv,
    FyTransformCheckArgs,
)

# ------------------------------------------------------------------
# Soft-data utilities (softconverter)
# ------------------------------------------------------------------
from .softconverter import (
    # Construction helpers
    probaGaussian,
    probaUniform,
    probaStudentT,
    probaoffset,
    probasplit,
    probaneighbours,
    probacat,
    probacombinedupli,
    # Conversion / query
    proba2val,
    proba2interval,
    proba2stat,
    proba2probdens,
    proba2quantile,
    # Format converters
    ud2zs,
    ud2zs_temp,
    zs2ud,
    ud2ud,
    gs2ud,
    uf2ud,
    uf2zs,
    softpdftypeCheckArgs,
)

__all__ = [
    # Options
    "BMEoptions",
    # Core estimators
    "BMEPosteriorMoments",
    "BMEPosteriorPDF",
    "BMEPosteriorPDF_grid",
    "BMEPosteriorMode",
    "BMEPosteriorCI",
    "BMEPosteriorQtl",
    "BMEprobaGaussian",
    "BMEpriorMean",
    "compute_bme_prior",       # backward-compat alias
    "compute_bme_local_prior", # backward-compat alias
    # T-variant estimators
    "BMEprobaTMode",
    "BMEprobaTPdf",
    "BMEprobaTCI",
    # NST class & primitives
    "NormalScoreTransform",
    "other2gauss",
    "pdfgauss2other",
    "probaother2gauss",
    "gausspdf",
    "gaussinv",
    "transformderiv",
    "FyTransformCheckArgs",
    # Soft-data utilities
    "probaGaussian",
    "probaUniform",
    "probaStudentT",
    "probaoffset",
    "probasplit",
    "probaneighbours",
    "probacat",
    "probacombinedupli",
    "proba2val",
    "proba2interval",
    "proba2stat",
    "proba2probdens",
    "proba2quantile",
    "ud2zs",
    "ud2zs_temp",
    "zs2ud",
    "ud2ud",
    "gs2ud",
    "uf2ud",
    "uf2zs",
    "softpdftypeCheckArgs",
]
