# -*- coding: utf-8 -*-
"""
stamps.analysis
===============
Spatiotemporal analysis methods.

Sub-modules
-----------
eof        Empirical Orthogonal Functions and matrix decomposition (HOSVD, PCA wrappers)
pca        Principal Component Analysis variants
mca        Maximum Covariance Analysis
cca        Canonical Correlation Analysis
ent        Entropy estimation (discrete and continuous)
mepdf      Maximum Entropy PDF estimation
ekf        Extended Kalman Filter utilities
gam        Generalised Additive Models (R/Python interface)
dlnm       Distributed Lag Non-Linear Models
stl        Seasonal-Trend decomposition (LOESS)
"""
from __future__ import division
import warnings

# ------------------------------------------------------------------
# EOF / Matrix decomposition (also exposes HOSVD, unfold, fold)
# ------------------------------------------------------------------
from .eof import (  # noqa: F401
    eof,
    eeof,
    sreof,
    varimax,
    meof,
    srmeof,
    HOSVD,
    unfold,
    fold,
)

# ------------------------------------------------------------------
# PCA variants
# ------------------------------------------------------------------
from .pca import (  # noqa: F401
    pca,
    srpca,
    stica,
    srica,
)

# ------------------------------------------------------------------
# Maximum Covariance Analysis
# ------------------------------------------------------------------
from .mca import (  # noqa: F401
    mca,
    smca,
)

# ------------------------------------------------------------------
# Canonical Correlation Analysis
# ------------------------------------------------------------------
from .cca import cca  # noqa: F401

# ------------------------------------------------------------------
# Entropy estimation
# ------------------------------------------------------------------
from .ent import (  # noqa: F401
    entropyD,
    entropyC,
    condentropy,
)

# ------------------------------------------------------------------
# Maximum Entropy PDF
# ------------------------------------------------------------------
from .mepdf import (  # noqa: F401
    maxentpdf_gh,
    maxentpdf_gcp,
    maxentpdf_gkhk,
    maxentcondpdf_gc,
    maxentpdf_gc,
    gkhk,
)

# ------------------------------------------------------------------
# Extended Kalman Filter
# ------------------------------------------------------------------
from .ekf import (  # noqa: F401
    getK,
    getXu,
    getPu,
)

# ------------------------------------------------------------------
# Generalised Additive Models (optional – requires rpy2)
# ------------------------------------------------------------------
try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="rpy2")
    from .gam import (  # noqa: F401
        gam,
        construct_gam_formula,
        construct_gam_model,
        predict_gam,
    )
except (ImportError, ModuleNotFoundError):
    pass  # rpy2 not installed; GAM functions unavailable

# ------------------------------------------------------------------
# Distributed Lag Non-Linear Models (optional – requires rpy2)
# ------------------------------------------------------------------
try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="rpy2")
    from .dlnm import (  # noqa: F401
        crossbasis,
    )
except (ImportError, ModuleNotFoundError):
    pass  # rpy2 not installed; DLNM functions unavailable

# ------------------------------------------------------------------
# STL decomposition (optional – requires rpy2)
# ------------------------------------------------------------------
try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="rpy2")
    from .stl import (  # noqa: F401
        stl,
        stl_plot,
    )
except (ImportError, ModuleNotFoundError):
    pass  # rpy2 not installed; STL functions unavailable

# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------
__all__ = [
    # EOF / decomposition
    "eof",
    "eeof",
    "sreof",
    "varimax",
    "meof",
    "srmeof",
    "HOSVD",
    "unfold",
    "fold",
    # PCA
    "pca",
    "srpca",
    "stica",
    "srica",
    # MCA
    "mca",
    "smca",
    # CCA
    "cca",
    # entropy
    "entropyD",
    "entropyC",
    "condentropy",
    # MaxEnt PDF
    "maxentpdf_gh",
    "maxentpdf_gcp",
    "maxentpdf_gkhk",
    "maxentcondpdf_gc",
    "maxentpdf_gc",
    "gkhk",
    # EKF
    "getK",
    "getXu",
    "getPu",
    # GAM
    "gam",
    "construct_gam_formula",
    "construct_gam_model",
    "predict_gam",
    # DLNM
    "crossbasis",
    # STL
    "stl",
    "stl_plot",
]
