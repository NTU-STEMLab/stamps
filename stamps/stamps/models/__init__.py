"""stamps.models
==============
Covariance and variogram model functions for spatial and space-time random
fields, ported from the bmelib MATLAB package.

Covariance models (C functions)
--------------------------------
nuggetC, exponentialC, sphericalC, gaussianC, holecosC, holesinC,
mexicanhatC, maternC

Non-separable space-time covariance aliases (CST)
-------------------------------------------------
nuggetCST, exponentialCST, gaussianCST, sphericalCST

Variogram models (V functions)
--------------------------------
nuggetV, exponentialV, sphericalV, gaussianV, holecosV, holesinV,
linearV, powerV

Utilities
---------
get_model
"""

from .covmodel import (  # noqa: F401
    # --- covariance models ---
    nuggetC,
    exponentialC,
    sphericalC,
    gaussianC,
    holecosC,
    holesinC,
    mexicanhatC,
    maternC,
    # --- space-time non-separable aliases ---
    nuggetCST,
    exponentialCST,
    gaussianCST,
    sphericalCST,
    # --- variogram models ---
    nuggetV,
    exponentialV,
    sphericalV,
    gaussianV,
    holecosV,
    holesinV,
    linearV,
    powerV,
    # --- utility ---
    get_model,
)

__all__ = [
    # covariance
    "nuggetC",
    "exponentialC",
    "sphericalC",
    "gaussianC",
    "holecosC",
    "holesinC",
    "mexicanhatC",
    "maternC",
    # ST aliases
    "nuggetCST",
    "exponentialCST",
    "gaussianCST",
    "sphericalCST",
    # variogram
    "nuggetV",
    "exponentialV",
    "sphericalV",
    "gaussianV",
    "holecosV",
    "holesinV",
    "linearV",
    "powerV",
    # utility
    "get_model",
]
