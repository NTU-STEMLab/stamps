# -*- coding: utf-8 -*-
"""
stamps.simulation
=================
Spatial random field generation.

Sub-modules
-----------
simulation    All simulation algorithms (Cholesky, sequential, circular embedding,
              soft-data conditioning).
"""
from __future__ import division

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

__all__ = [
    # Circular embedding (spectral)
    "stationary_gaussian_process",
    # Cholesky-based unconditional / conditional
    "simuchol",
    "simucholcond",
    "simucholcondME",
    "anisosimuchol",
    # Sequential simulation
    "simuseq",
    "simuseqcond",
    "simuseqcondME",
    "simuseqcondInt",
    # Soft-data conditioning
    "simuprobabilistic",
    "simuinterval",
]
