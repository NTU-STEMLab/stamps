# -*- coding: utf-8 -*-
"""
stamps.categorical
===================
Categorical spatial estimation: BME, MCP, and Hybrid BME.

Submodules
----------
estimation    BMEcatPdf, MCPcatPdf, HBMEcatPdf, producttablepdf
me_solvers    MaxEnt backends (GIS, IIS, MPL, MLE_reg, ME_dual)
_core         Shared helpers (_resolve_pmodel, _coerce_ps_soft, …)
prior         BMEcatPrior, BMEcatPriorLOOCV
first_order   kernel_first_order (2-D / 3-D q fields)
evaluate      confusion_matrix, loocv, …
"""
from .estimation import (
    BMEcatPdf,
    MCPcatPdf,
    HBMEcatPdf,
    producttablepdf,
    sumvaluesatindex,
    tune_regularization_loocv,
)
from .me_solvers import maxentropytable
from .first_order import kernel_first_order
from ._core import (
    _coerce_ps_soft,
    _resolve_pmodel,
    _find_effective_range,
    _auto_coord_cols,
    _p_marginal_from_pmodel,
    probamodel2bitable,
    neighbours_separable,
)
from .evaluate import (
    loocv_categorical,
    loocv_categorical_points,
)

__all__ = [
    "BMEcatPdf",
    "MCPcatPdf",
    "HBMEcatPdf",
    "producttablepdf",
    "sumvaluesatindex",
    "tune_regularization_loocv",
    "maxentropytable",
    "kernel_first_order",
    "_coerce_ps_soft",
    "_resolve_pmodel",
    "_find_effective_range",
    "_auto_coord_cols",
    "_p_marginal_from_pmodel",
    "probamodel2bitable",
    "neighbours_separable",
    "loocv_categorical",
    "loocv_categorical_points",
]
