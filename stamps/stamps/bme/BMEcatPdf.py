# -*- coding: utf-8 -*-
"""
Backward-compatibility shim for ``stamps.stamps.bme.BMEcatPdf``.

All public symbols have moved to :mod:`stamps.stamps.categorical`.
This module re-exports them so that existing ``from stamps.stamps.bme.BMEcatPdf import …``
statements continue to work.  A deprecation warning is emitted on first import.
"""
import warnings as _warnings

_warnings.warn(
    "Importing from 'stamps.stamps.bme.BMEcatPdf' is deprecated. "
    "Use 'stamps.stamps.categorical' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# --- Public estimators ---
from ..categorical.estimation import (  # noqa: F401
    BMEcatPdf,
    MCPcatPdf,
    HBMEcatPdf,
    producttablepdf,
    sumvaluesatindex,
    tune_regularization_loocv,
)

# --- ME solver backends ---
from ..categorical.me_solvers import (  # noqa: F401
    maxentropytable,
    sumoverallexcepttwo,
    multiplysubtable,
    iterativerescaling_GIS,
    iterativerescaling_IIS,
    MLEestimator,
    MPLestimator,
    ME_dual_estimator,
    ME_sequential_estimator,
    likeli_MLE,
    likeli_diff_MLE,
    likeli_MPL,
    likeli_diff_MPL,
    likeli_ME,
    likeli_diff_ME,
    _calculate_true_me_pdf,
)

# --- Shared front-layer helpers ---
from ..categorical._core import (  # noqa: F401
    _coerce_ps_soft,
    _resolve_pmodel,
    _find_effective_range,
    _auto_coord_cols,
    _p_marginal_from_pmodel,
    probamodel2bitable,
    neighbours,
)

# --- Deprecated: old neighbours_separable preserved for external code ---
# The function was decomposed into neighbours_cat + local_marginal in
# stamps.general, but we keep the old name here for backward compatibility.
try:
    from ..general.neighbours import neighbours_cat  # noqa: F401
    from ..general.local_marginal import local_marginal  # noqa: F401
except ImportError:
    pass
