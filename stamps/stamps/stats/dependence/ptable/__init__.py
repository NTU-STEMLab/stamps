# -*- coding: utf-8 -*-
"""
stamps.stats.dependence.ptable
================================
Categorical probability-table computation, fitting, and spatial analysis.
"""
from .empirical import (  # noqa: F401
    normalize_indicators,
    probatablecalc,
    probatablecalc_directional,
    probatablecalc_optimized,
    adaptive_bins,
    pair_distance_diagnostic,
)
from .spatial import (  # noqa: F401
    probatablecalc_spatial,
)
from .anisotropy import (  # noqa: F401
    polar_ptable_map,
    estimate_anisotropy_from_ptables,
    isotropy_test_categorical,
)
from .recenter import (  # noqa: F401
    ipf_recenter,
    recenter_pmodel,
)
from .fit import (  # noqa: F401
    probamodel2bitable,
    probamodel2bitable_separable,
    probatablefit,
    build_pmodel,
    spline_fit_pmodel,
    regularize_pmodel,
    alpha_empirical_bayes,
    loocv_tune_pmodel,
)
