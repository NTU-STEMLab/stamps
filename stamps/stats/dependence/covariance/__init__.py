# -*- coding: utf-8 -*-
"""
stamps.stats.dependence.covariance
===================================
Continuous spatial/temporal covariance estimation and model fitting.
"""
from .stcov import (  # noqa: F401
    stcov,
    stcov_kdtree,
    cov_avg_nd,
    stcov_split,
    stcov_dask,
    stcov_dask_half,
)
from .stcovfit import (  # noqa: F401
    covmodelwls,
    covmodelfit,
    covmodeldef,
    covmodelest,
    anisocovmodelest,
    covdownscale,
    coregfit,
)
from .mlecovfit import (  # noqa: F401
    mlecovfitv,
    mlecovfitg,
    mlecovfit_sub,
)
from .anisotropy import (  # noqa: F401
    polarstcovmap,
    iso2aniso,
    aniso2iso,
    covariancemap,
    estimate_anisotropy_params,
    isotropy_test,
)
