# -*- coding: utf-8 -*-
"""
Backward-compatibility shim — ``stamps.stest`` is now ``stamps.estimation``.

Lazy imports are used to avoid circular dependencies with ``categorical/``.
"""
import importlib as _importlib
import warnings as _warnings


def __getattr__(name):
    """Lazy-load from estimation/ and categorical/ to avoid circular deps."""

    _estimation_names = {
        "kriging", "krigingFac", "cokriging", "cokrigingT",
        "idw", "stmean", "designmatrix",
        "kernelsmoothing", "kernelsmoothing_truncate",
        "regression", "localmeanBME",
    }
    if name in _estimation_names:
        _warnings.warn(
            f"Importing '{name}' from 'stamps.stamps.stest' is deprecated. "
            "Use 'stamps.stamps.estimation' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        _mod = _importlib.import_module(".estimation", "stamps.stamps")
        try:
            return getattr(_mod, name)
        except AttributeError:
            _submod = _importlib.import_module(f".{name}", "stamps.stamps.estimation")
            return _submod

    _prior_names = {
        "BMEcatPrior", "BMEcatPriorLOOCV",
        "estimate_prior", "estimate_prior_xgboostlss",
        "estimate_prior_xgboost", "estimate_prior_uniform",
        "SpotpyHyperparameterSetup",
    }
    if name in _prior_names:
        _mod = _importlib.import_module(".BMEcatPrior", __name__)
        return getattr(_mod, name)

    _region_names = {"identify_homogeneous_regions"}
    if name in _region_names:
        _mod = _importlib.import_module("..stats.dependence.regions", __name__)
        return getattr(_mod, name)

    _analysis_names = {"identify_homogeneous_regions_data"}
    if name in _analysis_names:
        _mod = _importlib.import_module(".analysis", __name__)
        return getattr(_mod, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # estimation (from estimation/)
    "kriging", "krigingFac", "cokriging", "cokrigingT",
    "idw", "stmean", "designmatrix",
    # prior (from categorical/prior)
    "BMEcatPrior", "BMEcatPriorLOOCV",
    "estimate_prior", "estimate_prior_xgboostlss",
    "estimate_prior_xgboost", "estimate_prior_uniform",
    # regions
    "identify_homogeneous_regions",
    "identify_homogeneous_regions_data",
]
