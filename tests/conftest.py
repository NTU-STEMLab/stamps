# -*- coding: utf-8 -*-
"""
conftest.py
===========
Shared pytest fixtures for the stamps test suite.

All fixtures that are used by two or more test modules live here so they
are automatically discovered by pytest without any explicit imports.
"""
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Random number generator
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    """A seeded numpy Generator for reproducible draws in every test."""
    return np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Standard covariance setup (mirrors the base config in BMEPROBALIBtest.m)
# ---------------------------------------------------------------------------

@pytest.fixture
def standard_cov():
    """
    Nugget + exponential covariance matching the BMElib test baseline:

        cNugget = 0.05,  cc = 1.0,  aa = 1.0

    Returns a dict with keys:
        model    : list of str
        param    : list of (sill, range) tuples
        sill     : float  — sill of the exponential component
        ar       : float  — practical range of the exponential component
        nugget   : float  — nugget sill
    """
    cNugget = 0.05
    cc      = 1.0
    aa      = 1.0
    return dict(
        model  = ['nuggetC', 'exponentialC'],
        param  = [(cNugget, None), (cc, 3 * aa)],
        sill   = cc,
        ar     = 3 * aa,
        nugget = cNugget,
    )


# ---------------------------------------------------------------------------
# Small spatial grid (5 × 2 — safe for Cholesky and covariance tests)
# ---------------------------------------------------------------------------

@pytest.fixture
def small_grid():
    """
    A deterministic 5-point 2-D spatial grid suitable for simulation tests.
    Returns an ndarray of shape (5, 2).
    """
    return np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, 0.5],
    ])


# ---------------------------------------------------------------------------
# Standard BME soft-data setup (1 soft datum — test 1 in BMEPROBALIBtest.m)
# ---------------------------------------------------------------------------

@pytest.fixture
def bme_base(standard_cov):
    """
    Minimal BME test scenario with one hard datum and one soft datum,
    matching testnumber=1 in BMEPROBALIBtest.m:

        ck = [[0.5, 0.5]]
        ch = [[0.0, 0.0]]    zh = 1.4
        cs = [[1.0, 0.9]]    Gaussian soft pdf  (mean=0.9, var=0.04)

    The soft datum is encoded as type-2 (piecewise-linear).
    Returns a dict with keys: ck, ch, zh, cs, zs, covmodel, covparam.
    """
    from stamps.bme.softconverter import probaGaussian

    ck = np.array([[0.5, 0.5]])
    ch = np.array([[0.0, 0.0]])
    zh = np.array([[1.4]])
    cs = np.array([[1.0, 0.9]])

    softpdftype, nl, limi, probdens = probaGaussian(
        zm=np.array([0.9]), zv=np.array([0.04])
    )
    zs = (softpdftype, nl, limi, probdens)

    return dict(
        ck       = ck,
        ch       = ch,
        zh       = zh,
        cs       = cs,
        zs       = zs,
        covmodel = standard_cov['model'],
        covparam = standard_cov['param'],
    )


# ---------------------------------------------------------------------------
# VISTA-BME fixtures (branch feature/vista-bme)
# ---------------------------------------------------------------------------
# The sparse-precision modules are added Step by Step (docs/VISTA_INTEGRATION.md).
# Until a Step lands, its module is missing or its functions raise
# NotImplementedError; the two helpers below turn both situations into a skip so
# the suite stays green while the branch is under construction.

#: Defaults of the ``vista_*`` option keys (docs/VISTA_INTEGRATION.md §4).
VISTA_OPTION_DEFAULTS = {
    'vista_max_parents': 32,
    'vista_ordering': 'random',
    'vista_random_state': 0,
    'vista_hard_variance': 0.0,
    'vista_ep_damping': 0.7,
    'vista_ep_tolerance': 1e-6,
    'vista_ep_max_iterations': 50,
    'vista_ep_precision_cap': 1e8,
    'vista_soft_nnodes': 65,
    'vista_variance_exact_limit': 3000,
    'vista_variance_probes': 32,
    'vista_mixture_max_combinations': 128,
    'vista_mixture_variance_tolerance': 1e-2,
}


@pytest.fixture
def vista_require():
    """Import a VISTA module or member, skipping the test when it is absent.

    Usage::

        build_vecchia = vista_require('stamps.general.vecchia:build_vecchia')
        precision     = vista_require('stamps.mvn.precision')

    Call this *first* in a test, before computing any reference value, so an
    unimplemented Step costs no runtime.
    """
    def _require(dotted):
        module_name, _, attribute = dotted.partition(':')
        module = pytest.importorskip(module_name)
        if not attribute:
            return module
        if not hasattr(module, attribute):
            pytest.skip('{d} does not exist yet'.format(d=dotted))
        return getattr(module, attribute)
    return _require


@pytest.fixture
def vista_run():
    """Call a VISTA function, skipping the test while it is a skeleton."""
    def _run(function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except NotImplementedError:
            pytest.skip('{f} is not implemented yet'.format(
                f=getattr(function, '__name__', function)))
    return _run


@pytest.fixture
def vista_options():
    """Factory for a BMEoptions carrying the ``vista_*`` keys."""
    from stamps.bme import BMEoptions

    def _options(**overrides):
        options = BMEoptions()
        options.options_dict.update(VISTA_OPTION_DEFAULTS)
        options.options_dict.update(overrides)
        return options
    return _options


@pytest.fixture
def vista_field():
    """100 hard data, 40 targets, unit-sill exponential covariance.

    The hard values are drawn from the covariance model itself so that the
    sparse estimator and ``BMEPosteriorMoments`` share the same prior.

    Returns ``(ck, ch, zh, covmodel, covparam)``.
    """
    from stamps.general.coord2K import coord2K

    rng = np.random.default_rng(0)
    ch = rng.uniform(0, 1, size=(100, 2))
    ck = rng.uniform(0, 1, size=(40, 2))
    covmodel = ['exponentialC']
    covparam = [(1.0, 0.3)]
    K = np.asarray(coord2K(ch, ch, covmodel, covparam)[0], dtype=float)
    zh = rng.multivariate_normal(np.zeros(100), K)
    return ck, ch, zh, covmodel, covparam


@pytest.fixture
def vista_two_point():
    """One soft-data location and one target, with the 2x2 dense covariance.

    Reproduces the single-site geometry of the reference implementation's
    quadrature tests, where the posterior at the target is available in closed
    form from the soft datum's own first two moments:

        E[x_k]   = b * m_s
        Var[x_k] = K_kk - K_ks^2 / K_ss + b^2 * v_s,   b = K_ks / K_ss

    Returns a dict with keys ``cs, ck, covmodel, covparam, K, regression,
    conditional_variance``.
    """
    from stamps.general.coord2K import coord2K

    cs = np.array([[0.0, 0.0]])
    ck = np.array([[0.35, 0.15]])
    covmodel = ['exponentialC']
    covparam = [(1.7, 1.25)]
    coords = np.vstack((cs, ck))
    K = np.asarray(coord2K(coords, coords, covmodel, covparam)[0], dtype=float)
    return dict(
        cs=cs, ck=ck, covmodel=covmodel, covparam=covparam, K=K,
        regression=K[1, 0] / K[0, 0],
        conditional_variance=K[1, 1] - K[1, 0] ** 2 / K[0, 0],
    )
