# -*- coding: utf-8 -*-
"""
test_covariance_models.py
=========================
Tests for covariance model evaluation via ``coord2K``.

Adapted from ``MODELSLIBtest.m`` (BMElib testslib, Jan 1, 2001).

MATLAB → Python mapping
-----------------------
MODELSLIBtest testtype=1  →  test_nugget_spatial, test_exponential_spatial,
                              test_gaussian_spatial
MODELSLIBtest testtype=2  →  test_separable_spacetime
MODELSLIBtest testtype=3  →  test_nonseparable_spacetime
(new Python-only)         →  test_covariance_matrix_psd
"""
import numpy as np
import numpy.testing as npt
import pytest

from stamps.general.coord2K import coord2K
from stamps.models.covmodel import nuggetC, exponentialC, gaussianC


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _single_point():
    """Origin in 2-D (estimation location for lag-0 checks)."""
    return np.array([[0.0, 0.0]])


def _lag_grid(n: int = 50, rmax: float = 3.0) -> np.ndarray:
    """n points along the x-axis up to rmax — pure spatial grid."""
    r = np.linspace(0.0, rmax, n)
    return np.column_stack([r, np.zeros(n)])


def _lag_grid_st(ns: int = 20, nt: int = 20,
                  rmax: float = 2.0, tmax: float = 4.0) -> np.ndarray:
    """ns × nt grid of (x, 0, t) space-time coords (shape ns*nt × 3)."""
    r = np.linspace(0.0, rmax, ns)
    t = np.linspace(0.0, tmax, nt)
    rr, tt = np.meshgrid(r, t, indexing='ij')  # (ns, nt)
    return np.column_stack([rr.ravel(), np.zeros(ns * nt), tt.ravel()])


# ===========================================================================
# testtype=1  — Pure spatial covariance
# ===========================================================================

class TestSpatialCovariance:
    """Mirror MODELSLIBtest testtype=1."""

    def test_nugget_at_zero_equals_sill(self):
        """C_nugget(r=0) == sill  (nugget discontinuity at origin)."""
        sill   = 0.5
        c      = _single_point()
        K, _   = coord2K(c, c, ['nuggetC'], [(sill, None)])
        npt.assert_allclose(K[0, 0], sill, rtol=1e-12)

    def test_nugget_away_from_origin_is_zero(self):
        """C_nugget(r>0) == 0."""
        sill   = 0.5
        c1     = np.array([[0.0, 0.0]])
        c2     = np.array([[1.0, 0.0]])
        K, _   = coord2K(c1, c2, ['nuggetC'], [(sill, None)])
        npt.assert_allclose(K[0, 0], 0.0, atol=1e-15)

    def test_exponential_at_zero_equals_sill(self):
        """C_exp(r=0) == sill."""
        sill, ar = 1.0, 3.0
        c        = _single_point()
        K, _     = coord2K(c, c, ['exponentialC'], [(sill, ar)])
        npt.assert_allclose(K[0, 0], sill, rtol=1e-10)

    def test_exponential_decreases_with_distance(self):
        """C_exp must be strictly decreasing along positive lags."""
        sill, ar = 1.0, 3.0
        lags     = _lag_grid(n=30, rmax=5.0)
        c0       = _single_point()
        K, _     = coord2K(c0, lags, ['exponentialC'], [(sill, ar)])
        values   = K.ravel()
        # Skip r=0; each subsequent value should be ≤ the previous
        assert np.all(np.diff(values) <= 1e-12), \
            "Exponential covariance should be non-increasing with distance."

    def test_exponential_analytical_formula(self):
        """C_exp(r) == sill * exp(-3*r/ar) at several lag values."""
        sill, ar = 1.0, 3.0
        r_vals   = np.array([0.0, 0.5, 1.0, 2.0, 3.0])
        c0       = np.array([[0.0, 0.0]])
        c2       = np.column_stack([r_vals, np.zeros_like(r_vals)])
        K, _     = coord2K(c0, c2, ['exponentialC'], [(sill, ar)])
        expected = sill * np.exp(-3 * r_vals / ar)
        npt.assert_allclose(K.ravel(), expected, rtol=1e-10)

    def test_gaussian_analytical_formula(self):
        """C_gauss(r) == sill * exp(-3*(r/ar)^2) at several lag values."""
        sill, ar = 1.0, 2.0
        r_vals   = np.array([0.0, 0.5, 1.0, 2.0])
        c0       = np.array([[0.0, 0.0]])
        c2       = np.column_stack([r_vals, np.zeros_like(r_vals)])
        K, _     = coord2K(c0, c2, ['gaussianC'], [(sill, ar)])
        expected = sill * np.exp(-3 * (r_vals / ar) ** 2)
        npt.assert_allclose(K.ravel(), expected, rtol=1e-10)

    def test_nested_nugget_plus_exponential_at_zero(self):
        """Nested model: C(r=0) == nugget_sill + exp_sill."""
        c_nugget, c_exp, ar = 0.05, 1.0, 3.0
        c = _single_point()
        K, _ = coord2K(
            c, c,
            ['nuggetC', 'exponentialC'],
            [(c_nugget, None), (c_exp, ar)],
        )
        npt.assert_allclose(K[0, 0], c_nugget + c_exp, rtol=1e-10)

    def test_covariance_matrix_symmetric(self):
        """coord2K(c, c) must return a symmetric matrix."""
        rng  = np.random.default_rng(42)
        pts  = rng.uniform(0, 5, size=(10, 2))
        K, _ = coord2K(pts, pts,
                       ['nuggetC', 'exponentialC'],
                       [(0.05, None), (1.0, 3.0)])
        npt.assert_allclose(K, K.T, atol=1e-12)


# ===========================================================================
# testtype=2  — Space-time separable covariance
# ===========================================================================

class TestSeparableSpaceTimeCov:
    """Mirror MODELSLIBtest testtype=2."""

    def _get_st_cov(self, c1, c2, sill=1.0, as_=1.0, at=2.0):
        """nugget/nugget + gaussianC/exponentialC separable S/T model."""
        cNugget = 0.05
        # Separable model uses notation 'modelS/modelT'
        # Each param tuple must be (sill, param_s, param_t) — 3 elements
        model  = ['nuggetC/nuggetC', 'gaussianC/exponentialC']
        param  = [(cNugget, None, None), (sill, as_, at)]
        return coord2K(c1, c2, model, param)

    def test_zero_lag_equals_total_sill(self):
        """C(r=0, t=0) == nugget + sill."""
        c  = np.array([[0.0, 0.0, 0.0]])
        K, _ = self._get_st_cov(c, c)
        npt.assert_allclose(K[0, 0], 0.05 + 1.0, rtol=1e-9)

    def test_spatial_slice_decreases(self):
        """C(r, t=0) is non-increasing for increasing r."""
        lags = _lag_grid_st(ns=15, nt=1, rmax=3.0, tmax=0.0)
        c0   = np.array([[0.0, 0.0, 0.0]])
        K, _ = self._get_st_cov(c0, lags)
        values = K.ravel()
        assert np.all(np.diff(values) <= 1e-12), \
            "Separable C(r,t=0) should be non-increasing with r."

    def test_temporal_slice_decreases(self):
        """C(r=0, t) is non-increasing for increasing t."""
        nt   = 15
        t    = np.linspace(0.0, 5.0, nt)
        lags = np.column_stack([np.zeros((nt, 2)), t])
        c0   = np.array([[0.0, 0.0, 0.0]])
        K, _ = self._get_st_cov(c0, lags)
        values = K.ravel()
        assert np.all(np.diff(values) <= 1e-12), \
            "Separable C(r=0,t) should be non-increasing with t."

    def test_symmetry_in_lag(self):
        """C(r, t) ≈ C(-r, t) for a stationary model."""
        c0    = np.array([[0.5, 0.5, 1.0]])
        cpos  = np.array([[1.5, 0.5, 1.0]])
        cneg  = np.array([[-0.5, 0.5, 1.0]])
        Kp, _ = self._get_st_cov(c0, cpos)
        Kn, _ = self._get_st_cov(c0, cneg)
        npt.assert_allclose(Kp, Kn, rtol=1e-10)


# ===========================================================================
# testtype=3  — Space-time non-separable covariance
# ===========================================================================

class TestNonSeparableSpaceTimeCov:
    """Mirror MODELSLIBtest testtype=3."""

    def _get_st_cov_ns(self, c1, c2, sill=1.0, bs=1.0, s_t_ratio=0.5):
        """nuggetCST + gaussianCST non-separable S/T covariance."""
        cNugget = 0.05
        # Non-separable models: each param tuple is (sill, param_s, s_t_ratio)
        model  = ['nuggetCST', 'gaussianCST']
        param  = [(cNugget, None, 0.0), (sill, bs, s_t_ratio)]
        return coord2K(c1, c2, model, param)

    def test_zero_lag_equals_total_sill(self):
        """C(r=0, t=0) == nugget + sill."""
        c  = np.array([[0.0, 0.0, 0.0]])
        K, _ = self._get_st_cov_ns(c, c)
        npt.assert_allclose(K[0, 0], 0.05 + 1.0, rtol=1e-9)

    def test_spatial_slice_at_t0_decreases(self):
        """Non-separable C(r, t=0) must be non-increasing in r."""
        lags = _lag_grid_st(ns=15, nt=1, rmax=3.0, tmax=0.0)
        c0   = np.array([[0.0, 0.0, 0.0]])
        K, _ = self._get_st_cov_ns(c0, lags)
        values = K.ravel()
        assert np.all(np.diff(values) <= 1e-12), \
            "Non-separable C(r,t=0) should be non-increasing with r."

    def test_temporal_slice_at_r0_decreases(self):
        """Non-separable C(r=0, t) must be non-increasing in t."""
        nt   = 15
        t    = np.linspace(0.0, 5.0, nt)
        lags = np.column_stack([np.zeros((nt, 2)), t])
        c0   = np.array([[0.0, 0.0, 0.0]])
        K, _ = self._get_st_cov_ns(c0, lags)
        values = K.ravel()
        assert np.all(np.diff(values) <= 1e-12), \
            "Non-separable C(r=0,t) should be non-increasing with t."


# ===========================================================================
# PSD check (new Python-only test)
# ===========================================================================

class TestPositiveSemiDefinite:
    """The auto-covariance matrix K = coord2K(c, c, ...) must be PSD."""

    @pytest.mark.parametrize("model,param", [
        (['exponentialC'],                [(1.0, 3.0)]),
        (['gaussianC'],                   [(1.0, 2.0)]),
        (['nuggetC', 'exponentialC'],     [(0.05, None), (1.0, 3.0)]),
        (['nuggetC', 'gaussianC'],        [(0.1,  None), (0.9, 2.0)]),
    ])
    def test_spatial_covariance_matrix_is_psd(self, model, param):
        """All eigenvalues of coord2K(c, c) must be ≥ -1e-10."""
        rng  = np.random.default_rng(7)
        pts  = rng.uniform(0, 5, size=(12, 2))
        K, _ = coord2K(pts, pts, model, param)
        eigs = np.linalg.eigvalsh(K)
        assert np.all(eigs >= -1e-8), (
            f"Covariance matrix with {model} is not PSD: "
            f"min eigenvalue = {eigs.min():.3e}"
        )
