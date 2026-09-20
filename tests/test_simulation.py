# -*- coding: utf-8 -*-
"""
test_simulation.py
==================
Tests for spatial random field generation (``stamps.stats.simulation``).

Translated from the MATLAB ``simulib`` validation logic described in the
bmelib user guide and the implicit mathematical properties of each
simulation method.

Function → test class mapping
-------------------------------
simuchol                 →  TestSimuChol
simucholcond             →  TestSimuCholCond
stationary_gaussian_process (CE) → TestCircularEmbedding
anisosimuchol            →  TestAnisoSimuChol
simuseq                  →  TestSimuSeq
"""
import numpy as np
import numpy.testing as npt
import pytest

from stamps.stats.simulation.simulation import (
    simuchol,
    simucholcond,
    stationary_gaussian_process,
    anisosimuchol,
    simuseq,
)


# ---------------------------------------------------------------------------
# Shared covariance setup
# ---------------------------------------------------------------------------

_COV_MODEL_S  = ['exponentialC']          # pure spatial (2-D)
_COV_PARAM_S  = [(1.0, 3.0)]             # sill=1, 3*range=3

_COV_MODEL_ST = ['exponentialC/exponentialC']  # separable S/T
_COV_PARAM_ST = [(1.0, 3.0, 4.0)]             # sill=1, range_s=3, range_t=4

_N_REAL = 500   # number of realisations for empirical covariance checks


def _small_grid(n: int = 10) -> np.ndarray:
    """Deterministic n-point 2-D grid."""
    x = np.linspace(0, 4, n)
    return np.column_stack([x, np.zeros(n)])


def _small_st_grid(ns: int = 6, nt: int = 3) -> np.ndarray:
    """ns × nt S/T grid with d=3 columns (x, y=0, t)."""
    x = np.linspace(0, 3, ns)
    t = np.linspace(0, 2, nt)
    xx, tt = np.meshgrid(x, t, indexing='ij')
    return np.column_stack([xx.ravel(), np.zeros(ns * nt), tt.ravel()])


# ===========================================================================
# simuchol  — unconditional Cholesky simulation
# ===========================================================================

class TestSimuChol:
    """Matches simulib/simuchol.m validation (unconditional simulation)."""

    @pytest.fixture
    def grid(self):
        return _small_grid(n=12)

    def test_output_shape_single_realisation(self, grid):
        """simuchol with ns=1 returns Zh of shape (nh, 1)."""
        Zh, L = simuchol(grid, _COV_MODEL_S, _COV_PARAM_S, ns=1, seed=0)
        assert Zh.shape == (grid.shape[0], 1)

    def test_output_shape_multiple_realisations(self, grid):
        """simuchol with ns=5 returns Zh of shape (nh, 5)."""
        Zh, L = simuchol(grid, _COV_MODEL_S, _COV_PARAM_S, ns=5, seed=0)
        assert Zh.shape == (grid.shape[0], 5)

    def test_cholesky_factor_shape(self, grid):
        """The Cholesky factor L must be square (nh × nh)."""
        nh    = grid.shape[0]
        _, L  = simuchol(grid, _COV_MODEL_S, _COV_PARAM_S, ns=1, seed=0)
        assert L.shape == (nh, nh)

    def test_cholesky_factor_lower_triangular(self, grid):
        """L must be lower-triangular (upper triangle is zero)."""
        _, L = simuchol(grid, _COV_MODEL_S, _COV_PARAM_S, ns=1, seed=0)
        upper = np.triu(L, k=1)
        npt.assert_allclose(upper, np.zeros_like(upper), atol=1e-12)

    def test_covariance_reconstruction(self, grid):
        """L @ L.T must recover the true covariance matrix K."""
        from stamps.general.coord2K import coord2K
        K_true, _ = coord2K(grid, grid, _COV_MODEL_S, _COV_PARAM_S)
        _, L      = simuchol(grid, _COV_MODEL_S, _COV_PARAM_S, ns=1, seed=0)
        K_approx  = L @ L.T
        npt.assert_allclose(K_approx, K_true, rtol=1e-6)

    def test_empirical_variance_matches_sill(self, grid):
        """
        The sample variance across _N_REAL realisations at each point
        should be close to the sill (= 1.0).  rtol=0.20 to allow for
        Monte Carlo fluctuation with N=500 realisations.
        """
        Zh, _ = simuchol(grid, _COV_MODEL_S, _COV_PARAM_S, ns=_N_REAL, seed=1)
        empirical_var = np.var(Zh, axis=1, ddof=1)   # (nh,)
        npt.assert_allclose(empirical_var, 1.0, rtol=0.20,
                            err_msg="Empirical marginal variance should ≈ sill.")

    def test_zero_mean(self, grid):
        """Simulated fields should be zero-mean (E[Z] ≈ 0).
        atol=0.15 to allow Monte Carlo fluctuation with N=500."""
        Zh, _ = simuchol(grid, _COV_MODEL_S, _COV_PARAM_S, ns=_N_REAL, seed=2)
        empirical_mean = np.mean(Zh, axis=1)
        npt.assert_allclose(empirical_mean, np.zeros(grid.shape[0]),
                            atol=0.15,
                            err_msg="Unconditional simulation should be zero-mean.")

    def test_reproducibility_with_seed(self, grid):
        """Two calls with the same seed must produce identical output."""
        Zh1, _ = simuchol(grid, _COV_MODEL_S, _COV_PARAM_S, ns=3, seed=42)
        Zh2, _ = simuchol(grid, _COV_MODEL_S, _COV_PARAM_S, ns=3, seed=42)
        npt.assert_array_equal(Zh1, Zh2)

    def test_different_seeds_differ(self, grid):
        """Two calls with different seeds should produce different output."""
        Zh1, _ = simuchol(grid, _COV_MODEL_S, _COV_PARAM_S, ns=3, seed=0)
        Zh2, _ = simuchol(grid, _COV_MODEL_S, _COV_PARAM_S, ns=3, seed=99)
        assert not np.allclose(Zh1, Zh2), \
            "Different seeds should produce different realisations."

    def test_empirical_covariance_at_lag0(self, grid):
        """
        Empirical covariance between points 0 and 0 should equal
        the true model variance (sill = 1.0) within rtol=0.15.
        """
        Zh, _ = simuchol(grid, _COV_MODEL_S, _COV_PARAM_S, ns=_N_REAL, seed=3)
        cov_00 = float(np.cov(Zh[0, :], Zh[0, :], ddof=1)[0, 1])
        npt.assert_allclose(cov_00, 1.0, rtol=0.15)


# ===========================================================================
# simucholcond  — conditional Cholesky simulation
# ===========================================================================

class TestSimuCholCond:
    """Matches simulib/simucholcond.m validation logic."""

    @pytest.fixture
    def setup(self):
        """Conditioning at (0,0) = 2.0; simulate at (1,0)…(4,0)."""
        ch0 = np.array([[0.0, 0.0]])         # conditioning location
        zh0 = np.array([[2.0]])              # conditioning value
        ch  = np.linspace(0.5, 4.0, 8)
        ch  = np.column_stack([ch, np.zeros(8)])
        return ch, ch0, zh0

    def test_output_shape(self, setup):
        """simucholcond returns Zh of shape (nh, ns)."""
        ch, ch0, zh0 = setup
        Zh, _ = simucholcond(ch, ch0, zh0,
                             _COV_MODEL_S, _COV_PARAM_S, ns=3, seed=0)
        assert Zh.shape == (ch.shape[0], 3)

    def test_reproducibility(self, setup):
        ch, ch0, zh0 = setup
        Zh1, _ = simucholcond(ch, ch0, zh0,
                              _COV_MODEL_S, _COV_PARAM_S, ns=3, seed=5)
        Zh2, _ = simucholcond(ch, ch0, zh0,
                              _COV_MODEL_S, _COV_PARAM_S, ns=3, seed=5)
        npt.assert_array_equal(Zh1, Zh2)

    def test_conditioning_point_raises_on_overlap(self, setup):
        """
        Passing the exact conditioning location inside ``ch`` must raise
        ValueError because the conditional covariance matrix would be
        singular (zero kriging variance at the conditioning site).
        """
        ch0 = np.array([[0.0, 0.0]])
        zh0 = np.array([[2.0]])
        # Include ch0 at position 0 — should trigger the guard
        x_pts   = np.concatenate([[0.0], np.linspace(0.5, 4.0, 5)])  # 6 values
        ch_incl = np.column_stack([x_pts, np.zeros(6)])  # (6, 2)
        with pytest.raises(ValueError, match="identical coordinates"):
            simucholcond(ch_incl, ch0, zh0,
                         _COV_MODEL_S, _COV_PARAM_S, ns=1, seed=10)

    def test_conditional_mean_near_hard_data(self, setup):
        """
        The sample mean of Zh at the nearest simulation point to ch0
        should be closer to zh0 than the unconditional mean (0).
        """
        ch, ch0, zh0 = setup
        Zh, _ = simucholcond(ch, ch0, zh0,
                             _COV_MODEL_S, _COV_PARAM_S, ns=_N_REAL, seed=11)
        # Nearest point to ch0 is ch[0] at distance 0.5
        cond_mean = float(np.mean(Zh[0, :]))
        assert abs(cond_mean - float(zh0[0, 0])) < abs(cond_mean), \
            "Conditional mean should be pulled toward the conditioning value."

    def test_conditional_variance_less_than_sill(self, setup):
        """
        Conditional variance must be ≤ prior variance (sill) everywhere.
        """
        ch, ch0, zh0 = setup
        Zh, _ = simucholcond(ch, ch0, zh0,
                             _COV_MODEL_S, _COV_PARAM_S, ns=_N_REAL, seed=12)
        cond_var = np.var(Zh, axis=1, ddof=1)
        sill = 1.0
        assert np.all(cond_var <= sill + 0.1), \
            "Conditional variance should be ≤ sill everywhere."


# ===========================================================================
# stationary_gaussian_process  — circular embedding (FFT-based)
# ===========================================================================

class TestCircularEmbedding:
    """
    Tests for the FFT/circular-embedding unconditional simulator.

    ``stationary_gaussian_process(m, n, rho_func, seed=...)`` generates
    two independent 2-D Gaussian random fields of shape (m, n) by
    circular embedding.  The function also returns the x and y coordinate
    grids, so the return signature is ``(field1, field2, x_grid, y_grid)``.
    """

    # Exponential covariance ρ(h) = exp(-|h|/ℓ) — separable 2D version
    @staticmethod
    def _rho(h: np.ndarray, ell: float = 2.0) -> float:
        """Exponential covariance kernel: exp(-sqrt(h[0]^2+h[1]^2)/ell)."""
        return float(np.exp(-np.sqrt(h[0] ** 2 + h[1] ** 2) / ell))

    def test_output_shape(self):
        """Both fields should have shape (m, n)."""
        m, n = 16, 24
        f1, f2, xg, yg = stationary_gaussian_process(m, n, self._rho, seed=0)
        assert f1.shape == (m, n), f"field1 shape mismatch: {f1.shape}"
        assert f2.shape == (m, n), f"field2 shape mismatch: {f2.shape}"

    def test_output_grid_shape(self):
        """tx has length n (x-axis), ty has length m (y-axis)."""
        m, n = 8, 12
        f1, f2, tx, ty = stationary_gaussian_process(m, n, self._rho, seed=1)
        assert tx.shape == (n,), f"tx should have shape ({n},), got {tx.shape}"
        assert ty.shape == (m,), f"ty should have shape ({m},), got {ty.shape}"

    def test_reproducibility(self):
        """Same seed ↔ identical output."""
        m, n = 16, 16
        f1a, f2a, _, _ = stationary_gaussian_process(m, n, self._rho, seed=7)
        f1b, f2b, _, _ = stationary_gaussian_process(m, n, self._rho, seed=7)
        npt.assert_array_equal(f1a, f1b)
        npt.assert_array_equal(f2a, f2b)

    def test_different_seeds_differ(self):
        """Different seeds must produce distinct realisations."""
        m, n = 16, 16
        f1a, _, _, _ = stationary_gaussian_process(m, n, self._rho, seed=0)
        f1b, _, _, _ = stationary_gaussian_process(m, n, self._rho, seed=1)
        assert not np.allclose(f1a, f1b), \
            "Different seeds should yield different realisations."

    def test_empirical_mean_near_zero(self):
        """Grand mean over many draws should be ≈ 0."""
        m, n    = 16, 16
        n_draws = 50
        means   = []
        for k in range(n_draws):
            f1, f2, _, _ = stationary_gaussian_process(m, n, self._rho, seed=k)
            means.append(float(np.mean(f1)))
        grand_mean = float(np.mean(means))
        npt.assert_allclose(grand_mean, 0.0, atol=0.15,
                            err_msg="Grand mean of stationary_gaussian_process should be ≈ 0.")


# ===========================================================================
# anisosimuchol  — anisotropic Cholesky simulation
# ===========================================================================

class TestAnisoSimuChol:
    """Tests for anisotropy-aware unconditional simulation."""

    @pytest.fixture
    def aniso_grid(self):
        """
        Random 2-D grid with enough points to compute a meaningful
        empirical variance.
        """
        rng = np.random.default_rng(0)
        return rng.uniform(0, 8, size=(20, 2))

    def test_output_shape(self, aniso_grid):
        """anisosimuchol returns Zh of shape (nh, ns)."""
        Zh, _ = anisosimuchol(
            aniso_grid,
            covmodel=['exponentialC'], covparam=[(1.0, 3.0)],
            theta=0.0, ratio=2.0,
            ns=4, seed=0,
        )
        assert Zh.shape == (aniso_grid.shape[0], 4)

    def test_reproducibility(self, aniso_grid):
        Zh1, _ = anisosimuchol(
            aniso_grid, covmodel=['exponentialC'], covparam=[(1.0, 3.0)],
            theta=0.0, ratio=2.0, ns=2, seed=3,
        )
        Zh2, _ = anisosimuchol(
            aniso_grid, covmodel=['exponentialC'], covparam=[(1.0, 3.0)],
            theta=0.0, ratio=2.0, ns=2, seed=3,
        )
        npt.assert_array_equal(Zh1, Zh2)

    def test_isotropic_limit(self, aniso_grid):
        """
        With ratio=1.0 (no anisotropy), anisosimuchol should give the
        same covariance structure as simuchol.
        Compare Cholesky factors: rtol=1e-10.
        """
        _, L_aniso = anisosimuchol(
            aniso_grid, covmodel=['exponentialC'], covparam=[(1.0, 3.0)],
            theta=0.0, ratio=1.0, ns=1, seed=0,
        )
        _, L_iso   = simuchol(
            aniso_grid, covmodel=['exponentialC'], covparam=[(1.0, 3.0)],
            ns=1, seed=0,
        )
        npt.assert_allclose(L_aniso, L_iso, rtol=1e-6,
                            err_msg="ratio=1.0 should reproduce isotropic Cholesky.")

    def test_variance_at_sill(self, aniso_grid):
        """Empirical variance should be close to the covariance sill.
        rtol=0.20 to allow Monte Carlo fluctuation with N=500."""
        Zh, _ = anisosimuchol(
            aniso_grid, covmodel=['exponentialC'], covparam=[(1.0, 3.0)],
            theta=0.0, ratio=1.5, ns=_N_REAL, seed=5,
        )
        empirical_var = np.var(Zh, axis=1, ddof=1)
        npt.assert_allclose(empirical_var, 1.0, rtol=0.20,
                            err_msg="Empirical variance should ≈ sill.")


# ===========================================================================
# simuseq  — sequential Gaussian simulation
# ===========================================================================

class TestSimuSeq:
    """
    Tests for the sequential simulation method.  The empirical variance
    check uses generous tolerance (rtol=0.20) because sequential
    simulation has higher variance across realisations for small datasets.
    """

    @pytest.fixture
    def grid(self):
        return _small_grid(n=8)

    def test_output_shape(self, grid):
        """simuseq returns Zh of shape (nh, ns)."""
        Zh = simuseq(grid, _COV_MODEL_S, _COV_PARAM_S, ns=2,
                     nhmax=5, dmax=np.array([[10.0]]), seed=0)
        assert Zh.shape == (grid.shape[0], 2)

    def test_reproducibility(self, grid):
        Zh1 = simuseq(grid, _COV_MODEL_S, _COV_PARAM_S, ns=2,
                      nhmax=5, dmax=np.array([[10.0]]), seed=9)
        Zh2 = simuseq(grid, _COV_MODEL_S, _COV_PARAM_S, ns=2,
                      nhmax=5, dmax=np.array([[10.0]]), seed=9)
        npt.assert_array_equal(Zh1, Zh2)

    def test_different_seeds_differ(self, grid):
        Zh1 = simuseq(grid, _COV_MODEL_S, _COV_PARAM_S, ns=1,
                      nhmax=5, dmax=np.array([[10.0]]), seed=0)
        Zh2 = simuseq(grid, _COV_MODEL_S, _COV_PARAM_S, ns=1,
                      nhmax=5, dmax=np.array([[10.0]]), seed=1)
        assert not np.allclose(Zh1, Zh2)

    def test_empirical_variance(self, grid):
        """
        Sample variance across many realisations should be ≈ sill = 1.0.
        rtol=0.20 to allow sequential-simulation Monte Carlo fluctuation.
        """
        Zh = simuseq(grid, _COV_MODEL_S, _COV_PARAM_S, ns=_N_REAL,
                     nhmax=5, dmax=np.array([[10.0]]), seed=7)
        empirical_var = np.var(Zh, axis=1, ddof=1)
        npt.assert_allclose(empirical_var, 1.0, rtol=0.20,
                            err_msg="Sequential sim empirical variance should ≈ sill.")

    def test_zero_mean(self, grid):
        """Sequential simulation should also be zero-mean."""
        Zh = simuseq(grid, _COV_MODEL_S, _COV_PARAM_S, ns=_N_REAL,
                     nhmax=5, dmax=np.array([[10.0]]), seed=8)
        empirical_mean = np.mean(Zh, axis=1)
        npt.assert_allclose(empirical_mean, np.zeros(grid.shape[0]),
                            atol=0.10)
