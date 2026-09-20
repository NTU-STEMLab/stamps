# -*- coding: utf-8 -*-
"""
test_bme_transform.py
=====================
Tests for the Normal Score Transform and related CDF-transformation
utilities in ``stamps.bme.bme_transform``.

No direct MATLAB equivalent in testslib — these tests are new.
They validate the mathematical properties listed in the bme_transform
docstrings.

Scope
-----
* ``other2gauss``          — forward NST
* ``pdfgauss2other``       — back-transformation of probability density
* ``probaother2gauss``     — soft-data transformation to Gaussian domain
* ``gausspdf`` / ``gaussinv`` — Gaussian utilities
* ``transformderiv``       — CDF derivative (change-of-variable Jacobian)
* Round-trip tests (other2gauss → gaussinv → check identity)
"""
import numpy as np
import numpy.testing as npt
import pytest
from scipy import stats as sp_stats

from stamps.bme.bme_transform import (
    other2gauss,
    pdfgauss2other,
    gausspdf,
    gaussinv,
    transformderiv,
    probaother2gauss,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tabulated_cdf(dist, n: int = 200, nsig: float = 4.0):
    """
    Build a (yfile, Fyfile) tabulation of *dist* CDF over ±nsig σ.
    Returns (yfile, Fyfile) as 1-D arrays.
    """
    loc  = dist.mean()
    scl  = dist.std()
    y    = np.linspace(loc - nsig * scl, loc + nsig * scl, n)
    Fy   = dist.cdf(y)
    # Ensure strictly increasing CDF values for interpolation
    Fy   = np.clip(Fy, 1e-10, 1 - 1e-10)
    return y, Fy


# ===========================================================================
# gausspdf and gaussinv
# ===========================================================================

class TestGaussianUtilities:
    """Smoke tests for gausspdf and gaussinv (numerical helpers).

    Both functions accept ``params = [mean, variance]`` as their second
    argument (not separate mean and variance positional arguments).
    """

    def test_gausspdf_standard_normal_at_zero(self):
        """gausspdf(0, [0, 1]) == 1/sqrt(2π)."""
        expected = 1.0 / np.sqrt(2.0 * np.pi)
        result   = gausspdf(np.array([0.0]), [0.0, 1.0])
        npt.assert_allclose(float(np.asarray(result).flat[0]), expected, rtol=1e-10)

    def test_gausspdf_shape_preserved(self):
        """gausspdf must return an array with the same shape as input x."""
        x   = np.linspace(-3, 3, 50)
        out = gausspdf(x, [0.5, 0.1])
        assert out.shape == x.shape

    def test_gausspdf_non_negative(self):
        """PDF values must be ≥ 0."""
        x   = np.linspace(-5, 5, 100)
        out = gausspdf(x, [0.0, 1.0])
        assert np.all(out >= 0.0)

    def test_gaussinv_quantile_recovery(self):
        """gaussinv(p, [0, 1]) should recover standard-normal quantiles."""
        probs    = np.array([0.01, 0.1, 0.5, 0.9, 0.99])
        expected = sp_stats.norm.ppf(probs)
        result   = gaussinv(probs, [0.0, 1.0])
        npt.assert_allclose(result, expected, rtol=1e-6)


# ===========================================================================
# other2gauss  (forward Normal Score Transform)
# ===========================================================================

class TestOther2Gauss:
    """Tests for the forward Normal Score Transform."""

    @pytest.fixture
    def log_normal_cdf(self):
        """
        Tabulate a log-normal CDF with mu=0, sigma=1 as the reference
        distribution.
        """
        dist = sp_stats.lognorm(s=1.0)
        y, Fy = _make_tabulated_cdf(dist, n=300, nsig=3.0)
        # Make sure Fy starts above 0 and ends below 1
        Fy = np.clip(Fy, 1e-8, 1 - 1e-8)
        return y, Fy

    def test_returns_array_same_shape(self, log_normal_cdf):
        """other2gauss returns an array with the same shape as input y."""
        y_ref, Fy_ref = log_normal_cdf
        y_test = y_ref[10:20]
        z      = other2gauss(y_test, y_ref, Fy_ref)
        assert z.shape == y_test.shape

    def test_monotone_increasing(self, log_normal_cdf):
        """The transform z = Φ^{-1}(F(y)) must be monotone-increasing in y."""
        y_ref, Fy_ref = log_normal_cdf
        # Use interior values to avoid NaN from edge effects
        y_test = np.linspace(y_ref[5], y_ref[-6], 50)
        z      = other2gauss(y_test, y_ref, Fy_ref)
        # Drop NaNs at boundaries
        mask = np.isfinite(z)
        assert np.all(np.diff(z[mask]) > -1e-10), \
            "other2gauss should produce a monotone-increasing sequence."

    def test_median_maps_to_zero(self, log_normal_cdf):
        """
        F_Y(median) = 0.5  →  Φ^{-1}(0.5) = 0.
        For log-normal(0, 1) the median is exp(0) = 1.
        """
        y_ref, Fy_ref = log_normal_cdf
        z = other2gauss(np.array([1.0]), y_ref, Fy_ref)
        npt.assert_allclose(float(z[0]), 0.0, atol=0.05,
                            err_msg="Median of prior should map to z = 0.")

    def test_outside_domain_is_nan(self, log_normal_cdf):
        """Values outside [min(yfile), max(yfile)] must be NaN."""
        y_ref, Fy_ref = log_normal_cdf
        y_out = np.array([-1e6, 1e6])
        z     = other2gauss(y_out, y_ref, Fy_ref)
        assert np.all(np.isnan(z)), \
            "Values outside tabulated domain should map to NaN."


# ===========================================================================
# pdfgauss2other  (back-transformation of PDF)
# ===========================================================================

class TestPdfgauss2Other:
    """Tests for the change-of-variable back-transformation."""

    def test_pdf_integrates_to_one(self):
        """
        ∫ f_Y(y) dy ≈ 1.0  after back-transforming a standard Gaussian PDF.
        This verifies the Jacobian |dz/dy| is correctly applied.
        """
        # Build a normal CDF tabulation of a Gaussian Y ~ N(1.0, 0.25)
        mean_y, var_y = 1.0, 0.25
        dist  = sp_stats.norm(loc=mean_y, scale=np.sqrt(var_y))
        y_ref, Fy_ref = _make_tabulated_cdf(dist, n=400, nsig=4.5)

        # Compute the z values for each y
        z = other2gauss(y_ref, y_ref, Fy_ref)
        mask = np.isfinite(z)
        y_in = y_ref[mask]
        z_in = z[mask]

        # Gaussian PDF in z-domain (posterior in Gaussian space = N(0,1))
        zpdf_in = sp_stats.norm.pdf(z_in)

        # Back-transform to y-domain
        fy = pdfgauss2other(y_in, z_in, zpdf_in, y_ref, Fy_ref)

        # Integrate
        area = float(np.trapezoid(fy, y_in))
        npt.assert_allclose(area, 1.0, rtol=0.05,
                            err_msg="Back-transformed PDF should integrate to 1.")

    def test_back_transform_non_negative(self):
        """f_Y(y) must be ≥ 0 everywhere."""
        mean_y, var_y = 0.5, 0.1
        dist  = sp_stats.norm(loc=mean_y, scale=np.sqrt(var_y))
        y_ref, Fy_ref = _make_tabulated_cdf(dist, n=200, nsig=4.0)
        z = other2gauss(y_ref, y_ref, Fy_ref)
        mask = np.isfinite(z)
        fy = pdfgauss2other(
            y_ref[mask], z[mask],
            sp_stats.norm.pdf(z[mask]),
            y_ref, Fy_ref,
        )
        assert np.all(fy >= -1e-12), "Back-transformed PDF should be non-negative."


# ===========================================================================
# transformderiv  (CDF derivative — Jacobian)
# ===========================================================================

class TestTransformDeriv:
    """Tests for the dY/dZ Jacobian used in back-transformations.

    ``transformderiv(yfile, Fyfile)`` takes a CDF tabulation and returns
    the derivative dY/dZ at each tabulation node.  It does **not** accept
    a separate set of evaluation points as a third argument.
    """

    def test_derivative_positive(self):
        """The dY/dZ derivative must be positive (transform is monotone)."""
        mean_y, var_y = 0.0, 1.0
        dist  = sp_stats.norm(loc=mean_y, scale=np.sqrt(var_y))
        y_ref, Fy_ref = _make_tabulated_cdf(dist, n=200, nsig=3.5)
        deriv  = transformderiv(y_ref, Fy_ref)
        # Interior nodes only (boundary FD is one-sided and can be noisier)
        assert np.all(deriv[1:-1] > 0.0), \
            "Interior dY/dZ values should be strictly positive."

    def test_derivative_matches_finite_difference(self):
        """
        dY/dZ ≈ Δy/Δz  (finite-difference verification at interior nodes).
        transformderiv uses central FD on (z, y) pairs; this test
        cross-checks it against np.gradient applied the same way.
        rtol=0.10 to allow for numerical noise at the boundaries.
        """
        from scipy.stats import norm as _norm
        mean_y, var_y = 0.0, 1.0
        dist  = sp_stats.norm(loc=mean_y, scale=np.sqrt(var_y))
        y_ref, Fy_ref = _make_tabulated_cdf(dist, n=500, nsig=3.0)
        z_ref = _norm.ppf(Fy_ref)
        # Expected: dY/dZ computed via np.gradient on the tabulation pairs
        fd_expected = np.gradient(y_ref, z_ref)
        deriv = transformderiv(y_ref, Fy_ref)
        # Compare only interior nodes (first and last use one-sided FD)
        npt.assert_allclose(deriv[1:-1], fd_expected[1:-1], rtol=0.10,
                            err_msg="transformderiv should match finite difference.")


# ===========================================================================
# probaother2gauss  (soft-data transformation to Gaussian domain)
# ===========================================================================

class TestProbaOther2Gauss:
    """Verifies that soft data in Y-space transforms correctly to Z-space."""

    def test_output_structure(self):
        """
        probaother2gauss returns (softpdftype, nl, limi, probdens) for the
        transformed soft data.  Basic shape checks.
        """
        from stamps.bme.softconverter import probaGaussian

        # Use a Gaussian Y-distribution as the reference (NST maps Gaussian → Gaussian)
        mean_y, var_y = 1.0, 0.25
        dist  = sp_stats.norm(loc=mean_y, scale=np.sqrt(var_y))
        y_ref, Fy_ref = _make_tabulated_cdf(dist, n=300, nsig=4.5)

        ns = 3
        zm = np.ones(ns) * mean_y
        zv = np.ones(ns) * var_y
        stype, nl, limi, probdens = probaGaussian(zm, zv)

        stype_z, nl_z, limi_z, pd_z = probaother2gauss(
            stype, nl, limi, probdens, y_ref, Fy_ref
        )
        assert stype_z in (1, 2, 3, 4), \
            "probaother2gauss should return a valid softpdftype."
        assert nl_z.shape[0] == ns, \
            "nl_z should have one row per soft datum."
        assert limi_z.shape[0] == ns, \
            "limi_z should have one row per soft datum."

    def test_transformed_pdf_integrates_to_one(self):
        """
        The transformed PDF in Z-space should integrate to ≈ 1.0.
        """
        from stamps.bme.softconverter import probaGaussian

        mean_y, var_y = 0.5, 0.1
        dist  = sp_stats.norm(loc=mean_y, scale=np.sqrt(var_y))
        y_ref, Fy_ref = _make_tabulated_cdf(dist, n=300, nsig=4.5)

        ns = 2
        zm = np.ones(ns) * mean_y
        zv = np.ones(ns) * var_y
        stype, nl, limi, probdens = probaGaussian(zm, zv)

        stype_z, nl_z, limi_z, pd_z = probaother2gauss(
            stype, nl, limi, probdens, y_ref, Fy_ref
        )
        # Trapezoidal integration for each row
        for i in range(ns):
            n_i = int(nl_z[i, 0])
            x   = limi_z[i, :n_i]
            y   = pd_z[i, :n_i]
            area = float(np.trapezoid(y, x))
            npt.assert_allclose(area, 1.0, rtol=0.05,
                                err_msg=f"Row {i}: transformed PDF integral = {area:.4f}.")
