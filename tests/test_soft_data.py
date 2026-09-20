# -*- coding: utf-8 -*-
"""
test_soft_data.py
=================
Tests for soft probabilistic data generation and manipulation.

Adapted from ``probaGenerationTest.m`` and ``combinedupliTest.m``
(BMElib testslib, Jan 1, 2001).

MATLAB → Python mapping
-----------------------
probaGenerationTest  probaGaussian   →  TestProbaGaussian
probaGenerationTest  probaStudentT   →  TestProbaStudentT
probaGenerationTest  probaUniform    →  TestProbaUniform
combinedupliTest     probacat        →  TestProbaCat
combinedupliTest     combinedupli    →  TestProbaCombinedDupli
"""
import numpy as np
import numpy.testing as npt
import pytest

from stamps.bme.softconverter import (
    probaGaussian,
    probaStudentT,
    probaUniform,
    probacat,
    probacombinedupli,
    proba2stat,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _trapz_integrate(nl, limi, probdens) -> np.ndarray:
    """
    Numerically integrate each soft PDF row using the trapezoidal rule.
    Returns an array of shape (ns,) — each element should be ≈ 1.0.
    """
    ns = nl.shape[0]
    areas = np.empty(ns)
    for i in range(ns):
        n_i  = int(nl[i, 0])
        x    = limi[i, :n_i]
        y    = probdens[i, :n_i]
        areas[i] = float(np.trapezoid(y, x))
    return areas


# ===========================================================================
# probaGaussian  (mirrors probaGenerationTest)
# ===========================================================================

class TestProbaGaussian:

    @pytest.fixture
    def gaussian_soft(self):
        rng  = np.random.default_rng(0)
        ns   = 5
        zm   = rng.uniform(0, 2, size=ns)
        zv   = rng.uniform(0.01, 0.5, size=ns)
        return probaGaussian(zm, zv), zm, zv

    def test_softpdftype_is_2(self, gaussian_soft):
        """probaGaussian returns softpdftype = 2 (piecewise-linear)."""
        (stype, nl, limi, probdens), _, _ = gaussian_soft
        assert stype == 2

    def test_nl_is_13(self, gaussian_soft):
        """Each Gaussian soft datum uses 13 quantile nodes."""
        (stype, nl, limi, probdens), _, _ = gaussian_soft
        assert np.all(nl == 13)

    def test_limi_shape(self, gaussian_soft):
        """limi must have shape (ns, 13)."""
        (stype, nl, limi, probdens), zm, _ = gaussian_soft
        assert limi.shape == (len(zm), 13)

    def test_probdens_shape(self, gaussian_soft):
        """probdens must have shape (ns, 13)."""
        (stype, nl, limi, probdens), zm, _ = gaussian_soft
        assert probdens.shape == (len(zm), 13)

    def test_normalises_to_one(self, gaussian_soft):
        """Trapezoidal integral of each pdf row ≈ 1.0 (rtol=0.01)."""
        (stype, nl, limi, probdens), _, _ = gaussian_soft
        areas = _trapz_integrate(nl, limi, probdens)
        npt.assert_allclose(areas, np.ones_like(areas), rtol=0.01,
                            err_msg="probaGaussian PDFs should normalise to 1.")

    def test_mean_via_proba2stat(self, gaussian_soft):
        """proba2stat mean should recover zm (rtol=0.05 for piecewise approx)."""
        (stype, nl, limi, probdens), zm, zv = gaussian_soft
        means, _ = proba2stat(stype, nl, limi, probdens)
        npt.assert_allclose(means.ravel(), zm, rtol=0.05,
                            err_msg="proba2stat mean should match zm.")

    def test_var_via_proba2stat(self, gaussian_soft):
        """proba2stat variance should recover zv (rtol=0.10 for piecewise approx)."""
        (stype, nl, limi, probdens), zm, zv = gaussian_soft
        _, variances = proba2stat(stype, nl, limi, probdens)
        npt.assert_allclose(variances.ravel(), zv, rtol=0.10,
                            err_msg="proba2stat variance should match zv.")

    def test_limi_increases(self, gaussian_soft):
        """Each row of limi must be strictly increasing (valid support)."""
        (stype, nl, limi, probdens), zm, _ = gaussian_soft
        for i in range(limi.shape[0]):
            n_i = int(13)
            row = limi[i, :n_i]
            assert np.all(np.diff(row) > 0), \
                f"Row {i} of limi is not strictly increasing: {row}"

    def test_probdens_non_negative(self, gaussian_soft):
        """All probability density values must be ≥ 0."""
        (stype, nl, limi, probdens), _, _ = gaussian_soft
        assert np.all(probdens >= 0.0), "probdens should be non-negative."


# ===========================================================================
# probaStudentT  (mirrors probaGenerationTest)
# ===========================================================================

class TestProbaStudentT:

    @pytest.fixture
    def student_soft(self):
        rng   = np.random.default_rng(0)
        ns    = 4
        nobvs = np.ceil(2 + rng.uniform(0, 5, size=ns)).astype(int)
        zm    = rng.uniform(0, 2, size=ns)
        zv    = rng.uniform(0.01, 0.5, size=ns)
        return probaStudentT(zm, zv, nobvs), zm, zv, nobvs

    def test_softpdftype_is_2(self, student_soft):
        (stype, nl, limi, probdens), *_ = student_soft
        assert stype == 2

    def test_normalises_to_one(self, student_soft):
        """Student-T soft PDFs must also integrate to ≈ 1.0."""
        (stype, nl, limi, probdens), *_ = student_soft
        areas = _trapz_integrate(nl, limi, probdens)
        npt.assert_allclose(areas, np.ones_like(areas), rtol=0.01,
                            err_msg="probaStudentT PDFs should normalise to 1.")

    def test_heavier_tails_than_gaussian(self, student_soft):
        """
        For the same mean/variance, the Student-T 0.95-quantile should
        be farther from the mean than the Gaussian 0.95-quantile when
        nobvs is small (heavy tails for low dof).
        """
        (stype_t, nl_t, limi_t, pd_t), zm, zv, nobvs = student_soft
        stype_g, nl_g, limi_g, pd_g = probaGaussian(zm, zv / nobvs)

        # Check at least one datum with < 5 observations
        for i in range(len(zm)):
            if int(nobvs[i]) < 5:
                # Student-T should span a wider range than Gaussian
                range_t = float(limi_t[i, -1] - limi_t[i, 0])
                range_g = float(limi_g[i, -1] - limi_g[i, 0])
                assert range_t >= range_g, (
                    f"Student-T (dof={nobvs[i]-1}) should have wider "
                    f"support than Gaussian at point {i}."
                )
                break   # one example is sufficient

    def test_mean_via_proba2stat(self, student_soft):
        """proba2stat mean should recover zm for Student-T."""
        (stype, nl, limi, probdens), zm, zv, nobvs = student_soft
        means, _ = proba2stat(stype, nl, limi, probdens)
        npt.assert_allclose(means.ravel(), zm, rtol=0.05)


# ===========================================================================
# probaUniform  (mirrors probaGenerationTest)
# ===========================================================================

class TestProbaUniform:

    @pytest.fixture
    def uniform_soft(self):
        rng  = np.random.default_rng(0)
        ns   = 6
        zlow = rng.uniform(-1, 0, size=ns)
        zup  = zlow + rng.uniform(0.2, 2.0, size=ns)
        # probaUniform returns a list [softpdftype, nl, limi, probdens]
        result = probaUniform(zlow, zup)
        return result, zlow, zup

    def test_softpdftype_is_2(self, uniform_soft):
        """probaUniform returns softpdftype = 2."""
        (stype, nl, limi, probdens), *_ = uniform_soft
        assert stype == 2

    def test_nl_is_2(self, uniform_soft):
        """Uniform uses 2 limit nodes per datum."""
        (stype, nl, limi, probdens), *_ = uniform_soft
        assert np.all(nl == 2)

    def test_bounds_match(self, uniform_soft):
        """limi[:,0] == zlow and limi[:,1] == zup."""
        (stype, nl, limi, probdens), zlow, zup = uniform_soft
        npt.assert_allclose(limi[:, 0], zlow, rtol=1e-12)
        npt.assert_allclose(limi[:, 1], zup,  rtol=1e-12)

    def test_density_is_flat(self, uniform_soft):
        """The two density values per row should be equal (uniform)."""
        (stype, nl, limi, probdens), zlow, zup = uniform_soft
        npt.assert_allclose(probdens[:, 0], probdens[:, 1], rtol=1e-10,
                            err_msg="Uniform PDF should be flat.")

    def test_density_equals_inverse_width(self, uniform_soft):
        """Uniform density = 1/(zup - zlow)."""
        (stype, nl, limi, probdens), zlow, zup = uniform_soft
        expected = 1.0 / (zup - zlow)
        npt.assert_allclose(probdens[:, 0], expected, rtol=1e-10)

    def test_normalises_to_one(self, uniform_soft):
        """Trapezoidal integral of each uniform row ≈ 1.0."""
        (stype, nl, limi, probdens), *_ = uniform_soft
        areas = _trapz_integrate(nl, limi, probdens)
        npt.assert_allclose(areas, np.ones_like(areas), rtol=1e-10)


# ===========================================================================
# probacat  (mirrors combinedupliTest)
# ===========================================================================

class TestProbaCat:

    @pytest.fixture
    def two_sets(self):
        """One set of Uniform and one set of Gaussian soft data."""
        rng  = np.random.default_rng(2)
        nsU  = 3
        nsG  = 4
        zlow = rng.uniform(-1, 0, size=nsU)
        zup  = zlow + rng.uniform(0.2, 1.5, size=nsU)
        zm   = rng.uniform(0, 2, size=nsG)
        zv   = rng.uniform(0.05, 0.3, size=nsG)

        stype_u, nl_u, limi_u, pd_u = probaUniform(zlow, zup)
        stype_g, nl_g, limi_g, pd_g = probaGaussian(zm, zv)
        return (stype_u, nl_u, limi_u, pd_u,
                stype_g, nl_g, limi_g, pd_g, nsU, nsG)

    def test_output_row_count(self, two_sets):
        """probacat output must have nsU + nsG rows."""
        st_u, nl_u, li_u, pd_u, st_g, nl_g, li_g, pd_g, nsU, nsG = two_sets
        stype_c, nl_c, limi_c, pd_c = probacat(
            st_u, nl_u, li_u, pd_u,
            st_g, nl_g, li_g, pd_g,
        )
        assert nl_c.shape[0] == nsU + nsG, \
            f"Expected {nsU + nsG} rows, got {nl_c.shape[0]}."

    def test_output_softpdftype(self, two_sets):
        """Concatenated set should keep softpdftype = 2."""
        st_u, nl_u, li_u, pd_u, st_g, nl_g, li_g, pd_g, *_ = two_sets
        stype_c, *_ = probacat(
            st_u, nl_u, li_u, pd_u,
            st_g, nl_g, li_g, pd_g,
        )
        assert stype_c == 2

    def test_first_rows_preserved(self, two_sets):
        """The first nsU rows of the concatenated nl should match nl_u."""
        st_u, nl_u, li_u, pd_u, st_g, nl_g, li_g, pd_g, nsU, _ = two_sets
        stype_c, nl_c, limi_c, pd_c = probacat(
            st_u, nl_u, li_u, pd_u,
            st_g, nl_g, li_g, pd_g,
        )
        npt.assert_array_equal(nl_c[:nsU], nl_u)

    def test_last_rows_preserved(self, two_sets):
        """The last nsG rows of the concatenated nl should match nl_g."""
        st_u, nl_u, li_u, pd_u, st_g, nl_g, li_g, pd_g, nsU, nsG = two_sets
        stype_c, nl_c, limi_c, pd_c = probacat(
            st_u, nl_u, li_u, pd_u,
            st_g, nl_g, li_g, pd_g,
        )
        npt.assert_array_equal(nl_c[nsU:], nl_g)


# ===========================================================================
# probacombinedupli  (mirrors combinedupliTest)
# ===========================================================================

class TestProbaCombinedDupli:
    """
    Test that probacombinedupli correctly merges duplicated soft data.
    """

    @pytest.fixture
    def soft_with_dupes(self):
        """
        Build a soft dataset (Gaussian) where a cluster of 3 rows share
        identical coordinates, mimicking combinedupliTest.m.
        """
        rng  = np.random.default_rng(2)
        ns   = 8
        d    = 2
        cs   = rng.uniform(0, 1, size=(ns, d))

        # Force rows 2, 3, 4 to be duplicates of row 1
        cs[2, :] = cs[1, :]
        cs[3, :] = cs[1, :]
        cs[4, :] = cs[1, :]

        zm  = rng.uniform(0, 2, size=ns)
        zv  = rng.uniform(0.05, 0.3, size=ns)
        stype, nl, limi, probdens = probaGaussian(zm, zv)
        return cs, stype, nl, limi, probdens, ns

    def test_no_duplicate_coords_after_merge(self, soft_with_dupes):
        """
        After probacombinedupli, csC should have no repeated rows.
        """
        cs, stype, nl, limi, probdens, ns = soft_with_dupes
        csC, nlC, limiC, pdC, _ = probacombinedupli(cs, stype, nl, limi, probdens)

        # Check uniqueness of coordinates
        nC = csC.shape[0]
        for i in range(nC):
            for j in range(i + 1, nC):
                assert not np.allclose(csC[i], csC[j]), \
                    f"Duplicate coordinates found at rows {i} and {j} after merge."

    def test_output_row_count_reduced(self, soft_with_dupes):
        """nsC must be strictly less than ns when there are duplicates."""
        cs, stype, nl, limi, probdens, ns = soft_with_dupes
        csC, nlC, *_ = probacombinedupli(cs, stype, nl, limi, probdens)
        assert csC.shape[0] < ns, \
            "probacombinedupli should reduce the number of rows."

    def test_merged_pdf_normalises(self, soft_with_dupes):
        """Each merged soft PDF row should integrate to ≈ 1.0."""
        cs, stype, nl, limi, probdens, ns = soft_with_dupes
        csC, nlC, limiC, pdC, _ = probacombinedupli(cs, stype, nl, limi, probdens)
        areas = _trapz_integrate(nlC, limiC, pdC)
        npt.assert_allclose(areas, np.ones_like(areas), rtol=0.02,
                            err_msg="Merged soft PDFs should integrate to 1.")

    def test_additive_method_runs(self, soft_with_dupes):
        """probacombinedupli with method='add' should complete without error."""
        cs, stype, nl, limi, probdens, ns = soft_with_dupes
        csC, nlC, limiC, pdC, _ = probacombinedupli(
            cs, stype, nl, limi, probdens, method='add')
        assert csC.shape[0] < ns

    def test_auto_method_runs(self, soft_with_dupes):
        """probacombinedupli with method='auto' should complete without error."""
        cs, stype, nl, limi, probdens, ns = soft_with_dupes
        csC, nlC, limiC, pdC, _ = probacombinedupli(
            cs, stype, nl, limi, probdens, method='auto')
        assert csC.shape[0] < ns
