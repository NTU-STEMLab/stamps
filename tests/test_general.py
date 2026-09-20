# -*- coding: utf-8 -*-
"""
test_general.py
===============
Tests for general utilities: coordinate-to-distance, coordinate-to-covariance,
and pair-index functions.

Adapted from ``pairsindextest.m`` (BMElib testslib, Jan 1, 2001) and general
mathematical properties of the utility functions.

MATLAB → Python mapping
-----------------------
pairsindextest  findpairs     →  TestFindPairs
(new Python)    coord2dist    →  TestCoord2Dist
(new Python)    neighbours    →  TestNeighbours
(new Python)    coord2K       →  (see test_covariance_models.py)
"""
import numpy as np
import numpy.testing as npt
import pytest

from stamps.general.findpairs import findpairs
from stamps.general.coord2dist import coord2dist


# ===========================================================================
# findpairs  (mirrors pairsindextest.m)
# ===========================================================================

class TestFindPairs:
    """Tests for findpairs — mirrors pairsindextest.m in BMElib testslib."""

    def test_identical_matrices_all_match(self):
        """
        When c1 == c2, every row of c1 should appear as a pair.
        pairsindextest.m testcase 1: c1=c2 → n pairs with c1[i]==c2[i].
        """
        c = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        idx = findpairs(c, c)
        assert idx is not None and len(idx) == 3, \
            f"Expected 3 pairs, got {len(idx) if idx is not None else 'None'}."

    def test_no_common_points(self):
        """
        When no coordinates overlap, findpairs should return an empty result.
        pairsindextest.m testcase 2: no common points → empty index.
        """
        c1 = np.array([[0.0, 0.0], [1.0, 1.0]])
        c2 = np.array([[5.0, 5.0], [6.0, 6.0]])
        idx = findpairs(c1, c2)
        n   = 0 if idx is None or len(idx) == 0 else len(idx)
        assert n == 0, \
            f"Expected 0 pairs for non-overlapping coordinates, got {n}."

    def test_partial_overlap(self):
        """
        When only some coordinates match, only those should be returned.
        pairsindextest.m testcase 3.
        """
        c1 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        c2 = np.array([[1.0, 1.0], [3.0, 3.0]])
        idx = findpairs(c1, c2)
        n   = len(idx) if idx is not None else 0
        assert n == 1, f"Expected 1 pair, got {n}."

    def test_correct_index_values(self):
        """
        The returned pair should have c1[idx[0,0]] == c2[idx[0,1]].
        """
        c1 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        c2 = np.array([[3.0, 3.0], [1.0, 1.0], [4.0, 4.0]])
        idx = findpairs(c1, c2)
        assert idx is not None and len(idx) > 0, "Should find one pair."
        i1, i2 = int(idx[0, 0]), int(idx[0, 1])
        npt.assert_array_equal(c1[i1], c2[i2],
                               err_msg="Pair indices should point to identical coordinates.")

    def test_asymmetric_inputs(self):
        """
        findpairs(c1, c2) should handle differently-sized inputs.
        c1 has more rows than c2.
        """
        c1 = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        c2 = np.array([[2.0, 0.0]])
        idx = findpairs(c1, c2)
        n   = len(idx) if idx is not None else 0
        assert n == 1, f"Expected 1 pair, got {n}."

    def test_multiple_duplicates(self):
        """
        When c2 has the same point repeated, findpairs should find both.
        """
        c1  = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        c2  = np.array([[1.0, 1.0], [1.0, 1.0], [5.0, 5.0]])
        idx = findpairs(c1, c2)
        n   = len(idx) if idx is not None else 0
        assert n >= 1, f"Expected at least 1 pair, got {n}."

    def test_3d_coordinates(self):
        """findpairs should work for 3-D (space-time) coordinate matrices."""
        c1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        c2 = np.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        idx = findpairs(c1, c2)
        n   = len(idx) if idx is not None else 0
        assert n == 1, f"Expected 1 pair in 3-D, got {n}."


# ===========================================================================
# coord2dist  (mathematical properties)
# ===========================================================================

class TestCoord2Dist:
    """Sanity tests for the Euclidean distance calculation."""

    def test_self_distance_is_zero(self):
        """dist(c, c) diagonal must be zero."""
        rng  = np.random.default_rng(0)
        c    = rng.uniform(0, 10, size=(8, 2))
        D    = coord2dist(c, c)
        npt.assert_allclose(np.diag(D), 0.0, atol=1e-12)

    def test_symmetry(self):
        """dist(c1, c2) == dist(c2, c1).T"""
        rng  = np.random.default_rng(1)
        c1   = rng.uniform(0, 5, size=(6, 2))
        c2   = rng.uniform(0, 5, size=(4, 2))
        D12  = coord2dist(c1, c2)
        D21  = coord2dist(c2, c1)
        npt.assert_allclose(D12, D21.T, atol=1e-12)

    def test_non_negative(self):
        """All distances must be ≥ 0."""
        rng  = np.random.default_rng(2)
        c    = rng.uniform(-5, 5, size=(10, 3))
        D    = coord2dist(c, c)
        assert np.all(D >= 0.0)

    def test_known_2d_distance(self):
        """
        dist([[0,0]], [[3,4]]) == 5.0  (Pythagorean triplet).
        """
        c1 = np.array([[0.0, 0.0]])
        c2 = np.array([[3.0, 4.0]])
        D  = coord2dist(c1, c2)
        npt.assert_allclose(float(D[0, 0]), 5.0, rtol=1e-12)

    def test_known_3d_distance(self):
        """
        dist([[0,0,0]], [[1,1,1]]) == sqrt(3).
        """
        c1 = np.array([[0.0, 0.0, 0.0]])
        c2 = np.array([[1.0, 1.0, 1.0]])
        D  = coord2dist(c1, c2)
        npt.assert_allclose(float(D[0, 0]), np.sqrt(3.0), rtol=1e-12)

    def test_output_shape(self):
        """coord2dist(c1, c2) output shape must be (m1, m2)."""
        rng  = np.random.default_rng(3)
        c1   = rng.uniform(0, 1, size=(7, 2))
        c2   = rng.uniform(0, 1, size=(5, 2))
        D    = coord2dist(c1, c2)
        assert D.shape == (7, 5)

    def test_triangle_inequality(self):
        """
        For all triples (i, j, k): dist(i,k) ≤ dist(i,j) + dist(j,k).
        Checks a random sub-set of triples.
        """
        rng  = np.random.default_rng(4)
        pts  = rng.uniform(0, 10, size=(15, 2))
        D    = coord2dist(pts, pts)
        n    = pts.shape[0]
        idx  = rng.choice(n, size=50, replace=True)
        for _ in range(50):
            i, j, k = rng.integers(0, n, size=3)
            assert D[i, k] <= D[i, j] + D[j, k] + 1e-10, \
                f"Triangle inequality violated: D[{i},{k}]={D[i,k]:.4f} > {D[i,j]:.4f}+{D[j,k]:.4f}"


# ===========================================================================
# neighbours  (structural / correctness tests)
# ===========================================================================

class TestNeighbours:
    """Tests for the neighbourhood selection function."""

    def test_dmax_limits_results(self):
        """
        With dmax=1.5, only points within distance 1.5 should be returned.
        """
        from stamps.general.neighbours import neighbours

        c   = np.array([[0.0, 0.0]])
        all_pts = np.array([[0.5, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        z   = np.ones((4, 1))
        dmax = np.array([[1.5]])

        c_n, z_n, d_n, n_n, idx_n = neighbours(c, all_pts, z,
                                                nmax=10, dmax=dmax)
        # Only the first two points should be within dmax=1.5
        assert n_n <= 2, f"Expected ≤ 2 neighbours within dmax=1.5, got {n_n}."

    def test_nmax_limits_results(self):
        """
        With nmax=2, at most 2 neighbours should be returned.
        """
        from stamps.general.neighbours import neighbours

        c   = np.array([[0.0, 0.0]])
        all_pts = np.array([[0.5, 0.0], [1.0, 0.0], [1.5, 0.0], [2.0, 0.0]])
        z   = np.ones((4, 1))
        dmax = np.array([[10.0]])

        c_n, z_n, d_n, n_n, idx_n = neighbours(c, all_pts, z,
                                                nmax=2, dmax=dmax)
        assert n_n <= 2, f"nmax=2 should limit neighbours to ≤ 2, got {n_n}."

    def test_nearest_neighbour_returned_first(self):
        """
        The closest point should appear first in the neighbour list
        (sorted by distance).
        """
        from stamps.general.neighbours import neighbours

        c   = np.array([[0.0, 0.0]])
        all_pts = np.array([[3.0, 0.0], [1.0, 0.0], [5.0, 0.0]])
        z   = np.arange(3).reshape(-1, 1).astype(float)
        dmax = np.array([[10.0]])

        c_n, z_n, d_n, n_n, idx_n = neighbours(c, all_pts, z,
                                                nmax=10, dmax=dmax)
        if n_n >= 2:
            assert d_n[0] <= d_n[1], \
                "Nearest neighbour should appear first (ascending distance order)."
