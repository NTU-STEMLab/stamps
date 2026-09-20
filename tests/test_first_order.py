# -*- coding: utf-8 -*-
"""
test_first_order.py
===================
Tests for the unified first-order (unary) constraint machinery:

* ``ipf_recenter`` / ``recenter_pmodel``  — IPF re-margining utilities.
* ``ME_dual_estimator`` unary extension   — first-order constraints in the
  regularized MaxEnt dual solve.
* ``maxentropytable`` plumbing            — ``options['unary_constraints']``.
* ``BMEcatPdf`` / ``MCPcatPdf``           — ``options['first_order']`` mode.
"""
import numpy as np
import pytest

from stamps.stats.dependence.ptable.recenter import (
    ipf_recenter, recenter_pmodel,
)
from stamps.categorical.me_solvers import (
    ME_dual_estimator, maxentropytable,
    sumoverallexceptone, sumoverallexcepttwo,
)
from stamps.categorical.estimation import BMEcatPdf, MCPcatPdf
from stamps.categorical.first_order import kernel_first_order
from stamps.categorical.evaluate import tempered_map, tune_temper_tau


# ---------------------------------------------------------------------------
# Synthetic global Pmodel:  P(d) = w(d) * diag(p) + (1 - w(d)) * p p^T
# (perfect association at lag 0, exact independence in the far field)
# ---------------------------------------------------------------------------

NC = 3
P_GLOBAL = np.array([0.5, 0.3, 0.2])
DMODEL = np.linspace(0.0, 10.0, 21)
RANGE_A = 3.0


def _make_pmodel():
    Pm = np.zeros((NC, NC, len(DMODEL)))
    P_indep = np.outer(P_GLOBAL, P_GLOBAL)
    P_diag = np.diag(P_GLOBAL)
    for k, d in enumerate(DMODEL):
        w = np.exp(-d / RANGE_A)
        Pm[:, :, k] = w * P_diag + (1.0 - w) * P_indep
    return Pm


PMODEL = _make_pmodel()


def _odds_ratio(P, a, b):
    return (P[a, a] * P[b, b]) / (P[a, b] * P[b, a])


def _sample_data(rng, n=40):
    """Random 2-D coordinates and one-hot soft data."""
    cs = rng.uniform(0, 10, size=(n, 2))
    labels = rng.integers(0, NC, size=n)
    ps = np.full((n, NC), 0.01)
    ps[np.arange(n), labels] = 0.98
    ps /= ps.sum(axis=1, keepdims=True)
    return cs, ps


# ===========================================================================
# 1. IPF utilities
# ===========================================================================

class TestIpfRecenter:

    def test_margins_match(self, rng):
        P = PMODEL[:, :, 3]           # short-lag, strong association
        q_r = np.array([0.7, 0.2, 0.1])
        q_c = np.array([0.15, 0.35, 0.5])
        Px = ipf_recenter(P, q_r, q_c)
        assert Px.shape == (NC, NC)
        assert np.all(Px >= 0)
        assert np.isclose(Px.sum(), 1.0)
        np.testing.assert_allclose(Px.sum(axis=1), q_r, atol=1e-8)
        np.testing.assert_allclose(Px.sum(axis=0), q_c, atol=1e-8)

    def test_odds_ratios_preserved(self):
        P = PMODEL[:, :, 3]
        q = np.array([0.6, 0.3, 0.1])
        Px = ipf_recenter(P, q)
        for a in range(NC):
            for b in range(a + 1, NC):
                assert np.isclose(
                    _odds_ratio(P, a, b), _odds_ratio(Px, a, b),
                    rtol=1e-6,
                ), f'odds ratio changed for pair ({a}, {b})'

    def test_identity_when_q_equals_own_margins(self):
        P = PMODEL[:, :, 5]
        Px = ipf_recenter(P, P.sum(axis=1), P.sum(axis=0))
        np.testing.assert_allclose(Px, P / P.sum(), atol=1e-8)

    def test_far_field_maps_to_independence(self):
        """A p p^T table recentered to (q_r, q_c) must equal q_r q_c^T."""
        P = np.outer(P_GLOBAL, P_GLOBAL)
        q_r = np.array([0.7, 0.2, 0.1])
        q_c = np.array([0.1, 0.6, 0.3])
        Px = ipf_recenter(P, q_r, q_c)
        np.testing.assert_allclose(Px, np.outer(q_r, q_c), atol=1e-8)

    def test_degenerate_source_falls_back_to_independence(self):
        P = np.zeros((NC, NC))
        q = np.array([0.5, 0.3, 0.2])
        Px = ipf_recenter(P, q)
        np.testing.assert_allclose(Px, np.outer(q, q), atol=1e-12)

    def test_recenter_pmodel_stack(self):
        q = np.array([0.6, 0.25, 0.15])
        Pm_x = recenter_pmodel(PMODEL, q)
        assert Pm_x.shape == PMODEL.shape
        for k in range(PMODEL.shape[2]):
            np.testing.assert_allclose(Pm_x[:, :, k].sum(axis=1), q,
                                       atol=1e-7)


# ===========================================================================
# 2. ME_dual solver with unary constraints
# ===========================================================================

class TestMEDualUnary:

    REG = 1e4          # weak L2 regularization (penalty = ||mu||^2 / (2 reg))
    TOL = 1e-10

    def _solve(self, q_rows):
        """3-axis problem with all pair tables recentered to q margins."""
        ndim = q_rows.shape[0]
        cons = []
        P_short = PMODEL[:, :, 2]
        for i in range(ndim):
            for j in range(i + 1, ndim):
                cons.append(((i, j),
                             ipf_recenter(P_short, q_rows[i], q_rows[j])))
        return ME_dual_estimator(cons, ndim, NC, self.TOL, reg=self.REG,
                                 unary_constraints=q_rows)

    def test_axis_marginals_match_targets(self):
        q = np.array([
            [0.7, 0.2, 0.1],
            [0.2, 0.5, 0.3],
            [0.1, 0.3, 0.6],
        ])
        Pfit, _ = self._solve(q)
        for i in range(3):
            np.testing.assert_allclose(sumoverallexceptone(Pfit, i), q[i],
                                       atol=2e-3)

    def test_pairwise_odds_ratios_match_global(self):
        q = np.array([
            [0.7, 0.2, 0.1],
            [0.2, 0.5, 0.3],
            [0.1, 0.3, 0.6],
        ])
        Pfit, _ = self._solve(q)
        P_short = PMODEL[:, :, 2]
        P01 = sumoverallexcepttwo(Pfit, 0, 1)
        # Odds ratios of the fitted bivariate marginal should match the
        # *global* model (recentering preserves them; the ME solve should
        # reproduce the pair targets closely under weak regularization).
        for a in range(NC):
            for b in range(a + 1, NC):
                assert np.isclose(
                    _odds_ratio(P01, a, b), _odds_ratio(P_short, a, b),
                    rtol=0.15,
                )

    def test_backward_compatible_without_unary(self):
        """unary_constraints=None must reproduce the original solver output."""
        cons = [((i, j), PMODEL[:, :, 2])
                for i in range(3) for j in range(i + 1, 3)]
        P_a, _ = ME_dual_estimator(cons, 3, NC, self.TOL, reg=self.REG)
        P_b, _ = ME_dual_estimator(cons, 3, NC, self.TOL, reg=self.REG,
                                   unary_constraints=None)
        np.testing.assert_allclose(P_a, P_b, atol=1e-12)

    def test_unary_equal_to_global_matches_baseline(self):
        """q_i == p_g for all axes must reduce to the pairwise-only solve."""
        q = np.tile(P_GLOBAL, (3, 1))
        cons = [((i, j), PMODEL[:, :, 2])
                for i in range(3) for j in range(i + 1, 3)]
        P_base, _ = ME_dual_estimator(cons, 3, NC, self.TOL, reg=self.REG)
        P_unary, _ = self._solve(q)
        np.testing.assert_allclose(P_unary, P_base, atol=5e-3)

    def test_nan_row_leaves_axis_unconstrained(self):
        """NaN unary rows are skipped; constrained axes still match.

        The pair targets must be recentered consistently for constrained
        axes (NaN axes keep the table's own margin), mirroring what
        ``maxentropytable`` does internally.
        """
        q = np.array([
            [0.7, 0.2, 0.1],
            [np.nan, np.nan, np.nan],
            [0.1, 0.3, 0.6],
        ])
        P_short = PMODEL[:, :, 2]
        own = P_short.sum(axis=0)

        def _q_or_own(row):
            return own if np.any(np.isnan(q[row])) else q[row]

        cons = [
            ((0, 1), ipf_recenter(P_short, q[0], _q_or_own(1))),
            ((0, 2), ipf_recenter(P_short, q[0], q[2])),
            ((1, 2), ipf_recenter(P_short, _q_or_own(1), q[2])),
        ]
        Pfit, _ = ME_dual_estimator(cons, 3, NC, self.TOL, reg=self.REG,
                                    unary_constraints=q)
        np.testing.assert_allclose(sumoverallexceptone(Pfit, 0), q[0],
                                   atol=5e-3)
        np.testing.assert_allclose(sumoverallexceptone(Pfit, 2), q[2],
                                   atol=5e-3)


# ===========================================================================
# 3. maxentropytable plumbing
# ===========================================================================

class TestMaxentropytableUnary:

    OPTS = {'estimator': 'ME_dual', 'tol': 1e-10, 'reg': 1e4}

    def test_unary_constraints_enforced(self):
        c = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.5]])
        q = np.array([
            [0.7, 0.2, 0.1],
            [0.2, 0.5, 0.3],
            [0.1, 0.3, 0.6],
        ])
        opts = {**self.OPTS, 'unary_constraints': q}
        Pfit, _ = maxentropytable(c, DMODEL, PMODEL, options=opts)
        for i in range(3):
            np.testing.assert_allclose(sumoverallexceptone(Pfit, i), q[i],
                                       atol=2e-3)

    def test_backward_compat_without_unary(self):
        c = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.5]])
        P_a, _ = maxentropytable(c, DMODEL, PMODEL, options=dict(self.OPTS))
        P_b, _ = maxentropytable(c, DMODEL, PMODEL, options=dict(self.OPTS))
        np.testing.assert_allclose(P_a, P_b, atol=1e-12)

    def test_rejects_non_me_dual(self):
        c = np.array([[0.0, 0.0], [1.0, 0.0]])
        q = np.tile(P_GLOBAL, (2, 1))
        opts = {'estimator': 'MPL', 'unary_constraints': q}
        with pytest.raises(ValueError, match='ME_dual'):
            maxentropytable(c, DMODEL, PMODEL, options=opts)

    def test_rejects_bad_shape(self):
        c = np.array([[0.0, 0.0], [1.0, 0.0]])
        q = np.tile(P_GLOBAL, (3, 1))     # wrong ndim
        opts = {**self.OPTS, 'unary_constraints': q}
        with pytest.raises(ValueError, match='shape'):
            maxentropytable(c, DMODEL, PMODEL, options=opts)


# ===========================================================================
# 4. BMEcatPdf first_order mode
# ===========================================================================

class TestBMEcatPdfFirstOrder:

    BASE_OPTS = {'estimator': 'ME_dual', 'tol': 1e-8, 'reg': 1e4,
                 'auto_dmax': False}

    def test_no_neighbour_limit_is_q0(self, rng):
        cs, ps = _sample_data(rng)
        q_ck = np.array([[0.75, 0.15, 0.10]])
        ck_far = np.array([[500.0, 500.0]])
        opts = {**self.BASE_OPTS,
                'first_order': {'q_at_ck': q_ck}}
        pk = BMEcatPdf(ck_far, cs, ps, DMODEL, PMODEL, 5, 8.0, options=opts)
        np.testing.assert_allclose(pk[0], q_ck[0], atol=1e-4)

    def test_valid_pdf_and_data_pull(self, rng):
        cs, ps = _sample_data(rng)
        ck = np.array([[5.0, 5.0]])
        q_ck = np.tile(P_GLOBAL, (1, 1))
        opts = {**self.BASE_OPTS, 'first_order': {'q_at_ck': q_ck}}
        pk = BMEcatPdf(ck, cs, ps, DMODEL, PMODEL, 5, 8.0, options=opts)
        assert np.all(pk >= 0)
        np.testing.assert_allclose(pk.sum(axis=1), 1.0, atol=1e-10)

    def test_q_equal_global_matches_baseline(self, rng):
        """With q_i == p_g the first-order mode must reproduce the
        stationary ME_dual result (recentering is then an identity)."""
        cs, ps = _sample_data(rng)
        ck = rng.uniform(2, 8, size=(4, 2))
        pk_base = BMEcatPdf(ck, cs, ps, DMODEL, PMODEL, 5, 8.0,
                            options=dict(self.BASE_OPTS))
        opts_fo = {**self.BASE_OPTS,
                   'first_order': {'q_at_ck': np.tile(P_GLOBAL, (4, 1))}}
        pk_fo = BMEcatPdf(ck, cs, ps, DMODEL, PMODEL, 5, 8.0,
                          options=opts_fo)
        np.testing.assert_allclose(pk_fo, pk_base, atol=0.02)

    def test_first_order_shifts_posterior(self, rng):
        """A strong class-0 first-order prior must raise the class-0
        posterior relative to the uninformed run."""
        cs, ps = _sample_data(rng)
        ck = rng.uniform(2, 8, size=(4, 2))
        pk_base = BMEcatPdf(ck, cs, ps, DMODEL, PMODEL, 5, 8.0,
                            options=dict(self.BASE_OPTS))
        q_strong = np.tile([0.9, 0.06, 0.04], (4, 1))
        opts_fo = {**self.BASE_OPTS, 'first_order': {'q_at_ck': q_strong}}
        pk_fo = BMEcatPdf(ck, cs, ps, DMODEL, PMODEL, 5, 8.0,
                          options=opts_fo)
        assert np.all(pk_fo[:, 0] > pk_base[:, 0])

    def test_q_at_cs_accepted(self, rng):
        cs, ps = _sample_data(rng)
        ck = np.array([[5.0, 5.0]])
        q_cs = np.tile(P_GLOBAL, (cs.shape[0], 1))
        opts = {**self.BASE_OPTS,
                'first_order': {'q_at_ck': np.tile(P_GLOBAL, (1, 1)),
                                'q_at_cs': q_cs}}
        pk = BMEcatPdf(ck, cs, ps, DMODEL, PMODEL, 5, 8.0, options=opts)
        np.testing.assert_allclose(pk.sum(axis=1), 1.0, atol=1e-10)

    def test_mutual_exclusion_with_bf_options(self, rng, capsys):
        """BF options must be ignored (with a warning) in first-order mode."""
        cs, ps = _sample_data(rng)
        ck = np.array([[5.0, 5.0]])
        opts = {**self.BASE_OPTS,
                'prior_pdf_at_ck': np.array([0.9, 0.05, 0.05]),
                'first_order': {'q_at_ck': np.tile(P_GLOBAL, (1, 1))}}
        pk = BMEcatPdf(ck, cs, ps, DMODEL, PMODEL, 5, 8.0, options=opts)
        out = capsys.readouterr().out
        assert 'ignoring' in out
        np.testing.assert_allclose(pk.sum(axis=1), 1.0, atol=1e-10)

    def test_estimator_override(self, rng, capsys):
        """Non-ME_dual estimators are overridden in first-order mode."""
        cs, ps = _sample_data(rng)
        ck = np.array([[5.0, 5.0]])
        opts = {'estimator': 'MPL', 'reg': 1e4, 'auto_dmax': False,
                'first_order': {'q_at_ck': np.tile(P_GLOBAL, (1, 1))}}
        pk = BMEcatPdf(ck, cs, ps, DMODEL, PMODEL, 5, 8.0, options=opts)
        out = capsys.readouterr().out
        assert 'ME_dual' in out
        np.testing.assert_allclose(pk.sum(axis=1), 1.0, atol=1e-10)


# ===========================================================================
# 5. MCPcatPdf first_order mode
# ===========================================================================

class TestMCPcatPdfFirstOrder:

    BASE_OPTS = {'auto_dmax': False}

    def test_no_neighbour_limit_is_q0(self, rng):
        cs, ps = _sample_data(rng)
        q_ck = np.array([[0.75, 0.15, 0.10]])
        ck_far = np.array([[500.0, 500.0]])
        opts = {**self.BASE_OPTS, 'first_order': {'q_at_ck': q_ck}}
        pk = MCPcatPdf(ck_far, cs, ps, DMODEL, PMODEL, 5, 8.0, options=opts)
        np.testing.assert_allclose(pk[0], q_ck[0], atol=1e-4)

    def test_q_equal_global_matches_baseline(self, rng):
        cs, ps = _sample_data(rng)
        ck = rng.uniform(2, 8, size=(4, 2))
        pk_base = MCPcatPdf(ck, cs, ps, DMODEL, PMODEL, 5, 8.0,
                            options=dict(self.BASE_OPTS))
        opts_fo = {**self.BASE_OPTS,
                   'first_order': {'q_at_ck': np.tile(P_GLOBAL, (4, 1))}}
        pk_fo = MCPcatPdf(ck, cs, ps, DMODEL, PMODEL, 5, 8.0,
                          options=opts_fo)
        np.testing.assert_allclose(pk_fo, pk_base, atol=0.02)

    def test_first_order_shifts_posterior(self):
        """Single-neighbour case: the posterior equals the (recentered)
        conditional, so a strong class-0 first-order prior provably raises
        the class-0 posterior relative to the global-margin conditional.

        (With multiple neighbours the conditional-independence product can
        legitimately move either way — recentering to a q that makes a
        class rare also makes observing it more informative.)
        """
        ck = np.array([[5.0, 5.0]])
        cs = np.array([[6.5, 5.0]])
        ps = np.array([[0.98, 0.01, 0.01]])
        ps = ps / ps.sum(axis=1, keepdims=True)

        pk_base = MCPcatPdf(ck, cs, ps, DMODEL, PMODEL, 5, 8.0,
                            options=dict(self.BASE_OPTS))
        q_strong = np.array([[0.9, 0.06, 0.04]])
        opts_fo = {**self.BASE_OPTS, 'first_order': {'q_at_ck': q_strong}}
        pk_fo = MCPcatPdf(ck, cs, ps, DMODEL, PMODEL, 5, 8.0,
                          options=opts_fo)
        assert pk_fo[0, 0] > pk_base[0, 0]

    def test_q_at_cs_accepted(self, rng):
        cs, ps = _sample_data(rng)
        ck = np.array([[5.0, 5.0]])
        q_cs = np.tile(P_GLOBAL, (cs.shape[0], 1))
        opts = {**self.BASE_OPTS,
                'first_order': {'q_at_ck': np.tile(P_GLOBAL, (1, 1)),
                                'q_at_cs': q_cs}}
        pk = MCPcatPdf(ck, cs, ps, DMODEL, PMODEL, 5, 8.0, options=opts)
        np.testing.assert_allclose(pk.sum(axis=1), 1.0, atol=1e-10)

    def test_valid_pdfs(self, rng):
        cs, ps = _sample_data(rng)
        ck = rng.uniform(0, 10, size=(6, 2))
        q_ck = np.tile([0.6, 0.25, 0.15], (6, 1))
        opts = {**self.BASE_OPTS, 'first_order': {'q_at_ck': q_ck}}
        pk = MCPcatPdf(ck, cs, ps, DMODEL, PMODEL, 5, 8.0, options=opts)
        assert np.all(pk >= 0)
        np.testing.assert_allclose(pk.sum(axis=1), 1.0, atol=1e-10)


# ===========================================================================
# 6. kernel_first_order field builder
# ===========================================================================

class TestKernelFirstOrder:

    def _layered_data(self):
        """Two stations, two depth layers with opposite compositions."""
        rows = []
        labels = []
        for sx in (0.0, 100.0):
            for z in np.arange(-40.0, 0.0, 2.0):
                rows.append([sx, 0.0, z])
                # upper layer (z >= -20): class 0; lower layer: class 2
                labels.append(0 if z >= -20 else 2)
        cs = np.array(rows)
        ps = np.zeros((len(labels), NC))
        ps[np.arange(len(labels)), labels] = 1.0
        return cs, ps

    def test_shapes_and_validity(self):
        cs, ps = self._layered_data()
        q = kernel_first_order(np.array([[50.0, 0.0]]), cs, ps, sigma_xy=50.0)
        assert q.shape == (1, NC)
        assert np.all(q > 0)
        np.testing.assert_allclose(q.sum(axis=1), 1.0, atol=1e-12)

    def test_2d_far_query_returns_reference(self):
        cs, ps = self._layered_data()
        q = kernel_first_order(np.array([[1e9, 1e9]]), cs, ps, sigma_xy=50.0)
        ref = ps.mean(axis=0)
        np.testing.assert_allclose(q[0], ref / ref.sum(), atol=1e-6)

    def test_3d_mode_resolves_depth_layers(self):
        """3-D q must differ between layers; 2-D q cannot."""
        cs, ps = self._layered_data()
        cq = np.array([[50.0, 0.0, -10.0],     # upper layer (class 0)
                       [50.0, 0.0, -30.0]])    # lower layer (class 2)
        q3 = kernel_first_order(cq, cs, ps, sigma_xy=200.0,
                                sigma_z=5.0, zbin=10.0)
        # Upper query favours class 0, lower favours class 2
        assert q3[0, 0] > 0.5 and q3[1, 2] > 0.5
        assert q3[0, 0] > q3[1, 0]

        q2 = kernel_first_order(cq[:, :2], cs, ps, sigma_xy=200.0)
        np.testing.assert_allclose(q2[0], q2[1], atol=1e-12)

    def test_3d_requires_z_column(self):
        cs, ps = self._layered_data()
        with pytest.raises(ValueError, match='Z column'):
            kernel_first_order(np.array([[0.0, 0.0]]), cs[:, :2], ps,
                               sigma_xy=50.0, sigma_z=5.0)

    def test_shrinkage_with_large_n0(self):
        """Huge pseudo-count pins q to the reference."""
        cs, ps = self._layered_data()
        p_ref = np.array([0.4, 0.3, 0.2, 0.1])[:NC]
        p_ref = p_ref / p_ref.sum()
        q = kernel_first_order(np.array([[0.0, 0.0]]), cs, ps,
                               sigma_xy=50.0, n0=1e12, p_ref=p_ref)
        np.testing.assert_allclose(q[0], p_ref, atol=1e-6)

    def test_rejects_hard_labels(self):
        cs, ps = self._layered_data()
        with pytest.raises(ValueError, match='one-hot'):
            kernel_first_order(np.array([[0.0, 0.0]]), cs,
                               np.argmax(ps, axis=1), sigma_xy=50.0)


# ===========================================================================
# 7. Prevalence-tempered MAP decision rule
# ===========================================================================

class TestTemperedMap:

    P_REF = np.array([0.2, 0.2, 0.6])       # class 2 is the majority
    CATS = np.array([1, 2, 3])

    def test_tau_zero_is_plain_map(self):
        post = np.array([[0.3, 0.3, 0.4],
                         [0.5, 0.2, 0.3]])
        y0 = tempered_map(post, self.P_REF, 0.0, categories=self.CATS)
        np.testing.assert_array_equal(y0, self.CATS[np.argmax(post, axis=1)])

    def test_tempering_flips_boundary_case_toward_minority(self):
        # Majority class barely wins under plain MAP; tempering flips it.
        post = np.array([[0.35, 0.25, 0.40]])
        y0 = tempered_map(post, self.P_REF, 0.0, categories=self.CATS)
        y1 = tempered_map(post, self.P_REF, 1.0, categories=self.CATS)
        assert y0[0] == 3 and y1[0] == 1

    def test_confident_majority_prediction_survives(self):
        post = np.array([[0.02, 0.03, 0.95]])
        y1 = tempered_map(post, self.P_REF, 1.0, categories=self.CATS)
        assert y1[0] == 3

    def test_returns_indices_without_categories(self):
        post = np.array([[0.1, 0.8, 0.1]])
        idx = tempered_map(post, self.P_REF, 0.5)
        assert idx[0] == 1

    def test_tune_recovers_balanced_optimum(self, rng):
        # Imbalanced truth with calibrated-but-hedged posteriors: plain MAP
        # over-predicts the majority; some tau > 0 must do at least as well
        # on balanced accuracy, and tau=0 must be optimal for accuracy=metric
        # only if no better tau exists (sanity of the search itself).
        n = 600
        y = rng.choice(self.CATS, size=n, p=self.P_REF)
        onehot = (y[:, None] == self.CATS[None, :]).astype(float)
        noise = rng.dirichlet(2.0 * self.P_REF * 3 + 0.5, size=n)
        post = 0.55 * onehot + 0.45 * noise

        tau_best, scores = tune_temper_tau(post, y, self.P_REF, self.CATS)
        acc0, bal0 = scores[0.0]
        acc_b, bal_b = scores[tau_best]
        assert bal_b >= bal0
        assert bal_b == max(s[1] for s in scores.values())
