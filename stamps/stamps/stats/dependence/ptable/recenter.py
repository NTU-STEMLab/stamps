# -*- coding: utf-8 -*-
"""
stamps.stats.dependence.ptable.recenter
=======================================
IPF (Iterative Proportional Fitting / Sinkhorn) re-margining of bivariate
probability tables.

The core operation takes a fitted bivariate class-pair table
``P(a, b | d)`` — whose margins equal the *global* class proportions
``p_g`` — and rescales it so its margins become locally varying first-order
targets ``(q_row, q_col)`` while **preserving all odds ratios** (the
second-order association structure).  This is the minimum-KL projection of
the global table onto the set of tables with the requested margins:

.. math::

    P_x = \\arg\\min_{P \\in \\Pi(q_r, q_c)} D_{KL}(P \\,\\|\\, P_g)

Because row/column scaling cannot change cross-product ratios, the
"stationary dependence" of the global model is transported exactly to the
local first-order level.  Two useful limiting cases follow automatically:

* **Zero lag** — ``P_g(·,·,0)`` is diagonal, so the recentered table is
  diagonal with entries ``q(a)`` (same-point ⇒ same class, at local rates).
* **Far field** — ``P_g(·,·,d_max) = p_g p_gᵀ`` maps to ``q_r q_cᵀ``
  (exact independence at the local margins; no phantom correlation).

Public API
----------
ipf_recenter        Re-margin one (nc, nc) table.
recenter_pmodel     Vectorised re-margining of a (nc, nc, nd) Pmodel stack.
"""
import numpy as np

__all__ = ['ipf_recenter', 'recenter_pmodel']

_EPS = 1e-12


def _clean_margin(q, nc):
    """Validate/normalise a margin vector; clip away exact zeros."""
    q = np.asarray(q, dtype=float).ravel()
    if q.size != nc:
        raise ValueError(f'margin has size {q.size}, expected {nc}')
    q = np.clip(q, 1e-6, None)
    return q / q.sum()


def ipf_recenter(P, p_row, p_col=None, tol=1e-10, max_iter=200):
    """Rescale a bivariate probability table to prescribed margins.

    Applies IPF (alternating row/column scaling, also known as the
    Sinkhorn–Knopp algorithm) to ``P`` until its row margin equals
    ``p_row`` and its column margin equals ``p_col``.  Odds ratios of
    ``P`` are invariant under this operation.

    Parameters
    ----------
    P : (nc, nc) array
        Source bivariate probability table (need not be normalised;
        non-negative).  Typically a lag slice of a fitted global Pmodel.
    p_row : (nc,) array
        Target row margin (first-order probabilities for the row axis).
    p_col : (nc,) array or None
        Target column margin.  Defaults to ``p_row`` (symmetric case).
    tol : float
        Convergence tolerance on the L1 margin error.
    max_iter : int
        Maximum number of row+column sweeps.

    Returns
    -------
    P_out : (nc, nc) array
        Valid probability table (non-negative, sums to 1) with margins
        ``(p_row, p_col)`` and the odds ratios of ``P``.

    Notes
    -----
    Structural zeros in ``P`` are preserved (IPF cannot create support).
    If a full row/column of ``P`` is zero while its target margin is
    positive, that cell block is unreachable; the affected margin is
    then matched as closely as the support allows.
    """
    P = np.asarray(P, dtype=float)
    nc = P.shape[0]
    p_row = _clean_margin(p_row, nc)
    p_col = p_row if p_col is None else _clean_margin(p_col, nc)

    M = np.clip(P, 0.0, None).copy()
    s = M.sum()
    if not np.isfinite(s) or s <= _EPS:
        # Degenerate source table: fall back to independent margins
        return np.outer(p_row, p_col)
    M /= s

    for _ in range(max_iter):
        # Row scaling
        r = M.sum(axis=1)
        r_safe = np.where(r > _EPS, r, 1.0)
        M *= (p_row / r_safe)[:, None]
        # Column scaling
        c = M.sum(axis=0)
        c_safe = np.where(c > _EPS, c, 1.0)
        M *= (p_col / c_safe)[None, :]

        err = (np.abs(M.sum(axis=1) - p_row).sum()
               + np.abs(M.sum(axis=0) - p_col).sum())
        if err < tol:
            break

    total = M.sum()
    if total > _EPS:
        M /= total
    else:
        M = np.outer(p_row, p_col)
    return M


def recenter_pmodel(Pmodel, p_row, p_col=None, tol=1e-10, max_iter=200):
    """Re-margin every lag slice of a fitted Pmodel stack.

    Parameters
    ----------
    Pmodel : (nc, nc, nd) array
        Fitted bivariate probability tables over ``nd`` lags.
    p_row, p_col : (nc,) arrays
        Target margins (see :func:`ipf_recenter`).
    tol, max_iter :
        Forwarded to :func:`ipf_recenter`.

    Returns
    -------
    P_out : (nc, nc, nd) array
        Recentered Pmodel with identical odds ratios per lag.
    """
    Pmodel = np.asarray(Pmodel, dtype=float)
    out = np.empty_like(Pmodel)
    for k in range(Pmodel.shape[2]):
        out[:, :, k] = ipf_recenter(Pmodel[:, :, k], p_row, p_col,
                                    tol=tol, max_iter=max_iter)
    return out
