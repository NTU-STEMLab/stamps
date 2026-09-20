# -*- coding: utf-8 -*-
"""
stamps.general.local_marginal
==============================
Kernel-weighted local marginal probability estimation.

Provides a standalone function for estimating the local class distribution
at an estimation point from its neighbourhood soft data, with
Kish-effective-sample-size regularized shrinkage toward a reference
distribution.

This function is designed to work with the outputs of :func:`neighbours_cat`
(specifically the ``idx_all`` and ``D_all`` return values that expose ALL
candidates within the search window, not just the selected ``nsmax``
neighbours).

Public API
----------
local_marginal   Kernel-weighted local marginal with Kish regularization.
"""
from __future__ import annotations

import numpy as np
from typing import Optional, Union

__all__ = ["local_marginal"]


def local_marginal(
    Z_candidates: np.ndarray,
    distances: np.ndarray,
    bandwidths: Union[float, np.ndarray],
    p_reference: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Kernel-weighted local marginal with Kish-effective-sample regularization.

    Estimates the local class distribution at an estimation point by computing
    a Gaussian-kernel weighted average of soft-data probabilities from ALL
    candidate neighbours within the search window, then shrinking toward a
    reference distribution to prevent near-zero entries for absent classes.

    This function is intentionally decoupled from neighbour *selection* — it
    operates on the set of ALL valid candidates (not just the ``nsmax``
    selected for inference).  Pass the ``idx_all`` / ``D_all`` outputs of
    :func:`neighbours_cat` to obtain the correct inputs.

    Parameters
    ----------
    Z_candidates : (m, nc) ndarray
        Soft-data probability vectors for all *m* candidate neighbours
        within the search window.  Each row should be a valid probability
        vector; unnormalised rows are accepted and renormalised internally.
    distances : (m,) or (m, K) ndarray
        Distances from the estimation point to each of the *m* candidates.

        * ``(m,)`` — single Euclidean distance per candidate (non-separable).
        * ``(m, K)`` — per-axis distances for a separable model with *K*
          components.  The Gaussian kernel is a **product** of one per-axis
          kernel.

    bandwidths : float or (K,) array_like
        Gaussian kernel bandwidth(s) ``σ``.  The weight for candidate *i* is

        .. math::

            w_i = \\prod_j \\exp\\!\\left(-\\frac{d_{ij}^2}{2\\sigma_j^2}\\right)

        A common choice is ``σ_j = dmax_j / 3`` so the weight at the search
        boundary is ≈ 1 % (practical zero).

    p_reference : (nc,) ndarray or None
        Reference distribution for shrinkage toward when the neighbourhood
        is sparse (low Kish effective sample size).  When ``None``, the
        uniform distribution ``1/nc`` is used.  Typically set to the global
        class marginal ``p(c)`` estimated from all training data.

    Returns
    -------
    p_local : (nc,) ndarray or None
        Regularized local marginal.  ``None`` is returned when the total
        kernel weight is negligible (all candidates are very far from the
        estimation point), signalling that no reliable local estimate exists.

    Notes
    -----
    **Kish effective sample size regularization**

    .. math::

        N_{\\text{eff}} = \\frac{\\bigl(\\sum_k w_k\\bigr)^2}{\\sum_k w_k^2},
        \\qquad
        \\alpha = \\frac{1}{N_{\\text{eff}} + 1}

    The shrinkage coefficient ``α`` is close to 1 when very few informative
    candidates are available (sparse data), pulling the estimate toward
    ``p_reference`` to avoid explosive Bayes-factor ratios.  As more data
    accumulate ``α → 0`` and the estimate converges to the raw kernel average.

    Examples
    --------
    >>> import numpy as np
    >>> Z = np.array([[0.8, 0.2], [0.6, 0.4], [0.7, 0.3]])
    >>> d = np.array([10.0, 20.0, 30.0])
    >>> p = local_marginal(Z, d, bandwidths=50.0)
    >>> p.shape
    (2,)
    >>> abs(p.sum() - 1.0) < 1e-12
    True
    """
    Z_candidates = np.asarray(Z_candidates, dtype=float)
    distances = np.asarray(distances, dtype=float)

    if Z_candidates.ndim != 2:
        raise ValueError(
            f"Z_candidates must be 2-D (m, nc); got shape {Z_candidates.shape}"
        )
    m, nc = Z_candidates.shape

    if m == 0:
        return None

    # ── Compute Gaussian kernel weights ───────────────────────────────────────
    if distances.ndim == 1:
        # Non-separable: single Euclidean distance per candidate
        bw = np.asarray(bandwidths, dtype=float).ravel()
        sig = float(bw[0]) if bw.size >= 1 else float(bandwidths)
        if sig > 1e-15:
            kern = np.exp(-0.5 * (distances / sig) ** 2)
        else:
            kern = np.ones(m, dtype=float)
    else:
        # Separable: product of one Gaussian kernel per axis
        K = distances.shape[1]
        bw = np.asarray(bandwidths, dtype=float).ravel()
        if bw.size == 1:
            bw = np.full(K, bw[0])
        elif bw.size != K:
            raise ValueError(
                f"bandwidths has {bw.size} elements but distances has {K} columns."
            )
        kern = np.ones(m, dtype=float)
        for j in range(K):
            sig_j = float(bw[j])
            if sig_j > 1e-15:
                kern *= np.exp(-0.5 * (distances[:, j] / sig_j) ** 2)

    ksum = kern.sum()
    if ksum <= 1e-15:
        return None

    # ── Kernel-weighted class frequencies ─────────────────────────────────────
    # Normalise each row of Z so that unnormalised inputs are handled
    row_sums = Z_candidates.sum(axis=1, keepdims=True)
    row_sums[row_sums <= 0] = 1.0
    Z_norm = Z_candidates / row_sums
    p_local_raw = (kern @ Z_norm) / ksum

    # ── Kish effective sample size → shrinkage coefficient ────────────────────
    n_eff = (ksum ** 2) / np.sum(kern ** 2)
    alpha = 1.0 / (n_eff + 1.0)

    # ── Reference distribution ────────────────────────────────────────────────
    if p_reference is not None:
        p_ref = np.asarray(p_reference, dtype=float).ravel()
        if p_ref.size != nc:
            raise ValueError(
                f"p_reference has {p_ref.size} elements but Z_candidates has {nc} columns."
            )
        p_ref = np.clip(p_ref, 1e-15, None)
        p_ref = p_ref / p_ref.sum()
    else:
        p_ref = np.ones(nc, dtype=float) / nc

    # ── Shrinkage toward reference ─────────────────────────────────────────────
    p_local = (1.0 - alpha) * p_local_raw + alpha * p_ref
    p_local = np.clip(p_local, 1e-15, None)
    p_local = p_local / p_local.sum()
    return p_local
