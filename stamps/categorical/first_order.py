# -*- coding: utf-8 -*-
"""
stamps.categorical.first_order
===============================
Kernel-smoothed first-order (marginal) probability fields for categorical
estimation.

Provides ``kernel_first_order``, which estimates the local class
probabilities :math:`q(\\mathbf{s})` at arbitrary query locations from
borehole/station data.  The output plugs directly into the
``options['first_order']`` interface of
:func:`~stamps.categorical.estimation.BMEcatPdf` /
:func:`~stamps.categorical.estimation.MCPcatPdf` (keys ``q_at_ck`` /
``q_at_cs``), or into ``options['prior_pdf_at_ck']`` for the legacy
Bayes-factor path.

Two modes
---------
**2-D (column) mode** — ``sigma_z=None``.  Each station is summarised by its
whole-column class composition, then smoothed in XY with a Gaussian kernel.
Appropriate only when the field is vertically homogeneous.

**3-D (depth-binned) mode** — ``sigma_z`` given.  Each station's samples are
first grouped into depth bins of width *zbin*; the resulting
(station × depth-bin) composition cells are smoothed with a separable
Gaussian product kernel in XY and Z.  This captures the (X, Y) × Z
interaction typical of layered systems (e.g. clay interbeds inside
gravel-dominated columns) that a column-average field cannot represent.

Both modes apply Kish effective-sample-size shrinkage toward a reference
distribution (the per-depth-bin marginal in 3-D mode, the global marginal in
2-D mode) so that sparsely informed queries fall back gracefully instead of
echoing noise.

Leakage note
------------
For cross-validation, call this function **inside each fold** with training
rows only, so the held-out station never contributes to its own ``q``.
"""
from __future__ import annotations

import numpy as np
from typing import Optional

__all__ = ["kernel_first_order"]


def _station_cells(cs, ps, sigma_z, zbin):
    """Group samples into (station [, depth-bin]) composition cells.

    Returns a list of tuples ``(xy, z_center_or_None, prop, n)``.
    Stations are identified by their (rounded) XY coordinates.
    """
    xy = np.asarray(cs, dtype=float)[:, :2]
    keys = np.round(xy, 3)
    _, first_idx, inv = np.unique(keys, axis=0,
                                  return_index=True, return_inverse=True)
    cells = []
    if sigma_z is None:
        for g in range(first_idx.size):
            m = inv == g
            cells.append((xy[m][0], None, ps[m].mean(axis=0), int(m.sum())))
        return cells

    z = np.asarray(cs, dtype=float)[:, 2]
    z_lo = np.floor(z.min() / zbin) * zbin
    bin_idx = np.floor((z - z_lo) / zbin).astype(int)
    for g in range(first_idx.size):
        m_st = inv == g
        st_xy = xy[m_st][0]
        for b in np.unique(bin_idx[m_st]):
            m = m_st & (bin_idx == b)
            z_ctr = z_lo + (b + 0.5) * zbin
            cells.append((st_xy, z_ctr, ps[m].mean(axis=0), int(m.sum())))
    return cells


def kernel_first_order(c_query, cs, ps, sigma_xy, sigma_z=None,
                       zbin=20.0, n0=2.0,
                       p_ref: Optional[np.ndarray] = None):
    """Kernel-smoothed first-order class probabilities ``q`` at query points.

    Parameters
    ----------
    c_query : (nq, d) array
        Query coordinates.  Columns 0–1 are XY; column 2 (Z) is required in
        3-D mode.
    cs : (n, d) array
        Data coordinates (same convention).  Stations are identified by
        unique XY.
    ps : (n, nc) array
        Soft class probabilities (one-hot for hard data) aligned with *cs*.
    sigma_xy : float
        Horizontal Gaussian bandwidth.  A common choice is
        ``d_max_h / 3`` (≈ 1 % weight at the search boundary).
    sigma_z : float or None
        Vertical Gaussian bandwidth.  ``None`` (default) selects the 2-D
        column mode; a float selects the 3-D depth-binned mode.
    zbin : float
        Depth-bin width (metres) for the 3-D composition cells.
        Default ``20.0``.  Ignored in 2-D mode.
    n0 : float
        Shrinkage pseudo-count: the raw kernel estimate with Kish effective
        sample size ``n_eff`` is blended as
        ``q = (n_eff * q_raw + n0 * ref) / (n_eff + n0)``.
    p_ref : (nc,) array or None
        Override for the shrinkage reference.  When ``None`` (default), the
        reference is the count-weighted per-depth-bin marginal (3-D mode) or
        the global marginal of *ps* (2-D mode).

    Returns
    -------
    q : (nq, nc) ndarray
        Valid probability rows (clipped to ``>= 1e-6`` and normalised).

    Examples
    --------
    >>> q_at_ck = kernel_first_order(ck, cs, ps_onehot,
    ...                              sigma_xy=d_max_h/3,
    ...                              sigma_z=d_max_v/3, zbin=20.0)
    >>> pk = BMEcatPdf(ck, cs, ps_onehot, dmodel, Pmodel, nsmax, dmax,
    ...                options={'estimator': 'ME_dual',
    ...                         'first_order': {'q_at_ck': q_at_ck}})
    """
    c_query = np.atleast_2d(np.asarray(c_query, dtype=float))
    ps = np.asarray(ps, dtype=float)
    if ps.ndim != 2:
        raise ValueError(
            "ps must be a (n, nc) soft-probability matrix; convert hard "
            "labels to one-hot first.")
    nc = ps.shape[1]
    if sigma_z is not None and (c_query.shape[1] < 3 or
                                np.asarray(cs).shape[1] < 3):
        raise ValueError("3-D mode (sigma_z given) requires a Z column in "
                         "both c_query and cs.")

    cells = _station_cells(cs, ps, sigma_z, zbin)
    cell_xy = np.array([c[0] for c in cells])
    cell_pr = np.array([c[2] for c in cells])
    cell_n = np.array([c[3] for c in cells], dtype=float)

    # --- Shrinkage reference ------------------------------------------------
    if p_ref is not None:
        ref_global = np.asarray(p_ref, dtype=float).ravel()
        ref_global = ref_global / ref_global.sum()
        ref_bins = None
    elif sigma_z is None:
        ref_global = (cell_pr * cell_n[:, None]).sum(axis=0) / cell_n.sum()
        ref_bins = None
    else:
        # Count-weighted marginal per depth bin (fallback: global)
        cell_z = np.array([c[1] for c in cells], dtype=float)
        ref_global = (cell_pr * cell_n[:, None]).sum(axis=0) / cell_n.sum()
        uz = np.unique(cell_z)
        ref_bins = {}
        for zc in uz:
            m = cell_z == zc
            ref_bins[zc] = ((cell_pr[m] * cell_n[m, None]).sum(axis=0)
                            / cell_n[m].sum())

    if sigma_z is not None:
        cell_z = np.array([c[1] for c in cells], dtype=float)

    # --- Kernel evaluation ---------------------------------------------------
    q = np.empty((c_query.shape[0], nc))
    for i, pt in enumerate(c_query):
        w = cell_n * np.exp(
            -np.sum((cell_xy - pt[:2]) ** 2, axis=1) / (2.0 * sigma_xy ** 2))
        if sigma_z is not None:
            w = w * np.exp(-(cell_z - pt[2]) ** 2 / (2.0 * sigma_z ** 2))

        w_tot = w.sum()
        if sigma_z is not None and ref_bins is not None:
            zc_near = min(ref_bins, key=lambda zc: abs(zc - pt[2]))
            ref = ref_bins[zc_near]
        else:
            ref = ref_global

        if w_tot <= 1e-300:
            q[i] = ref
            continue
        q_raw = (w[:, None] * cell_pr).sum(axis=0) / w_tot
        n_eff = w_tot * w_tot / max((w * w).sum(), 1e-300)
        q[i] = (n_eff * q_raw + n0 * ref) / (n_eff + n0)

    q = np.clip(q, 1e-6, None)
    q /= q.sum(axis=1, keepdims=True)
    return q
