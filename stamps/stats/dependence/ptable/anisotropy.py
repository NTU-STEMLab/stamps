# -*- coding: utf-8 -*-
"""
stamps.stats.dependence.ptable.anisotropy
==========================================
Categorical-data anisotropy analysis (probability-table-based).

Functions: polar_ptable_map, estimate_anisotropy_from_ptables,
isotropy_test_categorical.

Also re-exports iso2aniso/aniso2iso from the covariance module (these
coordinate transforms are used by both continuous and categorical code).
"""
import numpy
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import chi2

from ..covariance.anisotropy import iso2aniso, aniso2iso  # noqa: F401



# ===========================================================================
# Categorical / probability-table anisotropy functions
# ===========================================================================

def _ptable_range(D, P_diag, p_bg):
    """Extract the effective range from one diagonal of a probability table.

    The effective range is defined as the lag ``h`` at which the joint
    self-transition departure from the **far-field independence level**
    (``p_c²``) drops to ``exp(−1)`` of the lag-0 departure:

    .. math::

        a_c = \\min\\bigl\\{h : P_{cc}(h) - p_c^2 \\leq
              e^{-1}\\,[P_{cc}(0) - p_c^2]\\bigr\\}

    where ``P_{cc}(0) = \\mathbb{E}[p_i(c)^2]`` (from self-pairs) and the
    far-field independence level is ``p_c^2 = (\\mathbb{E}[p_i(c)])^2``.

    **Why** ``p_c^2`` **and not** ``p_c``**?**
    The joint probability ``P[c,c,h]`` decays from ``\\mathbb{E}[p_i(c)^2]``
    at ``h = 0`` down to ``p_c^2`` at large ``h`` (statistical independence).
    Using ``p_c`` as the background would give a departure of
    ``\\mathbb{E}[p_i(c)^2] - p_c ≤ 0``, which is always non-positive and
    makes the range undefined.  Using ``p_c^2`` gives a positive departure
    equal to ``\\mathrm{Var}(p_i(c)) ≥ 0``.

    Parameters
    ----------
    D : (ncl,) array
        Mean distances per lag bin (output of :func:`probatablecalc_directional`).
    P_diag : (ncl,) array
        Self-transition values ``P[c, c, :]`` for one class ``c``.
    p_bg : float
        **Far-field independence level** for class ``c``, equal to ``p_c²``
        where ``p_c = E[p_i(c)]`` is the overall marginal proportion.
        Do **not** pass the marginal ``p_c`` itself — pass ``p_c ** 2``.

    Returns
    -------
    range_val : float or NaN
        Effective range in the same units as ``D``.
    """
    D      = np.asarray(D,      dtype=float)
    P_diag = np.asarray(P_diag, dtype=float)

    valid  = ~np.isnan(D) & ~np.isnan(P_diag)
    if valid.sum() < 2:
        return np.nan

    D_v = D[valid]
    P_v = P_diag[valid]

    departure_0 = float(P_v[0]) - p_bg
    if departure_0 <= 0:
        return np.nan

    threshold = p_bg + np.exp(-1) * departure_0
    below = np.where(P_v <= threshold)[0]
    if below.size == 0:
        return np.nan

    idx = below[0]
    if idx == 0:
        return float(D_v[0])

    h0, h1 = float(D_v[idx - 1]), float(D_v[idx])
    p0, p1 = float(P_v[idx - 1]), float(P_v[idx])
    if p1 >= p0:              # no decay → undefined
        return np.nan
    return h0 + (threshold - p0) / (p1 - p0) * (h1 - h0)


def polar_ptable_map(coord, value, coord_limit, rLag, rLagTol,
                     angLag=None, angTol=None,
                     group_ids=None,
                     chunk_size=None, plot=True, figsize=None):
    """Compute directional self-transition probability curves and optionally
    display a polar rose plot.

    This is the **categorical-data analogue** of :func:`polarstcovmap`.
    Instead of directional covariances it uses directional transition
    probability tables computed by :func:`probatablecalc_directional`.

    For each direction ``angLag[k]`` the self-transition probability
    ``P[c, c, h]`` is estimated in distance bins ``rLag``.  The polar plot
    shows, for each class, the mean self-transition probability at each
    direction as a rose petal, revealing the principal axis of spatial
    continuity.

    Parameters
    ----------
    coord : (n, 2) array
        2-D spatial coordinates (X, Y) of the observations.
    value : (n, nc) array
        One-hot indicator matrix (or soft category probabilities).
    coord_limit : (ncl,) array
        Distance-class limits used for the probability-table estimation.
        Must cover the range of ``rLag``.
    rLag : (nls,) array
        Reference spatial lags at which to read self-transition values
        for the polar plot (subset of the lag axis defined by ``coord_limit``).
    rLagTol : (nls,) array
        Lag tolerance for each ``rLag`` value (used to average nearby lags).
    angLag : (na,) array or None
        Directions in radians, in ``(-π/2, π/2]``.
        Default: 6 equally spaced directions at 30° intervals.
    angTol : (na,) array or None
        Angular tolerances.  Default: half the angular step.
    group_ids : (n,) array-like or None, optional
        Group labels forwarded to :func:`probatablecalc_directional`.
        When provided, only pairs within the same group contribute.
        Typical use: pass ``wells['Z'].values.astype(str)`` so that only
        same-depth-level pairs contribute to the horizontal anisotropy
        tables, consistent with the per-Z horizontal Pmodel.
    chunk_size : int or None
        Memory-control passed to :func:`probatablecalc_directional`.
    plot : bool
        If ``True``, show a polar rose figure with one panel per class.
    figsize : tuple or None
        Figure size passed to ``plt.subplots``; auto-derived if ``None``.

    Returns
    -------
    PP : list of (nc, nc, ncl) arrays
        Directional transition-probability tables, one per direction.
    OO : list of (ncl,) arrays
        Pair counts, one per direction.
    DD : list of (ncl,) arrays
        Mean distances, one per direction.
    angLag : (na,) array
        Evaluated directions in radians.

    Notes
    -----
    The effective range of categorical spatial dependence in direction ``θ``
    can be extracted from ``PP`` via :func:`estimate_anisotropy_from_ptables`.

    References
    ----------
    Carle, S.F. & Fogg, G.E. (1996). Transition probability-based indicator
        geostatistics. *Mathematical Geology*, 28(4), 453–476.
    Goovaerts, P. (1997). *Geostatistics for Natural Resources Evaluation*.
        Oxford University Press. Section 4.3 (polar covariance map).
    """
    from .empirical import probatablecalc_directional

    coord       = np.asarray(coord,        dtype=float)
    value       = np.asarray(value,        dtype=float)
    coord_limit = np.asarray(coord_limit,  dtype=float).ravel()
    rLag        = np.asarray(rLag,         dtype=float)
    rLagTol     = np.asarray(rLagTol,      dtype=float)

    if angLag is None:
        ang_step = np.pi / 6     # 30°
        angLag   = np.arange(-np.pi / 2, np.pi / 2, ang_step)
        angTol   = np.full_like(angLag, ang_step / 2)
    else:
        angLag = np.asarray(angLag, dtype=float)
        if angTol is None:
            ang_step = (angLag[1] - angLag[0]) if len(angLag) > 1 else np.pi / 6
            angTol   = np.full_like(angLag, ang_step / 2)
        else:
            angTol = np.asarray(angTol, dtype=float)

    nc  = value.shape[1]
    na  = len(angLag)
    ncl = len(coord_limit)

    PP, OO, DD = [], [], []

    for k, (ang, tol) in enumerate(zip(angLag, angTol)):
        D_k, P_k, O_k = probatablecalc_directional(
            coord, value, coord_limit,
            ang=ang, angtol=tol,
            group_ids=group_ids,
            chunk_size=chunk_size,
        )
        PP.append(P_k)
        OO.append(O_k)
        DD.append(D_k)

    if plot:
        n_cols  = nc
        fs      = figsize or (4.5 * nc, 4.5)
        fig, axes = plt.subplots(1, n_cols, figsize=fs,
                                 subplot_kw={'projection': 'polar'})
        if nc == 1:
            axes = [axes]

        # Background = far-field independence level p_c^2.
        # P[c,c,h] decays from E[p_i(c)^2] (at h=0) toward p_c^2 (at h→∞).
        # Using p_c (marginal) would give a non-positive departure everywhere.
        p_marginal = value.mean(axis=0)       # p_c = E[p_i(c)]
        p_bg_sq    = p_marginal ** 2          # p_c^2  — far-field independence

        for ci in range(nc):
            ax = axes[ci]

            # For each direction: mean departure from independence at lags rLag
            c_dirs = np.zeros(na)
            for k in range(na):
                D_k = DD[k]
                P_k = PP[k]
                vals = []
                for r, rt in zip(rLag, rLagTol):
                    mask = ~np.isnan(D_k) & (np.abs(D_k - r) <= rt)
                    if mask.any():
                        # departure from p_c^2 (far-field independence level)
                        dep = P_k[ci, ci, mask] - p_bg_sq[ci]
                        vals.append(float(np.nanmean(dep)))
                c_dirs[k] = float(np.nanmean(vals)) if vals else 0.0

            c_dirs = np.clip(c_dirs, 0, None)

            # Full 360° symmetric rose.
            # Under second-order stationarity P[c,c;h,θ] = P[c,c;h,θ+π],
            # so the half-plane [-π/2, π/2) mirrors exactly onto [π/2, 3π/2).
            ang_full = np.concatenate([angLag, angLag + np.pi])
            c_full   = np.concatenate([c_dirs, c_dirs])
            # close the circle
            ang_full = np.append(ang_full, ang_full[0] + 2 * np.pi)
            c_full   = np.append(c_full,   c_full[0])

            # Densify to 720 angles so that ax.fill() draws smooth arcs
            # instead of straight Cartesian chords that cut across the plot
            # interior when a petal drops to near-zero next to a large petal.
            ang_dense = np.linspace(ang_full[0], ang_full[-1], 721)
            c_dense   = np.interp(ang_dense, ang_full, c_full)

            ax.plot(ang_dense, c_dense, lw=2)
            ax.fill(ang_dense, c_dense, alpha=0.3)
            ax.set_title(f'Class {ci + 1}', pad=12)

        fig.suptitle(
            'Directional self-transition departure from independence\n'
            r'($P[c,c;\,h,\theta] - p_c^2$, probability-table based)',
            fontsize=12, y=1.02,
        )
        plt.tight_layout()
        plt.show()

    return PP, OO, DD, angLag


def estimate_anisotropy_from_ptables(DD, PP, OO, angLag,
                                     class_weights=None, sill=None,
                                     p_marginal=None):
    """Estimate the principal-axis angle and anisotropy ratio from directional
    transition probability tables.

    This is the **categorical-data analogue** of
    :func:`estimate_anisotropy_params`.  Instead of extracting effective
    ranges from directional covariances, it uses the self-transition
    probability departure curve ``P[c, c, h] − p_c`` computed by
    :func:`polar_ptable_map` / :func:`probatablecalc_directional`.

    The effective range in direction ``θ`` for class ``c`` is the lag ``a``
    where ``P[c,c,a] − p_c²`` drops to ``exp(−1)`` of its lag-0 departure
    ``P[c,c,0] − p_c²``.  Ranges are averaged across classes (weighted by
    ``class_weights``) to give a single directional range, and an anisotropy
    ellipse is fitted by weighted least squares.

    **Background level**: ``p_c²`` is the far-field *joint* probability under
    statistical independence.  Using the marginal ``p_c`` would give a
    non-positive departure (since ``P[c,c,h] ≤ p_c`` for all ``h``) and
    make range extraction impossible.

    Parameters
    ----------
    DD : list of (ncl,) arrays
        Mean distances per direction, returned by :func:`polar_ptable_map`.
    PP : list of (nc, nc, ncl) arrays
        Directional probability tables, returned by :func:`polar_ptable_map`.
    OO : list of (ncl,) arrays
        Pair counts per direction, returned by :func:`polar_ptable_map`.
    angLag : (na,) array
        Directions in radians evaluated by :func:`polar_ptable_map`.
    class_weights : (nc,) array or None
        Per-class weights for averaging ranges across classes.
        ``None`` uses equal weights.  Tip: use the class proportions to
        up-weight well-sampled classes.
    sill : float or None
        Not used (kept for API symmetry with
        :func:`estimate_anisotropy_params`).  Reserved for future use.
    p_marginal : (nc,) array or None
        Overall marginal class proportions ``p_c = E[p_i(c)]``.  The
        background is set to ``p_c²``.  When ``None`` (default), the
        marginals are estimated from the row sums of the probability tables
        (``sum_b P[c,b,h] = p_c`` for any non-zero bin) averaged across all
        directions and non-zero bins.  Passing the value array's column means
        directly (``value.mean(axis=0)``) is more accurate and is recommended.

    Returns
    -------
    theta_hat : float
        Estimated principal-axis angle in radians, in ``(-π/2, π/2]``.
    r_hat : float
        Estimated anisotropy ratio (secondary / principal range), in
        ``(0, 1]``.  ``r_hat = 1`` means isotropy.
    a_max_hat : float
        Estimated principal (maximum) effective range.
    a_dirs : (na,) array
        Effective ranges (class-averaged) extracted at each direction.
    a_dirs_per_class : (nc, na) array
        Effective ranges per class per direction.
    valid : (na,) bool array
        ``True`` if a range was found for that direction.

    Notes
    -----
    The anisotropy ellipse model is identical to the one in
    :func:`estimate_anisotropy_params`:

    .. math::

        a(\\phi) = \\frac{a_{\\max} \\cdot r}{\\sqrt{r^2 \\cos^2(\\phi-\\theta)
                   + \\sin^2(\\phi-\\theta)}}

    The fit minimises weighted squared residuals with weights equal to the
    mean pair count at each direction.

    References
    ----------
    Carle, S.F. & Fogg, G.E. (1996). Transition probability-based indicator
        geostatistics. *Mathematical Geology*, 28(4), 453–476.
    Goovaerts, P. (1997). *Geostatistics for Natural Resources Evaluation*.
        Oxford University Press. Section 4.3 (ellipse fitting, Eq. 4.5).
    """
    angLag = np.asarray(angLag, dtype=float)
    na     = len(angLag)
    nc     = PP[0].shape[0]

    if class_weights is None:
        class_weights = np.ones(nc) / nc
    else:
        class_weights = np.asarray(class_weights, dtype=float)
        class_weights = class_weights / class_weights.sum()

    # --- Background = far-field independence level p_c^2 --------------------
    # The joint probability P[c,c,h] decays from E[p_i(c)^2] at h=0 down to
    # p_c^2 at large h (statistical independence).  The correct background for
    # the e-folding range is p_c^2, NOT p_c (marginal) and NOT E[p_i(c)^2]
    # (lag-0 self-pair value).  Using E[p_i(c)^2] as the background causes
    # departure_0 = P[c,c,0] - E[p_i(c)^2] = 0 → NaN ranges everywhere.
    if p_marginal is not None:
        _p_marginal = np.asarray(p_marginal, dtype=float)
        if _p_marginal.shape[0] != nc:
            raise ValueError(
                f'p_marginal length ({len(_p_marginal)}) must match nc ({nc}).'
            )
    else:
        # Estimate p_c from the row sums of the directional tables:
        # sum_b P[c, b, h] = p_c for any non-zero bin (marginal property).
        _p_marginal = np.zeros(nc)
        _count = 0
        for P_k, O_k in zip(PP, OO):
            ncl_k = P_k.shape[2]
            for b_idx in range(1, ncl_k):   # skip lag-0 self-pair bin
                if float(O_k[b_idx]) > 0:
                    _p_marginal += P_k[:, :, b_idx].sum(axis=1)
                    _count += 1
        if _count > 0:
            _p_marginal /= _count
        else:
            _p_marginal[:] = 1.0 / nc      # uniform fallback

    p_bg = _p_marginal ** 2   # far-field independence level for joint P[c,c,h]

    # Extract effective range per class per direction
    a_dirs_per_class = np.full((nc, na), np.nan)
    weights_dir      = np.zeros(na)

    for k in range(na):
        D_k = np.asarray(DD[k], dtype=float)
        P_k = PP[k]
        O_k = np.asarray(OO[k], dtype=float)

        # Mean pair count across bins (excluding lag-0)
        valid_counts = O_k[1:][~np.isnan(O_k[1:])]
        weights_dir[k] = float(np.mean(valid_counts)) if len(valid_counts) else 0.0

        for ci in range(nc):
            a_dirs_per_class[ci, k] = _ptable_range(D_k, P_k[ci, ci, :], p_bg[ci])

    # Class-weighted average range per direction
    a_dirs = np.full(na, np.nan)
    for k in range(na):
        ranges_k = a_dirs_per_class[:, k]
        good     = ~np.isnan(ranges_k)
        if good.any():
            a_dirs[k] = float(np.average(ranges_k[good],
                                         weights=class_weights[good]))

    valid = ~np.isnan(a_dirs)

    if valid.sum() < 2:
        a0 = float(np.nanmean(a_dirs)) if valid.any() else 1.0
        return 0.0, 1.0, a0, a_dirs, a_dirs_per_class, valid

    # Fit anisotropy ellipse by WLS (identical to estimate_anisotropy_params)
    def ellipse_range(phi, theta, r, a_max):
        denom = np.sqrt(r**2 * np.cos(phi - theta)**2
                        + np.sin(phi - theta)**2)
        return a_max * r / denom

    def residuals(params):
        theta, r, a_max = params
        if r <= 0 or r > 1 or a_max <= 0:
            return 1e12
        pred  = ellipse_range(angLag[valid], theta, r, a_max)
        resid = weights_dir[valid] * (a_dirs[valid] - pred)**2
        return float(resid.sum())

    a0 = float(np.nanmean(a_dirs[valid]))
    bounds = [(-np.pi / 2, np.pi / 2), (0.01, 1.0), (1e-6, None)]
    try:
        result = minimize(residuals, [0.0, 0.8, a0],
                          method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 500, 'ftol': 1e-12})
        theta_hat, r_hat, a_max_hat = result.x
    except Exception:
        theta_hat, r_hat, a_max_hat = 0.0, 1.0, a0

    # Normalise to (-π/2, π/2]
    while theta_hat >  np.pi / 2:
        theta_hat -= np.pi
    while theta_hat <= -np.pi / 2:
        theta_hat += np.pi

    return float(theta_hat), float(r_hat), float(a_max_hat), \
        a_dirs, a_dirs_per_class, valid


def isotropy_test_categorical(coord, value, coord_limit, angLag, angTol,
                               class_weights=None,
                               n_boot=199, alpha=0.05, seed=None,
                               chunk_size=None):
    """Bootstrap isotropy test for **categorical** (probability-table) data.

    Tests whether the spatial dependence structure of a categorical variable
    is consistent with isotropy, using the effective ranges extracted from
    directional transition probability tables as the test statistic.

    The null hypothesis ``H₀`` is that the observed variation in effective
    range across directions is consistent with random spatial arrangement
    (isotropy).  The alternative is that the effective range is systematically
    larger in one direction (geometric anisotropy).

    **Test statistic**

    .. math::

        T_n = a_{\\text{principal}} - a_{\\text{secondary}}

    where the two directions are the estimated principal and secondary axes
    from :func:`estimate_anisotropy_from_ptables`.  A larger ``T_n`` means
    greater directional range contrast.

    **Null distribution**

    Generated by rotating all XY coordinates by ``n_boot`` uniformly drawn
    angles in ``[0, 2π)`` and recomputing ``T_n`` each time.  Under isotropy
    the rotation changes nothing about the dependence structure.

    Parameters
    ----------
    coord : (n, 2) array
        2-D spatial coordinates (X, Y).
    value : (n, nc) array
        One-hot indicator matrix (or soft category probabilities).
    coord_limit : (ncl,) array
        Distance-class limits for the probability-table estimation.
    angLag : (na,) array
        Directions in radians to evaluate (e.g. 6 directions at 30° steps).
    angTol : (na,) array
        Angular tolerances (half-widths of acceptance windows).
    class_weights : (nc,) array or None
        Per-class weights for range averaging. ``None`` → equal weights.
    n_boot : int
        Number of bootstrap rotations.  Recommended ≥ 199.
    alpha : float
        Significance level for the reject/do-not-reject decision.
    seed : int or None
        Random seed for reproducibility.
    chunk_size : int or None
        Memory-control passed to :func:`probatablecalc_directional`.

    Returns
    -------
    result : dict with keys
        ``'theta_hat'``      — float, estimated principal-axis angle (radians).
        ``'r_hat'``          — float, estimated anisotropy ratio.
        ``'a_max_hat'``      — float, estimated principal range.
        ``'a_dirs'``         — (na,) array, class-averaged directional ranges.
        ``'a_dirs_per_class'``— (nc, na) array, per-class directional ranges.
        ``'test_stat'``      — float, observed ``T_n``.
        ``'T_boot'``         — (n_boot,) array, bootstrap ``T_n`` values.
        ``'p_value'``        — float, bootstrap p-value.
        ``'reject_H0'``      — bool, whether H₀ is rejected at level ``alpha``.
        ``'message'``        — str, plain-English interpretation.

    References
    ----------
    Carle, S.F. & Fogg, G.E. (1996). Transition probability-based indicator
        geostatistics. *Mathematical Geology*, 28(4), 453–476.
    Guan, Y., Sherman, M. & Calvin, J.A. (2004). A nonparametric test for
        spatial isotropy using subsampling. *JASA*, 99(467), 810–821.
        [Rotation-bootstrap concept adapted here for categorical data.]
    """
    coord  = np.asarray(coord,  dtype=float)
    value  = np.asarray(value,  dtype=float)
    angLag = np.asarray(angLag, dtype=float)
    angTol = np.asarray(angTol, dtype=float)
    rng    = np.random.default_rng(seed)

    _p_marginal_iso = value.mean(axis=0)   # fixed across bootstrap rotations

    def _compute_T(gs):
        """Compute test statistic from coordinates ``gs``."""
        PP, OO, DD, _ = polar_ptable_map(
            gs, value, coord_limit,
            rLag    = np.array([]),   # rLag unused when plot=False
            rLagTol = np.array([]),
            angLag=angLag, angTol=angTol,
            chunk_size=chunk_size, plot=False,
        )
        theta, r, a_max, a_dirs, _, valid = estimate_anisotropy_from_ptables(
            DD, PP, OO, angLag,
            class_weights=class_weights,
            p_marginal=_p_marginal_iso,
        )
        if valid.sum() < 2:
            return np.nan, theta, r, a_max, a_dirs

        # T_n = range(principal) - range(secondary)
        phi2 = theta + np.pi / 2
        if phi2 > np.pi / 2:
            phi2 -= np.pi
        idx1 = int(np.argmin(np.abs(angLag - theta)))
        idx2 = int(np.argmin(np.abs(angLag - phi2)))
        a1 = float(a_dirs[idx1]) if not np.isnan(a_dirs[idx1]) else 0.0
        a2 = float(a_dirs[idx2]) if not np.isnan(a_dirs[idx2]) else 0.0
        return float(a1 - a2), theta, r, a_max, a_dirs

    # Observed statistic
    T_obs, theta_hat, r_hat, a_max_hat, a_dirs_obs = _compute_T(coord)

    # Bootstrap null distribution
    T_boot  = np.zeros(n_boot)
    center  = coord.mean(axis=0)
    for b in range(n_boot):
        angle_b      = rng.uniform(0, 2 * np.pi)
        cos_a, sin_a = np.cos(angle_b), np.sin(angle_b)
        R_b          = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        gs_rot       = (coord - center).dot(R_b.T) + center
        T_b, *_      = _compute_T(gs_rot)
        T_boot[b]    = T_b if not np.isnan(T_b) else 0.0

    p_value   = float(np.mean(T_boot >= T_obs)) if not np.isnan(T_obs) else np.nan
    reject_H0 = (p_value < alpha) if not np.isnan(p_value) else False

    # Plain-English interpretation (tier 1: effect size)
    if r_hat >= 0.85:
        tier1_msg = (f'r̂ = {r_hat:.3f} ≥ 0.85 — anisotropy is negligible.  '
                     f'Recommendation: use isotropic Pmodel.')
    elif r_hat >= 0.65:
        tier1_msg = (f'r̂ = {r_hat:.3f} ∈ [0.65, 0.85) — moderate anisotropy.  '
                     f'Formal test result is decisive.')
    else:
        tier1_msg = (f'r̂ = {r_hat:.3f} < 0.65 — strong anisotropy.  '
                     f'Principal axis: θ̂ = {np.degrees(theta_hat):.1f}°.')

    if not np.isnan(p_value):
        tier2_msg = (f'Bootstrap p-value = {p_value:.3f}.  '
                     f'H₀ (isotropy) '
                     f'{"rejected" if reject_H0 else "not rejected"} '
                     f'at α = {alpha}.')
    else:
        tier2_msg = 'Could not compute bootstrap test statistic.'

    return {
        'theta_hat':          float(theta_hat),
        'r_hat':              float(r_hat),
        'a_max_hat':          float(a_max_hat),
        'a_dirs':             a_dirs_obs,
        'test_stat':          T_obs,
        'T_boot':             T_boot,
        'p_value':            p_value,
        'reject_H0':          reject_H0,
        'message':            f'{tier1_msg}  {tier2_msg}',
    }
