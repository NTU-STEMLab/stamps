# -*- coding: utf-8 -*-
"""
Geographically Weighted Pmodel (GW-Pmodel) for nonstationary categorical BME.

Provides a container that stores per-region separable Pmodels (horizontal +
vertical) and marginals obtained from homogeneous region identification.
At estimation time, each location receives an **isotropic** Gaussian
distance-weighted blend of the regional Pmodels (bandwidth = per-region
``sigma``, i.e. max distance from centroid).

Per-region anisotropy parameters (``theta``, ``r``, ``a_max``) can still be
computed as diagnostic metadata via ``fit_regional_pmodels(..., anisotropy=True)``
but are **not** used in the blend weights.  Using anisotropic distance +
``a_max`` as bandwidth produced broad mixing kernels and a "phantom
correlation" artefact that destabilised BMEcatPdf — this module therefore
restores the original isotropic weighting scheme.

Workflow
--------
1. ``fit_regional_pmodels``  — compute per-region Pmodels from the output of
   ``probatablecalc_spatial`` + ``identify_homogeneous_regions``.
2. ``GWPmodel``              — container with ``blend_at(xy)`` method.
3. ``BMEcatPdf_GW``          — thin estimation wrapper that calls
   ``blend_at`` per estimation point, transforms coordinates, and delegates
   to the stationary ``BMEcatPdf``.
"""

import numpy as np
from scipy.spatial.distance import cdist

from ..fit import probatablefit, build_pmodel
from ..anisotropy import (
    polar_ptable_map,
    estimate_anisotropy_from_ptables,
)
from ...covariance.anisotropy import aniso2iso
from ..empirical import (
    probatablecalc,
    probatablecalc_directional,
    adaptive_bins,
)


# ---------------------------------------------------------------------------
# Offline: fit per-region models
# ---------------------------------------------------------------------------

def fit_regional_pmodels(
    coord,
    value,
    region_labels,
    region_stats,
    coord_limit_h,
    coord_limit_v=None,
    group_ids=None,
    n_fit=80,
    kstd_h=None,
    kstd_v=None,
    fit_method='spline',
    enforce_monotone=True,
    spline_smooth='auto',
    regularize=True,
    anisotropy=False,
    ang_step_deg=15,
    verbose=False,
):
    """Fit separable Pmodels (optionally plus diagnostic anisotropy) for each homogeneous region.

    .. note::
        Starting with the restored (stable) GW-Pmodel, the per-region
        horizontal Pmodel is fitted **isotropically** via
        :func:`probatablecalc` on raw Euclidean distances, and the
        :class:`GWPmodel` blend weights use an **isotropic** Gaussian
        kernel with bandwidth :math:`\\sigma_i` (max distance from
        centroid).  The earlier anisotropic weighting scheme (using
        ``theta``, ``r``, ``a_max``) produced broad mixing kernels and
        a "phantom correlation" artefact that destabilised BMEcatPdf.
        The ``anisotropy`` flag now defaults to ``False``.  Setting it
        to ``True`` computes ``theta``, ``r``, ``a_max`` as **diagnostic
        metadata only** — they are no longer used to weight the blend.
    

    Parameters
    ----------
    coord : (n, 3) array
        Full 3-D data coordinates ``[X, Y, Z]``.
    value : (n, nc) array
        One-hot or soft indicator matrix (same rows as *coord*).
    region_labels : (n,) array
        Region label per data point (from ``identify_homogeneous_regions``).
    region_stats : dict
        Output of ``identify_homogeneous_regions`` — must contain
        ``'n_centers'``, ``'centers'``, ``'P_mean'``, ``'P_std'``.
    coord_limit_h : (ncl_h,) array
        Lag bin edges for horizontal probability tables.
    coord_limit_v : (ncl_v,) array or None
        Lag bin edges for vertical tables.  If *None*, 12 equal-width bins
        spanning ``[0, max(|Z|)]`` are created automatically.
    group_ids : (n,) array or None
        Passed to ``probatablecalc`` for horizontal tables (e.g. depth-level
        grouping ``Z.astype(str)``).
    n_fit : int
        Number of points on the fitted distance grid for ``build_pmodel``.
    kstd_h, kstd_v : float or None
        Kernel bandwidth overrides; ``None`` → auto.
        **Only used when** ``fit_method='kernel'``.
    fit_method : ``{'spline', 'kernel'}``, optional
        Fitting back-end (default ``'spline'``).  The spline approach is
        robust to sparse regional data and produces no NaN gaps.
    enforce_monotone : bool
        Apply isotonic regression constraint in the spline fit (default True).
    spline_smooth : ``'auto'``, None, or float
        Smoothing parameter for the spline (default ``'auto'`` = LOOCV).
    regularize : bool
        Apply pseudo-count regularisation to each Pmodel.
    anisotropy : bool, default ``False``
        Whether to estimate per-region anisotropy (``theta``, ``r``,
        ``a_max``) as **diagnostic metadata**.  These values are stored
        in the region dict for user inspection only; they are **not**
        used by ``GWPmodel._weights`` (which always uses an isotropic
        Gaussian kernel with bandwidth ``sigma``).
    ang_step_deg : float
        Angular step for the directional sweep when ``anisotropy=True``.
    verbose : bool

    Returns
    -------
    regions : list of dict
        One dict per region with keys:

        - ``'centroid'``    : (2,) or (3,) XY(Z) centroid
        - ``'dmodel_h'``   : (nd_h,) fitted horizontal lag grid
        - ``'Pmodel_h'``   : (nc, nc, nd_h) fitted horizontal Pmodel
        - ``'dmodel_v'``   : (nd_v,) fitted vertical lag grid
        - ``'Pmodel_v'``   : (nc, nc, nd_v) fitted vertical Pmodel
        - ``'theta'``      : float, anisotropy angle (deg, CCW from X) — *diagnostic only*
        - ``'r'``          : float, anisotropy ratio in (0, 1] — *diagnostic only*
        - ``'a_max'``      : float, principal (max) effective range from
          the anisotropy ellipse fit — *diagnostic only*
        - ``'p_marginal'`` : (nc,) marginal class proportions
        - ``'N'``          : int, number of data points in region
        - ``'P_std_mean'`` : float, mean Pmodel table std (confidence)
        - ``'sigma'``      : float, max distance from centroid (m)
          — **used as Gaussian bandwidth in** ``GWPmodel._weights``
    """
    coord = np.asarray(coord, dtype=float)
    value = np.asarray(value, dtype=float)
    region_labels = np.asarray(region_labels)
    coord_limit_h = np.asarray(coord_limit_h, dtype=float).ravel()

    unique_labels = np.unique(region_labels)
    n_regions = len(unique_labels)
    nc = value.shape[1]

    if coord_limit_v is None:
        z_all = np.abs(coord[:, 2])
        z_max = z_all.max() if z_all.size > 0 else 1.0
        coord_limit_v = np.linspace(0, z_max, 13)

    n_bins_h = len(coord_limit_h) - 1
    d_max_h = float(coord_limit_h[-1])
    n_bins_v = len(coord_limit_v) - 1
    d_max_v = float(coord_limit_v[-1])

    regions = []

    for ri, label in enumerate(unique_labels):
        mask = region_labels == label
        c_reg = coord[mask]
        v_reg = value[mask]
        n_pts = int(mask.sum())

        if verbose:
            print(f'  Region {ri} (label={label}): {n_pts} points')

        # ── Centroid (XY only) ──────────────────────────────────────────
        centroid = c_reg[:, :2].mean(axis=0)

        # ── Geometric bandwidth: max distance from centroid ─────────────
        if n_pts > 1:
            dists_from_centroid = np.linalg.norm(
                c_reg[:, :2] - centroid[np.newaxis, :], axis=1)
            sigma = float(dists_from_centroid.max())
        else:
            sigma = float(d_max_h)

        # ── Horizontal Pmodel ───────────────────────────────────────────
        xy_reg = c_reg[:, :2]
        gids_reg = None
        if group_ids is not None:
            gids_reg = np.asarray(group_ids)[mask]

        D_h, P_h, O_h = probatablecalc(
            xy_reg, v_reg, coord_limit_h,
            group_ids=gids_reg, normalize_inputs=True)

        _bp_kwargs = dict(
            n_fit=n_fit,
            fit_method=fit_method,
            enforce_monotone=enforce_monotone,
            spline_smooth=spline_smooth,
            regularize=regularize,
            verbose=False,
        )
        if fit_method == 'kernel':
            _bp_kwargs['kstd'] = kstd_h if kstd_h is not None else 0.5 * d_max_h / n_bins_h
            _bp_kwargs['n_bins'] = n_bins_h

        try:
            dmodel_h, Pmodel_h, _ = probatablefit(
                D_h, P_h, O_h, d_max_emp=d_max_h, **_bp_kwargs)
        except Exception:
            dmodel_h = np.linspace(0, d_max_h, n_fit)
            Pmodel_h = np.zeros((nc, nc, n_fit))
            p_bg = v_reg.mean(axis=0)
            for lag in range(n_fit):
                Pmodel_h[:, :, lag] = np.outer(p_bg, p_bg)

        # ── Vertical Pmodel ─────────────────────────────────────────────
        z_reg = c_reg[:, 2:3]
        D_v, P_v, O_v = probatablecalc(
            z_reg, v_reg, coord_limit_v, normalize_inputs=True)

        _bp_kwargs_v = dict(
            n_fit=n_fit,
            fit_method=fit_method,
            enforce_monotone=enforce_monotone,
            spline_smooth=spline_smooth,
            regularize=regularize,
            verbose=False,
        )
        if fit_method == 'kernel':
            _bp_kwargs_v['kstd'] = kstd_v if kstd_v is not None else 0.5 * d_max_v / n_bins_v
            _bp_kwargs_v['n_bins'] = n_bins_v

        try:
            dmodel_v, Pmodel_v, _ = probatablefit(
                D_v, P_v, O_v, d_max_emp=d_max_v, **_bp_kwargs_v)
        except Exception:
            dmodel_v = np.linspace(0, d_max_v, n_fit)
            Pmodel_v = np.zeros((nc, nc, n_fit))
            p_bg = v_reg.mean(axis=0)
            for lag in range(n_fit):
                Pmodel_v[:, :, lag] = np.outer(p_bg, p_bg)

        # ── Marginal proportions ────────────────────────────────────────
        p_marginal = np.clip(v_reg.mean(axis=0), 1e-15, None)
        p_marginal /= p_marginal.sum()

        # ── Anisotropy ──────────────────────────────────────────────────
        theta_deg = 0.0
        r_hat = 1.0
        a_max_hat = float(d_max_h)          # fallback: horizontal Pmodel range
        if anisotropy and n_pts >= 20 and xy_reg.shape[0] >= 20:
            try:
                ang_step = ang_step_deg * np.pi / 180.0
                angLag = np.arange(-np.pi / 2, np.pi / 2, ang_step)
                rLag = coord_limit_h[1:]
                rLagTol = np.diff(coord_limit_h) / 2
                PP, OO, DD, angLag_out = polar_ptable_map(
                    xy_reg, v_reg, coord_limit_h,
                    rLag, rLagTol, angLag=angLag,
                    group_ids=gids_reg, plot=False)
                theta_rad, r_hat, _a_max, _, _, _ = (
                    estimate_anisotropy_from_ptables(
                        DD, PP, OO, angLag_out, p_marginal=p_marginal))
                theta_deg = float(np.degrees(theta_rad))
                if np.isfinite(_a_max) and _a_max > 0:
                    a_max_hat = float(_a_max)
            except Exception:
                theta_deg = 0.0
                r_hat = 1.0
        r_hat = float(max(min(r_hat, 1.0), 0.05))

        # ── Confidence metric ───────────────────────────────────────────
        P_std_arr = region_stats['P_std']
        if P_std_arr.ndim == 4 and ri < P_std_arr.shape[3]:
            _slice = P_std_arr[:, :, :, ri]
            _finite = _slice[np.isfinite(_slice)]
            P_std_mean = float(_finite.mean()) if _finite.size > 0 else 0.0
        else:
            P_std_mean = 0.0

        regions.append({
            'centroid':    centroid,
            'dmodel_h':   dmodel_h,
            'Pmodel_h':   Pmodel_h,
            'dmodel_v':   dmodel_v,
            'Pmodel_v':   Pmodel_v,
            'theta':      theta_deg,
            'r':          r_hat,
            'a_max':      a_max_hat,
            'p_marginal': p_marginal,
            'N':          n_pts,
            'P_std_mean': P_std_mean,
            'sigma':      sigma,
        })

    if verbose:
        print(f'Fitted {len(regions)} regional Pmodels')
    return regions


# ---------------------------------------------------------------------------
# GWPmodel container
# ---------------------------------------------------------------------------

class GWPmodel:
    """Geographically Weighted Pmodel for nonstationary BMEcatPdf.

    Stores a list of regional separable Pmodels and provides
    ``blend_at(xy)`` for per-point blending.
    """

    def __init__(self, regions, n_fit=80, fallback='global', scale_alpha=0.2):
        """
        Parameters
        ----------
        regions : list of dict
            Output of ``fit_regional_pmodels``.
        n_fit : int
            Shared lag-grid length for blending (all regional models are
            resampled to a common grid).
        fallback : {'global', 'nearest'}
            What to do when all weights are ≈ 0.
        scale_alpha : float
            Exponent for the data-size weight :math:`N_i^{\\alpha}`.
            Smaller values reduce the dominance of large regions.
            Default 0.2.
        """
        if not regions:
            raise ValueError('regions list is empty')
        self.regions = list(regions)
        self.n_regions = len(regions)
        self.n_fit = n_fit
        self.fallback = fallback
        self._scale_alpha = float(scale_alpha)

        nc = regions[0]['Pmodel_h'].shape[0]
        self.nc = nc

        # Build shared lag grids
        d_max_h = max(r['dmodel_h'][-1] for r in regions)
        d_max_v = max(r['dmodel_v'][-1] for r in regions)
        self.dmodel_h_common = np.linspace(0, d_max_h, n_fit)
        self.dmodel_v_common = np.linspace(0, d_max_v, n_fit)

        # Resample each regional Pmodel to the common grid
        for r in self.regions:
            r['_Pmodel_h_common'] = self._resample(
                r['dmodel_h'], r['Pmodel_h'], self.dmodel_h_common)
            r['_Pmodel_v_common'] = self._resample(
                r['dmodel_v'], r['Pmodel_v'], self.dmodel_v_common)

        # Pre-compute centroid array for fast distance calc
        self._centroids = np.array([r['centroid'][:2] for r in regions])

        # Global (equal-weight) model as fallback
        self._global_Pmodel_h = np.mean(
            [r['_Pmodel_h_common'] for r in regions], axis=0)
        self._global_Pmodel_v = np.mean(
            [r['_Pmodel_v_common'] for r in regions], axis=0)
        ws = np.array([r['N'] for r in regions], dtype=float)
        ws /= ws.sum()
        self._global_p_marginal = np.zeros(nc)
        for w, r in zip(ws, regions):
            self._global_p_marginal += w * r['p_marginal']
        self._global_p_marginal /= self._global_p_marginal.sum()
        self._global_theta = float(np.degrees(np.arctan2(
            np.sum(ws * np.sin(2 * np.radians([r['theta'] for r in regions]))),
            np.sum(ws * np.cos(2 * np.radians([r['theta'] for r in regions])))
        ) / 2))
        self._global_r = float(np.sum(ws * [r['r'] for r in regions]))

    # ---- internal helpers --------------------------------------------------

    @staticmethod
    def _resample(dmodel_old, Pmodel_old, dmodel_new):
        nc = Pmodel_old.shape[0]
        nd_new = len(dmodel_new)
        P_new = np.zeros((nc, nc, nd_new))
        for a in range(nc):
            for b in range(nc):
                P_new[a, b, :] = np.interp(
                    dmodel_new, dmodel_old, Pmodel_old[a, b, :])
        return P_new

    # ---- weight computation ------------------------------------------------

    def _weights(self, xy):
        """Combined importance weights for a single (x, y) location.

        .. math::

            W_i \\;\\propto\\; \\exp\\!\\left(-\\tfrac12 d_{\\text{norm},i}^2\\right)
                  \\;\\times\\; N_i^{\\alpha}

        where the **normalised isotropic distance** is

        .. math::

            d_{\\text{norm},i} = \\frac{\\|xy - c_i\\|}{\\sigma_i}

        *  :math:`\\|xy - c_i\\|` is the raw Euclidean distance from the
           query location to region *i*'s centroid.
        *  :math:`\\sigma_i` is the geometric bandwidth of region *i*
           (``reg['sigma']`` — max distance from centroid to any point
           in the region).  This is a tight localisation that keeps
           the blended Pmodel close to the nearest region's Pmodel
           and avoids the "phantom correlation" artefact that arises
           from arithmetic blending of joint Pmodels with differing
           marginals when weights are spread too broadly.
        *  The scale factor :math:`N_i^{\\alpha}` (default
           :math:`\\alpha=0.2`) gives mild preference to better-sampled
           regions without the double-counting that arises from using
           the raw count.

        Notes
        -----
        The earlier "anisotropic distance + ``a_max`` bandwidth" scheme
        was found to produce broad mixing kernels and unstable blends
        (phantom correlation), especially for minor classes.  Anisotropy
        parameters (``theta``, ``r``) from the regional fits are now
        retained only for diagnostic/inspection purposes in
        :func:`fit_regional_pmodels` and are **not** used here.
        """
        xy = np.asarray(xy, dtype=float).ravel()[:2]
        w = np.zeros(self.n_regions)
        for i, reg in enumerate(self.regions):
            cent = reg['centroid'][:2]
            sigma = float(reg.get('sigma', reg.get('a_max', 1.0)))

            dx = xy[0] - cent[0]
            dy = xy[1] - cent[1]
            d_iso = np.sqrt(dx * dx + dy * dy)

            d_norm = d_iso / max(sigma, 1e-6)
            geo_weight = np.exp(-0.5 * d_norm ** 2)

            scale_weight = float(reg['N']) ** self._scale_alpha

            w[i] = geo_weight * scale_weight

        w_sum = w.sum()
        if w_sum < 1e-30 or np.isnan(w_sum):
            return np.ones(self.n_regions) / self.n_regions
        return w / w_sum

    # ---- blending ----------------------------------------------------------

    def blend_at(self, xy):
        """Blend regional Pmodels at a single (x, y) location.

        Returns
        -------
        dmodel_h : (n_fit,) array
        Pmodel_h : (nc, nc, n_fit) array
        dmodel_v : (n_fit,) array
        Pmodel_v : (nc, nc, n_fit) array
        theta_eff : float  (degrees)
        r_eff : float
        p_marginal : (nc,) array
        """
        w = self._weights(xy)

        # Blend Pmodels on the common grid
        Pmodel_h_blend = np.zeros_like(self._global_Pmodel_h)
        Pmodel_v_blend = np.zeros_like(self._global_Pmodel_v)
        p_marginal_blend = np.zeros(self.nc)

        sin2t = 0.0
        cos2t = 0.0
        r_blend = 0.0

        for i, reg in enumerate(self.regions):
            wi = w[i]
            Pmodel_h_blend += wi * reg['_Pmodel_h_common']
            Pmodel_v_blend += wi * reg['_Pmodel_v_common']
            p_marginal_blend += wi * reg['p_marginal']

            # Circular mean for angle
            theta_rad = np.radians(reg['theta'])
            sin2t += wi * np.sin(2 * theta_rad)
            cos2t += wi * np.cos(2 * theta_rad)
            r_blend += wi * reg['r']

        # Normalise marginals
        p_marginal_blend = np.clip(p_marginal_blend, 1e-15, None)
        p_marginal_blend /= p_marginal_blend.sum()

        # Circular mean angle
        theta_eff = float(np.degrees(np.arctan2(sin2t, cos2t) / 2))
        r_eff = float(np.clip(r_blend, 0.05, 1.0))

        return (self.dmodel_h_common.copy(), Pmodel_h_blend,
                self.dmodel_v_common.copy(), Pmodel_v_blend,
                theta_eff, r_eff, p_marginal_blend)

    def blend_batch(self, xy_array):
        """Blend for multiple locations (vectorised convenience).

        Parameters
        ----------
        xy_array : (n, 2) array

        Returns
        -------
        results : list of 7-tuples (same as ``blend_at``)
        """
        xy_array = np.asarray(xy_array, dtype=float)
        return [self.blend_at(xy_array[i]) for i in range(xy_array.shape[0])]

    # ---- repr --------------------------------------------------------------

    def __repr__(self):
        return (f'GWPmodel(n_regions={self.n_regions}, nc={self.nc}, '
                f'n_fit={self.n_fit})')


# ---------------------------------------------------------------------------
# Online: estimation wrappers
# ---------------------------------------------------------------------------



def fit_covariate_pmodel(
    covariate_values,
    indicators,
    n_bins=20,
    n_fit=80,
    kstd=None,
):
    """Compute a 1-D bivariate probability model along a covariate axis.

    Bins sample pairs by the absolute difference in *covariate_values*,
    computes empirical bivariate tables in each bin, and fits a smooth
    Pmodel via :func:`build_pmodel`.

    Parameters
    ----------
    covariate_values : (n,) array
        Scalar covariate value per data point.
    indicators : (n, nc) array
        One-hot or soft-indicator matrix (same rows as *covariate_values*).
    n_bins : int
        Number of equal-width bins spanning ``[0, max_diff]``.
    n_fit : int
        Points on the fitted distance grid.
    kstd : float or None
        Kernel bandwidth override for ``build_pmodel``; ``None`` = auto.

    Returns
    -------
    dmodel : (n_fit,) ndarray
        Fitted distance (covariate-difference) grid.
    Pmodel : (nc, nc, n_fit) ndarray
        Fitted bivariate probability tables.
    D_emp : (nc, nc, n_bins) ndarray
        Empirical lag bin centres.
    P_emp : (nc, nc, n_bins) ndarray
        Empirical bivariate tables.
    """
    covariate_values = np.asarray(covariate_values, dtype=float).ravel()
    indicators = np.asarray(indicators, dtype=float)
    n = len(covariate_values)
    nc = indicators.shape[1]

    diffs = np.abs(covariate_values[:, None] - covariate_values[None, :])
    max_diff = np.nanmax(diffs)
    if max_diff < 1e-15:
        max_diff = 1.0

    bin_edges = np.linspace(0, max_diff, n_bins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    P_emp = np.zeros((nc, nc, n_bins))
    O_emp = np.zeros(n_bins)

    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if b == 0:
            mask = (diffs >= lo) & (diffs <= hi)
        else:
            mask = (diffs > lo) & (diffs <= hi)
        np.fill_diagonal(mask, False)

        ii, jj = np.where(mask)
        if len(ii) == 0:
            P_emp[:, :, b] = np.nan
            continue

        O_emp[b] = len(ii)
        for ci in range(nc):
            for cj in range(nc):
                P_emp[ci, cj, b] = np.sum(
                    indicators[ii, ci] * indicators[jj, cj]
                )

        bsum = P_emp[:, :, b].sum()
        if bsum > 0:
            P_emp[:, :, b] /= bsum

    # Keep only bins that have at least one pair (drop empty / nan bins)
    valid = O_emp > 0
    if not np.any(valid):
        raise ValueError(
            "fit_covariate_pmodel: no valid covariate-difference pairs found. "
            "Check that covariate_values has sufficient variance."
        )
    D_emp = bin_centres[valid]          # (n_valid,)
    P_emp = P_emp[:, :, valid]          # (nc, nc, n_valid)
    O_emp = O_emp[valid]                # (n_valid,)

    kwargs = {}
    if kstd is not None:
        kwargs['kstd'] = kstd

    dmodel, Pmodel, _ = probatablefit(
        D_emp, P_emp, O_emp, n_fit=n_fit, **kwargs
    )

    return dmodel, Pmodel, D_emp, P_emp


def fit_covariate_interaction_pmodel(
    cov1_values,
    cov2_values,
    indicators,
    n_bins_1=10,
    n_bins_2=10,
    n_fit_1=40,
    n_fit_2=40,
    sigma=None,
):
    """Fit a 2-D bivariate probability model on the joint covariate-difference space.

    Bins sample pairs by ``(|Δcov1|, |Δcov2|)`` simultaneously, computes
    empirical bivariate transition tables in each 2-D cell, smooths the
    surface with a Gaussian kernel, and evaluates onto a regular output grid.

    This captures **second-order interaction** effects between two covariates
    that the additive (main-effect-only) Approach B cannot represent.

    Parameters
    ----------
    cov1_values, cov2_values : (n,) array
        Scalar covariate values per data point.
    indicators : (n, nc) array
        One-hot or soft-indicator matrix (same rows as covariate arrays).
    n_bins_1, n_bins_2 : int
        Number of equal-width bins per covariate-difference axis.
    n_fit_1, n_fit_2 : int
        Points on the fitted output grids.
    sigma : float or None
        Bandwidth (in bin-index units) for 2-D Gaussian smoothing applied
        to each ``(cat_a, cat_b)`` slice.  ``None`` selects
        ``max(1.0, min(n_bins_1, n_bins_2) / 5)``.

    Returns
    -------
    dmodel1 : (n_fit_1,) ndarray
        Fitted distance grid for covariate-1 differences.
    dmodel2 : (n_fit_2,) ndarray
        Fitted distance grid for covariate-2 differences.
    Pmodel : (nc, nc, n_fit_1, n_fit_2) ndarray
        Fitted 2-D bivariate probability tables.
    P_emp : (nc, nc, n_bins_1, n_bins_2) ndarray
        Raw empirical tables (NaN where no pairs fell).
    """
    from scipy.ndimage import gaussian_filter
    from scipy.interpolate import RegularGridInterpolator

    cov1 = np.asarray(cov1_values, dtype=float).ravel()
    cov2 = np.asarray(cov2_values, dtype=float).ravel()
    indicators = np.asarray(indicators, dtype=float)
    n = len(cov1)
    nc = indicators.shape[1]

    diff1 = np.abs(cov1[:, None] - cov1[None, :])
    diff2 = np.abs(cov2[:, None] - cov2[None, :])

    max_d1 = float(np.nanmax(diff1))
    max_d2 = float(np.nanmax(diff2))
    if max_d1 < 1e-15:
        max_d1 = 1.0
    if max_d2 < 1e-15:
        max_d2 = 1.0

    edges1 = np.linspace(0, max_d1, n_bins_1 + 1)
    edges2 = np.linspace(0, max_d2, n_bins_2 + 1)
    centres1 = 0.5 * (edges1[:-1] + edges1[1:])
    centres2 = 0.5 * (edges2[:-1] + edges2[1:])

    P_emp = np.full((nc, nc, n_bins_1, n_bins_2), np.nan)
    O_emp = np.zeros((n_bins_1, n_bins_2))

    iu, ju = np.triu_indices(n, k=1)

    d1_pairs = diff1[iu, ju]
    d2_pairs = diff2[iu, ju]

    b1_idx = np.digitize(d1_pairs, edges1) - 1
    b2_idx = np.digitize(d2_pairs, edges2) - 1
    b1_idx = np.clip(b1_idx, 0, n_bins_1 - 1)
    b2_idx = np.clip(b2_idx, 0, n_bins_2 - 1)

    for p_idx in range(len(iu)):
        bi, bj = int(b1_idx[p_idx]), int(b2_idx[p_idx])
        ii, jj = int(iu[p_idx]), int(ju[p_idx])
        if np.isnan(P_emp[0, 0, bi, bj]):
            P_emp[:, :, bi, bj] = 0.0
        O_emp[bi, bj] += 1
        for ca in range(nc):
            for cb in range(nc):
                P_emp[ca, cb, bi, bj] += indicators[ii, ca] * indicators[jj, cb]

    for bi in range(n_bins_1):
        for bj in range(n_bins_2):
            s = np.nansum(P_emp[:, :, bi, bj])
            if s > 0:
                P_emp[:, :, bi, bj] /= s

    p_global = np.zeros((nc, nc))
    for ca in range(nc):
        for cb in range(nc):
            p_global[ca, cb] = np.sum(indicators[:, ca]) * np.sum(indicators[:, cb])
    p_global /= p_global.sum()

    P_smooth = np.zeros((nc, nc, n_bins_1, n_bins_2))
    if sigma is None:
        sigma = max(1.0, min(n_bins_1, n_bins_2) / 5.0)

    for ca in range(nc):
        for cb in range(nc):
            slc = P_emp[ca, cb].copy()
            nan_mask = np.isnan(slc)
            slc[nan_mask] = p_global[ca, cb]
            P_smooth[ca, cb] = gaussian_filter(slc, sigma=sigma)

    for bi in range(n_bins_1):
        for bj in range(n_bins_2):
            s = P_smooth[:, :, bi, bj].sum()
            if s > 0:
                P_smooth[:, :, bi, bj] /= s
            else:
                P_smooth[:, :, bi, bj] = p_global

    dmodel1 = np.linspace(0, max_d1, n_fit_1)
    dmodel2 = np.linspace(0, max_d2, n_fit_2)

    Pmodel = np.zeros((nc, nc, n_fit_1, n_fit_2))
    for ca in range(nc):
        for cb in range(nc):
            interp = RegularGridInterpolator(
                (centres1, centres2), P_smooth[ca, cb],
                method='linear', bounds_error=False, fill_value=None,
            )
            g1, g2 = np.meshgrid(dmodel1, dmodel2, indexing='ij')
            pts = np.column_stack([g1.ravel(), g2.ravel()])
            Pmodel[ca, cb] = interp(pts).reshape(n_fit_1, n_fit_2)

    for i1 in range(n_fit_1):
        for i2 in range(n_fit_2):
            s = Pmodel[:, :, i1, i2].sum()
            if s > 0:
                Pmodel[:, :, i1, i2] /= s
            else:
                Pmodel[:, :, i1, i2] = p_global

    return dmodel1, dmodel2, Pmodel, P_emp


