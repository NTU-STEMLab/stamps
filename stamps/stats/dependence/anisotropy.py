# -*- coding: utf-8 -*-
"""
Spatial and spatiotemporal anisotropy analysis functions.

Continuous-data functions (covariance-based)
--------------------------------------------
polarstcovmap              : Polar plot of directional empirical covariance.
iso2aniso                  : Convert isotropic to anisotropic coordinates.
aniso2iso                  : Convert anisotropic to isotropic coordinates.
covariancemap              : 2-D empirical covariance map (covariance surface).
estimate_anisotropy_params : Fit anisotropy ellipse to directional ranges.
isotropy_test              : Formal statistical test for spatial isotropy.

Categorical-data functions (probability-table-based)
-----------------------------------------------------
polar_ptable_map                : Directional probability-table sweep and plot.
estimate_anisotropy_from_ptables: Fit anisotropy ellipse from P-table ranges.
isotropy_test_categorical       : Bootstrap isotropy test for categorical data.
"""
import numpy
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import chi2

from .covariance.stcov import stcov
from ...estimation import stmean


# ---------------------------------------------------------------------------
# Existing functions (with bug fixes and updated docstrings)
# ---------------------------------------------------------------------------

def polarstcovmap(cMS, tME, Zst, rLag, rLagTol, tLag=None, tLagTol=None,
                  angLag=None, angTol=None, plot=True):
    """Plot polar map of the directional empirical spatial covariance.

    Parameters
    ----------
    cMS     : (ns, 2) spatial coordinates of observation stations.
    tME     : (nt,) temporal coordinates.
    Zst     : (ns, nt) observation values.
    rLag    : (nls,) spatial lag distances.
    rLagTol : (nls,) spatial lag tolerances.
    tLag    : (nlt,) temporal lags.  ``None`` for purely spatial data.
    tLagTol : (nlt,) temporal lag tolerances.
    angLag  : (na,) directions in radians.  Default: 6 directions in
              ``[-pi/2, pi/2)``.
    angTol  : (na,) angular tolerances.  Default: ``step/2``.
    plot    : bool.  ``True`` to display the polar covariance map.

    Returns
    -------
    CC     : list of (nls, nlt) arrays — directional empirical covariances.
    CCn    : list of (nls, nlt) arrays — directional pair counts.
    lagr   : (nls, nlt) spatial lag meshgrid.
    lagt   : (nls, nlt) temporal lag meshgrid.
    angLag : (na,) evaluated directions.

    References
    ----------
    Goovaerts, P. (1997). *Geostatistics for Natural Resources Evaluation*.
        Oxford University Press. Section 4.3 (directional covariance
        computation and polar plot).
    """
    if angLag is None:
        ang_inic = -numpy.pi / 2
        ang_step = 30 / 180. * numpy.pi
        angLag = numpy.arange(ang_inic, numpy.pi / 2, ang_step)
        angTol = numpy.ones(angLag.shape) * ang_step / 2

    CC = []
    CCn = []
    CC4plot = numpy.empty([rLag.size, 0])
    for i, ang in enumerate(angLag):
        if tLag is not None:
            C, Cn, lagr, lagt = stcov(cMS, tME, Zst, rLag, rLagTol,
                                      tLag, tLagTol, ang=ang,
                                      angtol=angTol[i])
        else:
            C, Cn, lagr, lagt = stcov(cMS, tME, Zst, rLag, rLagTol,
                                      ang=ang, angtol=angTol[i])
        CC.append(C)
        CCn.append(Cn)
        CC4plot = numpy.hstack((CC4plot, C[:, 0:1]))

    if plot is True:
        idx = numpy.where(numpy.isnan(CC4plot))
        if idx[0].size > 0:
            CC3 = numpy.hstack((CC4plot, numpy.hstack((CC4plot, CC4plot))))
            angLag3 = numpy.hstack(
                (angLag - numpy.pi, numpy.hstack((angLag, angLag + numpy.pi))))
            try:
                CC1 = stmean.stmeaninterp(lagr, angLag3, CC3, lagr, angLag,
                                          method='linear')
            except Exception:
                CC1 = stmean.stmeaninterp(lagr, angLag3, CC3, lagr, angLag,
                                          method='nearest')
            for k in range(int(idx[0].size)):
                CC4plot[idx[0][k], idx[1][k]] = CC1[idx[0][k], idx[1][k]]

        angLag4plot = numpy.hstack((angLag, angLag + numpy.pi))
        CC4plot = numpy.hstack((CC4plot, CC4plot))
        # M1 fix: use a step consistent with the first tolerance value.
        angLag4plot = numpy.append(angLag4plot,
                                   angLag4plot[-1] + angTol[0] * 2)
        CC4plot = numpy.hstack((CC4plot, CC4plot[:, 0:1]))

        angLag4plot = angLag4plot - angTol[0]
        theta, radius = numpy.meshgrid(angLag4plot, rLag)
        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)
        im = ax.pcolor(theta, radius, CC4plot, cmap='hot_r')
        plt.colorbar(im)
        plt.show()

    return CC, CCn, lagr, lagt, angLag


def iso2aniso(ciso, angle, ratio):
    """Convert isotropic coordinates to anisotropic coordinates.

    Applies the inverse of the :func:`aniso2iso` transformation: maps
    locations from the isotropic space back into the anisotropic space by
    rescaling the secondary axis and rotating back.

    Parameters
    ----------
    ciso  : (n, d) array  Coordinates in the isotropic space.
    angle : scalar or (d-1,) array  Principal-axis angle(s) in **degrees**,
            measured counter-clockwise from the x-axis.
    ratio : scalar or (d-1,) array  Ratio of secondary to principal axis
            length (2D), in (0, 1].

    Returns
    -------
    c : (n, d) array  Coordinates in the anisotropic space.

    Notes
    -----
    The input array ``ciso`` is **not** modified (a copy is made internally).

    References
    ----------
    Journel, A.G. & Huijbregts, C.J. (1978). *Mining Geostatistics*.
        Academic Press, London. Section I.4, inverse of Eq. I.4.3
        (anisotropic coordinate transformation).
    """
    if type(ciso) is numpy.ndarray:
        d = ciso.shape[1]
    else:
        ciso = numpy.array(ciso, ndmin=2)
        d = ciso.shape[1]
    # C2 fix: copy so the caller's array is never modified in-place.
    ciso = ciso.copy()
    angle = numpy.asarray(angle, dtype=float) * 2 * numpy.pi / 360
    angle = -angle

    if (d < 2) or (d > 3):
        raise ValueError(
            'iso2aniso requires coordinates in a 2D or 3D space')

    if d == 2:
        R = numpy.array([[numpy.cos(angle), numpy.sin(angle)],
                         [-numpy.sin(angle), numpy.cos(angle)]])
        ciso[:, 1] = ciso[:, 1] * ratio
        c = ciso.dot(R.T)

    if d == 3:
        phi = angle[0]
        teta = angle[1]
        ratioy = ratio[0]
        ratioz = ratio[1]

        R1 = numpy.array([[numpy.cos(phi), numpy.sin(phi), 0],
                          [-numpy.sin(phi), numpy.cos(phi), 0],
                          [0, 0, 1]])
        R2 = numpy.array([[numpy.cos(teta), 0, numpy.sin(teta)],
                          [0, 1, 0],
                          [-numpy.sin(teta), 0, numpy.cos(teta)]])
        # C5 fix: use the correct matrix product order for the 3D inverse.
        # aniso2iso uses R=(R1.T)@(R2.T); its transpose is R2@R1, which is
        # the correct rotation for iso2aniso.
        R = R2.dot(R1)
        ciso[:, 1] = ciso[:, 1] * ratioy
        ciso[:, 2] = ciso[:, 2] * ratioz
        c = ciso.dot(R)

    return c


def aniso2iso(c, angle, ratio):
    """Convert anisotropic coordinates to isotropic coordinates.

    Transforms 2-D or 3-D coordinates from an anisotropic space into an
    isotropic one by rotating to align the principal axis with the x-axis and
    then scaling the secondary axis(es).  The net effect is that an ellipse
    (ellipsoid) in the original space maps to a circle (sphere) whose radius
    equals the principal axis length.

    Parameters
    ----------
    c     : (n, d) array  Coordinates in the anisotropic space.
    angle : scalar or (d-1,) array  Principal-axis angle(s) in **degrees**,
            measured counter-clockwise from the x-axis.
    ratio : scalar or (d-1,) array  Ratio of secondary to principal axis
            length, in (0, 1].

    Returns
    -------
    ciso : (n, d) array  Coordinates in the isotropic space.

    References
    ----------
    Journel, A.G. & Huijbregts, C.J. (1978). *Mining Geostatistics*.
        Academic Press, London. Section I.4, Eq. I.4.3
        (anisotropic coordinate transformation).
    """
    if type(c) is numpy.ndarray:
        d = c.shape[1]
    else:
        c = numpy.array(c, ndmin=2)
        d = c.shape[1]
    angle = numpy.asarray(angle, dtype=float) * 2 * numpy.pi / 360

    if (d < 2) or (d > 3):
        raise ValueError(
            'aniso2iso requires coordinates in a 2D or 3D space')

    if d == 2:
        R = numpy.array([[numpy.cos(angle), numpy.sin(angle)],
                         [-numpy.sin(angle), numpy.cos(angle)]])
        ciso = c.dot(R.T)
        ciso[:, 1] = ciso[:, 1] * 1. / ratio

    if d == 3:
        phi = angle[0]
        teta = angle[1]
        ratioy = ratio[0]
        ratioz = ratio[1]

        R1 = numpy.array([[numpy.cos(phi), numpy.sin(phi), 0],
                          [-numpy.sin(phi), numpy.cos(phi), 0],
                          [0, 0, 1]])
        R2 = numpy.array([[numpy.cos(teta), 0, numpy.sin(teta)],
                          [0, 1, 0],
                          [-numpy.sin(teta), 0, numpy.cos(teta)]])
        R = (R1.T).dot(R2.T)
        ciso = c.dot(R)
        ciso[:, 1] = ciso[:, 1] * 1. / ratioy
        ciso[:, 2] = ciso[:, 2] * 1. / ratioz

    return ciso


# ---------------------------------------------------------------------------
# Phase 1 — New functions
# ---------------------------------------------------------------------------

def covariancemap(grid_s, grid_v, dx_range, dy_range, n_bins, tol_frac=1.0):
    """Compute the 2-D empirical covariance map C(hx, hy).

    Bins all pairs (i, j) by their spatial displacement vector
    ``(Δx, Δy) = s_i − s_j`` and computes the sample covariance for each
    bin.  Iso-covariance contours on the resulting map reveal the anisotropy
    structure: elliptical contours indicate geometric anisotropy.

    Parameters
    ----------
    grid_s   : (n, 2) array  Spatial coordinates of observations.
    grid_v   : (n,) or (n, nt) array  Observation values.  If 2-D, the
               spatial covariance is averaged over all available time steps.
    dx_range : float  Half-extent of the lag-x axis (e.g. half the max dist).
    dy_range : float  Half-extent of the lag-y axis.
    n_bins   : int    Number of bins along each axis (map is n_bins × n_bins).
    tol_frac : float  Fraction of the bin step to use as the bin half-width
               (1.0 = no overlap, 1.5 = 50 % overlap). Default 1.0.

    Returns
    -------
    C_map  : (n_bins, n_bins) array  Covariance values (NaN where no pairs).
    N_map  : (n_bins, n_bins) int array  Pair counts per bin.
    hx_grid : (n_bins,) array  Lag-x bin centres.
    hy_grid : (n_bins,) array  Lag-y bin centres.

    Notes
    -----
    The map exploits the symmetry C(hx, hy) = C(-hx, -hy) by accumulating
    both (i,j) and (j,i) pairs into their respective bins.  Pairs with
    ``hx = Δx < 0`` are mirrored by swapping sign, so each pair is counted
    twice (once in each symmetric bin), which is correct for the average.

    References
    ----------
    Cressie, N.A.C. (1993). *Statistics for Spatial Data* (rev. ed.).
        Wiley, New York. Section 2.4 (2-D covariance surface / covariance map).
    """
    grid_s = np.asarray(grid_s)
    grid_v = np.asarray(grid_v, dtype=float)
    if grid_v.ndim == 2:
        # average across time, ignoring NaN
        z = np.nanmean(grid_v, axis=1)
    else:
        z = grid_v.ravel()

    n = grid_s.shape[0]
    z = z - np.nanmean(z)  # remove mean before cross-product

    hx_grid = np.linspace(-dx_range, dx_range, n_bins)
    hy_grid = np.linspace(-dy_range, dy_range, n_bins)
    dx_step = hx_grid[1] - hx_grid[0] if n_bins > 1 else dx_range
    dy_step = hy_grid[1] - hy_grid[0] if n_bins > 1 else dy_range
    dx_tol = dx_step * tol_frac / 2.0
    dy_tol = dy_step * tol_frac / 2.0

    C_map = np.zeros((n_bins, n_bins))
    N_map = np.zeros((n_bins, n_bins), dtype=int)

    for i in range(n):
        for j in range(i + 1, n):
            dxij = grid_s[i, 0] - grid_s[j, 0]
            dyij = grid_s[i, 1] - grid_s[j, 1]
            zz = z[i] * z[j]
            # accumulate both (dxij, dyij) and (-dxij, -dyij) bins
            for sign in (1, -1):
                dx_s, dy_s = sign * dxij, sign * dyij
                ix = np.where(np.abs(hx_grid - dx_s) <= dx_tol)[0]
                iy = np.where(np.abs(hy_grid - dy_s) <= dy_tol)[0]
                for ixi in ix:
                    for iyi in iy:
                        C_map[iyi, ixi] += zz
                        N_map[iyi, ixi] += 1

    C_map_out = np.full((n_bins, n_bins), np.nan)
    mask = N_map > 0
    C_map_out[mask] = C_map[mask] / N_map[mask]

    return C_map_out, N_map, hx_grid, hy_grid


def estimate_anisotropy_params(CC, CCn, angLag, lagr, sill=None):
    """Estimate the principal-axis angle and anisotropy ratio from directional covariances.

    Extracts the effective spatial range at each direction from the output of
    :func:`polarstcovmap` and fits an anisotropy ellipse to the directional
    ranges by weighted least squares.

    Parameters
    ----------
    CC     : list of (nls, nlt) arrays  Directional covariances from
             :func:`polarstcovmap`.  Only the purely spatial lag (column 0)
             is used.
    CCn    : list of (nls, nlt) arrays  Pair counts from :func:`polarstcovmap`.
    angLag : (na,) array  Directions in radians evaluated by
             :func:`polarstcovmap`.
    lagr   : (nls, nlt) array  Spatial lag meshgrid from :func:`polarstcovmap`.
    sill   : float, optional  The sill value.  If ``None``, estimated as the
             maximum observed covariance across all directions.

    Returns
    -------
    theta_hat : float  Estimated principal-axis angle in radians, in
                ``(-pi/2, pi/2]``.
    r_hat     : float  Estimated anisotropy ratio (secondary / principal
                range), in ``(0, 1]``.  ``r_hat = 1`` means isotropy.
    a_max_hat : float  Estimated principal (maximum) effective range.
    a_dirs    : (na,) array  Effective ranges extracted at each direction.
    valid     : (na,) bool array  ``True`` if an effective range was found
                for that direction.

    Notes
    -----
    The effective range at direction ``φ`` is defined as the lag ``h`` where
    the covariance drops below ``exp(-1)`` times the sill.  The anisotropy
    ellipse model is:

    .. math::

        a(\\phi) = \\frac{a_{max} \\cdot r}{\\sqrt{r^2 \\cos^2(\\phi-\\theta)
                   + \\sin^2(\\phi-\\theta)}}

    The fit minimises the weighted sum of squared residuals with weights
    equal to the average pair count at each direction.

    References
    ----------
    Zimmerman, D.L. (1993). Another look at anisotropy in geostatistics.
        *Mathematical Geology*, 25(4), 453–470.
        [Directional range ratio as isotropy test statistic.]
    Goovaerts, P. (1997). *Geostatistics for Natural Resources Evaluation*.
        Oxford University Press. Section 4.3 (ellipse fitting to directional
        ranges, Eq. 4.5).
    """
    angLag = np.asarray(angLag)
    lagr_1d = lagr[:, 0]  # purely spatial lags (column 0)
    na = len(angLag)

    # --- estimate sill ---
    if sill is None:
        sill = max(float(np.nanmax(CC[i][:, 0])) for i in range(na))
    threshold = np.exp(-1) * sill

    # --- extract effective range per direction ---
    a_dirs = np.full(na, np.nan)
    weights = np.zeros(na)
    for i in range(na):
        c_dir = CC[i][:, 0]  # covariance at spatial lags for direction i
        n_dir = CCn[i][:, 0]
        valid_mask = ~np.isnan(c_dir) & (n_dir > 0)
        if valid_mask.sum() < 2:
            continue
        c_v = c_dir[valid_mask]
        h_v = lagr_1d[valid_mask]
        # find first lag where C drops below threshold
        below = np.where(c_v <= threshold)[0]
        if below.size == 0:
            continue
        idx = below[0]
        if idx == 0:
            a_dirs[i] = h_v[0]
        else:
            # linear interpolation between idx-1 and idx
            h0, h1 = h_v[idx - 1], h_v[idx]
            c0, c1 = c_v[idx - 1], c_v[idx]
            if c1 < c0:
                a_dirs[i] = h0 + (threshold - c0) / (c1 - c0) * (h1 - h0)
            else:
                a_dirs[i] = h_v[idx]
        weights[i] = np.nanmean(n_dir[valid_mask])

    valid = ~np.isnan(a_dirs)
    if valid.sum() < 2:
        # Cannot fit ellipse with fewer than 2 valid directions
        return 0.0, 1.0, np.nanmean(a_dirs[valid]) if valid.sum() > 0 else 1.0, a_dirs, valid

    # --- fit anisotropy ellipse by WLS ---
    def ellipse_range(phi, theta, r, a_max):
        denom = np.sqrt(r**2 * np.cos(phi - theta)**2 + np.sin(phi - theta)**2)
        return a_max * r / denom

    def residuals(params):
        theta, r, a_max = params
        if r <= 0 or r > 1 or a_max <= 0:
            return 1e12
        pred = ellipse_range(angLag[valid], theta, r, a_max)
        resid = weights[valid] * (a_dirs[valid] - pred)**2
        return resid.sum()

    # initial guess: isotropic with mean range
    a0 = np.nanmean(a_dirs[valid])
    x0 = [0.0, 0.8, a0]
    bounds = [(-np.pi / 2, np.pi / 2), (0.01, 1.0), (1e-6, None)]
    try:
        result = minimize(residuals, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 500, 'ftol': 1e-12})
        theta_hat, r_hat, a_max_hat = result.x
    except Exception:
        theta_hat, r_hat, a_max_hat = 0.0, 1.0, a0

    # normalize theta to (-pi/2, pi/2]
    while theta_hat > np.pi / 2:
        theta_hat -= np.pi
    while theta_hat <= -np.pi / 2:
        theta_hat += np.pi

    return theta_hat, float(r_hat), float(a_max_hat), a_dirs, valid


def isotropy_test(grid_s, grid_v, lagr, lagr_tol, angLag, angTol,
                  method='bootstrap', n_boot=499, alpha=0.05,
                  covmodel=None, covparam=None, seed=None):
    """Formal statistical test of the isotropy hypothesis for 2-D spatial data.

    Implements a tiered scientific workflow for deciding whether the observed
    spatial covariance is consistent with isotropy:

    * **'bootstrap'**  Spatial rotation bootstrap directional covariance
      contrast test (model-free, recommended for irregular data).
    * **'lrt'**        Gaussian likelihood ratio test (requires a fitted
      covariance model; ``covmodel`` and ``covparam`` must be provided).
    * **'symmetry'**   Axial symmetry screening test (fast, tests a necessary
      but not sufficient condition for isotropy).
    * **'all'**        Run all three and return combined results.

    Parameters
    ----------
    grid_s    : (n, 2) array  2-D spatial coordinates.
    grid_v    : (n,) or (n, nt) array  Observation values.
    lagr      : (L,) array  Reference spatial lags for the directional test.
    lagr_tol  : (L,) array  Lag tolerances.
    angLag    : (na,) array  Directions in radians, in ``(-pi/2, pi/2]``.
    angTol    : (na,) array  Angular tolerances.
    method    : str  ``'bootstrap'``, ``'lrt'``, ``'symmetry'``, or ``'all'``.
    n_boot    : int  Number of bootstrap rotations (default 499).
    alpha     : float  Significance level (default 0.05).
    covmodel  : list  Covariance model for ``'lrt'`` method.
    covparam  : list  Initial covariance parameters for ``'lrt'`` method.
    seed      : int or None  Random seed for reproducibility.

    Returns
    -------
    result : dict with keys
        ``'method'``    — str, method(s) used.
        ``'test_stat'`` — float or dict, observed test statistic(s).
        ``'p_value'``   — float or dict, p-value(s).
        ``'reject_H0'`` — bool or dict, whether H₀ is rejected.
        ``'r_hat'``     — float, estimated anisotropy ratio (effect size).
        ``'theta_hat'`` — float, estimated principal-axis angle (radians).
        ``'message'``   — str, plain-English interpretation.

    Notes
    -----
    The **bootstrap test** statistic is:

    .. math::

        T_n = \\max_{l=1,\\ldots,L} |\\hat{C}(h_l, \\phi_1) -
              \\hat{C}(h_l, \\phi_2)|

    where ``(φ₁, φ₂)`` are the estimated principal and secondary axes.  The
    null distribution is obtained by randomly rotating all coordinates by
    ``B`` uniformly drawn angles and recomputing ``T_n`` each time.

    The **LRT** statistic is ``Λ = 2[ℓ(aniso) − ℓ(iso)] ~ χ²(2)`` under
    ``H₀``.  It requires that ``mlecovfitv`` has been run (or can be run
    here) for both isotropic and anisotropic models.

    The **symmetry test** checks whether
    ``Ĉ(hx, hy) ≈ Ĉ(hy, hx)`` using the covariance map.

    References
    ----------
    Guan, Y., Sherman, M., & Calvin, J.A. (2004). A nonparametric test for
        spatial isotropy using subsampling. *Journal of the American
        Statistical Association*, 99(467), 810–821.
        [Test statistic concept; rotation bootstrap replaces subsampling for
        small n.]
    Maity, A. & Sherman, M. (2012). Testing for spatial isotropy under
        general designs. *Journal of Statistical Planning and Inference*,
        142(5), 1081–1091.
    Scaccia, L. & Martin, R.J. (2005). Testing axial symmetry and
        separability of lattice processes. *Journal of Statistical Planning
        and Inference*, 131(1), 19–39.
        [Axial symmetry statistic, Section 2.]
    Self, S.G. & Liang, K.-Y. (1987). Asymptotic properties of maximum
        likelihood estimators and likelihood ratio statistics under
        non-standard conditions. *Journal of the American Statistical
        Association*, 82(398), 605–610.
        [LRT boundary case: Λ ~ χ²(2) even though r=1 is a boundary point.]
    Davison, A.C. & Hinkley, D.V. (1997). *Bootstrap Methods and Their
        Application*. Cambridge University Press. Chapter 2 (rotation
        bootstrap as a symmetry bootstrap).
    Goovaerts, P. (1997). *Geostatistics for Natural Resources Evaluation*.
        Oxford University Press. Section 4.3 (practical anisotropy ratio
        criteria, r̂ thresholds).
    """
    from .stcov import stcov as _stcov

    grid_s = np.asarray(grid_s, dtype=float)
    grid_v = np.asarray(grid_v, dtype=float)
    lagr = np.asarray(lagr)
    lagr_tol = np.asarray(lagr_tol)
    angLag = np.asarray(angLag)
    angTol = np.asarray(angTol)
    rng = np.random.default_rng(seed)

    # ---- Tier 2: estimate effect size (r_hat, theta_hat) -----------------
    tME = np.array([0.])
    grid_v2d = grid_v if grid_v.ndim == 2 else grid_v.reshape(-1, 1)
    CC, CCn, _lagr, _lagt, _ = polarstcovmap(
        grid_s, tME, grid_v2d, lagr, lagr_tol,
        angLag=angLag, angTol=angTol, plot=False)
    theta_hat, r_hat, a_max_hat, a_dirs, valid = estimate_anisotropy_params(
        CC, CCn, angLag, _lagr)

    # Find two orthogonal directions closest to theta_hat and theta_hat+pi/2
    phi1_idx = np.argmin(np.abs(angLag - theta_hat))
    phi2_angle = theta_hat + np.pi / 2
    if phi2_angle > np.pi / 2:
        phi2_angle -= np.pi
    phi2_idx = np.argmin(np.abs(angLag - phi2_angle))

    def _directional_contrast(gs, gv):
        """Compute T_n for a given coordinate set."""
        gv2d = gv if gv.ndim == 2 else gv.reshape(-1, 1)
        tme = np.array([0.])
        contrasts = []
        for l_r, l_tol in zip(lagr, lagr_tol):
            lr_arr = np.array([l_r])
            lt_arr = np.array([l_tol])
            C1, N1, _, _ = _stcov(gs, tme, gv2d, lr_arr, lt_arr,
                                   ang=angLag[phi1_idx],
                                   angtol=angTol[phi1_idx])
            C2, N2, _, _ = _stcov(gs, tme, gv2d, lr_arr, lt_arr,
                                   ang=angLag[phi2_idx],
                                   angtol=angTol[phi2_idx])
            c1 = float(C1[0, 0]) if not np.isnan(C1[0, 0]) else 0.0
            c2 = float(C2[0, 0]) if not np.isnan(C2[0, 0]) else 0.0
            if N1[0, 0] > 0 and N2[0, 0] > 0:
                contrasts.append(abs(c1 - c2))
        return max(contrasts) if contrasts else 0.0

    result = {
        'method': method,
        'r_hat': float(r_hat),
        'theta_hat': float(theta_hat),
        'a_max_hat': float(a_max_hat),
    }

    # ---- Bootstrap test --------------------------------------------------
    if method in ('bootstrap', 'all'):
        T_obs = _directional_contrast(grid_s, grid_v)
        T_boot = np.zeros(n_boot)
        center = grid_s.mean(axis=0)
        for b in range(n_boot):
            alpha_b = rng.uniform(0, 2 * np.pi)
            cos_a, sin_a = np.cos(alpha_b), np.sin(alpha_b)
            R_b = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            gs_rot = (grid_s - center).dot(R_b.T) + center
            T_boot[b] = _directional_contrast(gs_rot, grid_v)
        p_boot = float(np.mean(T_boot >= T_obs))
        reject_boot = p_boot < alpha
        if method == 'bootstrap':
            result.update({
                'test_stat': T_obs,
                'p_value': p_boot,
                'reject_H0': reject_boot,
                'T_boot': T_boot,
            })
        else:
            result.setdefault('test_stat', {})['bootstrap'] = T_obs
            result.setdefault('p_value', {})['bootstrap'] = p_boot
            result.setdefault('reject_H0', {})['bootstrap'] = reject_boot

    # ---- LRT -------------------------------------------------------------
    if method in ('lrt', 'all'):
        if covmodel is None or covparam is None:
            lrt_result = {
                'test_stat': np.nan, 'p_value': np.nan, 'reject_H0': False,
                'error': 'covmodel and covparam required for LRT'
            }
        else:
            try:
                from .mlecovfit import mlecovfitv
                from ...general.coord2K import coord2K

                zh = np.nanmean(grid_v2d, axis=1) if grid_v2d.shape[1] > 1 else grid_v2d.ravel()
                zh = zh - np.nanmean(zh)

                def _loglik(params, aniso_angle=None, aniso_ratio=None):
                    from ...general.coord2K import coord2K as _c2k
                    cp = params
                    K, _ = _c2k(grid_s, grid_s, covmodel, cp)
                    K += np.eye(K.shape[0]) * 1e-10 * K[0, 0]
                    try:
                        Lv = np.linalg.cholesky(K)
                    except np.linalg.LinAlgError:
                        return np.inf
                    logdetV = 2.0 * np.sum(np.log(np.diag(Lv)))
                    Viy = np.linalg.solve(K, zh)
                    n = len(zh)
                    return 0.5 * (n * np.log(2 * np.pi) + logdetV + zh.dot(Viy))

                # isotropic MLE
                param_iso = mlecovfitv(grid_s, zh, covmodel, covparam)
                ell_iso = _loglik(param_iso)

                # anisotropic MLE (using coordinate transform)
                from .stcovfit import covmodelfit
                CC_all, CCn_all, lagr_all, _, angLag_all = polarstcovmap(
                    grid_s, tME, grid_v2d, lagr, lagr_tol,
                    angLag=angLag, angTol=angTol, plot=False)
                param_aniso, theta_opt, ratio_opt, _ = covmodelfit(
                    lagr_all, np.zeros_like(lagr_all),
                    [c[:, 0:1] for c in CC_all],
                    [n[:, 0:1] for n in CCn_all],
                    covmodel, covparam,
                    theta0=theta_hat, ratio0=r_hat, lag_th=angLag_all)

                gs_iso = aniso2iso(grid_s, theta_opt * 180 / np.pi, ratio_opt)
                K_aniso, _ = coord2K(gs_iso, gs_iso, covmodel, param_aniso)
                K_aniso += np.eye(K_aniso.shape[0]) * 1e-10 * K_aniso[0, 0]
                try:
                    Lv_a = np.linalg.cholesky(K_aniso)
                    logdetV_a = 2.0 * np.sum(np.log(np.diag(Lv_a)))
                    Viy_a = np.linalg.solve(K_aniso, zh)
                    ell_aniso = 0.5 * (len(zh) * np.log(2 * np.pi) + logdetV_a + zh.dot(Viy_a))
                except np.linalg.LinAlgError:
                    ell_aniso = np.inf

                Lambda = 2.0 * max(0.0, ell_iso - ell_aniso)
                p_lrt = float(1 - chi2.cdf(Lambda, df=2))
                lrt_result = {
                    'test_stat': float(Lambda),
                    'p_value': p_lrt,
                    'reject_H0': p_lrt < alpha,
                }
            except Exception as e:
                lrt_result = {
                    'test_stat': np.nan, 'p_value': np.nan,
                    'reject_H0': False, 'error': str(e)
                }

        if method == 'lrt':
            result.update(lrt_result)
        else:
            result.setdefault('test_stat', {})['lrt'] = lrt_result.get('test_stat', np.nan)
            result.setdefault('p_value', {})['lrt'] = lrt_result.get('p_value', np.nan)
            result.setdefault('reject_H0', {})['lrt'] = lrt_result.get('reject_H0', False)

    # ---- Axial symmetry test --------------------------------------------
    if method in ('symmetry', 'all'):
        dx_range = float(np.max(lagr))
        dy_range = float(np.max(lagr))
        n_bins = max(5, int(np.sqrt(grid_s.shape[0])))
        gv_1d = np.nanmean(grid_v2d, axis=1)
        C_map, N_map, hx_grid, hy_grid = covariancemap(
            grid_s, gv_1d, dx_range, dy_range, n_bins)

        S_obs = 0.0
        n_sym = 0
        for ix in range(n_bins):
            for iy in range(n_bins):
                ix_sym = n_bins - 1 - ix
                iy_sym = n_bins - 1 - iy
                if (ix_sym < n_bins and iy_sym < n_bins
                        and not np.isnan(C_map[iy, ix])
                        and not np.isnan(C_map[iy_sym, ix_sym])):
                    n_pair = min(N_map[iy, ix], N_map[iy_sym, ix_sym])
                    S_obs += n_pair * (C_map[iy, ix] - C_map[iy_sym, ix_sym])**2
                    n_sym += 1

        # bootstrap null distribution for S under isotropy
        S_boot = np.zeros(n_boot)
        center = grid_s.mean(axis=0)
        for b in range(n_boot):
            alpha_b = rng.uniform(0, 2 * np.pi)
            cos_a, sin_a = np.cos(alpha_b), np.sin(alpha_b)
            R_b = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            gs_rot = (grid_s - center).dot(R_b.T) + center
            C_r, N_r, _, _ = covariancemap(gs_rot, gv_1d, dx_range, dy_range, n_bins)
            S_b = 0.0
            for ix in range(n_bins):
                for iy in range(n_bins):
                    ix_sym = n_bins - 1 - ix
                    iy_sym = n_bins - 1 - iy
                    if (ix_sym < n_bins and iy_sym < n_bins
                            and not np.isnan(C_r[iy, ix])
                            and not np.isnan(C_r[iy_sym, ix_sym])):
                        n_pair = min(N_r[iy, ix], N_r[iy_sym, ix_sym])
                        S_b += n_pair * (C_r[iy, ix] - C_r[iy_sym, ix_sym])**2
            S_boot[b] = S_b

        p_sym = float(np.mean(S_boot >= S_obs))
        sym_result = {
            'test_stat': float(S_obs),
            'p_value': p_sym,
            'reject_H0': p_sym < alpha,
        }
        if method == 'symmetry':
            result.update(sym_result)
        else:
            result.setdefault('test_stat', {})['symmetry'] = sym_result['test_stat']
            result.setdefault('p_value', {})['symmetry'] = sym_result['p_value']
            result.setdefault('reject_H0', {})['symmetry'] = sym_result['reject_H0']

    # ---- Plain-English message -------------------------------------------
    if r_hat >= 0.85:
        tier2_msg = (f"Effect size: r̂ = {r_hat:.3f} ≥ 0.85 — anisotropy is "
                     f"negligible.  Recommendation: use isotropic model.")
    elif r_hat >= 0.65:
        tier2_msg = (f"Effect size: r̂ = {r_hat:.3f} ∈ [0.65, 0.85) — moderate "
                     f"anisotropy.  Formal test results are decisive.")
    else:
        tier2_msg = (f"Effect size: r̂ = {r_hat:.3f} < 0.65 — strong anisotropy.  "
                     f"Principal axis: θ̂ = {np.degrees(theta_hat):.1f}°.")

    pv = result.get('p_value', {})
    if isinstance(pv, dict):
        any_reject = any(result.get('reject_H0', {}).values())
        pv_str = ', '.join(f"{k}: {v:.3f}" for k, v in pv.items() if not np.isnan(v))
        test_msg = (f"Formal test p-values ({pv_str}).  "
                    f"H₀ (isotropy) {'rejected' if any_reject else 'not rejected'} "
                    f"at α = {alpha}.")
    elif not np.isnan(pv):
        reject = result.get('reject_H0', False)
        test_msg = (f"p-value = {pv:.3f}.  "
                    f"H₀ (isotropy) {'rejected' if reject else 'not rejected'} "
                    f"at α = {alpha}.")
    else:
        test_msg = ""

    result['message'] = f"{tier2_msg}  {test_msg}".strip()
    return result


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
    from .probatablecalc import probatablecalc_directional

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
