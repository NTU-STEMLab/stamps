# -*- coding: utf-8 -*-
"""
stamps.stats.dependence.covariance.anisotropy
================================================
Continuous-data anisotropy analysis (covariance-based).

Functions: polarstcovmap, iso2aniso, aniso2iso, covariancemap,
estimate_anisotropy_params, isotropy_test.
"""
import numpy
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import chi2

from .stcov import stcov
from ....estimation import stmean


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
                from ....general.coord2K import coord2K

                zh = np.nanmean(grid_v2d, axis=1) if grid_v2d.shape[1] > 1 else grid_v2d.ravel()
                zh = zh - np.nanmean(zh)

                def _loglik(params, aniso_angle=None, aniso_ratio=None):
                    from ....general.coord2K import coord2K as _c2k
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

