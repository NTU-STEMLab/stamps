# -*- coding: utf-8 -*-
"""
bme_transform.py — Normal Score Transformation & CDF-transform utilities
=========================================================================

This module provides the mathematical layer for transforming data between
an arbitrary random field Y and a Gaussian random field Z, where Y = g(Z)
is defined by the prior CDF of Y.  It is used by the BME T-variant
estimators (``BMEprobaTMode``, ``BMEprobaTPdf``, ``BMEprobaTCI``).

Sections
--------
A : Core CDF / NST primitives
    ``gausspdf``, ``gaussinv``, ``transformderiv``, ``other2gauss``
B : PDF back-transformation
    ``pdfgauss2other``
C : Soft-data transformation
    ``probaother2gauss``, ``FyTransformCheckArgs``
D : NormalScoreTransform class (sklearn-compatible)
E : T-variant public wrappers re-exported by BMEprobaEstimations
    ``BMEprobaTMode``, ``BMEprobaTPdf``, ``BMEprobaTCI``

Adapted from the BMElib MATLAB package (Jan 1, 2001).
All indices are 0-based (Python convention), adapted from 1-based MATLAB.
"""
from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
from scipy.stats import norm as _norm_dist
from scipy.interpolate import interp1d as _interp1d

from .softconverter import (
    proba2interval,
    proba2probdens,
    proba2val,
    probasplit,
    probacat,
    softpdftypeCheckArgs,
)

__all__ = [
    # Section A
    "gausspdf",
    "gaussinv",
    "transformderiv",
    "other2gauss",
    # Section B
    "pdfgauss2other",
    # Section C
    "probaother2gauss",
    "FyTransformCheckArgs",
    # Section D
    "NormalScoreTransform",
    # Section E
    "BMEprobaTMode",
    "BMEprobaTPdf",
    "BMEprobaTCI",
]


# =============================================================================
# Section A: Core CDF / NST primitives
# =============================================================================

def gausspdf(
    x: np.ndarray,
    params: Union[list, tuple, np.ndarray],
) -> np.ndarray:
    """Evaluate the Gaussian (normal) probability density function.

    Adapted from ``gausspdf.m`` (BMElib statlib).

    Parameters
    ----------
    x : array_like
        Points at which to evaluate the Gaussian PDF.
    params : array_like of length 2
        ``[mean, variance]`` of the Gaussian distribution.
        Note: ``params[1]`` is the **variance**, not the standard deviation.

    Returns
    -------
    f : ndarray
        PDF values at each element of ``x``.

    Examples
    --------
    >>> gausspdf(0.0, [0, 1])    # Standard normal at 0
    0.3989422804014327
    """
    x = np.asarray(x, dtype=float)
    mean = float(params[0])
    var  = float(params[1])
    return _norm_dist.pdf(x, loc=mean, scale=np.sqrt(var))


def gaussinv(
    p: np.ndarray,
    params: Union[list, tuple, np.ndarray],
) -> np.ndarray:
    """Evaluate the inverse CDF (quantile function) of the Gaussian distribution.

    Adapted from ``gaussinv.m`` (BMElib statlib).

    Parameters
    ----------
    p : array_like
        Probabilities in the interval (0, 1).
    params : array_like of length 2
        ``[mean, variance]`` of the Gaussian distribution.

    Returns
    -------
    x : ndarray
        Quantile values corresponding to each probability in ``p``.

    Examples
    --------
    >>> gaussinv(0.975, [0, 1])   # ≈ 1.96
    1.959963984540054
    """
    p = np.asarray(p, dtype=float)
    mean = float(params[0])
    var  = float(params[1])
    return _norm_dist.ppf(p, loc=mean, scale=np.sqrt(var))


def transformderiv(
    yfile: np.ndarray,
    Fyfile: np.ndarray,
) -> np.ndarray:
    """Compute the derivative of the NST back-transform dY/dZ at the grid nodes.

    Adapted from ``transformderiv.m`` (BMElib statlib / bmeprobalib).

    The Normal Score Transform (NST) defines Z = Φ^{-1}(F_Y(Y)), where Φ is
    the standard-normal CDF and F_Y is the empirical/prior CDF of Y.  This
    function returns the derivative of the inverse transform:

    .. math::

        \\frac{dY}{dZ}\\bigg|_{z_i} = \\frac{\\phi(z_i)}{f_Y(y_i)}

    where :math:`\\phi` is the standard-normal PDF and :math:`f_Y` is the
    PDF of Y at the corresponding Y node.

    The derivative is approximated numerically using central differences on
    the interior nodes and one-sided differences at the endpoints.

    Parameters
    ----------
    yfile : array_like of shape (k,)
        Y values sorted in ascending order defining the prior CDF.
    Fyfile : array_like of shape (k,)
        Corresponding CDF values, strictly increasing, in (0, 1).
        Must satisfy ``Fyfile[0] < 0.001`` and ``Fyfile[-1] > 0.999``.

    Returns
    -------
    dgfile : ndarray of shape (k,)
        Derivative dY/dZ at each node.  Values are guaranteed positive.

    Notes
    -----
    The derivative is computed as:

    1. Transform y-nodes to z-nodes: ``zfile = Φ^{-1}(Fyfile)``.
    2. Estimate dY/dZ at each node using finite differences on the
       (z, y) pairs.

    This approach is numerically more stable than computing f_Y from
    diff(Fyfile)/diff(yfile) because it avoids dividing by potentially
    small density values.
    """
    yfile  = np.asarray(yfile,  dtype=float).ravel()
    Fyfile = np.asarray(Fyfile, dtype=float).ravel()
    k = len(yfile)
    zfile = _norm_dist.ppf(Fyfile)  # Z values at nodes

    # Compute dY/dZ via finite differences on (z, y) pairs
    dgfile = np.empty(k, dtype=float)
    if k == 1:
        dgfile[:] = 1.0
        return dgfile

    # Central differences (interior)
    for i in range(1, k - 1):
        dz = zfile[i + 1] - zfile[i - 1]
        dy = yfile[i + 1] - yfile[i - 1]
        dgfile[i] = dy / dz if abs(dz) > 1e-30 else 0.0

    # One-sided differences (endpoints)
    dz0 = zfile[1] - zfile[0]
    dgfile[0] = (yfile[1] - yfile[0]) / dz0 if abs(dz0) > 1e-30 else 0.0

    dz_end = zfile[-1] - zfile[-2]
    dgfile[-1] = (yfile[-1] - yfile[-2]) / dz_end if abs(dz_end) > 1e-30 else 0.0

    # Guard: derivative must be positive (transform is monotone increasing)
    dgfile = np.maximum(dgfile, 1e-30)
    return dgfile


def other2gauss(
    y: np.ndarray,
    yfile: np.ndarray,
    Fyfile: np.ndarray,
) -> np.ndarray:
    """Transform arbitrary Y values to Gaussian Z values via NST.

    Adapted from ``other2gauss.m`` (BMElib statlib / bmeprobalib).

    Applies the Normal Score Transform Z = Φ^{-1}(F_Y(Y)), where the prior
    CDF F_Y is defined by the tabulated values ``(yfile, Fyfile)``.  The CDF
    value is obtained by linear interpolation, and the inverse is computed
    via ``scipy.stats.norm.ppf``.

    Parameters
    ----------
    y : array_like
        Y values to transform.  Values outside ``[min(yfile), max(yfile)]``
        are mapped to ``NaN``.
    yfile : array_like of shape (k,)
        Y values sorted in ascending order defining the prior CDF.
    Fyfile : array_like of shape (k,)
        Corresponding CDF values, strictly increasing, in (0, 1).

    Returns
    -------
    z : ndarray
        Transformed Z values (zero-mean, unit-variance Gaussian scale).
        Elements outside the domain are ``NaN``.

    Notes
    -----
    The transform is equivalent to the MATLAB expression::

        z = norminv(interp1(yfile, Fyfile, y, 'linear'))

    Index convention: 0-based (adapted from 1-based MATLAB code).
    """
    y      = np.asarray(y,      dtype=float)
    yfile  = np.asarray(yfile,  dtype=float).ravel()
    Fyfile = np.asarray(Fyfile, dtype=float).ravel()

    orig_shape = y.shape
    y_flat = y.ravel()

    # Linear interpolation of CDF; values outside domain → NaN
    Fy = np.interp(y_flat, yfile, Fyfile, left=np.nan, right=np.nan)

    # Inverse normal CDF
    z = _norm_dist.ppf(Fy)
    return z.reshape(orig_shape)


# =============================================================================
# Section B: PDF back-transformation
# =============================================================================

def pdfgauss2other(
    y: np.ndarray,
    z: np.ndarray,
    zpdf: np.ndarray,
    yfile: np.ndarray,
    Fyfile: np.ndarray,
) -> np.ndarray:
    """Transform the posterior PDF from Gaussian Z scale to original Y scale.

    Adapted from ``pdfgauss2other.m`` (BMElib bmeprobalib, Jan 1, 2001).

    Transforms the posterior PDF ``f_Z(z)`` for the Gaussian random variable
    Z to the corresponding posterior PDF ``f_Y(y)`` for the arbitrary random
    variable Y, where Y and Z are related by the one-to-one function Y = g(Z).
    The function g is defined by the prior CDF of Y such that the maximum-entropy
    distribution for Z is multivariate Gaussian.

    The transformation formula is:

    .. math::

        f_Y(y) = \\frac{f_Z(z(y))}{\\frac{dZ}{dY}\\bigg|_y}
               = f_Z(z) \\cdot \\left(\\frac{dY}{dZ}\\right)^{-1}

    Parameters
    ----------
    y : array_like of shape (ny,)
        Y values at which to compute the posterior PDF.
    z : array_like of shape (nz,)
        Z values defining the Gaussian posterior PDF grid
        (corresponds to the Z representation of y values).
    zpdf : array_like of shape (nz,)
        Gaussian posterior PDF values ``f_Z(z[i])``.
    yfile : array_like of shape (k,)
        Y values sorted in ascending order defining the prior CDF.
    Fyfile : array_like of shape (k,)
        Corresponding CDF values, strictly increasing, in (0, 1).

    Returns
    -------
    ypdf : ndarray of shape (ny,)
        Posterior PDF values on the original Y scale.  Elements outside
        ``[min(yfile), max(yfile)]`` are set to ``NaN``.

    Notes
    -----
    The derivative ``dY/dZ`` is computed via :func:`transformderiv` and then
    interpolated at the query Z values using the Z-grid ``zfile =
    Φ^{-1}(Fyfile)``.

    If a Y value is outside the definition domain of ``(yfile, Fyfile)``,
    the returned ``ypdf`` value is ``NaN``.

    Index convention: 0-based (adapted from 1-based MATLAB code).
    """
    y     = np.asarray(y,    dtype=float).ravel()
    z     = np.asarray(z,    dtype=float).ravel()
    zpdf  = np.asarray(zpdf, dtype=float).ravel()
    yfile  = np.asarray(yfile,  dtype=float).ravel()
    Fyfile = np.asarray(Fyfile, dtype=float).ravel()

    ypdf = np.full_like(y, np.nan, dtype=float)

    dgfile = transformderiv(yfile, Fyfile)
    zfile  = _norm_dist.ppf(Fyfile)  # Z grid corresponding to yfile

    in_domain = (y >= yfile[0]) & (y <= yfile[-1])
    if not in_domain.any():
        return ypdf

    # Interpolate dY/dZ at the query Z values
    dgtemp = np.interp(z[in_domain], zfile, dgfile)

    zpdf_in = zpdf[in_domain]

    # f_Y = f_Z / (dZ/dY) = f_Z * (dY/dZ)^{-1}
    # Since dgfile = dY/dZ, f_Y = f_Z / (1/dgtemp) = f_Z * dgtemp?
    # No: f_Y = f_Z * |dZ/dY| = f_Z / |dY/dZ| = f_Z / dgtemp
    # (See BMElib pdfgauss2other.m: ypdftemp = ypdftemp ./ dgtemp)
    with np.errstate(divide='ignore', invalid='ignore'):
        ypdf[in_domain] = np.where(dgtemp > 0, zpdf_in / dgtemp, np.nan)

    return ypdf


# =============================================================================
# Section C: Soft data transformation
# =============================================================================

def probaother2gauss(
    ysoftpdftype: int,
    ynl: np.ndarray,
    ylimi: np.ndarray,
    yprobdens: np.ndarray,
    yfile: np.ndarray,
    Fyfile: np.ndarray,
) -> tuple:
    """Transform probabilistic soft data from Y scale to Gaussian Z scale.

    Adapted from ``probaother2gauss.m`` (BMElib bmeprobalib, Jan 1, 2001).

    Transforms the soft probabilistic data for an arbitrary random variable Y
    to the corresponding soft probabilistic data for the Gaussian random
    variable Z, where Y and Z are related by the NST Y = g(Z).

    For each soft data row, the transformation maps the interval limits
    from Y-space to Z-space using ``other2gauss``, and adjusts the
    probability densities using the Jacobian of the transform.

    Parameters
    ----------
    ysoftpdftype : int
        Soft PDF type for Y (1 = Histogram, 2 = Linear, 3 = Grid histogram,
        4 = Grid linear).
    ynl : ndarray of shape (ns, 1)
        Number of interval limits for each soft datum.
    ylimi : ndarray of shape (ns, l)
        Interval limits in Y space.
    yprobdens : ndarray of shape (ns, p)
        Probability density values in Y space.
    yfile : array_like of shape (k,)
        Y values sorted in ascending order defining the prior CDF of Y.
    Fyfile : array_like of shape (k,)
        CDF values, strictly increasing, with ``Fyfile[0] < 0.001`` and
        ``Fyfile[-1] > 0.999``.

    Returns
    -------
    zsoftpdftype : int
        Soft PDF type for Z (may differ from ``ysoftpdftype`` for Grid types:
        type 3 → type 1, type 4 → type 2).
    znl : ndarray of shape (ns, 1)
        Number of interval limits in Z space (same as ``ynl``).
    zlimi : ndarray of shape (ns, l)
        Interval limits in Z space.
    zprobdens : ndarray of shape (ns, p)
        Probability density values in Z space, normalised.

    Notes
    -----
    The transformation of the PDF from Y to Z space uses the change-of-
    variable formula:

    * **Types 1 & 3 (Histogram)**:
      ``f_Z(z) Δz = f_Y(y) Δy``  →  ``zpd = ypd * Δy / Δz``

    * **Types 2 & 4 (Linear)**:
      ``f_Z(z) = f_Y(y) / (dZ/dY)``  →  ``zpd = ypd * dg`` where
      ``dg = dY/dZ`` (the inverse Jacobian), computed via
      :func:`transformderiv`.

    For Grid types (3 and 4), the grid limits are expanded to an explicit
    grid before transformation, because the Z-space limits are no longer
    uniformly spaced.

    Index convention: 0-based (adapted from 1-based MATLAB code).
    """
    ynl       = np.atleast_2d(np.asarray(ynl,       dtype=int))
    ylimi     = np.atleast_2d(np.asarray(ylimi,     dtype=float))
    yprobdens = np.atleast_2d(np.asarray(yprobdens, dtype=float))
    yfile     = np.asarray(yfile,  dtype=float).ravel()
    Fyfile    = np.asarray(Fyfile, dtype=float).ravel()
    ns        = ynl.shape[0]

    dgfile = transformderiv(yfile, Fyfile)

    spt = int(ysoftpdftype)

    if spt == 1:
        # Histogram: transform limits; adjust density by ratio of bin widths
        zsoftpdftype = 1
        znl       = ynl.copy()
        zlimi     = np.full_like(ylimi, np.nan)
        zprobdens = np.full_like(yprobdens, np.nan)
        for i in range(ns):
            ni = int(ynl[i, 0])
            yl = ylimi[i, :ni]
            yld = np.diff(yl)
            ypd = yprobdens[i, :ni - 1]
            zl = other2gauss(yl, yfile, Fyfile)
            zld = np.diff(zl)
            with np.errstate(divide='ignore', invalid='ignore'):
                zpd = np.where(np.abs(zld) > 1e-30, yld * ypd / zld, 0.0)
            zlimi[i, :ni]       = zl
            zprobdens[i, :ni - 1] = zpd
        return zsoftpdftype, znl, zlimi, zprobdens

    elif spt == 2:
        # Linear: transform limits and multiply by dY/dZ
        zsoftpdftype = 2
        znl       = ynl.copy()
        zlimi     = np.full_like(ylimi, np.nan)
        zprobdens = np.full_like(yprobdens, np.nan)
        for i in range(ns):
            ni  = int(ynl[i, 0])
            yl  = ylimi[i, :ni]
            ypd = yprobdens[i, :ni]
            # dY/dZ at Y-nodes: interpolate dgfile using yfile grid
            dg  = np.interp(yl, yfile, dgfile, left=np.nan, right=np.nan)
            zl  = other2gauss(yl, yfile, Fyfile)
            zpd = ypd * dg   # f_Z = f_Y * (dY/dZ) at each node
            # Normalise
            _, _, zpd_norm, _ = proba2probdens(zsoftpdftype,
                                               np.array([[ni]]),
                                               zl[np.newaxis, :],
                                               zpd[np.newaxis, :])
            zlimi[i, :ni]     = zl
            zprobdens[i, :ni] = zpd_norm.ravel()[:ni]
        return zsoftpdftype, znl, zlimi, zprobdens

    elif spt == 3:
        # Grid histogram: expand grid → type 1 in Z space
        zsoftpdftype = 1
        nl_max = int(ynl.max())
        znl       = ynl.copy()
        zlimi     = np.full((ns, nl_max), np.nan)
        zprobdens = np.full((ns, nl_max - 1), np.nan)
        for i in range(ns):
            ni   = int(ynl[i, 0])
            a, step, b = ylimi[i, 0], ylimi[i, 1], ylimi[i, 2]
            yl   = np.linspace(a, b, ni)
            if len(yl) != ni:
                raise ValueError(f"Row {i}: grid length mismatch for ynl/ylimi")
            yld  = np.diff(yl)
            ypd  = yprobdens[i, :ni - 1]
            zl   = other2gauss(yl, yfile, Fyfile)
            zld  = np.diff(zl)
            with np.errstate(divide='ignore', invalid='ignore'):
                zpd = np.where(np.abs(zld) > 1e-30, yld * ypd / zld, 0.0)
            zlimi[i, :ni]         = zl
            zprobdens[i, :ni - 1] = zpd
        return zsoftpdftype, znl, zlimi, zprobdens

    elif spt == 4:
        # Grid linear: expand grid → type 2 in Z space
        zsoftpdftype = 2
        nl_max = int(ynl.max())
        znl       = ynl.copy()
        zlimi     = np.full((ns, nl_max), np.nan)
        zprobdens = np.full((ns, nl_max), np.nan)
        for i in range(ns):
            ni    = int(ynl[i, 0])
            a, step, b = ylimi[i, 0], ylimi[i, 1], ylimi[i, 2]
            yl    = np.linspace(a, b, ni)
            ypd   = yprobdens[i, :ni]
            dg    = np.interp(yl, yfile, dgfile, left=np.nan, right=np.nan)
            zl    = other2gauss(yl, yfile, Fyfile)
            zpd   = ypd * dg
            _, _, zpd_norm, _ = proba2probdens(zsoftpdftype,
                                               np.array([[ni]]),
                                               zl[np.newaxis, :],
                                               zpd[np.newaxis, :])
            zlimi[i, :ni]     = zl
            zprobdens[i, :ni] = zpd_norm.ravel()[:ni]
        return zsoftpdftype, znl, zlimi, zprobdens

    else:
        raise ValueError(f"Unsupported ysoftpdftype: {spt}")


def FyTransformCheckArgs(
    yfile: np.ndarray,
    Fyfile: np.ndarray,
    ck: np.ndarray,
    ch: np.ndarray,
    cs: np.ndarray,
    yh: np.ndarray,
    ysoftpdftype: int,
    ynl: np.ndarray,
    ylimi: np.ndarray,
    yprobdens: np.ndarray,
) -> None:
    """Validate arguments for the BMEprobaT family of functions.

    Adapted from ``FyTransformCheckArgs.m`` (BMElib bmeprobalib, Jan 1, 2001).

    Checks that the prior CDF defined by ``(yfile, Fyfile)`` is well-formed
    and covers the full range of both hard and soft data.

    Parameters
    ----------
    yfile : array_like of shape (k,)
        Y values sorted in ascending order.
    Fyfile : array_like of shape (k,)
        CDF values, strictly increasing, all in [0, 1].
        Must satisfy ``Fyfile[0] < 0.001`` and ``Fyfile[-1] > 0.999``.
    ck : ndarray of shape (nk, d)
        Estimation point coordinates.
    ch : ndarray of shape (nh, d)
        Hard data coordinates.
    cs : ndarray of shape (ns, d)
        Soft data coordinates.
    yh : array_like of shape (nh,)
        Hard data values for Y.
    ysoftpdftype : int
        Soft PDF type for Y.
    ynl : ndarray of shape (ns, 1)
        Number of interval limits.
    ylimi : ndarray of shape (ns, l)
        Interval limits in Y space.
    yprobdens : ndarray of shape (ns, p)
        Probability density values in Y space.

    Raises
    ------
    ValueError
        If any consistency check fails.

    Notes
    -----
    Index convention: 0-based (adapted from 1-based MATLAB code).
    """
    yfile  = np.asarray(yfile,  dtype=float).ravel()
    Fyfile = np.asarray(Fyfile, dtype=float).ravel()

    # Empty yfile means no transform — nothing to check
    if yfile.size == 0:
        return

    yh  = np.asarray(yh,  dtype=float).ravel()
    ynl = np.atleast_2d(np.asarray(ynl, dtype=int))
    ylimi = np.atleast_2d(np.asarray(ylimi, dtype=float))
    nh = len(yh)
    ns = ynl.shape[0]

    if nh + ns == 0:
        return

    # Structural checks on yfile / Fyfile
    if yfile.shape != Fyfile.shape:
        raise ValueError("yfile and Fyfile must have the same length")
    if not np.all(np.diff(yfile) > 0):
        raise ValueError("yfile must be strictly increasing")
    if not np.all(np.diff(Fyfile) > 0):
        raise ValueError("Fyfile must be strictly increasing (no equal consecutive values)")
    if np.any((Fyfile < 0) | (Fyfile > 1)):
        raise ValueError("All Fyfile values must be in [0, 1]")
    if Fyfile[0] > 0.001:
        raise ValueError("Fyfile[0] must be smaller than 0.001")
    if Fyfile[-1] < 0.999:
        raise ValueError("Fyfile[-1] must be greater than 0.999")

    # Check that yfile covers all data
    if ns > 0:
        ys_min, ys_max = proba2interval(ysoftpdftype, ynl, ylimi)
        y_min = float(np.nanmin(np.concatenate([yh, ys_min])))
        y_max = float(np.nanmax(np.concatenate([yh, ys_max])))
    elif nh > 0:
        y_min = float(yh.min())
        y_max = float(yh.max())
    else:
        return

    if yfile[0] > y_min:
        raise ValueError(
            f"There is a hard or soft data value ({y_min:.4g}) smaller than "
            f"min(yfile) = {yfile[0]:.4g}"
        )
    if yfile[-1] < y_max:
        raise ValueError(
            f"There is a hard or soft data value ({y_max:.4g}) greater than "
            f"max(yfile) = {yfile[-1]:.4g}"
        )


# =============================================================================
# Section D: NormalScoreTransform class (sklearn-compatible)
# =============================================================================

class NormalScoreTransform:
    """Normal Score Transform (NST) for geostatistical applications.

    Transforms an arbitrary random field Y to a standard-normal (Gaussian)
    random field Z using the empirical or prior CDF of Y:

    .. math::

        Z = \\Phi^{-1}(\\hat{F}_Y(Y))

    where :math:`\\Phi` is the standard-normal CDF and :math:`\\hat{F}_Y` is
    the CDF of Y.  This is the standard Normal Score Transform used widely in
    geostatistics (Goovaerts 1997, Deutsch & Journel 1992).

    The class follows an ``sklearn``-style API: ``fit`` computes the
    empirical CDF, ``transform`` applies the NST, and ``inverse_transform``
    maps Z values back to Y.

    Parameters
    ----------
    n_quantiles : int, optional
        Number of quantile nodes used to build the empirical CDF when
        fitting from data.  Default is 1000.
    cdf_eps : float, optional
        Small offset added to the boundary CDF values so that the CDF
        spans (cdf_eps, 1 - cdf_eps) and avoids ±∞ in the transform.
        Default is 1e-4 (i.e., CDF spans approximately [0.0001, 0.9999]).

    Attributes
    ----------
    yfile_ : ndarray of shape (k,)
        Y nodes of the fitted CDF.
    Fyfile_ : ndarray of shape (k,)
        CDF values at the fitted nodes.
    is_fitted_ : bool
        ``True`` after :meth:`fit` or :meth:`fit_from_cdf` has been called.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> y_data = rng.lognormal(0, 1, 500)
    >>> nst = NormalScoreTransform()
    >>> nst.fit(y_data)
    NormalScoreTransform(n_quantiles=1000, cdf_eps=0.0001)
    >>> z = nst.transform(y_data)
    >>> y_back = nst.inverse_transform(z)

    References
    ----------
    Deutsch, C.V. & Journel, A.G. (1992). *GSLIB: Geostatistical Software
    Library and User's Guide*. Oxford University Press.
    """

    def __init__(self, n_quantiles: int = 1000, cdf_eps: float = 1e-4):
        self.n_quantiles = int(n_quantiles)
        self.cdf_eps = float(cdf_eps)
        self.yfile_: Optional[np.ndarray]  = None
        self.Fyfile_: Optional[np.ndarray] = None
        self.is_fitted_: bool = False

    def __repr__(self) -> str:
        return (f"NormalScoreTransform("
                f"n_quantiles={self.n_quantiles}, "
                f"cdf_eps={self.cdf_eps})")

    def fit(self, y: np.ndarray) -> "NormalScoreTransform":
        """Fit the NST using the empirical CDF of ``y``.

        Parameters
        ----------
        y : array_like of shape (n,)
            Observed data values used to build the empirical CDF.

        Returns
        -------
        self
        """
        y = np.asarray(y, dtype=float).ravel()
        y = y[np.isfinite(y)]
        if len(y) == 0:
            raise ValueError("fit received no finite data values")

        # Build empirical CDF at n_quantiles nodes
        probs = np.linspace(self.cdf_eps, 1.0 - self.cdf_eps, self.n_quantiles)
        quantiles = np.quantile(y, probs)

        # Ensure strictly increasing nodes (remove ties)
        unique_q, unique_idx = np.unique(quantiles, return_index=True)
        unique_p = probs[unique_idx]

        self.yfile_  = unique_q
        self.Fyfile_ = unique_p
        self.is_fitted_ = True
        return self

    def fit_from_cdf(
        self,
        yfile: np.ndarray,
        Fyfile: np.ndarray,
    ) -> "NormalScoreTransform":
        """Fit the NST from an externally supplied CDF table.

        Parameters
        ----------
        yfile : array_like of shape (k,)
            Y values sorted in ascending order.
        Fyfile : array_like of shape (k,)
            CDF values, strictly increasing, in (0, 1).

        Returns
        -------
        self
        """
        self.yfile_  = np.asarray(yfile,  dtype=float).ravel()
        self.Fyfile_ = np.asarray(Fyfile, dtype=float).ravel()
        self.is_fitted_ = True
        return self

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError(
                "NormalScoreTransform has not been fitted yet. "
                "Call fit() or fit_from_cdf() first."
            )

    def transform(self, y: np.ndarray) -> np.ndarray:
        """Apply the NST: Y → Z.

        Parameters
        ----------
        y : array_like
            Y values to transform.

        Returns
        -------
        z : ndarray
            Transformed Z values.
        """
        self._check_fitted()
        return other2gauss(y, self.yfile_, self.Fyfile_)

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        """Apply the inverse NST: Z → Y.

        Parameters
        ----------
        z : array_like
            Z values (Gaussian scale) to back-transform.

        Returns
        -------
        y : ndarray
            Back-transformed Y values.
        """
        self._check_fitted()
        z = np.asarray(z, dtype=float)
        Fy = _norm_dist.cdf(z)
        return np.interp(Fy, self.Fyfile_, self.yfile_,
                         left=np.nan, right=np.nan)

    def transform_pdf(
        self,
        y: np.ndarray,
        z: np.ndarray,
        zpdf: np.ndarray,
    ) -> np.ndarray:
        """Transform a posterior PDF from Z scale to Y scale.

        Parameters
        ----------
        y : array_like of shape (ny,)
            Y values at which to evaluate the posterior PDF.
        z : array_like of shape (nz,)
            Z-grid values.
        zpdf : array_like of shape (nz,)
            Posterior PDF values in Z space.

        Returns
        -------
        ypdf : ndarray of shape (ny,)
            Posterior PDF in Y space.
        """
        self._check_fitted()
        return pdfgauss2other(y, z, zpdf, self.yfile_, self.Fyfile_)

    def transform_soft_data(
        self,
        ysoftpdftype: int,
        ynl: np.ndarray,
        ylimi: np.ndarray,
        yprobdens: np.ndarray,
    ) -> tuple:
        """Transform probabilistic soft data from Y scale to Z (Gaussian) scale.

        Parameters
        ----------
        ysoftpdftype : int
            Soft PDF type for Y.
        ynl : ndarray of shape (ns, 1)
            Number of interval limits.
        ylimi : ndarray of shape (ns, l)
            Interval limits in Y space.
        yprobdens : ndarray of shape (ns, p)
            Probability density values in Y space.

        Returns
        -------
        zsoftpdftype, znl, zlimi, zprobdens : tuple
            Soft data representation in Z (Gaussian) space.
        """
        self._check_fitted()
        return probaother2gauss(
            ysoftpdftype, ynl, ylimi, yprobdens,
            self.yfile_, self.Fyfile_,
        )


# =============================================================================
# Section E: T-variant public wrappers
# =============================================================================

def _flat2zs(
    softpdftype: int,
    nl: np.ndarray,
    limi: np.ndarray,
    probdens: np.ndarray,
) -> list:
    """Convert MATLAB-style flat soft-data arrays to an internal ``zs`` list.

    The "flat format" stores ns soft data points as matrices:

    * ``nl``        – (ns, 1) int array, number of interval-limit entries per row.
    * ``limi``      – (ns, l_max) float array of interval limits.
    * ``probdens``  – (ns, p_max) float array of probability densities.

    This function converts the flat format to the internal list-of-tuples
    representation expected by :func:`BMEPosteriorMode`,
    :func:`BMEPosteriorCI`, :func:`BMEPosteriorPDF`, etc.::

        zs[i] = (softpdftype, np.array([ni]), limi_1d, probdens_1d)

    Parameters
    ----------
    softpdftype : int
        1 = Histogram, 2 = Linear piecewise, 3 = Grid histogram,
        4 = Grid linear.  Grid types store three ``limi`` values per row
        (min, step, max) regardless of ``nl``.
    nl : array_like of shape (ns, 1) or (ns,)
        Number of interval-limit entries per soft datum.
    limi : array_like of shape (ns, l_max)
        Interval limits.  For Grid types (3, 4) only the first 3 columns
        are used (min, step, max).
    probdens : array_like of shape (ns, p_max)
        Probability densities.  For Histogram types (1, 3) there are
        ``ni − 1`` density values per row; for Linear types (2, 4) there
        are ``ni`` values.

    Returns
    -------
    zs : list of tuple
        List of ns tuples ``(spt, np.array([ni]), limi_1d, probdens_1d)``.
    """
    nl       = np.atleast_2d(np.asarray(nl,       dtype=int))
    limi     = np.atleast_2d(np.asarray(limi,     dtype=float))
    probdens = np.atleast_2d(np.asarray(probdens, dtype=float))
    ns  = nl.shape[0]
    spt = int(softpdftype)
    zs  = []
    for i in range(ns):
        ni = int(nl[i, 0])
        # Grid types (3, 4) store grid definition as [a, step, b] — 3 columns
        if spt in (3, 4):
            limi_i = limi[i, :3].copy()
        else:
            limi_i = limi[i, :ni].copy()
        # Histogram types (1, 3): ni−1 density values; Linear (2, 4): ni values
        if spt in (1, 3):
            pd_i = probdens[i, :max(ni - 1, 0)].copy()
        else:
            pd_i = probdens[i, :ni].copy()
        zs.append((spt, np.array([ni]), limi_i, pd_i))
    return zs

def BMEprobaTMode(
    ck: np.ndarray,
    ch: np.ndarray,
    cs: np.ndarray,
    yh: np.ndarray,
    ysoftpdftype: int,
    ynl: np.ndarray,
    ylimi: np.ndarray,
    yprobdens: np.ndarray,
    covmodel,
    covparam: np.ndarray,
    nhmax: int,
    nsmax: int,
    dmax,
    order,
    options=None,
    yfile: Optional[np.ndarray] = None,
    cdfyfile: Optional[np.ndarray] = None,
) -> tuple:
    """BME mode prediction with Normal Score Transform and probabilistic data.

    Adapted from ``BMEprobaTMode.m`` (BMElib bmeprobalib, Jan 1, 2001).

    Computes the estimated **mode** of the BME posterior PDF at a set of
    estimation points, using both hard data and soft probabilistic data,
    with an optional Normal Score Transform (NST) from an arbitrary random
    field Y to a Gaussian random field Z.

    The transformation Y → Z is defined by the prior CDF of Y, specified
    through ``(yfile, cdfyfile)``.  The solution corresponds to the **mode of
    the posterior PDF on the original Y scale**.

    Parameters
    ----------
    ck : ndarray of shape (nk, d)
        Coordinates of estimation locations (0-based row index).
    ch : ndarray of shape (nh, d)
        Coordinates of hard data locations.
    cs : ndarray of shape (ns, d)
        Coordinates of soft data locations.
    yh : ndarray of shape (nh,)
        Hard data values on the Y scale.
    ysoftpdftype : int
        Soft PDF type for Y (1 = Histogram, 2 = Linear, 3 = Grid histogram,
        4 = Grid linear).
    ynl : ndarray of shape (ns, 1)
        Number of interval limits per soft datum.
    ylimi : ndarray of shape (ns, l)
        Interval limits in Y space.
    yprobdens : ndarray of shape (ns, p)
        Probability densities in Y space.
    covmodel : str or list
        Covariance model name(s) for the **Gaussian-transformed Z** field.
    covparam : array_like
        Parameters of the covariance model.
    nhmax : int
        Maximum number of hard data considered per estimation point.
    nsmax : int
        Maximum number of soft data considered per estimation point.
        **Must be ≤ 20** to avoid numerical issues.
    dmax : float or array_like
        Maximum distance threshold for neighbourhood selection.
    order : float or NaN
        Drift order for the mean trend removal.
        ``NaN`` for zero mean, ``0`` for constant, ``1`` for linear, etc.
    options : array_like or None, optional
        Optional algorithm parameters (see BMEoptions).
    yfile : array_like of shape (k,) or None, optional
        Y nodes of the prior CDF.  If ``None``, no transform is applied.
    cdfyfile : array_like of shape (k,) or None, optional
        CDF values of the prior CDF.

    Returns
    -------
    yk : ndarray of shape (nk,)
        Mode estimates on the **original Y scale**.
    info : ndarray of shape (nk,)
        Diagnostic codes per estimation point:

        * ``NaN`` – no hard or soft data in neighbourhood
        * ``0``   – BME computation with soft data
        * ``1``   – dubious: integration error above tolerance
        * ``2``   – dubious: mode search did not converge
        * ``3``   – kriging (no soft data)
        * ``4``   – hard data coincides with estimation point
        * ``10``  – dubious: severe integration problem

    Notes
    -----
    This function is a thin wrapper around :func:`BMEPosteriorMode` in
    ``BMEprobaEstimations.py``.  It applies the NST before calling the core
    estimator and back-transforms the result afterward.

    All the conventions for nested models, multivariate, and space-time
    cases are the same as for ``BMEprobaMode``.

    Index convention: 0-based (adapted from 1-based MATLAB code).
    """
    # Lazy import to avoid circular dependency at module load
    from .BMEprobaEstimations import BMEPosteriorMode

    do_transform = (yfile is not None) and (len(np.asarray(yfile)) > 0)

    if do_transform:
        yfile_arr    = np.asarray(yfile,    dtype=float).ravel()
        cdfyfile_arr = np.asarray(cdfyfile, dtype=float).ravel()
        FyTransformCheckArgs(yfile_arr, cdfyfile_arr, ck, ch, cs,
                             yh, ysoftpdftype, ynl, ylimi, yprobdens)
        # Transform hard data and soft data to Z scale
        zh = other2gauss(yh, yfile_arr, cdfyfile_arr)
        zsoftpdftype, znl, zlimi, zprobdens = probaother2gauss(
            ysoftpdftype, ynl, ylimi, yprobdens, yfile_arr, cdfyfile_arr
        )
    else:
        yfile_arr    = None
        cdfyfile_arr = None
        zh           = np.asarray(yh, dtype=float)
        zsoftpdftype = ysoftpdftype
        znl          = ynl
        zlimi        = ylimi
        zprobdens    = yprobdens

    # Convert flat soft-data format to internal zs list-of-tuples
    zs_list = _flat2zs(zsoftpdftype, znl, zlimi, zprobdens)

    # Ensure zh is column vector (nh, 1) as expected by BMEPosteriorMode
    zh_2d = np.asarray(zh, dtype=float).ravel()[:, np.newaxis]

    # Compute mode in Z space
    zk_mode, info = BMEPosteriorMode(
        ck=ck, ch=ch, cs=cs, zh=zh_2d,
        zs=zs_list,
        covmodel=covmodel, covparam=covparam,
        nhmax=nhmax, nsmax=nsmax, dmax=dmax,
        order=order, options=options,
    )
    zk = np.asarray(zk_mode, dtype=float).ravel()  # shape (nk,)

    # Back-transform mode estimates to Y scale
    if do_transform:
        nst = NormalScoreTransform()
        nst.fit_from_cdf(yfile_arr, cdfyfile_arr)
        yk = nst.inverse_transform(zk)
    else:
        yk = zk

    return yk, info.ravel()


def BMEprobaTPdf(
    y: np.ndarray,
    ck: np.ndarray,
    ch: np.ndarray,
    cs: np.ndarray,
    yh: np.ndarray,
    ysoftpdftype: int,
    ynl: np.ndarray,
    ylimi: np.ndarray,
    yprobdens: np.ndarray,
    covmodel,
    covparam: np.ndarray,
    nhmax: int,
    nsmax: int,
    dmax,
    order,
    options=None,
    yfile: Optional[np.ndarray] = None,
    cdfyfile: Optional[np.ndarray] = None,
) -> tuple:
    """BME posterior PDF prediction with Normal Score Transform.

    Adapted from ``BMEprobaTPdf.m`` (BMElib bmeprobalib, Jan 1, 2001).

    Computes the BME posterior probability density function at a **single**
    estimation point, using hard data, soft probabilistic data, and an
    optional Normal Score Transform (NST).  The result is reported on the
    **original Y scale**.

    Parameters
    ----------
    y : array_like of shape (ny,) or empty
        Y values at which to evaluate the posterior PDF.  If empty (``[]``),
        a grid of Y values is automatically constructed based on the mode
        and spread of the posterior.
    ck : ndarray of shape (1, d)
        Coordinate of the **single** estimation location.
    ch : ndarray of shape (nh, d)
        Coordinates of hard data locations.
    cs : ndarray of shape (ns, d)
        Coordinates of soft data locations.
    yh : ndarray of shape (nh,)
        Hard data values on the Y scale.
    ysoftpdftype : int
        Soft PDF type for Y.
    ynl : ndarray of shape (ns, 1)
        Number of interval limits per soft datum.
    ylimi : ndarray of shape (ns, l)
        Interval limits in Y space.
    yprobdens : ndarray of shape (ns, p)
        Probability densities in Y space.
    covmodel : str or list
        Covariance model name(s) for the Z field.
    covparam : array_like
        Covariance model parameters.
    nhmax : int
        Maximum number of hard data neighbours.
    nsmax : int
        Maximum number of soft data neighbours (≤ 20).
    dmax : float or array_like
        Neighbourhood distance threshold.
    order : float or NaN
        Drift polynomial order.
    options : array_like or None, optional
        Algorithm parameters.
    yfile : array_like of shape (k,) or None, optional
        Y nodes of the prior CDF.
    cdfyfile : array_like of shape (k,) or None, optional
        CDF values at the prior-CDF nodes.

    Returns
    -------
    y : ndarray of shape (ny,)
        Y values at which the posterior PDF is evaluated.
    ypdf : ndarray of shape (ny,)
        Posterior PDF values on the original Y scale.
    info : ndarray or scalar
        Diagnostic code(s) (see :func:`BMEprobaTMode` for code meanings).

    Notes
    -----
    This function is a thin wrapper around :func:`BMEPosteriorPDF` in
    ``BMEprobaEstimations.py``.  When ``yfile`` is provided, it applies the
    NST to hard/soft data before calling the core PDF estimator, then
    back-transforms the PDF result via :func:`pdfgauss2other`.

    Index convention: 0-based (adapted from 1-based MATLAB code).
    """
    from .BMEprobaEstimations import BMEPosteriorPDF

    _y_raw = np.asarray(y, dtype=float).ravel()
    y_arr  = _y_raw if _y_raw.size > 0 else np.array([], dtype=float)

    do_transform = (yfile is not None) and (len(np.asarray(yfile)) > 0)

    if do_transform:
        yfile_arr    = np.asarray(yfile,    dtype=float).ravel()
        cdfyfile_arr = np.asarray(cdfyfile, dtype=float).ravel()
        FyTransformCheckArgs(yfile_arr, cdfyfile_arr, ck, ch, cs,
                             yh, ysoftpdftype, ynl, ylimi, yprobdens)
        zh = other2gauss(yh, yfile_arr, cdfyfile_arr)
        zsoftpdftype, znl, zlimi, zprobdens = probaother2gauss(
            ysoftpdftype, ynl, ylimi, yprobdens, yfile_arr, cdfyfile_arr
        )
        # Transform query y values to z (None triggers auto-grid inside)
        z_query = other2gauss(y_arr, yfile_arr, cdfyfile_arr) if y_arr.size > 0 else None
    else:
        yfile_arr    = None
        cdfyfile_arr = None
        zh           = np.asarray(yh, dtype=float)
        zsoftpdftype = ysoftpdftype
        znl          = ynl
        zlimi        = ylimi
        zprobdens    = yprobdens
        z_query      = y_arr if y_arr.size > 0 else None

    # Convert flat soft-data to internal zs list-of-tuples
    zs_list = _flat2zs(zsoftpdftype, znl, zlimi, zprobdens)

    # Ensure zh is (nh, 1) column vector
    zh_2d = np.asarray(zh, dtype=float).ravel()[:, np.newaxis]

    # BMEprobaTPdf operates on a SINGLE estimation point; ck must be (1, d)
    ck_single = np.atleast_2d(ck)[:1, :]

    # Compute posterior PDF in Z space
    # BMEPosteriorPDF returns:
    #   (z_arr,  pdf_matrix (1, nz), info)  when z_grid is a 1-D array
    #   ([z1d],  [pdf1d],            info)  when z_grid is None (auto)
    z_result, pdf_result, info_arr = BMEPosteriorPDF(
        ck_single, ch, cs, zh_2d,
        zs=zs_list,
        covmodel=covmodel, covparam=covparam,
        nhmax=nhmax, nsmax=nsmax, dmax=dmax,
        order=order, options=options,
        z_grid=z_query,
    )

    # Extract 1-D arrays for the single estimation point
    if isinstance(z_result, list):
        z_out   = z_result[0]
        zpdf_out = np.asarray(pdf_result[0], dtype=float).ravel()
    else:
        z_out    = np.asarray(z_result,        dtype=float).ravel()
        zpdf_out = np.asarray(pdf_result[0, :], dtype=float).ravel()

    info = float(info_arr.ravel()[0])

    # Back-transform PDF to Y scale
    if do_transform:
        y_return = y_arr if y_arr.size > 0 else np.interp(
            z_out,
            _norm_dist.ppf(np.clip(cdfyfile_arr, 1e-9, 1 - 1e-9)),
            yfile_arr,
            left=np.nan, right=np.nan,
        )
        ypdf = pdfgauss2other(y_return, z_out, zpdf_out,
                              yfile_arr, cdfyfile_arr)
        mask_out = (y_return < yfile_arr[0]) | (y_return > yfile_arr[-1])
        ypdf = np.where(mask_out, 0.0, ypdf)
    else:
        ypdf     = zpdf_out
        y_return = z_out

    return y_return, ypdf, info


def BMEprobaTCI(
    ck: np.ndarray,
    ch: np.ndarray,
    cs: np.ndarray,
    yh: np.ndarray,
    ysoftpdftype: int,
    ynl: np.ndarray,
    ylimi: np.ndarray,
    yprobdens: np.ndarray,
    covmodel,
    covparam: np.ndarray,
    nhmax: int,
    nsmax: int,
    dmax,
    order,
    ci_level: float = 0.95,
    options=None,
    yfile: Optional[np.ndarray] = None,
    cdfyfile: Optional[np.ndarray] = None,
) -> tuple:
    """BME credible interval prediction with Normal Score Transform.

    Adapted from the BMElib T-variant framework (BMElib bmeprobalib).

    Computes the BME posterior credible interval (CI) on the **original Y
    scale** at a set of estimation points.

    Parameters
    ----------
    ck : ndarray of shape (nk, d)
        Estimation point coordinates.
    ch : ndarray of shape (nh, d)
        Hard data coordinates.
    cs : ndarray of shape (ns, d)
        Soft data coordinates.
    yh : ndarray of shape (nh,)
        Hard data values on the Y scale.
    ysoftpdftype : int
        Soft PDF type for Y.
    ynl : ndarray of shape (ns, 1)
        Number of interval limits.
    ylimi : ndarray of shape (ns, l)
        Interval limits in Y space.
    yprobdens : ndarray of shape (ns, p)
        Probability densities in Y space.
    covmodel : str or list
        Covariance model name(s) for the Z field.
    covparam : array_like
        Covariance model parameters.
    nhmax : int
        Maximum number of hard data neighbours.
    nsmax : int
        Maximum number of soft data neighbours (≤ 20).
    dmax : float or array_like
        Neighbourhood distance threshold.
    order : float or NaN
        Drift polynomial order.
    ci_level : float, optional
        Credible interval level in (0, 1).  Default is 0.95 (95 % CI).
    options : array_like or None, optional
        Algorithm parameters.
    yfile : array_like of shape (k,) or None, optional
        Y nodes of the prior CDF.
    cdfyfile : array_like of shape (k,) or None, optional
        CDF values at the prior-CDF nodes.

    Returns
    -------
    yk_lower : ndarray of shape (nk,)
        Lower bound of the credible interval on the Y scale.
    yk_upper : ndarray of shape (nk,)
        Upper bound of the credible interval on the Y scale.
    info : ndarray of shape (nk,)
        Diagnostic codes per estimation point.

    Notes
    -----
    This function is a thin wrapper around :func:`BMEPosteriorCI` in
    ``BMEprobaEstimations.py``.  When ``yfile`` is provided, the NST is
    applied to data before calling the core CI estimator, and the CI bounds
    are back-transformed to Y scale using the inverse NST.

    Index convention: 0-based (adapted from 1-based MATLAB code).
    """
    from .BMEprobaEstimations import BMEPosteriorCI

    do_transform = (yfile is not None) and (len(np.asarray(yfile)) > 0)

    if do_transform:
        yfile_arr    = np.asarray(yfile,    dtype=float).ravel()
        cdfyfile_arr = np.asarray(cdfyfile, dtype=float).ravel()
        FyTransformCheckArgs(yfile_arr, cdfyfile_arr, ck, ch, cs,
                             yh, ysoftpdftype, ynl, ylimi, yprobdens)
        zh = other2gauss(yh, yfile_arr, cdfyfile_arr)
        zsoftpdftype, znl, zlimi, zprobdens = probaother2gauss(
            ysoftpdftype, ynl, ylimi, yprobdens, yfile_arr, cdfyfile_arr
        )
    else:
        yfile_arr    = None
        cdfyfile_arr = None
        zh           = np.asarray(yh, dtype=float)
        zsoftpdftype = ysoftpdftype
        znl          = ynl
        zlimi        = ylimi
        zprobdens    = yprobdens

    # Convert flat soft-data to internal zs list-of-tuples
    zs_list = _flat2zs(zsoftpdftype, znl, zlimi, zprobdens)

    # Ensure zh is (nh, 1) column vector
    zh_2d = np.asarray(zh, dtype=float).ravel()[:, np.newaxis]

    # BMEPosteriorCI returns (zlCI, zuCI, pdfCI, PCI, z_grids, pdf_grids)
    # Each of zlCI / zuCI has shape (nk, nCI).  Here nCI=1 for a single level.
    zlCI, zuCI, _pdfCI, _PCI, _z_grids, _pdf_grids = BMEPosteriorCI(
        ck=ck, ch=ch, cs=cs, zh=zh_2d,
        zs=zs_list,
        covmodel=covmodel, covparam=covparam,
        nhmax=nhmax, nsmax=nsmax, dmax=dmax,
        order=order, ci_probs=[ci_level], options=options,
    )

    zk_lower = np.asarray(zlCI[:, 0], dtype=float).ravel()
    zk_upper = np.asarray(zuCI[:, 0], dtype=float).ravel()
    nk = len(zk_lower)
    info = np.zeros(nk)  # BMEPosteriorCI does not produce a separate info vector

    # Back-transform CI bounds to Y scale
    if do_transform:
        nst = NormalScoreTransform()
        nst.fit_from_cdf(yfile_arr, cdfyfile_arr)
        yk_lower = nst.inverse_transform(zk_lower)
        yk_upper = nst.inverse_transform(zk_upper)
    else:
        yk_lower = zk_lower
        yk_upper = zk_upper

    return yk_lower, yk_upper, info
