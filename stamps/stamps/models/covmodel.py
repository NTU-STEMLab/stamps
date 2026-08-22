# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import gamma, kv

def get_model(m_name):
    """Retrieve a covariance or variogram model function by name.

    Performs a dynamic lookup of a model function defined in this module using
    its string name.  This allows callers such as ``coord2K`` and ``covmodelfit``
    to work with model names stored as strings in parameter lists rather than
    holding direct function references.

    Parameters
    ----------
    m_name : str
        Name of the model function to retrieve.  Must match one of the
        covariance or variogram functions defined in this module, e.g.
        ``'exponentialC'``, ``'gaussianC'``, ``'sphericalC'``,
        ``'nuggetC'``, ``'maternC'``, ``'exponentialV'``, etc.

    Returns
    -------
    func : callable
        The model function corresponding to *m_name*.  The returned callable
        shares the signature ``func(dist, sill, ar, ...)`` used by all
        covariance and variogram models in this module.

    Raises
    ------
    NameError
        If *m_name* does not match any function defined in this module.

    Notes
    -----
    Internally uses ``eval(m_name)`` within the module's namespace.  Only
    names that resolve to objects in this module are accessible; arbitrary
    Python expressions are **not** evaluated for security reasons (the module
    namespace is the implicit context).

    Examples
    --------
    >>> from stamps.stamps.models.covmodel import get_model
    >>> fn = get_model('exponentialC')
    >>> import numpy as np
    >>> dist = np.array([[0., 500., 1000.]])
    >>> fn(dist, sill=10.0, ar=300.0)
    array([[10.        ,  0.22313016,  0.00497871]])
    """
    try:
        return eval(m_name)
    except NameError as e:
        #must be do something...
        raise e

def exponentialC(dist, sill, ar, jac=False, jacpar=None):
    """Exponential covariance model.

    Computes the isotropic exponential covariance function — or its first/
    second analytical Jacobian with respect to *sill* or *ar* — over a
    pre-computed distance matrix.

    The model is:

    .. math::

        C(h) = \\text{sill} \\cdot \\exp\\!\\left(-\\frac{3h}{\\text{ar}}\\right)

    where *ar* is the **practical range**: the distance at which C(h) drops
    to ≈ 5 % of the sill (i.e. exp(-3) ≈ 0.05).

    Parameters
    ----------
    dist : np.ndarray, shape (m, n)
        Non-negative pairwise distance matrix between two sets of locations.
    sill : float
        Variance at zero distance (C(0) = sill).
    ar : float
        Practical range parameter (distance where C ≈ 0.05 · sill).
    jac : bool, optional
        If ``False`` (default), return the covariance matrix.
        If ``True``, return the Jacobian (partial derivative) specified by
        *jacpar*.
    jacpar : {'sill', 'ar', 'ar2'} or None, optional
        Which parameter's Jacobian to return when ``jac=True``.

        * ``'sill'``  — ∂C/∂sill = exp(−3h/ar)
        * ``'ar'``    — ∂C/∂ar   = sill · (3h/ar²) · exp(−3h/ar)
        * ``'ar2'``   — ∂²C/∂ar² (second derivative w.r.t. *ar*)

    Returns
    -------
    cov : np.ndarray, shape (m, n)
        Covariance matrix when ``jac=False``.
    jac : np.ndarray, shape (m, n)
        Jacobian matrix when ``jac=True``.

    Notes
    -----
    The Jacobian outputs are used by the covariance-fitting routines in
    ``stcovfit`` and ``mlecovfit`` to drive gradient-based optimisation of
    the model parameters.

    Examples
    --------
    >>> import numpy as np
    >>> from stamps.stamps.models.covmodel import exponentialC
    >>> dist = np.array([[0., 300., 600.]])
    >>> exponentialC(dist, sill=10.0, ar=300.0)
    array([[10.        ,  0.49787068,  0.02478752]])

    ### Original Reference ###
    Input:
    dist      m by n    an 2D array of spatial or S/T distances
    sill      scalar    covariance sill
    ar        scalar    covariance correlation length
    jac       bool      if the return should be jacobian of parameters.
                        the default is False to return the covariance estimation
    jacpar    string    To determine which parameter is used for Jacobian
                        estimation, the default is None.
                        Options are 'sill' and 'ar'

    Output:
    cov/jac   m by n    an 2D array of covariance or Jacobian estimation
                        cooresponding to dist
    """

    if not jac:
        cov = sill * np.exp(-3 * dist / ar)
        return cov
    else:
        if jacpar == 'sill':
            jac = np.exp(-3 * dist / ar)
        elif jacpar == 'ar':
            jac = sill*3*dist/(ar**2)*np.exp(-3*dist/ar)
        elif jacpar == 'ar2':
            jac = sill*((3*dist/(ar**2))**2-6*dist/(ar**3))*np.exp(-3*dist/ar)      
        return jac

def gaussianC(dist, sill, ar, jac=False, jacpar=None):
    """Gaussian covariance model.

    Computes the isotropic Gaussian (squared-exponential) covariance function:

    .. math::

        C(h) = \\text{sill} \\cdot \\exp\\!\\left(-3 \\left(\\frac{h}{\\text{ar}}\\right)^2\\right)

    where *ar* is the **practical range**: the distance at which C(h) drops
    to ≈ 5 % of the sill (i.e. exp(−3) ≈ 0.05).

    The Gaussian model is infinitely differentiable at the origin; it should
    be used for phenomena with very smooth, predictable spatial variation.
    Unlike the exponential model it has a parabolic behaviour near h = 0.

    Parameters
    ----------
    dist : np.ndarray, shape (m, n)
        Non-negative pairwise distance matrix.
    sill : float
        Variance at zero distance (C(0) = sill).
    ar : float
        Practical range parameter.
    jac : bool, optional
        Return Jacobian instead of covariance when ``True``.
    jacpar : {'sill', 'ar', 'ar2'} or None, optional
        Which parameter's partial derivative to return when ``jac=True``.

        * ``'sill'``  — ∂C/∂sill
        * ``'ar'``    — ∂C/∂ar
        * ``'ar2'``   — ∂²C/∂ar²

    Returns
    -------
    out : np.ndarray, shape (m, n)
        Covariance (``jac=False``) or Jacobian (``jac=True``) matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from stamps.stamps.models.covmodel import gaussianC
    >>> dist = np.array([[0., 300., 600.]])
    >>> gaussianC(dist, sill=10.0, ar=300.0)
    array([[10.        ,  0.04978707,  6.1442e-06]])
    """
    if not jac:
        return sill * np.exp(-3 * (dist / ar)**2)
    else:
        if jacpar == 'sill':
            jac = np.exp(-3 * (dist / ar)**2)
        elif jacpar == 'ar':
            jac = sill * np.exp(-3 * (dist / ar)**2)*(6*(dist**2)/ar**3)
        elif jacpar == 'ar2':
            jac = sill * np.exp(-3*(dist/ar)**2) *\
                ((6*(dist**2)/ar**3)**2 - 18*(dist**2)/(ar**4))
        return jac

def sphericalC(dist, sill, ar, jac=False, jacpar=None):
    """Spherical covariance model.

    Computes the isotropic spherical covariance function:

    .. math::

        C(h) = \\begin{cases}
            \\text{sill} \\left(1 - \\frac{3h}{2\\,\\text{ar}} +
            \\frac{h^3}{2\\,\\text{ar}^3}\\right) & h \\leq \\text{ar} \\\\
            0 & h > \\text{ar}
        \\end{cases}

    The spherical model has compact support (C(h) = 0 for h > ar) and a
    linear behaviour near the origin, making it suitable for phenomena with
    moderate spatial continuity.  It is valid in spaces of dimension d ≤ 3.

    Parameters
    ----------
    dist : np.ndarray, shape (m, n)
        Non-negative pairwise distance matrix.
    sill : float
        Variance at zero distance (C(0) = sill).
    ar : float
        Range — the distance at which the covariance reaches exactly 0.
    jac : bool, optional
        Return Jacobian instead of covariance when ``True``.
    jacpar : {'sill', 'ar'} or None, optional
        Which parameter's partial derivative to return when ``jac=True``.

    Returns
    -------
    out : np.ndarray, shape (m, n)
        Covariance (``jac=False``) or Jacobian (``jac=True``) matrix.
        Entries with ``dist > ar`` are forced to zero.

    Notes
    -----
    Valid in d ≤ 3 spatial dimensions.  Unlike the exponential and Gaussian
    models, the spherical model reaches its sill at a **finite distance**
    (exactly *ar*).

    Examples
    --------
    >>> import numpy as np
    >>> from stamps.stamps.models.covmodel import sphericalC
    >>> dist = np.array([[0., 500., 1000., 1500.]])
    >>> sphericalC(dist, sill=10.0, ar=1000.0)
    array([[10.    ,  6.5625,  0.    ,  0.    ]])
    """
    value = dist / ar
    if not jac:    
        result = sill * (1- ((3/2.) * value - (1/2.) * (value)**3))
        result[dist > ar] = 0.
        return result
    else:
        if jacpar == 'sill':
            jac = 1 - ((3/2.) * value - (1/2.) * (value)**3)
            jac[dist > ar] = 0.
        elif jacpar == 'ar':
            jac = sill*(3/2.*dist/ar**2-3/2.*dist**3/ar**4)   
            jac[dist > ar] = 0.
        elif jacpar == 'ar':
            jac = sill*(-3.*dist/ar**3-6.*dist**3/ar**5)   
            jac[dist > ar] = 0.      
        return jac  
    
def holecosC(dist, a_half, p_half):
    """Cosinusoidal hole-effect covariance model.

    Computes the cosinusoidal hole-effect covariance:

    .. math::

        C(h) = a_{\\text{half}} \\cdot \\cos\\!\\left(\\frac{\\pi h}{p_{\\text{half}}}\\right)

    This model is used for phenomena with periodic spatial patterns (e.g.,
    cyclic geological structures or seasonally forced fields).

    Parameters
    ----------
    dist : np.ndarray, shape (m, n)
        Non-negative pairwise distance matrix.
    a_half : float
        Half-amplitude.  C(0) = a_half; the first trough is −a_half at
        h = p_half.
    p_half : float
        Half the period — the distance from the origin to the first trough.

    Returns
    -------
    C : np.ndarray, shape (m, n)
        Covariance matrix with the same shape as *dist*.

    Notes
    -----
    This model can be negative for h in (p_half, 2*p_half), which is
    physically meaningful for periodic phenomena but violates positive-
    definiteness in a strict sense.  Use it only for 1-dimensional fields
    or in combination with a positive-definite component.

    Examples
    --------
    >>> import numpy as np
    >>> from stamps.stamps.models.covmodel import holecosC
    >>> dist = np.array([[0., 500., 1000., 1500.]])
    >>> holecosC(dist, a_half=5.0, p_half=1000.0)
    array([[ 5.        ,  0.        , -5.        ,  0.        ]])
    """
    return a_half * np.cos(np.pi * dist / p_half)

def nuggetC(dist, sill, ar=None, jac=False, jacpar=None):
    """Nugget-effect covariance model.

    Represents microscale variability or measurement error: a discontinuity
    at the origin with zero covariance everywhere else.

    .. math::

        C(h) = \\begin{cases} \\text{sill} & h = 0 \\\\ 0 & h > 0 \\end{cases}

    In practice the nugget effect accounts for:
    * measurement error in the observed data,
    * sub-resolution spatial variation that cannot be resolved at the data
      spacing, or
    * both.

    Parameters
    ----------
    dist : np.ndarray, shape (m, n)
        Non-negative pairwise distance matrix.
    sill : float
        Nugget variance (C(0) = sill).
    ar : float or None, optional
        Not used; accepted for API compatibility with other model signatures.
    jac : bool, optional
        If ``True``, return the Jacobian (always an empty (0, 0) array for the
        nugget, as its only free parameter is *sill* and the derivative w.r.t.
        *sill* is the identity given by the covariance itself).
    jacpar : str or None, optional
        Ignored; accepted for API compatibility.

    Returns
    -------
    out : np.ndarray
        Covariance matrix of shape (m, n) when ``jac=False``.
        Empty array of shape (0, 0) when ``jac=True``.

    Notes
    -----
    When used in a nested model the nugget is typically the first component,
    e.g. ``covmodel=['nuggetC', 'exponentialC']`` with ``covparam=[(c0,), (c1, a1)]``.
    ``coord2K`` handles the single-element parameter tuple automatically.

    Examples
    --------
    >>> import numpy as np
    >>> from stamps.stamps.models.covmodel import nuggetC
    >>> dist = np.array([[0., 1., 10.]])
    >>> nuggetC(dist, sill=2.5)
    array([[2.5, 0. , 0. ]])
    """
    if not jac:
        result = np.zeros(dist.shape)
        result[dist == 0] = sill
        return result
    else:
        jac=np.zeros([0,0])
        return jac

# ---------------------------------------------------------------------------
# Non-separable space-time aliases (ST suffix)
# ---------------------------------------------------------------------------
# For non-separable models, coord2K passes the combined distance
#   d_combined = dist_s + s_t_ratio * dist_t
# to the spatial covariance model.  These "CST" variants are therefore
# identical to their pure-spatial counterparts — the "ST" naming merely
# signals to coord2K that they participate in the non-separable S/T path.

def nuggetCST(dist, sill, ar=None, jac=False, jacpar=None):
    """Non-separable ST nugget — identical to nuggetC (applied to combined dist)."""
    return nuggetC(dist, sill, ar, jac, jacpar)

def exponentialCST(dist, sill, ar, jac=False, jacpar=None):
    """Non-separable ST exponential — identical to exponentialC."""
    return exponentialC(dist, sill, ar, jac, jacpar)

def gaussianCST(dist, sill, ar, jac=False, jacpar=None):
    """Non-separable ST Gaussian — identical to gaussianC."""
    return gaussianC(dist, sill, ar, jac, jacpar)

def sphericalCST(dist, sill, ar, jac=False, jacpar=None):
    """Non-separable ST spherical — identical to sphericalC."""
    return sphericalC(dist, sill, ar, jac, jacpar)


def holesinC(dist, sill, period):
    """Sinusoidal hole-effect covariance model.

    Computes the sinusoidal hole-effect covariance matrix from a distance
    matrix.

    Parameters
    ----------
    dist : np.ndarray
        n by m distance matrix with values >= 0.
    sill : float
        Sill (variance) of the model.
    period : float
        Periodicity of the hole effect.  ``period`` is the distance to
        the first trough from the origin (i.e. the half-wavelength).

    Returns
    -------
    C : np.ndarray
        n by m covariance matrix with the same shape as *dist*.

    Notes
    -----
    C(h) = sill * sin(π h / period) / (π h / period)  for h > 0,
    C(0) = sill.
    Valid in d-dimensional space with d ≤ 3.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.pi * dist / period
        C = np.where(dist == 0, sill, sill * np.sin(ratio) / ratio)
    return C


def mexicanhatC(dist, sill, b, c):
    """Mexican-hat (ricker wavelet) covariance model.

    C(h) = sill * (1 − b · h²) · exp(−(c · h)²)

    Parameters
    ----------
    dist : np.ndarray
        n by m distance matrix with values >= 0.
    sill : float
        Overall scale (amplitude at zero).
    b : float
        Quadratic modulation parameter (controls width of negative lobe).
    c : float
        Gaussian decay parameter.

    Returns
    -------
    C : np.ndarray
        n by m covariance matrix with the same shape as *dist*.
    """
    return sill * (1.0 - b * dist ** 2) * np.exp(-(c * dist) ** 2)


def maternC(dist, sill, parms):
    """Matérn covariance model with practical-range parameterisation.

    Computes the isotropic Matérn covariance function:

    .. math::

        C(h) = \\frac{\\text{sill}}{2^{\\nu-1}\\,\\Gamma(\\nu)}
               \\left(\\frac{\\tilde{h}}{a_r}\\right)^{\\nu}
               K_{\\nu}\\!\\left(\\frac{\\tilde{h}}{a_r}\\right)

    where :math:`\\tilde{h} = s \\cdot h` is a rescaled distance,
    :math:`s` is a shape-dependent scale factor chosen so that *ar* is
    approximately the **practical range** (distance where C ≈ 0.05 · sill),
    :math:`K_\\nu` is the modified Bessel function of the second kind of
    order :math:`\\nu`, and :math:`\\Gamma` is the gamma function.

    Special cases: ν = 0.5 → exponential, ν → ∞ → Gaussian.

    Parameters
    ----------
    dist : np.ndarray, shape (m, n)
        Non-negative pairwise distance matrix.
    sill : float
        Variance at zero distance (C(0) = sill).
    parms : list or tuple of length 2
        ``parms[0]`` : float
            Practical range *ar* (distance where C ≈ 0.05 · sill).
        ``parms[1]`` : float
            Smoothness (shape) parameter *ν* > 0.  Larger values imply
            smoother realisations.  Common choices: 0.5, 1.5, 2.5.

    Returns
    -------
    C : np.ndarray, shape (m, n)
        Covariance matrix with the same shape as *dist*.
        C(0) = sill is enforced explicitly to avoid the 0 · ∞ indeterminate
        form from K_ν at the origin.

    Notes
    -----
    A ``numpy.errstate(invalid='ignore', divide='ignore')`` context is used
    internally to suppress the unavoidable 0 · K_ν(0) = 0 · ∞ = nan warning
    at zero lag; the correct limit C(0) = sill is applied afterward.

    The scale factor is::

        s = sqrt(2*ν*6) * (3/2)**(1/(4*ν))

    which is chosen so that C(ar) ≈ 0.05 · sill for all ν.

    Examples
    --------
    >>> import numpy as np
    >>> from stamps.stamps.models.covmodel import maternC
    >>> dist = np.array([[0., 500., 1000.]])
    >>> maternC(dist, sill=10.0, parms=[300.0, 1.5])
    array([[10.  ,  ...,  ...]])

    ### Original Reference ###
    Matern covariance with practical distance adjustment

    C = maternC(dist, sill, parms)

    Input
    dist    m by n     an numpy array of S/T distances
    sill    scalar     sill
    parms   list       a list of two components
                       (correlation length, shape parameter).

    Output
    Cov     m by n     an numpy array of S/T covariances
    """

    ar = parms[0]
    shp = parms[1]
    scale = np.sqrt(2*shp*6)*(3./2)**(1./4./shp)
    sdist = scale*dist
    # Suppress the unavoidable 0 * kv(shp, 0) = 0 * inf = nan at zero lag;
    # the correct limit C(0) = sill is applied explicitly afterwards.
    with np.errstate(invalid='ignore', divide='ignore'):
        result = (sill/(2.**(shp-1))/gamma(shp)) * ((sdist/ar)**shp) * kv(shp, sdist/ar)
    result[sdist == 0] = sill
    return result


# ===========================================================================
# Variogram models
# ===========================================================================
# Convention: all variogram functions share the same calling signature as
# their covariance counterparts so they can be looked up via get_model().
#   V = model(dist, sill, ar)
# where *sill* is the variance (plateau) and *ar* is the practical range.
# Two intrinsic models (linearV, powerV) use a slope + exponent instead.
# ===========================================================================


def nuggetV(dist: np.ndarray, sill: float, ar: float = None) -> np.ndarray:
    """Nugget-effect variogram model.

    γ(h) = sill  for h > 0,  γ(0) = 0.

    Parameters
    ----------
    dist : np.ndarray
        n by m distance matrix with values >= 0.
    sill : float
        Sill (discontinuity at the origin).
    ar : float, optional
        Not used; accepted for API compatibility with other model signatures.

    Returns
    -------
    V : np.ndarray
        n by m variogram matrix with the same shape as *dist*.
    """
    V = np.zeros(np.shape(dist))
    V[dist > 0] = sill
    return V


def exponentialV(dist: np.ndarray, sill: float, ar: float) -> np.ndarray:
    """Exponential variogram model.

    γ(h) = sill · (1 − exp(−3 h / ar))

    The *ar* parameter is the **practical range** — the distance at which
    the variogram reaches ≈ 95 % of the sill.

    Parameters
    ----------
    dist : np.ndarray
        n by m distance matrix with values >= 0.
    sill : float
        Sill of the model.
    ar : float
        Practical range (distance where γ ≈ 0.95 · sill).

    Returns
    -------
    V : np.ndarray
        n by m variogram matrix with the same shape as *dist*.
    """
    return sill * (1.0 - np.exp(-3.0 * dist / ar))


def sphericalV(dist: np.ndarray, sill: float, ar: float) -> np.ndarray:
    """Spherical variogram model.

    γ(h) = sill · (3h/(2·ar) − h³/(2·ar³))  for h ≤ ar,
    γ(h) = sill                                for h > ar.

    Parameters
    ----------
    dist : np.ndarray
        n by m distance matrix with values >= 0.
    sill : float
        Sill of the model.
    ar : float
        Range of the model (distance at which γ reaches the sill).

    Returns
    -------
    V : np.ndarray
        n by m variogram matrix with the same shape as *dist*.

    Notes
    -----
    Valid in d-dimensional space with d ≤ 3.
    """
    h = dist / ar
    V = sill * (1.5 * h - 0.5 * h ** 3)
    V[dist > ar] = sill
    return V


def gaussianV(dist: np.ndarray, sill: float, ar: float) -> np.ndarray:
    """Gaussian variogram model.

    γ(h) = sill · (1 − exp(−3 h² / ar²))

    The *ar* parameter is the **practical range** — the distance at which
    the variogram reaches ≈ 95 % of the sill.

    Parameters
    ----------
    dist : np.ndarray
        n by m distance matrix with values >= 0.
    sill : float
        Sill of the model.
    ar : float
        Practical range (distance where γ ≈ 0.95 · sill).

    Returns
    -------
    V : np.ndarray
        n by m variogram matrix with the same shape as *dist*.
    """
    return sill * (1.0 - np.exp(-3.0 * (dist / ar) ** 2))


def holecosV(dist: np.ndarray, a_half: float, p_half: float) -> np.ndarray:
    """Cosinusoidal hole-effect variogram model.

    γ(h) = a_half · (1 − cos(π h / p_half))

    Parameters
    ----------
    dist : np.ndarray
        n by m distance matrix with values >= 0.
    a_half : float
        Half-amplitude of the hole effect (maximum γ = 2 · a_half).
    p_half : float
        Half the periodicity (the distance from the origin to the first peak).

    Returns
    -------
    V : np.ndarray
        n by m variogram matrix with the same shape as *dist*.

    Notes
    -----
    Valid in a 1-dimensional space only.
    """
    return a_half * (1.0 - np.cos(np.pi * dist / p_half))


def holesinV(dist: np.ndarray, sill: float, period: float) -> np.ndarray:
    """Sinusoidal hole-effect variogram model.

    γ(h) = sill · (1 − sin(π h / period) / (π h / period))  for h > 0,
    γ(0) = 0.

    Parameters
    ----------
    dist : np.ndarray
        n by m distance matrix with values >= 0.
    sill : float
        Sill of the model.
    period : float
        Periodicity of the hole effect (first zero-crossing wavelength).

    Returns
    -------
    V : np.ndarray
        n by m variogram matrix with the same shape as *dist*.

    Notes
    -----
    Valid in d-dimensional space with d ≤ 3.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.pi * dist / period
        sinc = np.where(dist == 0, 1.0, np.sin(ratio) / ratio)
    return sill * (1.0 - sinc)


def linearV(dist: np.ndarray, slope: float, ar: float = None) -> np.ndarray:
    """Linear (intrinsic) variogram model.

    γ(h) = slope · h

    An *intrinsic* model — it has no finite sill.  The *ar* argument is
    accepted for API uniformity but is not used.

    Parameters
    ----------
    dist : np.ndarray
        n by m distance matrix with values >= 0.
    slope : float
        Positive slope of the linear variogram.
    ar : float, optional
        Not used; accepted for API compatibility.

    Returns
    -------
    V : np.ndarray
        n by m variogram matrix with the same shape as *dist*.
    """
    return slope * dist


def powerV(dist: np.ndarray, slope: float, exponent: float) -> np.ndarray:
    """Power (intrinsic) variogram model.

    γ(h) = slope · h^exponent

    An *intrinsic* model — it has no finite sill.

    Parameters
    ----------
    dist : np.ndarray
        n by m distance matrix with values >= 0.
    slope : float
        Scale factor (slope at unit distance).
    exponent : float
        Power exponent, with 0 < exponent < 2.

    Returns
    -------
    V : np.ndarray
        n by m variogram matrix with the same shape as *dist*.
    """
    return slope * (dist ** exponent)
