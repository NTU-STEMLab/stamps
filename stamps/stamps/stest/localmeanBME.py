# -*- coding: utf-8 -*-
import numpy

from .regression import regression
from .designmatrix import designmatrix


def localmeanBME( ck, ch, cs, zh, ms, vs,
                  Khh, Ksh, Kss, order ):
    """Estimate the local prior mean (and its variance) for a BME estimation node.

    Within each BME estimation neighbourhood this function fits the spatial
    trend (drift) coefficients by **Generalised Least Squares** (GLS) using
    all available hard and soft data, and then evaluates the fitted trend at
    the estimation node ``ck``.

    Internally it calls :func:`regression` (which calls :func:`designmatrix`)
    to solve the GLS system for the trend coefficients and their covariance
    matrix, then uses :func:`designmatrix` again to evaluate the design matrix
    at ``ck``.

    Parameters
    ----------
    ck : np.ndarray, shape (1, nd)
        Coordinates of the single estimation node.
    ch : np.ndarray, shape (nh, nd)
        Coordinates of the hard (exact) data points.
    cs : np.ndarray, shape (ns, nd)
        Coordinates of the soft (probabilistic) data points.
    zh : np.ndarray, shape (nh, 1)
        Hard data values at ``ch``.
    ms : np.ndarray, shape (ns,) or (ns, 1)
        Prior means (expected values) of the soft data PDFs at ``cs``.
    vs : np.ndarray, shape (ns,) or (ns, 1)
        Prior variances of the soft data PDFs at ``cs``.
    Khh : np.ndarray, shape (nh, nh)
        Covariance matrix between hard data locations.
    Ksh : np.ndarray, shape (ns, nh)
        Cross-covariance matrix between soft and hard data locations.
    Kss : np.ndarray, shape (ns, ns)
        Covariance matrix between soft data locations.
    order : int or float
        Trend order used by :func:`designmatrix`:

        * ``numpy.nan`` — no trend (zero mean); returns zeros everywhere.
        * ``0`` — constant drift (ordinary kriging style); a scalar mean is
          estimated from the combined data.

    Returns
    -------
    mkest : float
        Estimated prior mean at the estimation node ``ck``.
    mhest : np.ndarray, shape (nh, 1)
        Estimated mean at each hard data location (fitted trend values).
    msest : np.ndarray, shape (ns, 1)
        Estimated mean at each soft data location (fitted trend values).
    vkest : float
        Variance of the estimated mean at ``ck``, i.e. the propagated
        GLS coefficient uncertainty.

    Notes
    -----
    **Algorithm overview**

    1. Stack the hard and soft coordinates: ``c = vstack([ch, cs])``.
    2. Stack the observed/expected values: ``z = vstack([zh, ms])``.
    3. Build the full spatial covariance matrix ``K`` from the three blocks
       ``Khh``, ``Ksh``, ``Kss``.
    4. Call :func:`regression` to solve the GLS system:

       .. math::

           \\hat{\\mathbf{b}} = (\\mathbf{X}^T \\mathbf{K}^{-1} \\mathbf{X})^{-1}
                                 \\mathbf{X}^T \\mathbf{K}^{-1} \\mathbf{z}

    5. Evaluate the trend at ``ck``:

       .. math::

           \\hat{m}_k = \\mathbf{x}_k \\hat{\\mathbf{b}}, \\quad
           \\hat{v}_k = \\mathbf{x}_k \\hat{\\mathbf{V}}_{\\mathbf{b}} \\mathbf{x}_k^T

    When ``order = numpy.nan`` the function immediately returns zeros for all
    outputs because :func:`designmatrix` returns an empty design matrix.

    This function is called once per estimation node inside
    :func:`~stamps.stamps.bme.BMEprobaEstimations.BMEPosteriorMoments` when
    the ``order`` parameter is an integer (non-NaN).

    Examples
    --------
    >>> import numpy as np
    >>> from stamps.stamps.stest.localmeanBME import localmeanBME
    >>> ch = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
    >>> cs = np.array([[0.3, 0.5]])
    >>> zh = np.array([[10.0], [12.0], [11.0]])
    >>> ms = np.array([[11.5]])
    >>> vs = np.array([[1.0]])
    >>> Khh = np.eye(3)
    >>> Ksh = np.ones((1, 3)) * 0.5
    >>> Kss = np.eye(1)
    >>> ck = np.array([[0.5, 0.5]])
    >>> mk, mh, ms_est, vk = localmeanBME(ck, ch, cs, zh, ms, vs, Khh, Ksh, Kss, 0)
    >>> round(mk, 2)   # estimated constant mean near 11
    11...
    """

    nh = zh.shape[0]
    ns = ms.shape[0]
    mkest = 0.
    mhest = numpy.empty((nh, 1))*numpy.NaN
    msest = numpy.empty((ns, 1))*numpy.NaN
    vkest = 0.

    c = numpy.vstack((ch, cs))
    ms=ms.reshape((ns,1)) 
    vs=vs.reshape((ns,1))       
    Ksh=Ksh.reshape((ns,nh))
    Kss=Kss.reshape((ns,ns))
    #add '_' b/s we won't to change vs and Kss inplace
#    vs_ = numpy.tile( vs,(1,ns ) ) if len(vs) else vs
#    Kss_ = Kss + numpy.diag(vs_)
    K = numpy.vstack( (numpy.hstack((Khh, Ksh.T)), numpy.hstack((Ksh, Kss)) ) )
    z = numpy.vstack((zh, ms))

    best, Vbest, mm, index = regression(c, z, order, K)

    if mm.size > 0:
        mhest = mm[:nh,0:1]
        msest = mm[nh:,0:1]
    else:
        mhest[:] = 0.0 #numpy.array([])
        msest[:] = 0.0 #numpy.array([])

    x, index = designmatrix(ck, order)
    if x.size > 0:
        mkest = ( x.dot(best) )[0][0]
        vkest = ( x.dot(Vbest).dot(x.T) )[0][0]

    return mkest , mhest, msest, vkest
