# -*- coding: utf-8 -*-
import numpy

from .designmatrix import designmatrix


def regression(c, z, order, K):
    """Fit a polynomial trend by Ordinary or Generalised Least Squares (GLS).

    Estimates the trend coefficients **b** and their covariance matrix
    **Vb** for a polynomial drift of order ``order``, given observations
    **z** at locations **c** and an optional spatial covariance matrix **K**.

    When ``K`` is non-empty, the system is solved by **Generalised Least
    Squares** (GLS):

    .. math::

        \\hat{\\mathbf{b}} =
            \\left(\\mathbf{X}^T \\mathbf{K}^{-1} \\mathbf{X}\\right)^{-1}
            \\mathbf{X}^T \\mathbf{K}^{-1} \\mathbf{z}, \\quad
        \\hat{\\mathbf{V}}_{\\mathbf{b}} =
            \\left(\\mathbf{X}^T \\mathbf{K}^{-1} \\mathbf{X}\\right)^{-1}

    When ``K`` is empty (size 0), **Ordinary Least Squares** (OLS) is used
    and the residual variance is estimated from the data:

    .. math::

        \\hat{\\mathbf{b}} =
            \\left(\\mathbf{X}^T \\mathbf{X}\\right)^{-1} \\mathbf{X}^T \\mathbf{z}, \\quad
        \\hat{s}^2 = \\frac{\\mathbf{r}^T \\mathbf{r}}{n - p}, \\quad
        \\hat{\\mathbf{V}}_{\\mathbf{b}} =
            \\left(\\mathbf{X}^T \\mathbf{X}\\right)^{-1} \\hat{s}^2

    Parameters
    ----------
    c : np.ndarray, shape (n, nd)
        Coordinate matrix for all data points.  Passed directly to
        :func:`designmatrix` to build the trend design matrix **X**.
    z : np.ndarray, shape (n, 1)
        Observed (or expected) values at each location.
    order : float or int
        Trend order forwarded to :func:`designmatrix`.

        * ``numpy.nan`` — no trend; the function returns four empty arrays.
        * ``0`` — constant drift.
    K : np.ndarray, shape (n, n) or empty
        Spatial covariance matrix of the observations.  If ``K.size == 0``
        OLS is used; otherwise GLS is applied.

    Returns
    -------
    best : np.ndarray, shape (p, 1)
        Estimated trend coefficients.  Empty array if ``order = numpy.nan``.
    Vbest : np.ndarray, shape (p, p)
        Covariance matrix of the trend coefficients.  Empty if no trend.
    zest : np.ndarray, shape (n, 1)
        Fitted trend values at the input locations ``X @ best``.  Empty if
        no trend.
    index : np.ndarray, shape (1, 2) or empty
        Index array from :func:`designmatrix` (transposed), indicating which
        trend components are active.

    Notes
    -----
    This function is a **helper** called exclusively by :func:`localmeanBME`,
    which is in turn called within
    :func:`~stamps.stamps.bme.BMEprobaEstimations.BMEPosteriorMoments` to
    compute the local prior mean at each estimation node.

    When ``order = numpy.nan``, :func:`designmatrix` returns an empty **X**
    (shape ``(n, 0)``), and this function immediately returns four empty
    ``ndarray`` objects without performing any computation.

    Examples
    --------
    >>> import numpy as np
    >>> from stamps.stamps.stest.regression import regression
    >>> c = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
    >>> z = np.array([[10.0], [12.0], [11.0]])
    >>> K = np.eye(3)
    >>> best, Vbest, zest, idx = regression(c, z, 0, K)
    >>> float(best)   # constant mean ≈ (10+12+11)/3
    11.0
    """

    X, index = designmatrix(c, order)

    index = index.T
    n, p = X.shape

    if X.size > 0:
        Xt = X.T
        if K.size == 0:
            invXtX = numpy.linalg.inv( Xt.dot(X) )
            best = invXtX.dot(Xt).dot(z)
            zest = X.dot(best)
            resi = z - zest
            s2 = ( resi.T.dot(resi) ) / float(n - p)
            Vbest = invXtX.dot( s2 )
        else:
            XtinvK = X.T.dot( numpy.linalg.inv(K) )
            invXtinvKX = numpy.linalg.inv( XtinvK.dot(X) )
            best = invXtinvKX.dot(XtinvK).dot(z)
            zest = X.dot(best)
            Vbest = invXtinvKX
    else: # set default size
        return numpy.array([],ndmin=2), numpy.array([],ndmin = 2),numpy.array([],ndmin=2), numpy.array([],ndmin = 2)


    return best, Vbest, zest, index
