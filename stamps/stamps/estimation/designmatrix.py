# -*- coding: utf-8 -*-
import numpy


#only accept order = numpy.nan or 0
def designmatrix( c, order ):
    """Build the spatial trend design matrix for GLS/BME drift estimation.

    Constructs the design matrix **X** for fitting a polynomial drift to
    spatial (or space-time) data.  Currently supports two trend orders:

    * ``numpy.nan`` — no trend (zero-mean model).
    * ``0`` — constant trend (one intercept per location).

    Parameters
    ----------
    c : np.ndarray, shape (n, nd)
        Coordinate matrix.  Rows are data/estimation points; columns are
        spatial (or space-time) dimensions.  ``n`` is the number of locations
        and ``nd`` is the number of dimensions.
    order : float or int
        Trend order.

        * ``numpy.nan`` — returns an empty design matrix and an empty index
          array (equivalent to a zero-mean prior).
        * ``0`` — returns a column of ones ``(n, 1)`` (constant drift).

    Returns
    -------
    X : np.ndarray, shape (n, p)
        Design matrix where ``p`` is the number of trend coefficients:

        * ``p = 0`` when ``order = numpy.nan`` (shape ``(n, 0)``).
        * ``p = 1`` when ``order = 0``  (column of ones).
    index : np.ndarray
        Index array describing which trend components are active.
        Shape ``(2, 1)`` of zeros for ``order = 0``; empty ``(1, 0)``
        array for ``order = numpy.nan``.

    Notes
    -----
    Only ``order = numpy.nan`` and ``order = 0`` are currently implemented.
    Support for higher-order polynomial drifts (``order = 1`` for linear,
    ``order = 2`` for quadratic) is stubbed out in comments but not active.

    This function is used internally by :func:`regression` and
    :func:`localmeanBME`, and is called once per estimation node inside
    :func:`~stamps.stamps.bme.BMEprobaEstimations.BMEPosteriorMoments`.

    Examples
    --------
    >>> import numpy as np
    >>> from stamps.stamps.stest.designmatrix import designmatrix
    >>> c = np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 1.0]])
    >>> X, idx = designmatrix(c, 0)
    >>> X          # column of ones
    array([[1.],
           [1.],
           [1.]])
    >>> X_nan, _ = designmatrix(c, np.nan)
    >>> X_nan.shape   # empty — no trend
    (3, 0)
    """

    if numpy.isnan(order):
        return numpy.array([],ndmin=2).reshape(c.shape[0],0), numpy.array([],ndmin=2)
    else:
        n, nd = c.shape
        X = numpy.ones((n, 1))
        index = numpy.zeros((2, 1))
        if nd == 1:
            return X, index[0,:]
        else:
            return X, index

    # order = numpy.array([ [order, order] ])
    # if ~numpy.isnan( order[0][0] ) or ~numpy.isnan( order[0][1] ):
    # 	X = numpy.ones((n, 1))
    # 	index = numpy.zeros((2, 1))
    # if ~numpy.isnan( order[0][0] ):
