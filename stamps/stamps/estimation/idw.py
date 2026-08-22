# -*- coding: utf-8 -*-
# @Author: Chieh-Han Lee
# @Date:   2015-08-05 19:40:44
# @Last Modified by:   Chieh-Han Lee
# @Last Modified time: 2016-10-31 23:26:00
# -*- coding: utf-8 -*-
'''
Created on 2012/4/11

@author: KSJ
'''

import numpy as np

from scipy.spatial import cKDTree as KDTree
from scipy.spatial.distance import cdist as scipy_cdist

def idw_est( x, y, z, x_est, y_est ,power = 2):
    """Inverse Distance Weighting interpolation with separate x/y/z inputs.

    Estimates values at query locations (x_est, y_est) using all data
    points (x, y, z) via inverse distance weighting:

    .. math::

        \\hat{z}(\\mathbf{x}_0) = \\frac{\\sum_i w_i z_i}{\\sum_i w_i},
        \\quad w_i = \\frac{1}{d(\\mathbf{x}_0, \\mathbf{x}_i)^p + \\epsilon}

    where :math:`\\epsilon = 10^{-10}` prevents division by zero at
    coincident locations.

    Parameters
    ----------
    x : array-like, shape (n,)
        Easting (x-coordinate) of the *n* data locations.
    y : array-like, shape (n,)
        Northing (y-coordinate) of the *n* data locations.
    z : array-like, shape (n,)
        Observed values at the *n* data locations.
    x_est : array-like, shape (m,)
        Easting of the *m* estimation locations.
    y_est : array-like, shape (m,)
        Northing of the *m* estimation locations.
    power : int or float, optional
        IDW power exponent *p* (default 2).  Larger values concentrate
        weight on the nearest data point (Voronoi-like for very large p).

    Returns
    -------
    z_est : np.ndarray, shape (1, m)
        IDW estimates at the *m* query locations.

    Notes
    -----
    All inputs are internally converted to 2-D arrays via ``np.array(x, ndmin=2)``.
    For coordinate-array inputs prefer ``idw_est_coord_value``, which has a
    cleaner interface.
    """
    x, y, z, x_est, y_est =\
    list(map( lambda x : np.array( x, ndmin = 2 ),
         ( x, y, z, x_est, y_est ) ))
    #dist_matrix = np.linalg.norm(
    #   np.hstack((x.T - x_est, y.T - y_est)) , axis=0 ) + 10**-10
    dist_matrix =\
        np.sqrt( ( x.T - x_est ) **2 + ( y.T - y_est ) **2 ) + 10**-10
    weight_matrix = np.reciprocal( dist_matrix ** power )
    up_matrix = weight_matrix * z.T
    up_matrix = up_matrix.sum( axis = 0 ) #sum column
    down_matrix = weight_matrix.sum( axis = 0 ) #sum column
    z_est = up_matrix / down_matrix
    return z_est

def idw_est_coord_value(coord, value, coord_est, power = 2):
    """Inverse Distance Weighting interpolation with coordinate-array inputs.

    Estimates values at query coordinates using all data points via IDW.
    Generalises ``idw_est`` to arbitrary spatial dimensions and provides a
    cleaner array-based interface.

    .. math::

        \\hat{z}(\\mathbf{x}_0) = \\frac{\\sum_i w_i z_i}{\\sum_i w_i},
        \\quad w_i = \\frac{1}{d(\\mathbf{x}_0, \\mathbf{x}_i)^p}

    Coincident locations (zero distance) receive infinite weight, which is
    handled by zeroing the corresponding weight so that the exact observed
    value is returned (no averaging with neighbours).

    Parameters
    ----------
    coord : np.ndarray, shape (n, d)
        Coordinates of the *n* data locations in *d*-dimensional space.
    value : np.ndarray, shape (n, 1)
        Observed values at the *n* data locations.
    coord_est : np.ndarray, shape (m, d)
        Coordinates of the *m* estimation locations.
    power : int or float, optional
        IDW power exponent *p* (default 2).

    Returns
    -------
    value_est : np.ndarray, shape (m, 1)
        IDW estimates at the *m* query locations.

    Notes
    -----
    Uses ``scipy.spatial.distance.cdist`` for vectorised pairwise distance
    computation, making this function efficient for large datasets.

    Duplicate locations in *coord_est* that exactly coincide with a data
    point will receive a weight of zero for that data point (``np.isinf``
    guard), effectively inheriting the weighted average of other neighbours.

    ### Original Reference ###
    coord: a 2d array, r x d, row is data count, column is dimension
    value: a 2d array, r x 1, row is data count, column is value
    coord_est: dito coord
    """
    coord_matrix = scipy_cdist(coord_est, coord) #coord_est by coord
    weight_matrix = np.reciprocal(coord_matrix**power)
    # remove dupliacted localtion (Set 0 wieght)
    weight_matrix[np.isinf(weight_matrix)] = 0.
    up_matrix = weight_matrix * value.T
    up_matrix = up_matrix.sum(axis=1, keepdims=True) #sum column
    down_matrix = weight_matrix.sum(axis=1, keepdims=True) #sum column
    value_est = up_matrix / down_matrix
    return value_est
 
def idw_kdtree( grid_s, grid_v, grid_s_est, nnear=10, eps=0, power=2, weights=None, leafsize=16 ):
    """Inverse Distance Weighting interpolation accelerated by a k-d tree.

    Estimates values at query locations using the *nnear* nearest data
    neighbours found via ``scipy.spatial.cKDTree``, weighted by the inverse
    of their distance raised to *power*.  This is substantially faster than
    the brute-force ``idw_est_coord_value`` for large datasets.

    Parameters
    ----------
    grid_s : np.ndarray, shape (n, d)
        Coordinates of the *n* data points in *d*-dimensional space.
    grid_v : np.ndarray, shape (n,) or (n, 1)
        Observed values at the *n* data points.
    grid_s_est : np.ndarray, shape (m, d)
        Coordinates of the *m* estimation points.
    nnear : int, optional
        Number of nearest neighbours to use in the weighted average
        (default 10).  Larger values give smoother but less locally
        adaptive estimates.
    eps : float, optional
        Approximate nearest-neighbour tolerance passed to ``cKDTree.query``
        (default 0 = exact).  The *k*-th returned value is guaranteed within
        a factor (1 + eps) of the true *k*-th nearest distance.
    power : int or float, optional
        IDW power exponent *p* (default 2).
    weights : np.ndarray, shape (n,) or None, optional
        Optional per-data-point multiplier applied to the distance-based
        weights.  If ``None`` (default), only distance weights are used.
    leafsize : int, optional
        k-d tree leaf size.  Smaller values speed up queries for large
        datasets but increase build time (default 16).

    Returns
    -------
    interp : np.ndarray, shape (m,)
        IDW estimates at the *m* query locations.

    Notes
    -----
    **Exact coincidence** — if an estimation point is at the same location
    as a data point (distance < 1e-10), the data value is returned directly
    without averaging.

    **NaN handling** — data points with NaN values are excluded from the
    weighted average of their neighbours.

    **Large datasets** — for datasets with tens of thousands of points this
    function is orders of magnitude faster than ``idw_est_coord_value``
    because it limits computation to the *nnear* closest neighbours rather
    than computing the full pairwise distance matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from stamps.stamps.stest.idw import idw_kdtree
    >>> rng = np.random.default_rng(42)
    >>> grid_s = rng.random((100, 2)) * 1000.0
    >>> grid_v = rng.random(100)
    >>> grid_s_est = rng.random((10, 2)) * 1000.0
    >>> interp = idw_kdtree(grid_s, grid_v, grid_s_est, nnear=5)
    >>> interp.shape
    (10,)

    ### Original Reference ###
    Inverse distance weighting (IDW) method using KDtree

    Syntax
        interp = idw_kdtree( grid_s, grid_v, grid_s_est, nnear=10, eps=0, power=2, weights=None, leafsize=10 ):

    Input
        grid_s:
            [r1 x d]. Coordinates in grid format.
        grid_v:
            [r1 x 1].
        grid_s_est:
            [r2 x d].
        nnear:
            integer. The list of k-th nearest neighbors to return.
        eps:
            nonnegative float. Return approximate nearest neighbors.
        power:
            integer. Power parameter.
        weights:
            []. Weighted matrix.
        leafsize:
            positive integer.

    Output
        interp:
        [r2 x 1].Interpolation result of IDW.
    """


    tree = KDTree(grid_s, leafsize=leafsize)

    distances, indices = tree.query(grid_s_est, k=nnear, eps=eps)
    # When nnear=1, scipy returns 1-D arrays; reshape to 2-D for uniform handling
    if nnear == 1:
        distances = distances.reshape(-1, 1)
        indices   = indices.reshape(-1, 1)
    interp = np.zeros( (len(grid_s_est),) + np.shape(grid_v[0]) )
    iternum = 0
    for dist, idx in zip(distances, indices):
        z0 = grid_v[idx[0]]
        if nnear == 1:
            weighted_v = grid_v[idx[0]]
        elif dist[0] < 1e-10 and ~np.isnan(z0):
            weighted_v = grid_v[idx[0]]
        else:
            ix = np.where(dist==0)[0]
            if ix.size:
                dist = np.delete(dist, ix)
                idx = np.delete(idx, ix)
            ix = np.where(np.isnan(grid_v[idx]))[0]
            dist = np.delete(dist, ix)
            idx = np.delete(idx, ix)

            weight_matrix = np.reciprocal( dist ** power )
            if weights is not None:
                weight_matrix *= weights[idx]

            weight_matrix /= np.sum(weight_matrix)
            weighted_v = np.dot(weight_matrix, grid_v[idx])

        interp[iternum] = weighted_v
        iternum += 1

    return interp


if __name__ == "__main__":
    x = np.random.random(5)
    y = np.random.random(5)
    z = np.random.random(5)
    
    x_est = np.random.random(7)
    y_est = np.random.random(7)
    
    print (idw_est( x, y, z, x_est, y_est))

    grid_s = np.random.random((100,2))
    grid_v = np.random.random((100,1))
    grid_s_est = np.random.random((7000,2))
    print (idw_kdtree( grid_s, grid_v, grid_s_est ))
