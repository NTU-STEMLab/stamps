# -*- coding:utf-8 -*-
import numpy as np
import multiprocessing as mp
CPU_COUNT = mp.cpu_count()

from six.moves import range
from scipy import sparse
from scipy.spatial import cKDTree as KDTree
from scipy.spatial.distance import cdist


def coord2dist_mp( c1, c2, workers=max(1, CPU_COUNT - 1) ):
    """Parallelised pairwise Euclidean distance computation.

    Computes the full (r1 × r2) Euclidean distance matrix between two sets
    of coordinates.  For small matrices (r1 × r2 ≤ 10⁷) it falls through to
    the single-process ``coord2dist``; for larger ones it splits work across
    *workers* processes using ``multiprocessing.Pool``.

    Parameters
    ----------
    c1 : np.ndarray, shape (r1, d)
        First set of coordinates.
    c2 : np.ndarray, shape (r2, d)
        Second set of coordinates.
    workers : int, optional
        Number of worker processes.  Defaults to ``cpu_count − 1`` (at
        least 1).

    Returns
    -------
    d : np.ndarray, shape (r1, r2)
        Pairwise Euclidean distance matrix.

    Notes
    -----
    The parallelisation strategy partitions both *c1* and *c2* into
    *workers* chunks, computes all *workers²* block distances in parallel,
    then reassembles the result via ``np.vstack`` / ``np.hstack``.

    For most practical use cases the overhead of process spawning outweighs
    the benefit unless *workers* × block size is very large.  Prefer
    ``coord2dist`` (or ``scipy.spatial.distance.cdist`` directly) for
    moderate-sized datasets.
    """

    if c1.shape[0] * c2.shape[0] <= 10 ** 7 or workers == 1:
        return coord2dist(c1, c2)
    else:
        print('used worker: {w}'.format(w = workers))
        d_c1 = int(np.ceil(c1.shape[0] / float(workers)))
        d_c2 = int(np.ceil(c2.shape[0] / float(workers)))
        c12 = [(c1[d_c1 * i:d_c1 * (i + 1)],
                c2[d_c2 * i:d_c2 * (i + 1)]) for i in range(workers)]
        res_ij = [(c12[i][0], c12[j][1]) for i in range(workers)
                  for j in range(workers)]
        pool = mp.Pool(processes=workers)
        d = pool.map(_warp_coord2dist, res_ij)

        d = [np.vstack(d[i::workers]) for i in range(workers)]
        d = np.hstack(d)
        return d
    return d

def _warp_coord2dist(args):
    return coord2dist(*args)

def coord2dist( c1, c2, norm=2 ):
    """Compute the pairwise Minkowski distance matrix between two sets of coordinates.

    Primary distance utility used throughout the ``stamps`` package.  Uses
    ``scipy.spatial.distance.cdist`` for efficient vectorised computation and
    falls back to a manual Kronecker-product implementation if ``cdist``
    raises an exception (e.g., incompatible shapes).

    Parameters
    ----------
    c1 : np.ndarray, shape (r1, d)
        First set of coordinates.  Each row is one point in *d*-dimensional
        space.
    c2 : np.ndarray, shape (r2, d)
        Second set of coordinates.
    norm : int or float, optional
        Minkowski *p*-norm exponent (default 2 = Euclidean).  ``norm=1``
        gives Manhattan distance; ``norm=np.inf`` gives Chebyshev distance.

    Returns
    -------
    d : np.ndarray, shape (r1, r2)
        Pairwise distance matrix.  ``d[i, j]`` is the distance between
        ``c1[i]`` and ``c2[j]``.

    Notes
    -----
    The fallback path uses Kronecker products and explicit broadcasting to
    compute distances, which is slower but works for unusual array shapes
    that confuse ``cdist``.

    In the ``stamps`` convention coordinates always follow the layout
    ``(easting, northing)`` for 2-D spatial data and
    ``(easting, northing, time)`` for 3-D space-time data.

    Examples
    --------
    >>> import numpy as np
    >>> from stamps.general.coord2dist import coord2dist
    >>> c1 = np.array([[0., 0.], [3., 4.]])
    >>> c2 = np.array([[0., 0.]])
    >>> coord2dist(c1, c2)
    array([[0.],
           [5.]])

    ### Original Reference ###
    Calculate the distance between coordinates c1 and c2

    Syntax: result = coord2dist(c1, c2)

    Input
        c1        [r1 x d]     np.array of coordinates
        c2        [r2 x d]     np.array of coordinates
    Output
        d         [r1 x r2]    np.array of distances
    """

    try:
        d = cdist(c1, c2, 'minkowski', p=norm)
    except Exception:
        ones_c1 = np.ones((c1.shape[0], 1))
        ones_c2 = np.ones((c2.shape[0], 1))
        a = np.kron(c1, ones_c2)
        b = np.kron(ones_c1, c2)
        d = ((a - b) ** 2)
        d = d.sum(axis=1)
        d = np.sqrt(d)
        d = d.reshape((c1.shape[0], c2.shape[0]))
    return d

def coord2dist_sparse( c1, c2, distance_upper_bound=1000 ):
    """Compute a sparse pairwise distance matrix using a k-d tree.

    Returns only the pairwise distances that are within *distance_upper_bound*,
    stored in a ``scipy.sparse.coo_matrix``.  This is memory-efficient for
    large datasets where most pairs are far apart (e.g., spatial neighbourhood
    computations with a small search radius).

    Parameters
    ----------
    c1 : np.ndarray, shape (r1, d)
        First set of coordinates.
    c2 : np.ndarray, shape (r2, d)
        Second set of coordinates.  A k-d tree is built on *c2*.
    distance_upper_bound : float, optional
        Maximum distance to include in the sparse matrix (default 1000).
        Pairs with distance > *distance_upper_bound* are omitted (stored
        as structural zeros).

    Returns
    -------
    d_sparse : scipy.sparse.coo_matrix, shape (r1, r2)
        Sparse distance matrix in COOrdinate format.  Only non-zero (within
        *distance_upper_bound*) distances are stored.

    Notes
    -----
    Internally queries up to k = 1000 nearest neighbours per point; the
    result is then trimmed to at most ``count_col`` finite entries per row,
    where ``count_col`` is the number of neighbours within the bound for
    the first query point.  For very dense datasets where many points fall
    within the bound, increase *k* in the ``tree.query`` call.

    ### Original Reference ###
    Sparse method for calculating the distance between coordinates c1 and c2

    Syntax: d_sparse = coord2dist_sparse(c1, c2, distance_upper_bound)

    Input
        c1:        [r1 x d]. numpy.array of coordinates
        c2:        [r2 x d]. numpy.array of coordinates
        distance_upper_bound:  nonnegative float.

    Output
        d_sparse:  [r1 x r2]. Sparse COO matrix of distances.
    """
    
    tree = KDTree(c2)
    d, i = tree.query(c1, k=1000, distance_upper_bound=distance_upper_bound)
    count_col = np.isfinite(d[0,:]).sum()
    d = d[:, :count_col]
    i = i[:, :count_col]

    row = np.repeat(range(len(c1)), d.shape[1])
    col = i.ravel()
    d_sparse = sparse.coo_matrix( ( d.ravel(), (row, col) )
                                  , shape=(len(c1), len(c2)) )

    return d_sparse
