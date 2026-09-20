# -*- coding:utf-8 -*-
import numpy
from scipy.spatial import cKDTree
try:
  from scipy.spatial.distance import cdist as coord2dist
except:
  from .coord2dist import coord2dist


def neighbours_index_kd(ck, ctree, nmax, dmax):
    """Group estimation points by their shared set of nearest neighbours.

    For each estimation point in *ck*, finds up to *nmax* neighbours in
    *ctree* within distance *dmax*, then builds an inverse index: a
    dictionary mapping each unique neighbour-set (as a tuple of indices)
    to the list of estimation-point indices that share that exact set.

    This grouping allows BME to batch-process estimation points that have
    identical neighbourhood structure, substantially reducing redundant
    covariance computation.

    Parameters
    ----------
    ck : np.ndarray, shape (nk, nd)
        Coordinates of the *nk* estimation points.
    ctree : np.ndarray, shape (n, nd) or scipy.spatial.cKDTree
        Data coordinates or a pre-built k-d tree.  If an array is passed,
        a ``cKDTree`` is constructed internally.
    nmax : int
        Maximum number of neighbours to return per estimation point.
    dmax : float
        Maximum search distance.  Points beyond this distance are excluded.

    Returns
    -------
    res : dict
        Keys are tuples of data indices (the neighbour set), values are
        lists of estimation-point indices in *ck* that share that set.

        Example::

            {(3, 7, 12): [0, 5, 9],   # ck[0], ck[5], ck[9] all share data[3,7,12]
             (2, 3, 7):  [1, 6]}       # ck[1], ck[6] share data[2,3,7]

    Notes
    -----
    Uses ``cKDTree.query`` with ``k=range(1, nmax+1)`` and
    ``distance_upper_bound=dmax``.  Indices for points beyond *dmax* are
    masked (infinite distance) and excluded via ``masked_array.compressed()``.

    ### Original Reference ###
    let ck group by ctree
        ctree can be a numpy array with shape (n, nd) or scipy kd-tree
        nmax is first nearest nmax neighbors
        dmax here is only max distance
        return a dict; key is a tuple of ctree indices; value is a list of ck indices.
    """
    if not isinstance(ctree, cKDTree):
        try:
            ctree = cKDTree(ctree)
        except Exception as e:
            # import ipdb
            # ipdb.set_trace()
            raise e
    dd, ii = ctree.query(ck, k=range(1, nmax+1), distance_upper_bound=dmax)
    marr = numpy.ma.masked_array(ii, numpy.isinf(dd))
    marr = numpy.sort(marr)

    res = {}
    for k_idx, marr_i in enumerate(marr):
        try:
            res[tuple(marr_i.compressed())].append(k_idx)
        except KeyError:
            res[tuple(marr_i.compressed())] = [k_idx]
    return res
        
def neighbours_kd(c_one, c, z, nmax, dmax, tree=None):
  """Find nearest neighbours using a pre-built or on-the-fly k-d tree.

  Accelerated version of ``neighbours`` that uses ``scipy.spatial.cKDTree``
  for fast approximate/exact nearest-neighbour search.  Supports both
  pure-spatial (nd <= 2) and space-time (nd = 3) coordinate systems.

  Parameters
  ----------
  c_one : np.ndarray, shape (1, nd)
      Coordinates of the single query point.  ``nd`` can be 1, 2, or 3.
  c : np.ndarray, shape (n, nd)
      Data coordinates.
  z : np.ndarray, shape (n, pz)
      Observed values at the *n* data locations.
  nmax : int
      Maximum number of neighbours to return.
  dmax : np.ndarray, shape (1, rd)
      Search radius.  For pure spatial ``rd = 1`` (dmax[0][0] is max dist).
      For space-time ``rd = 3``: [dmax_space, dmax_time, s_t_ratio].
  tree : scipy.spatial.cKDTree or None, optional
      Pre-built k-d tree on *c*.  If ``None``, built internally.

  Returns
  -------
  c_nebr : np.ndarray, shape (n_nebr, nd)
      Coordinates of the selected neighbours.
  z_nebr : np.ndarray, shape (n_nebr, pz)
      Values at the selected neighbours.
  d_nebr : np.ndarray
      Distances to the selected neighbours.
  n_nebr : int
      Number of neighbours found.
  idx_nebr : np.ndarray, shape (n_nebr,) or (n_nebr, 1)
      Indices of the selected neighbours in *c*.

  Notes
  -----
  For best performance with repeated queries, pre-build the tree with
  ``scipy.spatial.cKDTree(c)`` and pass it via *tree*.

  ### Original Reference ###
  input
  c_one: 1 by nd (nd = 1/2/3)
  c: n by nd
  z: n by ??
  nmax: int
  dmax: 1 by rd float (rd = 1 or 3)
  return
  c_nebr, z_nebr, d_nebr, n_nebr, idx_nebr
  """  
  empty_result = [ numpy.array([]).reshape( ( 0, c_one.shape[1] ) ),
                   numpy.array([]).reshape( ( 0, 1 ) ),
                   numpy.array([]).reshape( ( 0, 1 ) ),
                   0,
                   numpy.array([]).reshape( ( 0, 1 ) ) ]

  isST = 1 if dmax.size == 3 else 0

  if c.size == 0:
    print ('no data')
    return empty_result

  if nmax == 0:
    print ('nmax is 0')
    return empty_result  
    
  if isST==0: # pure spatial or temporal cases
    if tree is None:
      try:
        tree=cKDTree(c,leafsize=15)
      except:
        import sys
        sys.setrecursionlimit(10000)
        tree=cKDTree(c,leafsize=30)
    d_nebr,idx_nebr=tree.query(
        c_one, k=range(1, nmax+1), distance_upper_bound=dmax[0][0])
    idx_nebr=idx_nebr[0]    
    c_nebr=c[idx_nebr,:]
    z_nebr=z[idx_nebr,:]    
    n_nebr=idx_nebr.size
    return c_nebr, z_nebr, d_nebr, n_nebr, idx_nebr
  
  elif isST == 1:#space time case  
    if tree is None:
      #get distance of time
      d_t = numpy.abs( c[:,2:3] - c_one[:,2:3] )
      index_t = numpy.where( d_t <= dmax[0][1] )
      if len(index_t[0]) == 0:
        print ("noneighbor")
        return empty_result
  
      #get distance of space which already match time
      d_xy = coord2dist( c[index_t[0],0:2], c_one[:,0:2] )
      index_s = numpy.where( d_xy <= dmax[0][0] )
      if len(index_s[0]) == 0:
        print ("noneighbor")
        return empty_result  
    
      #calculate all distance which matched perfectly
      c_one_n=c_one
      c_n=c[index_t[0][index_s[0]],:]
      c_one_n[:,2]=c_one[:,2]*dmax[0][2]
      c_n[:,2]=c_n[:,2]*dmax[0][2]
      tree=cKDTree(c_n,leafsize=15)

    d_nebr,idx_nebr=tree.query(c_one_n, k=range(1, nmax+1))  
    idx_nebr=index_t[0][index_s[0]][idx_nebr[0]]
    d_nebr=d_nebr.T
    c_nebr=c[idx_nebr,:]
    z_nebr=z[idx_nebr]
    n_nebr=idx_nebr.size   
    return c_nebr, z_nebr, d_nebr, n_nebr,\
        numpy.sort(idx_nebr.reshape((-1,1)), axis=0)
    
    
def neighbours( c_one, c, z, nmax, dmax ):
    """Find the nearest data neighbours of a query point within a search window.

    For a single query point *c_one*, selects up to *nmax* data points from
    *c* that lie within the spatial (and temporal) search window defined by
    *dmax*.  Neighbours are sorted by ascending distance and the closest
    *nmax* are returned if more are found within the window.

    Supports pure-spatial (nd = 1 or 2) and space-time (nd = 3) coordinate
    systems.

    Parameters
    ----------
    c_one : np.ndarray, shape (1, nd)
        Coordinates of the single query point.  ``nd`` can be:

        * 1 — 1-D spatial or temporal
        * 2 — 2-D spatial (easting, northing)
        * 3 — space-time (easting, northing, time)
    c : np.ndarray, shape (n, nd)
        Coordinates of the *n* candidate data points.
    z : np.ndarray, shape (n, pz)
        Observed values at the *n* data points.  Any number of value
        columns ``pz`` is accepted.
    nmax : int
        Maximum number of neighbours to select.  If fewer data points
        exist within *dmax*, all are returned.
    dmax : scalar, 1-D array, or np.ndarray of shape (1, rd)
        Search radius specification.  Internally normalised to shape (1, rd)
        via ``np.atleast_2d``.

        * ``rd = 1`` — scalar/1-element: spatial or temporal max distance.
        * ``rd = 3`` — ``[dmax_space, dmax_time, s_t_ratio]`` for space-time.
          The composite distance used for trimming is
          ``d_xy + s_t_ratio * d_t``.

    Returns
    -------
    c_nebr : np.ndarray, shape (n_nebr, nd)
        Coordinates of the selected neighbours.
    z_nebr : np.ndarray, shape (n_nebr, pz)
        Values at the selected neighbours.
    d_nebr : np.ndarray, shape (n_nebr, 1)
        Distances (spatial or composite) to each selected neighbour.
    n_nebr : int
        Number of neighbours found (0 if none within *dmax*).
    idx_nebr : np.ndarray, shape (n_nebr, 1)
        0-based row indices of the selected neighbours in *c*.

    Notes
    -----
    When no neighbours are found within *dmax*, all five outputs are empty
    arrays of compatible shapes and ``n_nebr = 0``.

    This function uses brute-force distance computation via ``coord2dist``.
    For repeated queries against the same large dataset, prefer
    ``neighbours_kd``, which builds a k-d tree for O(log n) query time.

    Examples
    --------
    >>> import numpy as np
    >>> from stamps.general.neighbours import neighbours
    >>> c_one = np.array([[5., 5.]])
    >>> c = np.array([[0., 0.], [3., 4.], [10., 10.], [6., 6.]])
    >>> z = np.array([[1.0], [2.0], [3.0], [4.0]])
    >>> dmax = np.array([[8.0]])
    >>> c_nebr, z_nebr, d_nebr, n_nebr, idx_nebr = neighbours(c_one, c, z, nmax=3, dmax=dmax)
    >>> n_nebr
    3

    ### Original Reference ###
    input
    c_one: 1 by nd (nd = 1/2/3 for space, space, space-time)
    c: n by nd
    z: n by ?? (observations)
    nmax: int
    dmax: 1 by rd float (rd = 1 or 3)
    return
    c_nebr, z_nebr, d_nebr, n_nebr, idx_nebr
    """

    empty_result = [ numpy.array([]).reshape( ( 0, c_one.shape[1] ) ),
                     numpy.array([]).reshape( ( 0, 1 ) ),
                     numpy.array([]).reshape( ( 0, 1 ) ),
                     0,
                     numpy.array([]).reshape( ( 0, 1 ) ) ]

    # Accept scalar, 1-D or 2-D dmax; normalise to 2-D so dmax[0][0] always works
    dmax = numpy.atleast_2d(numpy.asarray(dmax, dtype=float))

    isST = 1 if dmax.size == 3 else 0

    if c.size == 0:
    #    print 'no data'
        return empty_result

    if nmax == 0:
    #    print 'nmax is 0'
        return empty_result

    if isST == 0: #
        #get distance of space (only)
        d_xy = coord2dist( c, c_one )
        index_s = numpy.where( d_xy <= dmax[0][0] )
        if len(index_s[0]) == 0:
            print ("noneighbor")
            return empty_result
        elif len( index_s[0] ) <= nmax:
            # Sort by distance (ascending) for consistent ordering
            sort_order = d_xy[index_s[0], 0].argsort()
            sorted_idx = index_s[0][sort_order]
            c_nebr = c[sorted_idx,:]
            z_nebr = z[sorted_idx,:]
            d_nebr = d_xy[sorted_idx,:]
            n_nebr = len( sorted_idx )
            idx_nebr = sorted_idx.reshape( ( -1, 1 ) )
            return c_nebr, z_nebr, d_nebr, n_nebr, idx_nebr
        elif len( index_s[0] ) > nmax:
            d_nebr = d_xy[index_s[0],:]
            index_s = ( numpy.sort( d_nebr[:,0].argsort()[:nmax] ), 0 ) #dummy 0 for consistence
            c_nebr = c[index_s[0],:]
            z_nebr = z[index_s[0],:]
            d_nebr = d_xy[index_s[0],:]
            n_nebr = len( index_s[0] )
            idx_nebr = index_s[0].reshape( ( -1, 1 ) )
            return c_nebr, z_nebr, d_nebr, n_nebr, idx_nebr

    elif isST == 1:#space time case

        #get distance of time
        d_t = numpy.float64(numpy.abs( c[:,2:3] - c_one[:,2:3] ))
        index_t = numpy.where( d_t <= dmax[0][1] )
        if len(index_t[0]) == 0:
            # print "noneighbor"
            return empty_result

        #get distance of space which already match time
        d_xy = coord2dist( c[index_t[0],0:2], c_one[:,0:2] )
        index_s = numpy.where( d_xy <= dmax[0][0] )
        if len(index_s[0]) == 0:
            # print "noneighbor"
            return empty_result
        
        #calculate all distance which matched perfectly
        d_r = d_xy[index_s[0],0:1] + dmax[0][2] * d_t[ index_t[0] [ index_s[0] ],0:1]
        index_r = numpy.where( d_r <= dmax[0][0] + dmax[0][2] * dmax[0][1] )
        
        if len( index_r[0] ) == 0:
            n_nebr = 0
            return empty_result
        elif len( index_r[0] ) <= nmax:
            c_nebr = c[index_t[0],:][index_s[0],:][index_r[0],:]
            z_nebr = z[index_t[0],:][index_s[0],:][index_r[0],:]
            d_nebr = d_r[index_r[0],:]
            n_nebr = len( index_r[0] )
            idx_nebr = index_t[0].reshape( ( -1, 1 ) )[index_s[0][index_r[0]],:]
            return c_nebr, z_nebr, d_nebr, n_nebr, idx_nebr
        elif len( index_r[0] ) > nmax:
            d_nebr = d_r[index_r[0],:]
            index_r = ( numpy.sort( d_nebr[:,0].argsort()[:nmax] ), 0 ) #dummy 0 for consistence
            c_nebr = c[index_t[0],:][index_s[0],:][index_r[0],:]
            z_nebr = z[index_t[0],:][index_s[0],:][index_r[0],:]
            d_nebr = d_r[index_r[0],:]
            n_nebr = len( index_r[0] )
            idx_nebr = index_t[0].reshape( ( -1, 1 ) )[index_s[0][index_r[0]],:]
            return c_nebr, z_nebr, d_nebr, n_nebr, idx_nebr


# ---------------------------------------------------------------------------
# Unified categorical-aware neighbour search
# ---------------------------------------------------------------------------

def neighbours_cat(c0, c, Z, nmax, dmax, options=None):
    """Unified categorical-aware neighbour search with separable axes and
    information-theoretic ranking.

    Extends the basic nearest-neighbour search with two additional capabilities:

    1. **Separable-axis filtering** — when ``options['separable_axes']`` is
       provided, distances are computed independently for each coordinate-axis
       group and a point is kept only if it is within ``dmax`` on **every**
       axis.  Required for anisotropic or 3-D separable Pmodels (e.g.
       horizontal + vertical).

    2. **Pmodel-based ranking** — instead of ranking by Euclidean distance,
       candidates can be ranked by *total variation* (TV) or *mutual
       information* (MI) between their bivariate joint table and the
       independence product ``p(c)·p(c')``.  Higher TV/MI means the data
       point is more informative for the estimation.

    The function also returns ``idx_all`` and ``D_all`` — the indices and
    per-axis distances of **all** valid candidates within the search window
    (not just the selected ``nmax``).  These are required by
    :func:`~stamps.general.local_marginal.local_marginal` to estimate the
    kernel-weighted local class marginal.

    Parameters
    ----------
    c0 : (1, d) ndarray
        Coordinates of the estimation point.
    c : (n, d) ndarray
        Coordinates of all candidate data points.
    Z : (n, nc) ndarray
        Soft-data probability vectors at each candidate point.
    nmax : int or None
        Maximum number of neighbours to select.  ``None`` or ``np.inf``
        selects all valid candidates.
    dmax : float or None
        Fallback maximum search distance (scalar) used when
        ``'separable_axes'`` is **not** in options.
    options : dict, optional
        Search options (all optional):

        ``'separable_axes'`` *(list of dict)* — per-axis specification.
        Each element must have:

        * ``'coord_cols'`` or ``'dims'`` *(list of int)*
        * ``'dmax'`` *(float)* — capped at ``dmodel[-1]/2`` when
          ``'dmodel'`` is also given.
        * ``'dmodel'`` *(array, optional)*

        ``'neighbour_ranking'`` *(str, default ``'distance'``)* —
        ``'distance'``, ``'tv'``, or ``'mutual_info'``.

        ``'ranking_pmodel'`` *(dict)* — required for TV/MI ranking. Keys:
        ``'dmodels'``, ``'Pmodels'``, ``'p_marginal'``, ``'mode'``.

        ``'neighborhood_structure'`` / ``'nmax_per_sector'`` — for the
        non-separable path only (sectored quadrant/6-sector search).

    Returns
    -------
    csub : (m, d) ndarray
    Zsub : (m, nc) ndarray
    dsub : (m,) ndarray
        Full Euclidean distances to selected neighbours.
    nsub : int
    idx_selected : (m,) ndarray of int
    d_components : (m, K) ndarray
        Per-axis distances for selected neighbours (K=1 if non-separable).
    idx_all : (M,) ndarray of int
        Indices of ALL valid candidates (M >= m).
    D_all : (M, K) ndarray
        Per-axis distances for all valid candidates.
    """
    import warnings as _warnings

    if options is None:
        options = {}
    if nmax is not None and numpy.isinf(nmax):
        nmax = None

    c0 = numpy.asarray(c0, dtype=float).reshape(1, -1)
    c  = numpy.asarray(c,  dtype=float)
    Z  = numpy.asarray(Z,  dtype=float)
    n  = c.shape[0]

    separable_axes = options.get('separable_axes', None)

    # ── Separable path ────────────────────────────────────────────────────────
    if separable_axes is not None:
        K = len(separable_axes)
        D = numpy.zeros((n, K), dtype=float)
        dmaxs = numpy.zeros(K, dtype=float)
        for j, ax in enumerate(separable_axes):
            cols = numpy.asarray(
                ax.get('coord_cols', ax.get('dims', [])), dtype=int
            ).ravel()
            if cols.size == 0:
                raise ValueError(
                    f"separable_axes[{j}] must have 'coord_cols' or 'dims'."
                )
            ax_dmax = float(ax['dmax'])
            if 'dmodel' in ax:
                safe = float(numpy.asarray(ax['dmodel'])[-1]) / 2.0
                ax_dmax = min(ax_dmax, safe)
            dmaxs[j] = ax_dmax
            D[:, j] = coord2dist(c0[:, cols], c[:, cols]).ravel()

        valid = numpy.ones(n, dtype=bool)
        for j in range(K):
            valid &= D[:, j] < dmaxs[j]
        idx_all = numpy.where(valid)[0]
        d_euc   = coord2dist(c0, c).ravel()

        if idx_all.size == 0:
            emp = numpy.empty((0, K), dtype=float)
            return (
                numpy.empty((0, c.shape[1])),
                numpy.empty((0, Z.shape[1])),
                numpy.empty(0),
                0,
                numpy.empty(0, dtype=int),
                emp, idx_all, emp,
            )

        D_all = D[idx_all, :]

        # Ranking
        ranking = options.get('neighbour_ranking', 'distance')
        rp = options.get('ranking_pmodel', None)
        use_prob_rank = (ranking in ('tv', 'mutual_info')) and rp is not None

        if use_prob_rank:
            try:
                from ..stats.dependence.probatablefit import (
                    probamodel2bitable_separable,
                )
            except ImportError:
                use_prob_rank = False
                _warnings.warn(
                    "probamodel2bitable_separable not available; "
                    "falling back to distance ranking.", stacklevel=2,
                )

        if use_prob_rank:
            p_marg  = numpy.asarray(rp['p_marginal'], dtype=float).ravel()
            p_marg  = numpy.clip(p_marg, 1e-15, None)
            p_marg /= p_marg.sum()
            P_indep = numpy.outer(p_marg, p_marg)
            scores  = numpy.empty(idx_all.size, dtype=float)
            for m_i in range(idx_all.size):
                P3d = probamodel2bitable_separable(
                    D_all[m_i, :], rp['dmodels'], rp['Pmodels'],
                    p_marg, mode=rp.get('mode', 'product'),
                )
                if ranking == 'tv':
                    scores[m_i] = 0.5 * numpy.abs(P3d - P_indep).sum()
                else:
                    pio = numpy.maximum(P_indep, 1e-300)
                    with numpy.errstate(divide='ignore', invalid='ignore'):
                        ratio = P3d / pio
                        log_r = numpy.log(
                            numpy.where(P3d > 1e-300,
                                        numpy.maximum(ratio, 1e-300), 1.0)
                        )
                    scores[m_i] = float(numpy.sum(P3d * log_r))
            order = numpy.argsort(-scores)
        else:
            scores = numpy.sqrt(
                numpy.sum(
                    (D_all / numpy.maximum(dmaxs, 1e-30)) ** 2, axis=1
                )
            )
            order = numpy.argsort(scores)

        if nmax is not None:
            order = order[:nmax]
        sltidx = idx_all[order]

        return (
            c[sltidx, :], Z[sltidx, :], d_euc[sltidx], len(sltidx),
            sltidx, D[sltidx, :], idx_all, D_all,
        )

    # ── Non-separable path ────────────────────────────────────────────────────
    if dmax is None:
        raise ValueError(
            "dmax must be provided when 'separable_axes' is not in options."
        )
    dmax_f   = float(numpy.asarray(dmax).ravel()[0])
    d_euc    = coord2dist(c0, c).ravel()
    idx_all  = numpy.where(d_euc < dmax_f)[0]
    D_all_1d = d_euc[idx_all].reshape(-1, 1)

    if idx_all.size == 0:
        return (
            numpy.empty((0, c.shape[1])),
            numpy.empty((0, Z.shape[1])),
            numpy.empty(0),
            0,
            numpy.empty(0, dtype=int),
            numpy.empty((0, 1), dtype=float),
            idx_all,
            numpy.empty((0, 1), dtype=float),
        )

    ranking = options.get('neighbour_ranking', 'distance')
    rp = options.get('ranking_pmodel', None)
    use_prob_rank = (ranking in ('tv', 'mutual_info')) and rp is not None

    if use_prob_rank:
        try:
            from ..stats.dependence.probatablefit import probamodel2bitable
        except ImportError:
            use_prob_rank = False
            _warnings.warn(
                "probamodel2bitable not available; "
                "falling back to distance ranking.", stacklevel=2,
            )

    if use_prob_rank:
        p_marg  = numpy.asarray(rp['p_marginal'], dtype=float).ravel()
        p_marg  = numpy.clip(p_marg, 1e-15, None)
        p_marg /= p_marg.sum()
        P_indep = numpy.outer(p_marg, p_marg)
        dmodel  = rp['dmodels'][0]
        Pmodel  = rp['Pmodels'][0]
        scores  = numpy.empty(idx_all.size, dtype=float)
        for m_i in range(idx_all.size):
            P3d = probamodel2bitable(d_euc[idx_all[m_i]], dmodel, Pmodel)
            if ranking == 'tv':
                scores[m_i] = 0.5 * numpy.abs(P3d - P_indep).sum()
            else:
                pio = numpy.maximum(P_indep, 1e-300)
                with numpy.errstate(divide='ignore', invalid='ignore'):
                    ratio = P3d / pio
                    log_r = numpy.log(
                        numpy.where(P3d > 1e-300,
                                    numpy.maximum(ratio, 1e-300), 1.0)
                    )
                scores[m_i] = float(numpy.sum(P3d * log_r))
        order = numpy.argsort(-scores)
    else:
        order = numpy.argsort(d_euc[idx_all])

    # Sectored search (non-separable only)
    neighborhood_structure = options.get('neighborhood_structure', 'non-sectored')
    nmax_per_sector = int(options.get('nmax_per_sector', 1))

    if neighborhood_structure == 'sectored' and not use_prob_rank:
        import pandas as _pd
        ndim   = c0.shape[1]
        deltas = c[idx_all, :] - c0
        df     = _pd.DataFrame(deltas, index=idx_all)
        df['_d'] = d_euc[idx_all]
        sectors = []
        if ndim == 2:
            df.columns = ['x', 'y', '_d']
            sectors = [
                (df['x'] >= 0) & (df['y'] >= 0),
                (df['x'] <  0) & (df['y'] >= 0),
                (df['x'] <  0) & (df['y'] <  0),
                (df['x'] >= 0) & (df['y'] <  0),
            ]
        elif ndim == 3:
            df.columns = ['x', 'y', 'z', '_d']
            z_tol  = 1e-6
            horiz  = numpy.abs(df['z']) <= z_tol
            sectors = [
                horiz & (df['x'] >= 0) & (df['y'] >= 0),
                horiz & (df['x'] <  0) & (df['y'] >= 0),
                horiz & (df['x'] <  0) & (df['y'] <  0),
                horiz & (df['x'] >= 0) & (df['y'] <  0),
                df['z'] >  z_tol,
                df['z'] < -z_tol,
            ]
        if sectors:
            all_sect = []
            for mask in sectors:
                pts = df[mask]
                if not pts.empty:
                    all_sect.extend(
                        pts.nsmallest(nmax_per_sector, '_d').index.tolist()
                    )
            sltidx = numpy.array(list(dict.fromkeys(all_sect)), dtype=int)
        else:
            sltidx = idx_all[order[:nmax] if nmax is not None else order]
    else:
        sltidx = idx_all[order[:nmax] if nmax is not None else order]

    return (
        c[sltidx, :], Z[sltidx, :], d_euc[sltidx], len(sltidx),
        sltidx, d_euc[sltidx].reshape(-1, 1), idx_all, D_all_1d,
    )
