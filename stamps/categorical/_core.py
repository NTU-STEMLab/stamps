# -*- coding: utf-8 -*-
"""
stamps.categorical._core
=========================
Shared front-layer helpers used by both BME and MCP categorical estimators.

These functions handle soft-data coercion, Pmodel detection (separable vs
stationary), effective-range estimation, coordinate-column assignment, and
bivariate table interpolation.
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d

from ..general.coord2dist import coord2dist
from ..stats.dependence.ptable.fit import probamodel2bitable_separable


def _coerce_ps_soft(ps, ncat, options=None):
    """Return soft-data matrix *ps* with shape ``(n, ncat)``.

    Accepts either:

    * ``(n, ncat)`` — already class probabilities / soft indicators; or
    * ``(n,)`` — hard class labels.  If ``options['category_codes']`` is set
      (length ``ncat``, same order as ``Pmodel`` rows), labels are matched
      with ``==``.  Otherwise labels must be integer indices ``0 … ncat-1``.
    """
    ps = np.asarray(ps)
    if options is None:
        options = {}
    if ps.ndim == 2:
        if ps.shape[1] != ncat:
            raise ValueError(
                f'ps must have shape (n, {ncat}) for this Pmodel; got {ps.shape}'
            )
        return ps.astype(float, copy=False)
    if ps.ndim == 1:
        codes = options.get('category_codes', None)
        if codes is not None:
            cats = np.asarray(codes).ravel()
            if cats.size != ncat:
                raise ValueError(
                    f"options['category_codes'] must have length {ncat}; "
                    f'got {cats.size}'
                )
            return (ps[:, None] == cats[None, :]).astype(float)
        # Auto-infer category_codes when labels are not 0-indexed
        lab_raw = ps
        try:
            lab_int = ps.astype(int, copy=False)
        except (ValueError, TypeError):
            lab_int = None
        if lab_int is not None and ((lab_int < 0).any() or (lab_int >= ncat).any()):
            unique_codes = np.sort(np.unique(lab_int))
            if unique_codes.size == ncat:
                # Sorted unique values map to columns 0…nc-1
                return (lab_int[:, None] == unique_codes[None, :]).astype(float)
            raise ValueError(
                'ps has shape (n,) but values are not in [0, ncat). '
                f'Found {unique_codes.size} unique codes; expected {ncat}. '
                "Set options['category_codes'] to match Pmodel row order."
            )
        lab = lab_int.astype(int, copy=False)
        out = np.zeros((ps.shape[0], ncat), dtype=float)
        out[np.arange(ps.shape[0], dtype=int), lab] = 1.0
        return out
    raise ValueError(
        f'ps must be 1D (n,) hard labels or 2D (n, {ncat}) soft data; '
        f'got shape {ps.shape}'
    )


def probamodel2bitable(d,dmodel,Pmodel):
    """
    Builds a bivariate probability table for a given distance via interpolation.
    
    Linearly interpolates from a set of known bivariate probability tables
    (`Pmodel`) defined at specific distances (`dmodel`) to estimate the
    table at a new distance `d`.

    Parameters
    ----------
    d : float
        The target distance for which to compute the probability table.
    dmodel : np.ndarray
        1D array of distances at which `Pmodel` is defined, shape ``(nd,)``.
    Pmodel : np.ndarray
        3D array of bivariate probability tables, shape ``(nc, nc, nd)``.
        `Pmodel[:, :, i]` corresponds to the table at `dmodel[i]`.

    Returns
    -------
    Pd : np.ndarray
        The interpolated bivariate probability table, shape ``(nc, nc)``.
    """
    
    nc=Pmodel.shape[0] # number of catogories
    Pmodel_vec = Pmodel.reshape(nc**2,-1)
    nd = Pmodel_vec.shape[1] # Number of distance models

    ## find the distance in dmodel
    loc = np.interp(d,dmodel,range(len(dmodel)))
    
    ## do interpolation in Pmodel
    idx_low = int(np.floor(loc))
    idx_high = int(np.ceil(loc))
    
    # Handle edges
    if idx_low == idx_high:
        if idx_high >= nd - 1: # At or beyond the end
            idx_low = max(0, nd - 2)
            idx_high = nd - 1
        elif idx_high == 0: # Exactly at the start
            idx_high = min(1, nd - 1)
            
    if idx_high >= nd:
        idx_high = nd - 1
        
    if idx_low < 0:
        idx_low = 0

    if idx_low == idx_high: # This can happen if dmodel has length 1
         if nd > 1:
            idx_high = idx_low + 1
         else:
             # This case means dmodel has only 1 entry, so just use it
             return Pmodel_vec[:, idx_low].reshape(nc, nc)

    frac = loc - idx_low # Use frac based on idx_low
    if idx_high >= Pmodel_vec.shape[1]: # Bounds check
        idx_high = Pmodel_vec.shape[1] - 1
    
    Pd = (Pmodel_vec[:,idx_low]*(1-frac) + Pmodel_vec[:,idx_high]*(frac)).reshape(nc,nc)

    return Pd


def _p_marginal_from_pmodel(Pmodel):
    """Estimate the marginal category proportions from the far-field bivariate table.

    At the largest tabulated distance the joint table should be close to the
    independence product ``p(c) * p(c')``.  Row-summing gives ``p(c)``.

    Parameters
    ----------
    Pmodel : (nc, nc, nd) ndarray
        Bivariate probability tables along a single axis.

    Returns
    -------
    p : (nc,) ndarray
    """
    P_far = Pmodel[:, :, -1]
    p = P_far.sum(axis=1)
    p = np.clip(p, 1e-15, None)
    return p / p.sum()


def _auto_coord_cols(nd, K):
    """Default coordinate-column assignment for K separable components in nd dimensions.

    Convention (mirrors the stamps space-time convention in ``coord2K``):

    * K == 2, nd == 3 → ``[[0, 1], [2]]``  (XY horizontal, Z vertical)
    * K == 2, nd == 2 → ``[[0], [1]]``     (X, Y)
    * K == 3, nd == 4 → ``[[0, 1], [2], [3]]``
    * General: first K-1 components get one column each; last gets the rest.

    Returns
    -------
    coord_cols : list of list of int
    """
    if K == 2 and nd == 3:
        return [[0, 1], [2]]
    if K == 2 and nd == 2:
        return [[0], [1]]
    if K == 3 and nd >= 4:
        return [[0, 1], [2], [3]]
    # Generic fallback: one column per component
    cols = list(range(nd))
    if K > nd:
        raise ValueError(
            f"Cannot auto-assign coord_cols: K={K} components but cs has only nd={nd} columns. "
            "Set options['coord_cols'] explicitly."
        )
    return [[c] for c in cols[:K]]


def _resolve_pmodel(dmodel, Pmodel, dmax, cs, options):
    """Detect separable vs 1-D Pmodel from input types and build ``sep_opt`` dict.

    Separable mode is detected when both *dmodel* and *Pmodel* are Python lists
    or tuples (analogous to ``isspacetime(models)`` in ``coord2K``).

    For the separable case this function:

    * Reads ``mode`` from ``options`` (default ``'product'``).
    * Reads or auto-computes ``p_marginal`` (from far-field row sums).
    * Reads or auto-assigns ``coord_cols`` based on ``cs.shape[1]`` and ``len(dmodel)``.
    * Converts a list or scalar ``dmax`` into per-component values (``None`` →
      will be filled by ``auto_dmax`` later).
    * Packs everything into the ``separable_pmodel`` dict that internal functions
      (``neighbours_separable``, ``maxentropytable``) expect.

    Parameters
    ----------
    dmodel : array or list of arrays
    Pmodel : array or list of arrays
    dmax   : scalar, list, or None
    cs     : (n, nd) coordinate array (used to infer nd for auto coord_cols)
    options : dict

    Returns
    -------
    sep_opt : dict or None
        ``None`` when 1-D mode is detected.
    dmodel_1d, Pmodel_1d : the (possibly unchanged) scalar arrays for 1-D path.
    dmax_1d : the (possibly unchanged) scalar dmax for 1-D path.
    """
    is_sep = isinstance(dmodel, (list, tuple)) and isinstance(Pmodel, (list, tuple))
    if not is_sep:
        return None, dmodel, Pmodel, dmax

    K = len(dmodel)
    if len(Pmodel) != K:
        raise ValueError(
            f"dmodel list length ({K}) must match Pmodel list length ({len(Pmodel)})."
        )

    nd = cs.shape[1]
    mode = options.get('mode', 'product')

    # --- p_marginal: auto-compute per component, then average ----------------
    p_raw = options.get('p_marginal', None)
    if p_raw is None:
        p_list = [_p_marginal_from_pmodel(np.asarray(Pmodel[k])) for k in range(K)]
        p_avg = np.mean(np.stack(p_list, axis=0), axis=0)
        p_marginal = p_avg / p_avg.sum()
    else:
        p_marginal = np.asarray(p_raw, dtype=float).ravel()
        p_marginal = np.clip(p_marginal, 1e-15, None)
        p_marginal /= p_marginal.sum()

    # --- coord_cols ----------------------------------------------------------
    coord_cols = options.get('coord_cols', None)
    if coord_cols is None:
        coord_cols = _auto_coord_cols(nd, K)

    # --- per-component dmax --------------------------------------------------
    if dmax is None or (isinstance(dmax, (int, float)) and np.isnan(float(dmax))):
        dmax_list = [np.inf] * K
    elif isinstance(dmax, (list, tuple, np.ndarray)):
        dmax_list = [float(d) if d is not None else np.inf for d in dmax]
    else:
        dmax_list = [float(dmax)] * K

    # --- build components list -----------------------------------------------
    components = []
    for k in range(K):
        components.append({
            'dmodel':     np.asarray(dmodel[k], dtype=float),
            'Pmodel':     np.asarray(Pmodel[k], dtype=float),
            'coord_cols': list(np.asarray(coord_cols[k], dtype=int).ravel()),
            'dmax':       dmax_list[k],
        })

    sep_opt = {
        'mode':       mode,
        'p_marginal': p_marginal,
        'components': components,
    }
    return sep_opt, None, None, None


def _find_effective_range(dmodel, Pmodel, threshold=0.05):
    """
    Estimate the effective correlation range from bivariate probability tables.

    The function measures the total variation (TV) between the bivariate table
    ``Pmodel[:, :, k]`` and the corresponding **independence product**
    ``p(c) * p(c')`` at each tabulated distance ``dmodel[k]``.  It then
    returns the interpolated distance at which the normalised TV drops to
    ``threshold`` × its value at the smallest distance.

    Mathematically:

    .. math::

        TV(d) = \\frac{1}{2}\\sum_{c,c'} \\left| P(c,c';d) - p(c)p(c') \\right|

        d^* = \\inf\\{ d : TV(d)/TV(d_0) \\le \\text{threshold} \\}

    Parameters
    ----------
    dmodel : np.ndarray
        1-D array of tabulated distances, shape ``(nd,)``.
    Pmodel : np.ndarray
        3-D array of bivariate probability tables, shape ``(nc, nc, nd)``.
        ``Pmodel[:, :, k]`` is the joint table at distance ``dmodel[k]``.
    threshold : float, optional
        Fraction of the zero-distance TV below which the data are considered
        decorrelated.  Default ``0.05`` (5 %).

    Returns
    -------
    d_range : float
        Estimated effective correlation range (in the same units as
        ``dmodel``).  If the TV never drops below ``threshold`` within the
        range of ``dmodel``, the largest tabulated distance is returned as a
        conservative upper bound.

    Notes
    -----
    The marginal ``p(c)`` is estimated from the table at the **largest**
    tabulated distance (where the table is closest to independence).  If the
    largest-distance table is still far from independence the returned range
    will be a lower bound rather than the true range.
    """
    nd = len(dmodel)

    # Estimate marginal p(c) from the table at the largest distance
    P_far = Pmodel[:, :, -1]
    p_marginal = P_far.sum(axis=1)          # sum over c' for each c
    p_marginal = np.clip(p_marginal, 1e-15, None)
    p_marginal /= p_marginal.sum()

    # Independence product  P_indep[c, c'] = p(c) * p(c')
    P_indep = np.outer(p_marginal, p_marginal)

    # Compute normalised TV at every tabulated distance
    tv = np.zeros(nd)
    for k in range(nd):
        P_k = Pmodel[:, :, k].copy()
        s = P_k.sum()
        if s > 1e-30:
            P_k /= s           # ensure sums to 1
        tv[k] = 0.5 * np.abs(P_k - P_indep).sum()

    tv_max = tv[0]
    if tv_max < 1e-15:
        # No dependence at all — any dmax is fine; return smallest distance
        return float(dmodel[0])

    tv_norm = tv / tv_max      # 1 at d=dmodel[0], approaching 0 at large d

    # Find the first index where tv_norm <= threshold
    below = np.where(tv_norm <= threshold)[0]
    if len(below) == 0:
        # TV never drops below threshold in the tabulated range
        return float(dmodel[-1])

    idx = below[0]
    if idx == 0:
        return float(dmodel[0])

    # Linear interpolation between dmodel[idx-1] and dmodel[idx]
    d_lo, d_hi   = float(dmodel[idx - 1]), float(dmodel[idx])
    tv_lo, tv_hi = tv_norm[idx - 1],       tv_norm[idx]
    if abs(tv_hi - tv_lo) < 1e-15:
        return d_lo

    d_range = d_lo + (d_hi - d_lo) * (threshold - tv_lo) / (tv_hi - tv_lo)
    return float(d_range)
# ---------------------------------------------------------------------------


def neighbours(c0, c, Z, nmax, dmax, options=None):
    """
    Selects a subset of coordinates and data based on neighborhood rules.
    
    Finds neighbors to an estimation point `c0` from a set of data points
    `c` and `Z`, based on distance and neighborhood structure rules.

    Parameters
    ----------
    c0 : np.ndarray
        Coordinates of the estimation point, shape ``(1, d)``.
    c : np.ndarray
        Coordinates of all available data points, shape ``(n, d)``.
    Z : np.ndarray
        Data values (e.g., PDFs) at the `c` locations, shape ``(n, k)``.
    nmax : int
        Maximum number of neighbors for a 'non-sectored' search.
    dmax : float or list
        Maximum distance. Can be a single float for spatial, or a list
        for spatio-temporal (e.g., ``[dmax_s, dmax_t, temporal_rate]``).
    options : dict, optional
        Dictionary of search options:
        - 'neighborhood_structure' (str): 'non-sectored' (default) finds the
          `nmax` closest points. 'sectored' finds points using spatial
          sectors.
        - 'nmax_per_sector' (int): Max points per sector (default 1).
          Used only if `neighborhood_structure='sectored'`.

    Returns
    -------
    csub : np.ndarray
        Subset of coordinates for the found neighbors, shape ``(m, d)``.
    Zsub : np.ndarray
        Subset of data values for the found neighbors, shape ``(m, k)``.
    dsub : np.ndarray
        Vector of distances from `c0` to `csub`, shape ``(m,)``.
    nsub : int
        Number of neighbors found (m).
    index : np.ndarray
        Indices of the neighbors in the original `c` and `Z` arrays, shape ``(m,)``.
    """
    
    # --- START: Options Handling ---
    if options is None:
        options = {}
    neighborhood_structure = options.get('neighborhood_structure', 'non-sectored')
    nmax_per_sector = options.get('nmax_per_sector', 1)
    # --- END: Options Handling ---

    if isinstance(dmax,np.ndarray):
        dmax = dmax.ravel().tolist()
    if isinstance(dmax,int)|isinstance(dmax,float):
        dmax = [dmax]
    if np.isinf(nmax):
        nmax = None
    
    c0 = c0.reshape(1,-1)
    ndim_coords = c0.shape[1]
    isST = len(dmax)==3 and ndim_coords > 1 # Check for ST case

    if isST:
        # --- START: Original Spatio-Temporal Logic (Unchanged) ---
        ## spatio and temporal case 
        dt = cdist(c0[:,-1].reshape(-1,1),c[:,-1].reshape(-1,1)).ravel() ## calculate the temporal distance between c0 and c
        ds = cdist(c0[:,:-1],c[:,:-1]).ravel() ## calculate the spatial distance between c0 and c
        d = ds + dt*dmax[2] ## combine distance in space and time
        index=np.where((ds<=dmax[0])&(dt<=dmax[1]))[0] ## find distances in space & time<=dmax
        dstdf= pd.DataFrame(d.reshape(-1,1)).loc[index] ## add the idx on each distance by create the dataframe and remove the points that the distance more than dmax
        sltidx = dstdf.sort_values(0).iloc[:nmax].index## get the index of at most the closest nsmax amount of point.
        # --- END: Original Spatio-Temporal Logic ---
        
    else:
        # --- START: Pure Spatial Logic (2D or 3D) ---
        d = cdist(c0, c).ravel()
        idx_in_range = np.where(d < dmax[0])[0]

        if neighborhood_structure == 'non-sectored':
            # Original non-sectored logic: find nmax closest
            if idx_in_range.size == 0:
                sltidx = np.array([], dtype=int)
            else:
                ddf = pd.DataFrame(d[idx_in_range], index=idx_in_range, columns=['distance'])
                sltidx = ddf.sort_values('distance').iloc[:nmax].index
        
        elif neighborhood_structure == 'sectored':
            # New sectored logic: find nmax_per_sector in each sector
            if idx_in_range.size == 0:
                sltidx = np.array([], dtype=int)
            else:
                deltas = c - c0 # (n_all, ndim_coords)
                valid_deltas = deltas[idx_in_range, :]
                
                df_neighbors = pd.DataFrame(valid_deltas, index=idx_in_range)
                df_neighbors['distance'] = d[idx_in_range]
                
                sectors = []
                if ndim_coords == 2:
                    # 2D: 4 Quadrants
                    df_neighbors.columns = ['x', 'y', 'distance']
                    sectors.append( (df_neighbors['x'] >= 0) & (df_neighbors['y'] >= 0) ) # Q1 (NE)
                    sectors.append( (df_neighbors['x'] < 0)  & (df_neighbors['y'] >= 0) ) # Q2 (NW)
                    sectors.append( (df_neighbors['x'] < 0)  & (df_neighbors['y'] < 0)  ) # Q3 (SW)
                    sectors.append( (df_neighbors['x'] >= 0) & (df_neighbors['y'] < 0)  ) # Q4 (SE)
                
                elif ndim_coords == 3:
                    # 3D: 6 Sectors (4 horizontal + Up/Down)
                    # Assuming (x,y,z) -> (0,1,2)
                    df_neighbors.columns = ['x', 'y', 'z', 'distance']
                    z_tol = 1e-6 # Tolerance for horizontal plane
                    
                    # Horizontal Sectors
                    is_horizontal = (np.abs(df_neighbors['z']) <= z_tol)
                    sectors.append( is_horizontal & (df_neighbors['x'] >= 0) & (df_neighbors['y'] >= 0) ) # H-NE
                    sectors.append( is_horizontal & (df_neighbors['x'] < 0)  & (df_neighbors['y'] >= 0) ) # H-NW
                    sectors.append( is_horizontal & (df_neighbors['x'] < 0)  & (df_neighbors['y'] < 0)  ) # H-SW
                    sectors.append( is_horizontal & (df_neighbors['x'] >= 0) & (df_neighbors['y'] < 0)  ) # H-SE
                    
                    # Vertical Sectors
                    sectors.append( df_neighbors['z'] > z_tol )  # Up
                    sectors.append( df_neighbors['z'] < -z_tol ) # Down
                
                else:
                    # 1D or >3D: Fall back to non-sectored
                    sltidx = df_neighbors.sort_values('distance').iloc[:nmax].index
                    csub,Zsub,dsub,nsub,index = c[sltidx,:],Z[sltidx,:],d[sltidx],len(sltidx),sltidx
                    return csub,Zsub,dsub,nsub,index

                # Collect indices
                all_sector_indices = []
                for s_mask in sectors:
                    points_in_sector = df_neighbors[s_mask]
                    if not points_in_sector.empty:
                        closest_indices = points_in_sector.nsmallest(nmax_per_sector, 'distance').index
                        all_sector_indices.extend(closest_indices)
                
                sltidx = pd.Index(all_sector_indices).unique() # Use pd.Index to keep order and uniqueness
        
        else:
            raise ValueError(f"Unknown neighborhood_structure: {neighborhood_structure}")
        # --- END: Pure Spatial Logic ---
        
    csub,Zsub,dsub,nsub,index = c[sltidx,:],Z[sltidx,:],d.ravel()[sltidx],len(sltidx),sltidx
    return csub,Zsub,dsub,nsub,index



def neighbours_separable(
    c0, c, Z, nmax, components, options=None, p_marginal=None, mode='product'
):
    """Neighbour search with per-axis distance limits (for separable Pmodels).

    .. deprecated::
        Use :func:`stamps.general.neighbours.neighbours_cat` with
        ``options['separable_axes']`` and
        :func:`stamps.general.local_marginal.local_marginal` for the local
        marginal estimation.  This function is preserved for backward
        compatibility but is no longer called internally.

    A point is kept only if **all** component distances are below the
    corresponding ``dmax``.  Among valid points, up to ``nmax`` are kept and
    ordered by ``options['neighbour_ranking']`` (default ``'tv'``).

    * ``'distance'`` — ascending normalised geometric score
      :math:`\\sqrt{\\sum_k (d_k / d^{\\max}_k)^2}` (legacy behaviour).
    * ``'tv'`` — descending total variation
      :math:`\\tfrac{1}{2}\\sum_{a,b}|P_{3\\mathrm{D}}(a,b)-p_ap_b|` (default).
    * ``'mutual_info'`` — descending mutual information
      :math:`\\sum_{a,b} P_{3\\mathrm{D}}(a,b)\\log(P_{3\\mathrm{D}}(a,b)/(p_ap_b))`.

    TV and MI require ``p_marginal`` (and use ``probamodel2bitable_separable``);
    if ``p_marginal`` is missing, ranking falls back to ``'distance'``.

    Parameters
    ----------
    c0 : (1, d) ndarray
        Estimation location.
    c, Z : (n, d) and (n, k)
        Data sites and soft data.
    nmax : int
        Maximum neighbours.
    components : list of dict
        Each dict has ``coord_cols``, ``dmax``, ``dmodel``, ``Pmodel``.
    options : dict, optional
        May include ``neighbour_ranking`` in ``{'distance', 'tv', 'mutual_info'}``.
    p_marginal : (nc,) array_like, optional
        Marginal category proportions for separable Pmodel composition.
    mode : {'product', 'additive'}, optional
        Passed to ``probamodel2bitable_separable`` for TV/MI ranking.

    Returns
    -------
    csub, Zsub, dsub, nsub, index, d_components
        ``dsub`` is full Euclidean distance in original ``c`` space (for
        collocation checks).  ``d_components`` has shape ``(nsub, ncomp)``.
    """
    if options is None:
        options = {}
    if np.isinf(nmax):
        nmax = None

    c0 = np.asarray(c0, dtype=float).reshape(1, -1)
    c = np.asarray(c, dtype=float)
    Z = np.asarray(Z)
    n = c.shape[0]
    ncomp = len(components)

    d_euc = cdist(c0, c).ravel()
    D = np.zeros((n, ncomp), dtype=float)
    dmaxs = np.zeros(ncomp, dtype=float)
    for j, comp in enumerate(components):
        cols = np.asarray(comp.get('coord_cols', comp.get('dims', [])), dtype=int)
        dmaxs[j] = float(comp['dmax'])
        if cols.size == 0:
            raise ValueError(f"components[{j}]['coord_cols'] is empty")
        if cols.max() >= c.shape[1] or cols.min() < 0:
            raise ValueError(
                f"components[{j}]['coord_cols'] out of bounds for c.shape[1]={c.shape[1]}"
            )
        D[:, j] = cdist(c0[:, cols], c[:, cols]).ravel()

        # --- Pairwise-range compliance check ---------------------------------
        # Two candidates within dmax_j of c0 are at most 2*dmax_j apart
        # (triangle inequality).  Cap dmax_j at dmodel[-1]/2 so that no
        # data-to-data pair in maxentropytable exceeds the interpolation range.
        dmodel_max_j = float(comp['dmodel'][-1])
        safe_dmax_j  = dmodel_max_j / 2.0
        if dmaxs[j] > safe_dmax_j:
            dmaxs[j] = safe_dmax_j
        # ---------------------------------------------------------------------

    valid = np.ones(n, dtype=bool)
    for j in range(ncomp):
        valid &= D[:, j] < dmaxs[j]

    idx_all = np.where(valid)[0]
    if idx_all.size == 0:
        return (
            np.empty((0, c.shape[1])),
            np.empty((0, Z.shape[1])),
            np.empty(0),
            0,
            np.empty(0, dtype=int),
            np.empty((0, ncomp)),
            None,   # p_local: no candidates → no local marginal
        )

    ranking = options.get('neighbour_ranking', 'tv')
    if ranking not in ('distance', 'tv', 'mutual_info'):
        raise ValueError(
            "options['neighbour_ranking'] must be one of "
            "'distance', 'tv', 'mutual_info'; "
            f"got {ranking!r}"
        )

    use_prob_rank = ranking in ('tv', 'mutual_info') and p_marginal is not None
    if use_prob_rank:
        p = np.asarray(p_marginal, dtype=float).ravel()
        p = np.clip(p, 1e-15, None)
        p = p / p.sum()
        dmodels = [comp['dmodel'] for comp in components]
        Pmodels = [comp['Pmodel'] for comp in components]
        n_idx = idx_all.size
        scores = np.empty(n_idx, dtype=float)
        for m in range(n_idx):
            dvec = D[idx_all[m], :]
            P3d = probamodel2bitable_separable(
                dvec, dmodels, Pmodels, p, mode=mode
            )
            P_indep = np.outer(p, p)
            if ranking == 'tv':
                scores[m] = 0.5 * np.abs(P3d - P_indep).sum()
            else:
                pio = np.maximum(P_indep, 1e-300)
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = P3d / pio
                    log_r = np.log(np.where(P3d > 1e-300, np.maximum(ratio, 1e-300), 1.0))
                scores[m] = float(np.sum(P3d * log_r))
        order = np.argsort(-scores)
    else:
        scores = np.sqrt(
            np.sum((D[idx_all, :] / np.maximum(dmaxs, 1e-30)) ** 2, axis=1)
        )
        order = np.argsort(scores)

    if nmax is not None:
        order = order[:nmax]
    sltidx = idx_all[order]

    csub = c[sltidx, :]
    Zsub = Z[sltidx, :]
    dsub = d_euc[sltidx]
    d_components = D[sltidx, :]
    nsub = len(sltidx)

    # --- Kernel-weighted local marginal from ALL valid candidates (not just nsmax) ---
    # Uses the already-computed D[idx_all, :] — zero additional cdist calls.
    # σ_j = dmaxs[j] / 3  →  kernel weight ≈ 1 % at d = dmax (practical zero).
    #
    # Regularization via Kish effective sample size:
    #   N_eff = (Σ w_k)² / Σ w_k²
    # Shrinkage toward the global marginal p_marginal (if supplied) or uniform:
    #   α = 1 / (N_eff + 1)
    #   p_local_reg = (1-α)·p_local_raw + α·p_reference
    # This prevents near-zero entries for absent classes that would cause
    # explosive Bayes-factor ratios when p_local appears in the numerator of
    # the composite prior  q_eff = p_local × q_XGB / p_global.
    _kern = np.ones(idx_all.size, dtype=float)
    for j in range(ncomp):
        _sig_j = dmaxs[j] / 3.0
        if _sig_j > 1e-15:
            _kern *= np.exp(-0.5 * (D[idx_all, j] / _sig_j) ** 2)
    _ksum = _kern.sum()
    if _ksum > 1e-15:
        _p_local_raw = (_kern @ Z[idx_all]) / _ksum   # weighted class frequencies

        # Kish effective sample size
        _n_eff = (_ksum ** 2) / np.sum(_kern ** 2)

        # Reference distribution for shrinkage: p_marginal if provided, else uniform
        if p_marginal is not None:
            _p_ref = np.asarray(p_marginal, dtype=float).ravel()
            _p_ref = np.clip(_p_ref, 1e-15, None)
            _p_ref /= _p_ref.sum()
        else:
            _nc = Z.shape[1]
            _p_ref = np.ones(_nc, dtype=float) / _nc

        # Shrinkage coefficient: decays toward 0 as N_eff grows
        _alpha = 1.0 / (_n_eff + 1.0)
        _p_local = (1.0 - _alpha) * _p_local_raw + _alpha * _p_ref

        _p_local = np.clip(_p_local, 1e-15, None)
        _p_local /= _p_local.sum()
    else:
        _p_local = None
    # -------------------------------------------------------------------------------

    return csub, Zsub, dsub, nsub, sltidx, d_components, _p_local


