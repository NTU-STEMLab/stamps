# -*- coding: utf-8 -*-
from six.moves import range
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy import integrate as _sci_integrate

from ..general.neighbours import neighbours
from ..general.coord2K import coord2K
from ..general.coord2dist import coord2dist
from ..models.covmodel import get_model
from .designmatrix import designmatrix

def kriging(ck,ch,zh,model,param,nhmax,dmax,order=None,options=None,
            aniso_angle=None,aniso_ratio=None):
  ''' 
% kriging                   - prediction using kriging methods 
%
% Standard linear kriging algorithm for processing hard data
% that can reasonably be assumed as Gaussian distributed. The
% function is intendend to be as general as possible, covering
% various situations, like non-stationarity of the mean,
% multivariate cases, nested models, space-time estimations,
% etc. Depending on the case, specific format are needed for
% the input variables. This function is a special case of the more
% general BMEprobaMoments.mfunction, which processes both hard and 
% soft data.
%
% SYNTAX :
%
% [zk,vk]=kriging(ck,ch,zh,model,param,nhmax,dmax,order,options);
%
% INPUT :
%
% ck        nk by d   matrix of coordinates for the estimation locations.
%                     A line corresponds to the vector of coordinates at
%                     an estimation location, so the number of columns
%                     corresponds to the dimension of the space. There is
%                     no restriction on the dimension of the space.
% ch        nh by d   matrix of coordinates for the hard data locations,
%                     with the same convention as for ck.
% zh        nh by 1   vector of values for the hard data at the coordinates
%                     specified in ch.
  covmodel  list      list of m nested covariance models in each of 
                      which the spatial and temporal components are put 
                      in a list as [covmodelS,covmodelT] 
  covparam0 list      list of intial values covariance parameters for 
                      m covmodels in each component the parameters are 
                      listed as [sill,[covparamS1,covparamS2,..],
                      [covparamT1,covparamT2,..]]
% nhmax     scalar    maximum number of hard data values that are considered
%                     for the estimation at the locations specified in ck.
% dmax      scalar    maximum distance between an estimation location and
%                     existing hard data locations. All hard data locations
%                     separated by a distance smaller than dmax from an
%                     estimation location will be included in the estimation
%                     process for that location, whereas other hard data
%                     locations are neglected.
% order     scalar    order of the polynomial mean along the spatial axes at
%                     the estimation locations. For the zero-mean case, NaN
%                     (Not-a-Number) is used. Note that order=NaN can only be
%                     used with covariance models and not with variogram models.
% options   scalar    optional parameter that can be used if the default value
%                     is not satisfactory (otherwise it can simply be omitted
%                     from the input list of variables). options(1) is taking
%                     the value 1 or 0 if the user wants or does not want to
%                     display the order number of the location which is
%                     currently processed, respectively.
%
% OUTPUT :
%
% zk        nk by 1   vector of estimated values at the estimation locations. A
%                     value coded as NaN means that no estimation has been performed
%                     at that location due to the lack of available data. 
% vk        nk by 1   vector of estimation (kriging) variances at the estimation
%                     locations. As for zk, a value coded as NaN means that no
%                     estimation has been performed at the corresponding location.
%       
    
    
    input
    ck: nk by d float
    ch: nh by d float
    zh: nh by 1 float
    model: mn by 1 string
    param: mn by 3 float
    nhmax: int
    dmax: 1 by 3 float
    order: None or 0
    options: not no use here
    
    return
    zk: nk by 1 float
    vk: nk by 1 float
  '''
    
    #initial return variable
  # Normalise inputs: zh must be (n,1); ck/ch must be 2-D
  ck = np.atleast_2d(np.asarray(ck, dtype=float))
  ch = np.atleast_2d(np.asarray(ch, dtype=float))
  zh = np.asarray(zh, dtype=float)
  if zh.ndim == 1:
      zh = zh.reshape(-1, 1)

  nk = ck.shape[0]
  zk = np.zeros( ( nk, 1 ) )
  vk = np.zeros( ( nk, 1 ) )
  zk[:] = np.nan
  vk[:] = np.nan
  
  if order is None:
    order = np.nan
    
  for i in range( ck.shape[0] ):
    ci = ck[i:i+1]
    ci_nebr,zi_nebr,di_nebr,ni_nebr,idxi_nebr=neighbours(ci,ch,zh,nhmax,dmax)
    if ni_nebr > 0:
      K, dummyKK = coord2K( ci_nebr, ci_nebr, model, param,
                            aniso_angle=aniso_angle, aniso_ratio=aniso_ratio )
      k, dummykk = coord2K( ci_nebr, ci, model, param,
                            aniso_angle=aniso_angle, aniso_ratio=aniso_ratio )
      k0, dummykk0 = coord2K(ci, ci, model, param,
                              aniso_angle=aniso_angle, aniso_ratio=aniso_ratio)
      X=designmatrix(ci_nebr,order)[0]  
      x=designmatrix(ci,order)[0]
#            unit = np.ones(k.shape) # n by 1, X in matlab
#            unit_t_add = np.append( unit.T, [[0]], axis = 1 ) # 1 by n+1, x in matlab
                 
            #change shape for kriging
#            Kadd = np.hstack( ( K, unit ) )
#            Kadd = np.vstack( ( Kadd, unit_t_add ) )
      Kadd = K
      Xm,Xn = X.shape
      Kadd = np.vstack([np.hstack([K,X]),
                           np.hstack([X.T,np.zeros((Xn,Xn))])])
#            kadd = np.append( k, [[0]], axis = 0 )
      kadd = np.vstack([k,x.T])
      weight = np.linalg.solve(Kadd,kadd)[0:Xm]
      weight_t = weight.T
            #compute zk, vk
      zk[i] = weight_t.dot( zi_nebr )
      vk[i] = (k0-2*weight_t.dot(k)+weight_t.dot(K).dot(weight))[0]
            
    else:
      pass #already give NaN
  
  return zk.ravel(), vk.ravel()

def krigconstr(c0, c, order=None):
    """Return the trend basis matrices for kriging (Simple or Ordinary).

    Parameters
    ----------
    c0 : ndarray, shape (1, d)
        Estimation location.
    c : ndarray, shape (n, d)
        Data locations.
    order : {None, 0}
        Kriging type.  ``None`` → Simple Kriging (no trend constraint);
        ``0`` → Ordinary Kriging (constant mean, unbiasedness constraint).

    Returns
    -------
    X : ndarray, shape (n, nx)
        Design matrix for data locations.
    x : ndarray, shape (1, nx)
        Design row for estimation location.
    """
    if order is None or np.isnan(order):
        # Simple Kriging — no trend constraints
        X = np.zeros((c.shape[0], 0))
        x = np.zeros((1, 0))
    elif order == 0:
        # Ordinary Kriging — unbiasedness via a unit constant column
        X = np.ones((c.shape[0], 1))
        x = np.ones((1, 1))
    else:
        # Polynomial trend of given order (basic implementation up to order 1)
        cols = [np.ones(c.shape[0])]
        for dim in range(c.shape[1]):
            cols.append(c[:, dim])
        X = np.column_stack(cols)
        cols0 = [np.ones(1)]
        for dim in range(c0.shape[1]):
            cols0.append(c0[:, dim])
        x = np.column_stack(cols0)
    return X, x
    
def krigingFac( ck, ch, zh, model, param, nhmax, dmax, order = 0, options = None):

    ''' 
    input
    ck: nk by d float
    ch: nh by d float
    zh: nh by 1 float
    model: mn by 1 string ( ex. "exponential/gaussian" )
    param: mn by 3 float (ex. [ c, s, t ] )
    nhmax: int
    dmax: 1 by 3 float
    order: always 0, equals NaN in matlab
    options: not no use here, 0 or 1 in matlab for display echo 
    
    return
    zk: mn+1 by 1 float
    vk: mn+1 by 1 float
    '''
    
    #initial return variable
    nk = ck.shape[0]
    mn = len(model) #model.shape[0]
    zk = np.zeros( ( nk, mn+1 ) )
    vk = np.zeros( ( nk, mn+1 ) )
    zk[:] = np.nan
    vk[:] = np.nan
    
    for i in range( ck.shape[0] ):
        ci = ck[i:i+1]
        ci_nebr, zi_nebr, di_nebr, ni_nebr, idxi_nebr = neighbours(ci, ch, zh, nhmax, dmax )
        if ni_nebr > 0:
            K, dummyKK = coord2K( ci_nebr, ci_nebr, model, param )
            dummyk, kk = coord2K( ci_nebr, ci, model, param ) #kk is a list contain ki
            dummyk0, kk0 = coord2K(ci, ci, model, param)
            unit = np.ones(kk[0].shape) # n by 1
            unit_t_add = np.append( unit.T, [[0]], axis = 1 ) # 1 by n+1
            for idx_k, (ki,k0i) in enumerate( zip( kk, kk0 ) ):          
                #change shape for kriging
                Kadd = np.hstack( ( K, unit ) )
                Kadd = np.vstack( ( Kadd, unit_t_add ) )
                kkadd = np.append( ki, [[0]], axis = 0 )
                weight = np.dot( np.linalg.inv( Kadd ), kkadd )[:-1,:]
                weight_t = weight.T
                #compute zk, vk
                zk[i:i+1,idx_k:idx_k+1] = weight_t.dot( zi_nebr )
                vk[i:i+1,idx_k:idx_k+1] = ( k0i - 2 * weight_t.dot( ki ) + weight_t.dot( K ).dot( weight ) )[0]
            #compute local mean trend
            kkadd = np.append( np.zeros( kk[0].shape ), [[1]], axis = 0 )
            weight = np.dot( np.linalg.inv( Kadd ), kkadd )[:-1,:]
            weight_t = weight.T
            zk[i:i+1,-1:] = weight_t.dot( zi_nebr )
            vk[i:i+1,-1:] = weight_t.dot( K ).dot( weight )[0]
        else:
            pass #already give NaN
    return zk, vk


# ---------------------------------------------------------------------------
# LMC helper: build a covariance value between two sets of (multi-variable)
# observations given a list of LMC structure dicts from coregfit().
# ---------------------------------------------------------------------------

def _lmc_eval(D: np.ndarray, var_rows: np.ndarray, var_cols: np.ndarray,
              lmc_models: list) -> np.ndarray:
    """Evaluate the LMC covariance matrix for a pair of observation sets.

    Parameters
    ----------
    D : ndarray, shape (n_rows, n_cols)
        Euclidean distance matrix between the two observation sets.
    var_rows : ndarray of int, shape (n_rows,)
        Variable index (0-based) for each row observation.
    var_cols : ndarray of int, shape (n_cols,)
        Variable index (0-based) for each column observation.
    lmc_models : list of dict
        LMC parameter list from :func:`coregfit` — each dict has keys
        ``'model'``, ``'sill_matrix'``, ``'range_param'``.

    Returns
    -------
    K : ndarray, shape (n_rows, n_cols)
        LMC covariance matrix.
    """
    K = np.zeros(D.shape)
    for lmc in lmc_models:
        model_fn = get_model(lmc["model"])
        C_k = np.asarray(lmc["sill_matrix"])
        # nugget structures have no range_param
        rp = lmc.get("range_param", [None])
        g_k = np.asarray(model_fn(D, 1.0, rp))          # (n_rows, n_cols)
        sill_pairs = C_k[np.ix_(var_rows, var_cols)]     # (n_rows, n_cols)
        K += g_k * sill_pairs
    return K


def cokriging(
    ck: np.ndarray,
    ck_var: np.ndarray,
    ch: np.ndarray,
    ch_var: np.ndarray,
    zh: np.ndarray,
    lmc_models: list,
    nhmax: int,
    dmax: float,
    order: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Multivariate co-kriging using a Linear Model of Coregionalization.

    Performs Simple or Ordinary co-kriging for multiple coregionalized
    variables.  The joint spatial structure is described by a Linear Model
    of Coregionalization (LMC) whose parameters are obtained from
    :func:`~stamps.stats.dependence.stcovfit.coregfit`.

    Under the LMC the cross-covariance between variable ``i`` at location
    ``a`` and variable ``j`` at location ``b`` is:

    .. math::

        C_{ij}(h_{ab}) = \\sum_k B_k[i,j]\\cdot g_k(h_{ab})

    The co-kriging estimator for variable ``q`` at location ``u`` is:

    .. math::

        Z_q^*(u) = \\sum_{\\alpha} \\lambda_\\alpha Z_{v_\\alpha}(u_\\alpha)

    where the weights solve the standard kriging system built from the
    full cross-covariance matrix.

    Adapted from ``krigingMP.m`` in the BMElib package (Jan 1, 2001).

    Parameters
    ----------
    ck : ndarray, shape (nk, d)
        Coordinates of the estimation locations.  One row per location.
    ck_var : array_like of int, shape (nk,)
        Variable index (0-based) for each estimation location.  If all
        estimation locations target the same variable, pass a constant
        array (e.g. ``np.zeros(nk, dtype=int)``).
    ch : ndarray, shape (nh, d)
        Coordinates of the data locations.  One row per observation.
    ch_var : array_like of int, shape (nh,)
        Variable index (0-based) for each observation.
    zh : array_like, shape (nh,)
        Observed values at ``ch`` locations.
    lmc_models : list of dict
        LMC parameters returned by :func:`coregfit`.  Each dict must have
        keys ``'model'``, ``'sill_matrix'`` (``nv × nv`` ndarray), and
        ``'range_param'``.
    nhmax : int
        Maximum number of data observations to use per estimation location
        (neighbourhood truncation).
    dmax : float
        Maximum spatial search radius.  Observations farther than ``dmax``
        from an estimation location are ignored.
    order : {None, 0}
        Kriging type:

        * ``None`` — Simple co-kriging (assumes zero mean everywhere).
        * ``0``    — Ordinary co-kriging (adds per-variable unbiasedness
          constraints).
    verbose : bool, optional
        If ``True``, print progress every 100 estimation locations.

    Returns
    -------
    zk : ndarray, shape (nk,)
        Co-kriging estimates.  ``NaN`` where no neighbours were found.
    vk : ndarray, shape (nk,)
        Co-kriging estimation variances.  ``NaN`` where no data were found.

    Notes
    -----
    1. Neighbourhood selection is based purely on Euclidean distance from the
       estimation location (spatial coordinates only).  All observations within
       ``dmax`` and among the ``nhmax`` closest are included, regardless of
       variable index.
    2. The self-covariance ``k0 = C_{qq}(0) = \\sum_k B_k[q,q]`` is
       computed directly from the LMC parameters (no model call at lag 0
       needed since ``g_k(0) = 1`` by the unit-sill convention).
    3. For numerical stability, the kriging system is solved with
       :func:`numpy.linalg.lstsq` (minimum-norm solution when the system is
       rank-deficient).

    Examples
    --------
    >>> import numpy as np
    >>> from stamps.stats.dependence.stcovfit import coregfit
    >>> from stamps.stest.kriging import cokriging
    >>> rng = np.random.default_rng(42)
    >>> ch = rng.uniform(0, 100, (30, 2))
    >>> ch_var = np.array([0]*15 + [1]*15)
    >>> zh = rng.standard_normal(30)
    >>> ck = np.array([[50., 50.]])
    >>> ck_var = np.array([0])
    >>> d = np.linspace(0, 100, 10)
    >>> o = np.ones(10) * 50
    >>> C_true = np.array([[1.0, 0.5], [0.5, 0.8]])
    >>> V = [[C_true[i,j] * np.exp(-3*d/30) for j in range(2)] for i in range(2)]
    >>> lmc = coregfit(d, V, o, 'exponentialC', (1.0, 30.0))
    >>> zk, vk = cokriging(ck, ck_var, ch, ch_var, zh, lmc, nhmax=10, dmax=60.)
    """
    ck = np.atleast_2d(ck)
    ch = np.atleast_2d(ch)
    ch_var = np.asarray(ch_var, dtype=int).ravel()
    ck_var = np.asarray(ck_var, dtype=int).ravel()
    zh = np.asarray(zh, dtype=float).ravel()

    nk = ck.shape[0]
    nh = ch.shape[0]

    zk_out = np.full(nk, np.nan)
    vk_out = np.full(nk, np.nan)

    dmax_arr = np.array([[float(dmax)]])  # shape (1,1) for neighbours()

    for i in range(nk):
        if verbose and i % 100 == 0:
            print(f"cokriging: {i}/{nk}")

        ck_i = ck[i : i + 1, :]       # (1, d)
        q = int(ck_var[i])             # target variable index

        # ---- neighbourhood selection (spatial distance only) ---- #
        _, zh_local, _, n_local, idx_local = neighbours(
            ck_i, ch, zh.reshape(-1, 1), nhmax, dmax_arr
        )
        if n_local == 0:
            continue

        idx_local = idx_local.ravel().astype(int)
        ch_local = ch[idx_local, :]           # (n_local, d)
        zh_local = zh_local.ravel()           # (n_local,)
        var_local = ch_var[idx_local]         # (n_local,)

        # ---- build K (n_local × n_local) ---- #
        D_hh = coord2dist(ch_local, ch_local)
        K = _lmc_eval(D_hh, var_local, var_local, lmc_models)

        # ---- build k (n_local,) cross-covariances to target (q, ck_i) ---- #
        D_hk = coord2dist(ch_local, ck_i)     # (n_local, 1)
        k = _lmc_eval(D_hk, var_local, np.array([q]), lmc_models).ravel()

        # ---- self-covariance k0 = sum_k B_k[q, q] * g_k(0) ---- #
        # g_k(0) = 1 (unit sill, distance = 0)
        k0 = sum(float(np.asarray(lmc["sill_matrix"])[q, q]) for lmc in lmc_models)

        # ---- kriging system ---- #
        if order is None or (isinstance(order, float) and np.isnan(order)):
            # Simple co-kriging: K λ = k
            lam, *_ = np.linalg.lstsq(K, k, rcond=None)
        else:
            # Ordinary co-kriging: add one unbiasedness constraint per variable
            # present in the neighbourhood
            vars_present = np.unique(var_local)
            n_con = len(vars_present)

            # Augmented system: [[K, F], [F^T, 0]] [λ, μ]^T = [k, f0]
            F = np.zeros((n_local, n_con))
            f0 = np.zeros(n_con)
            for ci, vi in enumerate(vars_present):
                F[var_local == vi, ci] = 1.0
                if vi == q:
                    f0[ci] = 1.0

            Kaug = np.block([[K, F], [F.T, np.zeros((n_con, n_con))]])
            kaug = np.concatenate([k, f0])
            sol, *_ = np.linalg.lstsq(Kaug, kaug, rcond=None)
            lam = sol[:n_local]

        # ---- estimate and variance ---- #
        zk_out[i] = float(lam.dot(zh_local))
        vk_out[i] = float(k0 - 2.0 * lam.dot(k) + lam.dot(K).dot(lam))

    return zk_out, vk_out


def cokrigingT(
    ck: np.ndarray,
    ck_var: np.ndarray,
    ch: np.ndarray,
    ch_var: np.ndarray,
    yh: np.ndarray,
    lmc_models: list,
    nhmax: int,
    dmax: float,
    yfiles: list,
    cdfy_files: list,
    order: Optional[int] = None,
    n_grid: int = 200,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Multivariate co-kriging with Normal Score Transformation.

    Extends :func:`cokriging` to non-Gaussian data by applying a Normal
    Score Transformation (NST) prior to estimation and back-transforming the
    posterior distribution to the original data scale.

    The workflow mirrors ``krigingT.m`` (BMElib) generalised to the
    multivariate setting:

    1. **Forward transform** — each observation ``yh[a]`` (of variable
       ``ch_var[a]``) is mapped to the Gaussian scale via the empirical
       CDF of that variable:
       ``zh[a] = other2gauss(yh[a], yfiles[ch_var[a]], cdfy_files[ch_var[a]])``.
    2. **Co-kriging** — :func:`cokriging` is applied on the transformed
       data to obtain a Gaussian-scale estimate ``(zk_i, vzk_i)`` for each
       estimation location.
    3. **Back-transformation** — for each estimation location ``i``:

       * If ``vzk_i ≈ 0`` (exact data hit), just map ``zk_i`` back via
         ``gauss2other``.
       * Otherwise, the Gaussian posterior ``N(zk_i, vzk_i)`` is
         discretised over a fine grid in ``y``-space, the density is
         normalised, and the posterior mean and variance on the original
         scale are computed via numerical integration.

    Adapted from ``krigingT.m`` in the BMElib package (Jan 1, 2001).

    Parameters
    ----------
    ck : ndarray, shape (nk, d)
        Estimation coordinates.
    ck_var : array_like of int, shape (nk,)
        Variable index (0-based) for each estimation location.
    ch : ndarray, shape (nh, d)
        Data coordinates.
    ch_var : array_like of int, shape (nh,)
        Variable index (0-based) for each observation.
    yh : array_like, shape (nh,)
        Observed values on the **original** (non-Gaussian) scale.
    lmc_models : list of dict
        LMC parameters from :func:`~stamps.stats.dependence.stcovfit.coregfit`.
        These must have been fitted to the **Gaussian-scale** covariances.
    nhmax : int
        Maximum neighbourhood size.
    dmax : float
        Search radius.
    yfiles : list of array_like
        Length-``nv`` list.  ``yfiles[v]`` is a sorted vector of original-
        scale values used to define the empirical CDF for variable ``v``.
    cdfy_files : list of array_like
        Length-``nv`` list.  ``cdfy_files[v]`` is the empirical CDF
        evaluated at ``yfiles[v]``.
    order : {None, 0}, optional
        Kriging type (same as :func:`cokriging`).
    n_grid : int, optional
        Number of grid points used to discretise the posterior PDF in the
        original scale (default 200).
    verbose : bool, optional
        Print progress.

    Returns
    -------
    yk : ndarray, shape (nk,)
        Posterior **mean** of the original-scale variable at each estimation
        location.  ``NaN`` where estimation was not possible.
    vk : ndarray, shape (nk,)
        Posterior **variance** on the original scale.  ``NaN`` where
        estimation was not possible.

    Notes
    -----
    * The LMC model passed to this function should be fitted to the
      *Gaussian-transformed* data.  Transform each variable first using
      :func:`~stamps.bme.bme_transform.other2gauss`, then call
      :func:`~stamps.stats.dependence.stcov.stcov` and
      :func:`~stamps.stats.dependence.stcovfit.coregfit` on the
      Gaussian-scale empirical covariances.
    * This function calls :func:`cokriging` internally to obtain the
      Gaussian-scale estimate and variance, then performs
      back-transformation for each location independently.
    * Back-transformation uses trapezoidal integration over the
      Gaussian posterior evaluated on a grid in the original scale.
    * If a variable has a perfectly known value at an estimation location
      (co-kriging variance = 0), the estimate is simply the inverse CDF
      evaluated at that value.

    Examples
    --------
    >>> import numpy as np
    >>> from stamps.stats.dependence.stcovfit import coregfit
    >>> from stamps.stest.kriging import cokrigingT
    >>> from stamps.bme.bme_transform import other2gauss
    >>> rng = np.random.default_rng(0)
    >>> # Original-scale lognormal data for one variable
    >>> yh = np.exp(rng.standard_normal(20))
    >>> ch = rng.uniform(0, 100, (20, 2))
    >>> ch_var = np.zeros(20, dtype=int)
    >>> ck = np.array([[50., 50.]])
    >>> ck_var = np.array([0])
    >>> # Build empirical CDF
    >>> yfile = np.sort(yh)
    >>> cdfyfile = np.linspace(0.01, 0.99, len(yfile))
    >>> # Gaussian-scale LMC (single variable)
    >>> d = np.linspace(0, 100, 8)
    >>> o = np.ones(8) * 20
    >>> V = [[np.exp(-3*d/30)]]   # synthetic
    >>> lmc = coregfit(d, V, o, 'exponentialC', (1.0, 30.0))
    >>> yk, vk = cokrigingT(ck, ck_var, ch, ch_var, yh, lmc, nhmax=8,
    ...                      dmax=80., yfiles=[yfile], cdfy_files=[cdfyfile])
    """
    from ..bme.bme_transform import other2gauss, gaussinv, pdfgauss2other

    ck = np.atleast_2d(ck)
    ch = np.atleast_2d(ch)
    ch_var = np.asarray(ch_var, dtype=int).ravel()
    ck_var = np.asarray(ck_var, dtype=int).ravel()
    yh = np.asarray(yh, dtype=float).ravel()
    yfiles = [np.asarray(yf, dtype=float).ravel() for yf in yfiles]
    cdfy_files = [np.asarray(cf, dtype=float).ravel() for cf in cdfy_files]

    nk = ck.shape[0]

    # ------------------------------------------------------------------ #
    # Step 1: forward transform each observation to Gaussian scale
    # ------------------------------------------------------------------ #
    zh = np.empty_like(yh)
    for a in range(len(yh)):
        v = ch_var[a]
        zh[a] = float(np.asarray(
            other2gauss(yh[a:a+1], yfiles[v], cdfy_files[v])
        ).flat[0])

    # ------------------------------------------------------------------ #
    # Step 2: co-kriging on the Gaussian scale
    # ------------------------------------------------------------------ #
    zk_gauss, vzk_gauss = cokriging(
        ck, ck_var, ch, ch_var, zh, lmc_models, nhmax, dmax, order, verbose
    )

    # ------------------------------------------------------------------ #
    # Step 3: back-transform
    # ------------------------------------------------------------------ #
    yk = np.full(nk, np.nan)
    vk_out = np.full(nk, np.nan)

    for i in range(nk):
        if np.isnan(zk_gauss[i]):
            continue

        q = int(ck_var[i])
        yfile_q = yfiles[q]
        cdfy_q = cdfy_files[q]

        zk_i = float(zk_gauss[i])
        vzk_i = max(float(vzk_gauss[i]), 0.0)

        if vzk_i < 1e-12:
            # Exact hit: direct back-transform
            yk[i] = float(np.asarray(
                pdfgauss2other(
                    yfile_q,
                    np.array([zk_i]),
                    np.ones(1),          # dummy pdf, just need gauss2other
                    yfile_q,
                    cdfy_q,
                )
            ).flat[0])
            # Use the simpler direct inversion via gaussinv for exact case
            from ..bme.bme_transform import gaussinv as _ginv
            z_vals = _ginv(cdfy_q, [zk_i, 1e-12])
            y_vals = yfile_q  # already sorted
            # back-transform: just get nearest in CDF
            # Use the pdfgauss2other approach: pdf=delta_approx → mean = gauss2other
            yk[i] = float(np.interp(zk_i,
                                    np.asarray(other2gauss(
                                        yfile_q, yfile_q, cdfy_q
                                    )).ravel(),
                                    yfile_q))
            vk_out[i] = 0.0
        else:
            # Non-zero variance: integrate the posterior PDF in y-space
            # Build a grid in y-space (original scale)
            y_lo, y_hi = yfile_q[0], yfile_q[-1]
            y_grid = np.linspace(y_lo, y_hi, n_grid)

            # Evaluate the back-transformed posterior PDF using pdfgauss2other
            # The Gaussian posterior is N(zk_i, vzk_i) — use gaussinv to
            # convert the y_grid CDF values to the Gaussian scale, then
            # evaluate the Gaussian pdf and apply the Jacobian.
            try:
                pdf_y = np.asarray(pdfgauss2other(
                    y_grid,
                    np.array([zk_i]),
                    np.array([1.0 / np.sqrt(2 * np.pi * vzk_i)]),  # N(mu,var) at z=mu
                    yfile_q,
                    cdfy_q,
                )).ravel()
            except Exception:
                # Fallback: construct PDF via gaussinv + Jacobian manually
                from scipy.stats import norm as _norm
                cdf_grid = np.interp(y_grid,
                                     yfile_q,
                                     cdfy_q,
                                     left=cdfy_q[0],
                                     right=cdfy_q[-1])
                z_grid = _norm.ppf(np.clip(cdf_grid, 1e-12, 1 - 1e-12))
                gauss_pdf = _norm.pdf(z_grid, loc=zk_i, scale=np.sqrt(vzk_i))
                # Jacobian: dz/dy ≈ d(other2gauss)/dy (numerical)
                dcdf_dy = np.gradient(cdf_grid, y_grid)
                from scipy.stats import norm as _norm2
                dz_dcdf = 1.0 / np.maximum(
                    _norm2.pdf(z_grid), 1e-300
                )
                jacobian = np.abs(dcdf_dy * dz_dcdf)
                pdf_y = gauss_pdf * jacobian

            pdf_y = np.maximum(pdf_y, 0.0)

            # Normalise
            area = np.trapezoid(pdf_y, y_grid)
            if area > 1e-300:
                pdf_y = pdf_y / area

            # Posterior mean and variance
            yk[i] = float(np.trapezoid(y_grid * pdf_y, y_grid))
            vk_out[i] = float(np.trapezoid(
                (y_grid - yk[i]) ** 2 * pdf_y, y_grid
            ))

        if verbose and i % 100 == 0:
            print(f"cokrigingT back-transform: {i}/{nk}")

    return yk, vk_out
