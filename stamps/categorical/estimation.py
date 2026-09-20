# -*- coding: utf-8 -*-
"""
stamps.categorical.estimation
===============================
Categorical spatial estimation: BME, MCP, and Hybrid BME.

Public API
----------
BMEcatPdf               Full categorical BME (MaxEnt tensor x marginals).
MCPcatPdf               Fast MCP approximation (conditional independence).
HBMEcatPdf              Hybrid: MCP regional prior then BME local update.
tune_regularization_loocv   LOOCV grid search over ME regularization.
producttablepdf         Multiply full prior by 1-D marginals (BME data step).
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d

from ..stats.dependence.ptable.fit import probamodel2bitable_separable
from ..stats.dependence.ptable.recenter import ipf_recenter
from ..general.coord2dist import coord2dist
from ..general.neighbours import neighbours_cat
from ..general.local_marginal import local_marginal

from ._core import (
    _coerce_ps_soft,
    _resolve_pmodel,
    _find_effective_range,
    _auto_coord_cols,
    _p_marginal_from_pmodel,
    probamodel2bitable,
    neighbours,
)
from .me_solvers import maxentropytable


def producttablepdf(P,pdf):
    """
    Multiplies a full PDF by its 1D marginals (prior and data).
    
    This function performs the "data integration" step of BME. It takes the
    full *prior* PDF `P(X_0, ..., X_n)` and multiplies it by the
    independent marginal PDFs (the prior at `X_0` and the data at `X_1...X_n`)
    to get the unnormalized posterior.

    Parameters
    ----------
    P : np.ndarray
        The full multivariate *prior* PDF, shape ``(nc, ..., nc)``
        with `ndim` dimensions.
    pdf : np.ndarray
        A 2D array of marginal probabilities, shape ``(ndim, nc)``.
        `pdf[0, :]` is the prior at the estimation point.
        `pdf[1:, :]` are the soft data PDFs at the neighbor points.

    Returns
    -------
    Pprod : np.ndarray
        The unnormalized multivariate posterior PDF, shape ``(nc, ..., nc)``.
    """
    ndim = P.ndim
    nc = pdf.shape[-1]
    
    Pprod = P.copy()
    
    # for i in range(ndim):   
    #    slices = [slice(None)] * ndim
    #    for j in range(nc):
    #        slices[i] = j
    #        Pprod[tuple(slices)]*= pdf[i,j]
            
    for i in range(ndim):
      expand_shape = [1] * ndim
      expand_shape[i] = nc # Use nc from pdf shape
      Pprod *= pdf[i,:].reshape(expand_shape)

    return Pprod

def sumvaluesatindex(P,index):
    """
    (Legacy) Sums values at specific indices for hard data conditioning.
    
    This function is designed to handle hard data by summing the
    probability of specific outcomes.
    
    Note: This function uses `eval()` and is not robust. It appears to
    be part of the original MATLAB port.

    Parameters
    ----------
    P : np.ndarray
        The full multivariate PDF, shape ``(nc, ..., nc)``.
    index : np.ndarray
        A 2D boolean-like array, shape ``(ndim, nc)``, where `index[i, j] = 1`
        if category `j` is the known hard data for point `i`.

    Returns
    -------
    S : float
        The sum of probabilities at the specified hard data indices.
    """
    
    ndim=P.ndim
    stridx=''
    for i in range(ndim):
        stridx_ = [':']*ndim
        stridx_[i] = str(np.where(index[i,:]==1)[0]).replace(' ',',')
        stridx_ = ','.join(stridx_)
        stridx = stridx+ '['+stridx_+']'
    ###
    S = eval(f'P{stridx}.sum()')
    return S

def BMEcatPdf(ck, cs, ps, dmodel, Pmodel, nsmax, dmax, options=None):
    """Perform categorical BME estimation with an optional spatially-varying prior.

    Parameters
    ----------
    ck : (nk, d) ndarray
        Estimation point coordinates.
    cs : (n, d) ndarray
        Data point coordinates.
    ps : (n, nc) or (n,) ndarray
        Soft data (class probabilities) or hard class labels.  Label handling
        is identical to :func:`MCPcatPdf`.
    dmodel : (nd,) ndarray **or** list of (nd_k,) arrays
        Tabulated distances.  Pass a Python **list** to activate separable
        mode (auto-detected); see :func:`MCPcatPdf` for full description.
    Pmodel : (nc, nc, nd) ndarray **or** list of such arrays
        Bivariate probability tables.  Pass a **list** to activate separable
        mode.
    nsmax : int
        Maximum number of neighbours.
    dmax : float, list, or None
        Maximum search distance.  ``None`` or a per-axis list for separable
        mode; scalar for 1-D mode.
    options : dict, optional
        All keys from :func:`MCPcatPdf` are supported, plus:

        ``'prior_pdf_at_ck'`` *(ndarray)* — externally supplied prior
        ``q(c)``, shape ``(nc,)`` or ``(nk, nc)``.  A Bayes-factor correction
        is applied automatically to avoid double-counting the global marginal.

        ``'global_marginal'`` *(ndarray, nc)* — override for ``p(c)`` used in
        the Bayes-factor correction.  Auto-estimated from ``ps.mean(axis=0)``
        when absent.

        ``'show_progress'`` *(int, default 0)*, ``'estimator'``
        *(default ``'MPL'``)*,  ``'model_structure'``
        *(default ``'full'``)*,  ``'tol'``, ``'reg'``, ``'k_neighbors'`` —
        ME solver options.

        ``'separable_pmodel'`` *(dict)* — legacy dict-form separable model;
        still accepted for backward compatibility.

        ``'covariate_mode'`` *(str or None)* — ``None`` (default, pure
        spatial), ``'separable'`` (Approach A), ``'constraint'``
        (Approach B), or ``'combined'`` (A + B).

        ``'covariate_constraints'`` *(list of dict)* — required for modes
        ``'constraint'`` and ``'combined'``.  Each dict needs ``'name'``,
        ``'dmodel'``, ``'Pmodel'``, ``'values_cs'``, ``'values_ck'``,
        and optionally ``'weight'`` (default 1.0).

        ``'covariate_solver'`` *(str, default ``'joint'``)*:
        ``'joint'`` or ``'sequential'``.

        ``'first_order'`` *(dict or None)* — unified first-order (unary)
        constraint mode.  When set, local first-order probabilities are
        enforced *inside* the MaxEnt table: every bivariate pair target is
        IPF-recentered to the per-axis marginals and matching unary
        constraints are added to the ``ME_dual`` solve.  The posterior data
        step then performs pure soft-data integration (no Bayes-factor tilt
        is applied — the BF options ``prior_pdf_at_ck`` / ``local_marginal``
        are ignored with a warning when this mode is active).  Keys:

        * ``'q_at_ck'`` *(ndarray (nk, nc) or (nc,), optional)* — first-order
          probabilities at the estimation points (e.g. from
          :func:`~stamps.categorical.prior.BMEcatPrior`).
        * ``'q_at_cs'`` *(ndarray (n, nc), optional)* — first-order
          probabilities at the data points (aligned with the *original*
          ``cs`` rows).  When absent, each neighbour axis falls back to the
          estimation point's ``q_0`` (local-stationarity / IPF mode).
        * ``'use_local_marginal'`` *(bool, default False)* — compose the
          kernel-weighted local marginal with ``q_at_ck`` via odds
          (Case-A logic); requires a separable Pmodel.

        The no-neighbour limit of the posterior is ``q_0`` (the composed
        first-order target at the estimation point).  ``estimator`` is
        forced to ``'ME_dual'`` in this mode.

    Returns
    -------
    pk : (nk, nc) ndarray
        Posterior marginal PDFs at the estimation locations.
    """
    if options is None:
        options = {}

    # --- GW-Pmodel nonstationarity support -----------------------------------
    # When options['gw_pmodel'] is a GWPmodel instance, the Pmodel, coordinate
    # transformation, and p_marginal are blended per estimation point inside
    # the main loop.  dmodel / Pmodel arguments are ignored in GW mode.
    _gw = options.get('gw_pmodel', None)

    # --- Detect separable vs 1-D from dmodel/Pmodel types -------------------
    cs = np.asarray(cs, dtype=float)
    if _gw is None:
        sep_bme, dmodel, Pmodel, dmax = _resolve_pmodel(dmodel, Pmodel, dmax, cs, options)
        if sep_bme is None:
            sep_bme = options.get('separable_pmodel')
        if sep_bme is not None:
            ncat = int(sep_bme['components'][0]['Pmodel'].shape[0])
        else:
            ncat = int(Pmodel.shape[0])
    else:
        # GW mode: ncat from GWPmodel; dmodel/Pmodel resolved per-point in loop
        sep_bme = None
        dmodel = None
        Pmodel = None
        ncat = _gw.nc

    ps = _coerce_ps_soft(ps, ncat, options)

    # --- Auto dmax (stationary only) -----------------------------------------
    if _gw is None:
        if sep_bme is not None:
            if options.get('auto_dmax', False):
                _thresh = options.get('auto_dmax_threshold', 0.05)
                for co in sep_bme['components']:
                    co['dmax'] = _find_effective_range(
                        co['dmodel'], co['Pmodel'], threshold=_thresh)
        elif options.get('auto_dmax', False):
            _thresh = options.get('auto_dmax_threshold', 0.05)
            dmax = _find_effective_range(dmodel, Pmodel, threshold=_thresh)
    # -------------------------------------------------------------------------

    show_progress = options.get('show_progress', 0)

    # Ensure sep_bme is visible to maxentropytable (which reads options dict)
    if sep_bme is not None and 'separable_pmodel' not in options:
        options = {**options, 'separable_pmodel': sep_bme}

    isnotOK=np.sum(abs(np.diff(ps.T,axis = 0)),axis = 0)==0
    cs = cs[~isnotOK,:]
    ps = ps[~isnotOK,:]

    # --- Covariate mode / data bookkeeping ---
    _cov_mode = options.get('covariate_mode')
    _cov_layers = options.get('covariate_constraints')
    _has_cov_b = (_cov_mode in ('constraint', 'combined') and
                  _cov_layers is not None)
    if _has_cov_b:
        _cov_cs_all = {}
        for lay in _cov_layers:
            if lay.get('type') == 'interaction':
                continue
            v = np.asarray(lay['values_cs'], dtype=float).ravel()
            _cov_cs_all[lay['name']] = v[~isnotOK]
        _cov_ck_all = {}
        for lay in _cov_layers:
            if lay.get('type') == 'interaction':
                continue
            _cov_ck_all[lay['name']] = np.asarray(
                lay['values_ck'], dtype=float).ravel()
    else:
        _cov_cs_all = _cov_ck_all = None

    nk = len(ck)
    
    # Default uniform prior for general cases
    uniform_prior = np.ones(ncat) / ncat

    # --- START: MODIFIED PRIOR LOGIC ---
    external_priors = options.get('prior_pdf_at_ck')
    has_array_of_priors = False
    
    if external_priors is not None:
        try:
            # Case 1: A 2D array [nk x ncat] of priors is provided
            if external_priors.ndim == 2 and external_priors.shape == (nk, ncat):
                has_array_of_priors = True
            # Case 2: A 1D array [ncat,] single prior is provided
            elif external_priors.size == ncat:
                external_priors = external_priors.flatten() # Ensure it's 1D
            else:
                # Shape is invalid
                print(f"Warning: 'prior_pdf_at_ck' shape {external_priors.shape} is invalid. Expected ({nk}, {ncat}) or ({ncat},). Ignoring.")
                external_priors = None
        except Exception as e:
            print(f"Warning: Error processing 'prior_pdf_at_ck'. Ignoring. Error: {e}")
            external_priors = None
    # --- END: MODIFIED PRIOR LOGIC ---

    # --- Global marginal (used as BF denominator when local_marginal is off) ---
    # In GW mode this is computed per-point from the blended p_marginal so that
    # the Bayes-factor denominator always matches the Pmodel's own marginal,
    # preventing the double-amplification documented in the GW-Pmodel analysis.
    _gm_raw = options.get('global_marginal', None)
    if _gm_raw is not None:
        _global_marginal_base = np.asarray(_gm_raw, dtype=float).ravel()
    else:
        _global_marginal_base = ps.mean(axis=0).astype(float).ravel()
    _global_marginal_base = np.clip(_global_marginal_base, 1e-15, None)
    _global_marginal_base /= _global_marginal_base.sum()
    # In stationary mode _global_marginal is fixed; in GW mode it is replaced
    # per-point inside the loop.
    _global_marginal = _global_marginal_base

    # When local_marginal=True the per-point p_local replaces _global_marginal as
    # BF denominator; _global_marginal is still used as fallback and for Case B.
    _use_local_marginal = options.get('local_marginal', False)
    # ---

    # --- First-order (unary) constraint mode ---------------------------------
    # See docstring: first-order information is enforced inside the ME table
    # (IPF-recentered pair targets + unary constraints) instead of the
    # posterior Bayes-factor tilt.
    _fo = options.get('first_order', None)
    _fo_q_ck = _fo_q_cs = None
    _fo_use_lm = False
    if _fo is not None:
        # Mutual exclusion: exactly one first-order entry point.  Ignoring the
        # BF options eliminates the double-counting class of bugs by
        # construction.
        if external_priors is not None or _use_local_marginal:
            print("Warning: options['first_order'] is active — ignoring the "
                  "Bayes-factor options 'prior_pdf_at_ck' and "
                  "'local_marginal' to avoid double-counting first-order "
                  "information.")
            external_priors = None
            has_array_of_priors = False
            _use_local_marginal = False

        # The unary block requires the true dual solver: MPL distorts axis
        # marginals, which is incompatible with first-order constraints.
        if options.get('estimator', 'MPL') != 'ME_dual':
            print("Note: options['first_order'] requires estimator="
                  "'ME_dual'; overriding.")
            options = {**options, 'estimator': 'ME_dual'}

        _fo_use_lm = bool(_fo.get('use_local_marginal', False))

        _fo_q_ck = _fo.get('q_at_ck', None)
        if _fo_q_ck is not None:
            _fo_q_ck = np.asarray(_fo_q_ck, dtype=float)
            if _fo_q_ck.ndim == 1 and _fo_q_ck.size == ncat:
                _fo_q_ck = np.tile(_fo_q_ck.ravel(), (nk, 1))
            if _fo_q_ck.shape != (nk, ncat):
                raise ValueError(
                    f"first_order['q_at_ck'] must have shape ({nk}, {ncat}) "
                    f"or ({ncat},), got {_fo_q_ck.shape}")

        _fo_q_cs = _fo.get('q_at_cs', None)
        if _fo_q_cs is not None:
            _fo_q_cs = np.asarray(_fo_q_cs, dtype=float)
            if _fo_q_cs.shape[0] == isnotOK.size:
                # Aligned with the original cs rows: apply the same
                # constant-row filter used on cs/ps above.
                _fo_q_cs = _fo_q_cs[~isnotOK]
            if _fo_q_cs.shape != (cs.shape[0], ncat):
                raise ValueError(
                    f"first_order['q_at_cs'] must have one row per data "
                    f"point ({cs.shape[0]} after filtering), "
                    f"got {_fo_q_cs.shape}")
    # ---------------------------------------------------------------------------

    pk=np.ones((nk,ncat))*np.nan;## 先創一個都是nan的矩陣再慢慢取代
    for i in range(nk):
        ck0=ck[[i],:]

        # --- Per-point GW-Pmodel blending ------------------------------------
        # When a GWPmodel is provided, we blend the Pmodel at ck0's XY
        # position and use the blended p_marginal as the BF denominator
        # instead of the dataset-wide mean.
        #
        # IMPORTANT: we do NOT transform cs/ck0 via _aniso2iso here.  The
        # regional horizontal Pmodels are fitted with `probatablecalc` on
        # **raw XY coordinates** (omnidirectional, isotropic fit), so Pmodel
        # lookups must be done with raw Euclidean distances.  Applying an
        # isotropic-space transform would stretch distances by up to 1/r,
        # weakening bivariate constraints, shrinking the effective neighbour
        # radius, and destabilising the ME solver.  The anisotropy parameters
        # (theta, r) from the regional fits are used internally by
        # ``GWPmodel._weights`` for blend-weight computation and are not
        # applied to estimation-time coordinates.
        if _gw is not None:
            (_dm_h, _Pm_h, _dm_v, _Pm_v,
             _theta_i, _r_i, _p_marg_i) = _gw.blend_at(ck0[0, :2])
            _p_marg_i      = np.asarray(_p_marg_i, dtype=float).ravel()
            _p_marg_i      = np.clip(_p_marg_i, 1e-15, None)
            _p_marg_i     /= _p_marg_i.sum()

            # Use raw coordinates (consistent with Pmodel fitting space)
            _cs_i  = cs
            _ck0_i = ck0

            # Build per-point separable model dict matching _resolve_pmodel output
            _nf_h = len(_dm_h)
            _dmax_h = float(_dm_h[-1]) if _nf_h > 0 else 1.0
            _nf_v = len(_dm_v)
            _dmax_v = float(_dm_v[-1]) if _nf_v > 0 else 1.0

            # Per-point auto_dmax: shrink the neighbour-search radius to the
            # effective correlation range of the *blended* Pmodel.  This
            # replaces the full common-grid maximum (often far beyond the
            # true range) with a physically meaningful localisation.
            if options.get('auto_dmax', False):
                _thresh = options.get('auto_dmax_threshold', 0.05)
                try:
                    _dmax_h = float(_find_effective_range(
                        _dm_h, _Pm_h, threshold=_thresh))
                except Exception:
                    pass
                try:
                    _dmax_v = float(_find_effective_range(
                        _dm_v, _Pm_v, threshold=_thresh))
                except Exception:
                    pass

            _sep_i = {
                'mode': 'product',
                'p_marginal': _p_marg_i,
                'components': [
                    {'dmodel': _dm_h, 'Pmodel': _Pm_h,
                     'dmax': _dmax_h, 'coord_cols': [0, 1]},
                    {'dmodel': _dm_v, 'Pmodel': _Pm_v,
                     'dmax': _dmax_v, 'coord_cols': [2]},
                ],
            }
            # Override BF denominator with the local blended marginal
            _global_marginal = _p_marg_i
            _loop_options = {**options, 'separable_pmodel': _sep_i}
        else:
            _cs_i   = cs
            _ck0_i  = ck0
            _sep_i  = sep_bme
            _loop_options = options
            _global_marginal = _global_marginal_base
        # ---------------------------------------------------------------------

        ## V~d臨近的位置 ds為每個被找的點與ck0之距離 sumnslocal為總共找到的已知點數 ck0為未知點位, cs與ps分別為已知點位與機率
        ## cslocal,pslocal分別為鄰近已知點座標與機率
        if _sep_i is not None:
            _comps = _sep_i['components']
            _sep_opts = {
                **_loop_options,
                'separable_axes': [
                    {
                        'coord_cols': comp.get('coord_cols', comp.get('dims', [])),
                        'dmax':       comp['dmax'],
                        'dmodel':     comp['dmodel'],
                    }
                    for comp in _comps
                ],
                'ranking_pmodel': {
                    'dmodels':   [c['dmodel'] for c in _comps],
                    'Pmodels':   [c['Pmodel'] for c in _comps],
                    'p_marginal': _sep_i.get('p_marginal'),
                    'mode':      _sep_i.get('mode', 'product'),
                },
            }
            (cslocal, pslocal, ds, sumnslocal,
             _nb_idx, _dcomp, _idx_all, _D_all) = neighbours_cat(
                _ck0_i, _cs_i, ps, nsmax, None, options=_sep_opts
            )
            _p_local_k = None
            if (_use_local_marginal or _fo_use_lm) and _idx_all.size > 0:
                _dmaxs = np.array([c['dmax'] for c in _comps], dtype=float)
                _p_local_k = local_marginal(
                    ps[_idx_all], _D_all,
                    bandwidths=_dmaxs / 3.0,
                    p_reference=_sep_i.get('p_marginal'),
                )
        else:
            cslocal, pslocal, ds, sumnslocal, _nb_idx = neighbours(
                _ck0_i, _cs_i, ps, nsmax, dmax, options=_loop_options)
            _dcomp = None
            _p_local_k = None   # local_marginal not supported in non-separable path

        # --- Determine the desired no-data-limit target q_eff(c) ---
        #
        # The ME table has p_global baked into every axis marginal.  For the
        # BME posterior to converge to a target q_eff(c) when no neighbours
        # are found, we must pass prior_to_use = q_eff / p_global (normalised)
        # into producttablepdf — the p_global in S(c) then cancels.
        #
        # Three cases based on which information sources are active:
        #
        #   Case A — external prior AND local_marginal
        #     q_eff(c) = p_local(c) × q_XGB(c) / p_global(c)   (composite)
        #     → sparse-data limit: posterior ∝ p_local × XGB-odds
        #     → uninformative XGB (q_XGB≈p_global): reduces to p_local  ✓
        #     → no local variation (p_local≈p_global): reduces to q_XGB ✓
        #
        #   Case B — local_marginal only (no external prior)
        #     q_eff(c) = p_local(c)
        #     → sparse-data limit: posterior → p_local             ✓
        #
        #   Default — external prior only (original BF)
        #     q_eff(c) = q_XGB(c)
        #     → sparse-data limit: posterior → q_XGB              ✓
        #
        # In all cases the BF denominator is always _global_marginal, which
        # is the only correct cancellation of the ME-table marginal.

        if _fo is not None:
            # --- First-order mode: build q_0 at ck0 --------------------------
            # The first-order target enters the ME table (unary constraints +
            # IPF-recentered pair targets); the posterior data step is pure
            # soft-data integration with a *uniform* pseudo-prior.  The
            # no-neighbour limit of the ME table is the product of its unary
            # constraints, so the posterior automatically converges to q_0.
            _src_ck = _fo_q_ck[i, :] if _fo_q_ck is not None else None
            if _src_ck is not None and _fo_use_lm and _p_local_k is not None:
                # Case-A composition relocated to the constraint level:
                # q_0 = p_local × (q_cov / p_global)  (odds composite)
                _odds = np.clip(_src_ck / _global_marginal, 1e-15, None)
                _q0 = _p_local_k * _odds
            elif _fo_use_lm and _p_local_k is not None:
                _q0 = _p_local_k.copy()
            elif _src_ck is not None:
                _q0 = _src_ck.copy()
            else:
                _q0 = _global_marginal.copy()
            _q0 = np.asarray(_q0, dtype=float).ravel()
            _q0 = np.clip(_q0, 1e-6, None)
            _q0 /= _q0.sum()

            prior_to_use = uniform_prior.copy().flatten()
            _fallback_pdf = _q0
        else:
            _q0 = None
            # 1. Raw desired target
            if has_array_of_priors:
                _q_ext = np.asarray(external_priors[i, :], dtype=float).ravel()
            elif external_priors is not None:
                _q_ext = np.asarray(external_priors, dtype=float).ravel()
            else:
                _q_ext = None

            if _q_ext is not None and _use_local_marginal and _p_local_k is not None:
                # Case A: composite q_eff = p_local × (q_XGB / p_global)
                _xgb_odds = _q_ext / _global_marginal          # odds ratio
                _xgb_odds = np.clip(_xgb_odds, 1e-15, None)
                _q_eff = _p_local_k * _xgb_odds
            elif _q_ext is None and _use_local_marginal and _p_local_k is not None:
                # Case B: q_eff = p_local
                _q_eff = _p_local_k.copy()
            elif _q_ext is not None:
                # Default: q_eff = q_XGB
                _q_eff = _q_ext.copy()
            elif _loop_options.get('use_neighbor_average_as_prior', False) and sumnslocal > 0:
                _q_eff = np.mean(pslocal, axis=0)
            else:
                _q_eff = uniform_prior.copy()

            _q_eff = np.asarray(_q_eff, dtype=float).ravel()
            _q_eff = np.clip(_q_eff, 1e-15, None)
            _q_eff /= _q_eff.sum()

            # 2. BF correction: prior_to_use = q_eff / p_global  (normalised)
            #    This is applied whenever we have a non-uniform target (any of the
            #    three cases above).  When q_eff == uniform we skip it (no effect).
            needs_bf = (_q_ext is not None) or (_use_local_marginal and _p_local_k is not None)
            if needs_bf:
                _bf = _q_eff / _global_marginal
                _bf = np.clip(_bf, 1e-15, None)
                prior_to_use = _bf / _bf.sum()
            else:
                prior_to_use = _q_eff   # uniform — no BF needed

            prior_to_use = prior_to_use.flatten()
            _fallback_pdf = prior_to_use
        # ---

        if sumnslocal>0:
            # Normalize pslocal to ensure each row sums to 1
            row_sums = pslocal.sum(axis=1, keepdims=True)
            # Avoid division by zero for rows that are all zero
            row_sums[row_sums == 0] = 1
            pslocal = pslocal / row_sums

            isdszero=np.where(ds==0)[0]
            if isdszero.size ==0: # check if any collocated data at ck
                test=0            # No
            else:
                ps0=pslocal[isdszero[0],:]
                pslocal = np.delete(pslocal, isdszero[0], 0)
                cslocal = np.delete(cslocal, isdszero[0], 0)
                sumnslocal=sumnslocal-1
                test=1            # If yes, remove the collocated data

            # --- Build per-point covariate values for maxentropytable ---
            me_opts = _loop_options
            if _has_cov_b:
                me_opts = dict(_loop_options)
                _cpv = {}
                nb_local = _nb_idx
                if isdszero.size > 0:
                    nb_local = np.delete(nb_local, isdszero[0])
                for lay in _cov_layers:
                    if lay.get('type') == 'interaction':
                        continue
                    nm = lay['name']
                    ck_val = _cov_ck_all[nm][i]
                    cs_local_vals = _cov_cs_all[nm][nb_local]
                    _cpv[nm] = np.concatenate([[ck_val], cs_local_vals])
                me_opts['_cov_point_values'] = _cpv

            # --- First-order mode: per-axis unary targets for the ME table ---
            if _fo is not None:
                _nb_local_fo = _nb_idx
                if isdszero.size > 0:
                    _nb_local_fo = np.delete(_nb_local_fo, isdszero[0])
                if _fo_q_cs is not None:
                    _q_nb = np.asarray(_fo_q_cs[_nb_local_fo], dtype=float)
                else:
                    # Local stationarity fallback: every neighbour axis shares
                    # q_0 (this is exactly the IPF-recentering proposal).
                    _q_nb = np.tile(_q0, (sumnslocal, 1))
                _q_unary_arr = np.vstack([_q0[None, :], _q_nb])
                _q_unary_arr = np.clip(_q_unary_arr, 1e-6, None)
                _q_unary_arr /= _q_unary_arr.sum(axis=1, keepdims=True)
                me_opts = dict(me_opts)
                me_opts['unary_constraints'] = _q_unary_arr

            try:
                Pkh_,niter = maxentropytable(np.vstack((_ck0_i,cslocal)),
                                             dmodel,
                                             Pmodel,
                                             options=me_opts)
            except Exception as e:
                # Error reporting modified to catch 'staid' if available in a test
                err_loc = f"(point {i})"
                try: err_loc = f"(station {staid}, point {i})" 
                except: pass
                
                print(f"ERROR in maxentropytable {err_loc}: {e}")
                print(f"  Estimator={options.get('estimator', 'MPL')}, nsmax={nsmax}, ndim={sumnslocal+1}")
                print("  Falling back to prior for this location.")
                pk[i,:] = _fallback_pdf
                if show_progress==1:
                    print(f'{i+1}/{nk} (FAILED)')
                continue # Skip to the next estimation point

            ## updatting probability table by given points
            # --- THIS LINE IS NOW SAFE ---
            Pkh = producttablepdf(Pkh_,np.vstack((prior_to_use,pslocal))) 
            
            # --- ROBUSTNESS CHECK 1: Numerical Underflow ---
            Pkh_sum = Pkh.sum()
            if Pkh_sum > 1e-30: # Check if the sum is non-zero
                pk[i,:] = Pkh.sum(axis = tuple(np.arange(1,sumnslocal+1))) / Pkh_sum
            else:
                # Underflow occurred (all probs ~0). Fall back to the prior.
                pk[i,:] = _fallback_pdf
            # ---
            
            # --- ROBUSTNESS CHECK 2: Collocation Test (test==1) ---
            if test==1:
                product = np.multiply(ps0, pk[i,:])
                product_sum = np.sum(product)
                
                if product_sum > 1e-30: # Check for non-zero product
                    pk[i,:] = product / product_sum
                else:
                    # Total conflict between hard data (ps0) and neighbor belief (pk).
                    # Trust the hard data.
                    pk[i,:] = ps0
            # ---
            
            del Pkh
            del Pkh_
        else:
            # No neighbors found, just use the prior (q_0 in first-order mode).
            pk[i,:] = _fallback_pdf

        if show_progress==1:
            print(f'{i+1}/{nk}') # Use print for better compatibility

    return pk

# --- START: New tune_regularization_loocv function ---
def tune_regularization_loocv(cs, ps, dmodel, Pmodel, nsmax, dmax, reg_grid, base_options=None):
    """
    Tunes the regularization hyperparameter 'reg' using Leave-One-Out (LOOCV).
    
    Predicts each data point `i` using all *other* data points and finds the
    'reg' value from `reg_grid` that maximizes the total log-likelihood
    of the "held-out" predictions.
    
    Note: This is intended for 'MLE_reg' or 'ME_dual' estimators.

    Parameters
    ----------
    cs : np.ndarray
        Data point coordinates, shape ``(n, d)``.
    ps : np.ndarray
        Soft data (PDFs) at data points, shape ``(n, nc)``.
    dmodel : np.ndarray
        1D array of distances for `Pmodel`, shape ``(nd,)``.
    Pmodel : np.ndarray
        3D array of bivariate probability tables, shape ``(nc, nc, nd)``.
    nsmax : int
        Maximum number of neighbors to use.
    dmax : float
        Maximum search distance.
    reg_grid : list or np.ndarray
        A list of 'reg' values to test (e.g., [1e-8, 1e-7, 1e-6]).
    base_options : dict, optional
        Other options for BMEcatPdf (e.g., 'tol', 'model_structure',
        'estimator'). 'estimator' should be 'MLE_reg' or 'ME_dual'.
    
    Returns
    -------
    scores : dict
        A dictionary of {reg: total_log_likelihood} scores.
    best_reg : float
        The 'reg' value from `reg_grid` with the highest score.
    """
    
    if base_options is None:
        base_options = {}
    
    if 'estimator' not in base_options:
        print("Warning: 'estimator' not in base_options. Defaulting to 'MLE_reg' for tuning.")
        base_options['estimator'] = 'MLE_reg'
    
    n_data = cs.shape[0]
    scores = {}
    
    print(f"--- Starting LOOCV Hyperparameter Tuning ---")
    print(f"Testing {n_data} data points with {len(reg_grid)} reg values...")
    
    for reg in reg_grid:
        current_options = base_options.copy()
        current_options['reg'] = reg
        total_log_likelihood = 0.0
        
        print(f"  Testing reg = {reg} ...")
        
        for i in range(n_data):
            # 1. Hold out point i
            ck_i = cs[i:i+1, :]         # The point to predict
            ps_i_known = ps[i, :]       # The known PDF at that point
            
            cs_minus_i = np.delete(cs, i, axis=0) # All other data coords
            ps_minus_i = np.delete(ps, i, axis=0) # All other data PDFs
            
            # Check if there's any data left to predict from
            if cs_minus_i.shape[0] == 0:
                continue 
                
            # 2. Predict point i using the rest of the data
            pk_i_pred = BMEcatPdf(ck_i, 
                                  cs_minus_i, 
                                  ps_minus_i, 
                                  dmodel, 
                                  Pmodel, 
                                  nsmax, 
                                  dmax, 
                                  options=current_options)
            
            pk_i_pred_vector = pk_i_pred[0, :]
            
            # 3. Score the prediction using log-likelihood
            # We clip the predicted probabilities to avoid log(0)
            pk_i_pred_vector = np.clip(pk_i_pred_vector, 1e-15, 1.0)
            
            # This is the "log score"
            score_i = np.sum(ps_i_known * np.log(pk_i_pred_vector))
            
            if not np.isnan(score_i):
                total_log_likelihood += score_i
                
        scores[reg] = total_log_likelihood
        print(f"    -> Score (Log-Likelihood): {total_log_likelihood:.4f}")
    
    # 4. Find the best reg
    best_reg = max(scores, key=scores.get)
    print(f"--- LOOCV Tuning Complete ---")
    print(f"All scores: {scores}")
    print(f"Best reg: {best_reg} (Score: {scores[best_reg]:.4f})")
    
    return scores, best_reg
# --- END: New tune_regularization_loocv function ---


def MCPcatPdf(ck, cs, ps, dmodel, Pmodel, nsmax, dmax, options=None):
    """Estimate categorical PDF using Maximum Conditional Probability (MCP).

    A fast log-space approximation of BME that assumes conditional independence
    of neighbouring data points given the estimation point.

    Parameters
    ----------
    ck : (nk, d) ndarray
        Estimation point coordinates.
    cs : (n, d) ndarray
        Data point coordinates.
    ps : (n, nc) or (n,) ndarray
        Soft data (class probabilities) or hard class labels.  When 1-D:

        * If values are in ``0 … nc-1`` they are used as column indices.
        * Otherwise the function auto-infers ``category_codes`` from
          ``np.sort(np.unique(ps))`` (assuming sorted codes match Pmodel
          row order).  Provide ``options['category_codes']`` explicitly
          for non-sorted or custom orderings.

    dmodel : (nd,) ndarray **or** list of (nd_k,) arrays
        Tabulated distances for the Pmodel.

        * **1-D mode** — a single 1-D array, shape ``(nd,)``.
        * **Separable mode** — a Python list of per-axis arrays
          ``[dmodel_h, dmodel_v, …]``.  Separable mode is detected
          automatically from this type difference (no extra option needed).

    Pmodel : (nc, nc, nd) ndarray **or** list of such arrays
        Bivariate probability tables.

        * **1-D mode** — shape ``(nc, nc, nd)``.
        * **Separable mode** — list ``[Pmodel_h, Pmodel_v, …]`` whose
          length matches ``dmodel``.

    nsmax : int
        Maximum number of neighbours per estimation point.
    dmax : float, list, or None
        Maximum search distance.

        * **1-D mode** — scalar float.  Ignored when ``auto_dmax=True``.
        * **Separable mode** — ``None`` (all axes use ``auto_dmax``), a
          scalar applied to all axes, or a list ``[dmax_h, dmax_v, …]``.

    options : dict, optional
        Algorithmic options (all have sensible defaults):

        ``'mode'`` *(str, default ``'product'``)* — separable composition rule:
        ``'product'`` (Carle–Fogg chain) or ``'additive'``.  Ignored for 1-D.

        ``'p_marginal'`` *(array, optional)* — category marginal proportions
        ``p(c)``, shape ``(nc,)``.  When absent the function auto-computes
        ``p`` from the far-field row sums of each component Pmodel and
        averages across components.  Only relevant for separable mode.

        ``'coord_cols'`` *(list of lists, optional)* — which columns of *cs*
        belong to each separable axis.  Auto-assigned by convention when
        absent: ``[[0,1],[2]]`` for 3-D (XY horizontal + Z vertical),
        ``[[0],[1]]`` for 2-D, ``[[0,1],[2],[3]]`` for 4-D.

        ``'category_codes'`` *(array-like, length nc)* — explicit label-to-column
        mapping when ``ps`` is 1-D with non-standard codes.

        ``'auto_dmax'`` *(bool, default ``True``)* — replace *dmax* with the
        effective correlation range from :func:`_find_effective_range`.

        ``'auto_dmax_threshold'`` *(float, default ``0.05``)* — TV threshold
        for ``auto_dmax``.

        ``'neighbour_ranking'`` *(str, default ``'tv'``)* — ranking method for
        separable neighbours: ``'tv'``, ``'mutual_info'``, or ``'distance'``.

        ``'prior_pdf_at_ck'``, ``'use_neighbor_average_as_prior'``,
        ``'neighborhood_structure'``, ``'nmax_per_sector'`` — see
        :func:`BMEcatPdf` for descriptions.

        ``'separable_pmodel'`` *(dict)* — legacy dict-form separable model;
        still accepted for backward compatibility.

        ``'first_order'`` *(dict or None)* — unified first-order constraint
        mode (same keys as in :func:`BMEcatPdf`: ``'q_at_ck'``,
        ``'q_at_cs'``, ``'use_local_marginal'``).  Each neighbour's bivariate
        table is IPF-recentered to margins ``(q_j, q_0)`` before the
        conditional-independence product, and the double-counting correction
        divides by ``q_0`` instead of the global marginal — the MCP analogue
        of the ME unary constraints.  The BF options ``prior_pdf_at_ck`` /
        ``local_marginal`` are ignored with a warning when active.

    Returns
    -------
    pk : (nk, nc) ndarray
        Posterior marginal PDFs at the estimation locations.
    """
    if options is None:
        options = {}

    # --- GW-Pmodel nonstationarity support -----------------------------------
    _gw = options.get('gw_pmodel', None)

    # --- Detect separable vs 1-D from dmodel/Pmodel types -------------------
    cs = np.asarray(cs, dtype=float)
    if _gw is None:
        sep_opt, dmodel, Pmodel, dmax = _resolve_pmodel(dmodel, Pmodel, dmax, cs, options)
        if sep_opt is None:
            sep_opt = options.get('separable_pmodel')
        if sep_opt is not None:
            ncat = int(sep_opt['components'][0]['Pmodel'].shape[0])
        else:
            ncat = int(Pmodel.shape[0])
    else:
        sep_opt = None
        dmodel = None
        Pmodel = None
        ncat = _gw.nc

    ps = _coerce_ps_soft(ps, ncat, options)

    # --- Auto dmax (stationary only) -----------------------------------------
    if _gw is None:
        if sep_opt is not None:
            if options.get('auto_dmax', True):
                _thresh = options.get('auto_dmax_threshold', 0.05)
                for co in sep_opt['components']:
                    co['dmax'] = _find_effective_range(
                        co['dmodel'], co['Pmodel'], threshold=_thresh)
        elif options.get('auto_dmax', True):
            _thresh = options.get('auto_dmax_threshold', 0.05)
            dmax = _find_effective_range(dmodel, Pmodel, threshold=_thresh)
    # -------------------------------------------------------------------------

    # Ensure sep_opt is visible to inner calls that read options dict
    if sep_opt is not None and 'separable_pmodel' not in options:
        options = {**options, 'separable_pmodel': sep_opt}

    n_est = ck.shape[0]
    nc = ncat

    # --- Structural marginal p(c) from the Pmodel --------------------------------
    # Used as the MCP double-counting correction: the (1-n) log p_marg term.
    # In GW mode this is blended per-point; here we set a fallback for the 
    # stationary case.
    if _gw is None:
        if sep_opt is not None:
            _mcp_marginal_base = np.asarray(sep_opt['p_marginal'], dtype=float).ravel()
        else:
            _mcp_marginal_base = Pmodel[:, :, -1].sum(axis=1).astype(float)
    # Filter out constant (non-informative) rows first so marginal is accurate
    isnotOK=np.sum(abs(np.diff(ps.T,axis = 0)),axis = 0)==0
    cs = cs[~isnotOK,:]
    ps = ps[~isnotOK,:]

    if _gw is not None:
        # GW mode: data-wide mean as fallback (overridden per-point in loop)
        _mcp_marginal_base = ps.mean(axis=0).astype(float).ravel()
    _mcp_marginal_base = np.clip(_mcp_marginal_base, 1e-15, None)
    _mcp_marginal_base /= _mcp_marginal_base.sum()
    _mcp_marginal = _mcp_marginal_base  # may be overridden per-point in GW mode

    # External prior array (nk, nc) or (nc,), or None
    _ext_prior_raw = options.get('prior_pdf_at_ck', None)
    _has_ext_array = (
        _ext_prior_raw is not None
        and isinstance(_ext_prior_raw, np.ndarray)
        and _ext_prior_raw.ndim == 2
        and _ext_prior_raw.shape == (n_est, nc)
    )
    _has_ext_prior = _ext_prior_raw is not None

    _use_local_marginal = options.get('local_marginal', False)

    # --- First-order (unary) constraint mode (MCP analogue) ------------------
    _fo = options.get('first_order', None)
    _fo_q_ck = _fo_q_cs = None
    _fo_use_lm = False
    if _fo is not None:
        if _has_ext_prior or _use_local_marginal:
            print("Warning: options['first_order'] is active — ignoring the "
                  "Bayes-factor options 'prior_pdf_at_ck' and "
                  "'local_marginal' to avoid double-counting first-order "
                  "information.")
            _ext_prior_raw = None
            _has_ext_array = False
            _has_ext_prior = False
            _use_local_marginal = False

        _fo_use_lm = bool(_fo.get('use_local_marginal', False))

        _fo_q_ck = _fo.get('q_at_ck', None)
        if _fo_q_ck is not None:
            _fo_q_ck = np.asarray(_fo_q_ck, dtype=float)
            if _fo_q_ck.ndim == 1 and _fo_q_ck.size == nc:
                _fo_q_ck = np.tile(_fo_q_ck.ravel(), (n_est, 1))
            if _fo_q_ck.shape != (n_est, nc):
                raise ValueError(
                    f"first_order['q_at_ck'] must have shape ({n_est}, {nc}) "
                    f"or ({nc},), got {_fo_q_ck.shape}")

        _fo_q_cs = _fo.get('q_at_cs', None)
        if _fo_q_cs is not None:
            _fo_q_cs = np.asarray(_fo_q_cs, dtype=float)
            if _fo_q_cs.shape[0] == isnotOK.size:
                _fo_q_cs = _fo_q_cs[~isnotOK]
            if _fo_q_cs.shape != (cs.shape[0], nc):
                raise ValueError(
                    f"first_order['q_at_cs'] must have one row per data "
                    f"point ({cs.shape[0]} after filtering), "
                    f"got {_fo_q_cs.shape}")
    # ---------------------------------------------------------------------------

    # Allocate memory for the output posterior probabilities
    pk = np.zeros((n_est, nc))

    # --- 2. Main loop over each estimation point ---
    for i in range(n_est):

        # --- Per-point GW-Pmodel blending ------------------------------------
        # The regional horizontal Pmodels are fitted on raw XY coordinates
        # (omnidirectional), so Pmodel lookups must be done with raw
        # Euclidean distances.  Do NOT apply _aniso2iso here; the anisotropy
        # parameters (theta, r) from regional fits are used only for blend-
        # weight computation in ``GWPmodel._weights``.
        if _gw is not None:
            (_dm_h, _Pm_h, _dm_v, _Pm_v,
             _theta_i, _r_i, _p_marg_i) = _gw.blend_at(ck[i, :2])
            _p_marg_i = np.asarray(_p_marg_i, dtype=float).ravel()
            _p_marg_i = np.clip(_p_marg_i, 1e-15, None)
            _p_marg_i /= _p_marg_i.sum()

            # Use raw coordinates (consistent with Pmodel fitting space)
            _cs_i  = cs
            _ck0_i = ck[i:i+1, :]

            _dmax_h = float(_dm_h[-1]) if len(_dm_h) > 0 else 1.0
            _dmax_v = float(_dm_v[-1]) if len(_dm_v) > 0 else 1.0

            # Per-point auto_dmax: shrink search radius to the effective
            # correlation range of the blended Pmodel (see BMEcatPdf for
            # the same logic).  Default ON for MCPcatPdf to match the
            # stationary branch convention.
            if options.get('auto_dmax', True):
                _thresh = options.get('auto_dmax_threshold', 0.05)
                try:
                    _dmax_h = float(_find_effective_range(
                        _dm_h, _Pm_h, threshold=_thresh))
                except Exception:
                    pass
                try:
                    _dmax_v = float(_find_effective_range(
                        _dm_v, _Pm_v, threshold=_thresh))
                except Exception:
                    pass

            _sep_i = {
                'mode': 'product',
                'p_marginal': _p_marg_i,
                'components': [
                    {'dmodel': _dm_h, 'Pmodel': _Pm_h,
                     'dmax': _dmax_h, 'coord_cols': [0, 1]},
                    {'dmodel': _dm_v, 'Pmodel': _Pm_v,
                     'dmax': _dmax_v, 'coord_cols': [2]},
                ],
            }
            _mcp_marginal = _p_marg_i   # per-point BF denominator
            _loop_options = {**options, 'separable_pmodel': _sep_i}
        else:
            _cs_i = cs
            _ck0_i = ck[i:i+1, :]
            _sep_i = sep_opt
            _mcp_marginal = _mcp_marginal_base
            _loop_options = options
        # ---------------------------------------------------------------------

        # --- 3. Find neighbors for the current estimation point ---
        if _sep_i is not None:
            _comps = _sep_i['components']
            _sep_opts = {
                **_loop_options,
                'separable_axes': [
                    {
                        'coord_cols': comp.get('coord_cols', comp.get('dims', [])),
                        'dmax':       comp['dmax'],
                        'dmodel':     comp['dmodel'],
                    }
                    for comp in _comps
                ],
                'ranking_pmodel': {
                    'dmodels':    [c['dmodel'] for c in _comps],
                    'Pmodels':    [c['Pmodel'] for c in _comps],
                    'p_marginal': _sep_i.get('p_marginal'),
                    'mode':       _sep_i.get('mode', 'product'),
                },
            }
            (csub, Zsub, dsub, nsub,
             index, d_comp, _idx_all, _D_all) = neighbours_cat(
                _ck0_i, _cs_i, ps, nsmax, None, options=_sep_opts
            )
            _p_local_k = None
            if (_use_local_marginal or _fo_use_lm) and _idx_all.size > 0:
                _dmaxs = np.array([c['dmax'] for c in _comps], dtype=float)
                _p_local_k = local_marginal(
                    ps[_idx_all], _D_all,
                    bandwidths=_dmaxs / 3.0,
                    p_reference=_sep_i.get('p_marginal'),
                )
        else:
            csub, Zsub, dsub, nsub, index = neighbours(
                _ck0_i, _cs_i, ps, nsmax, dmax, options=_loop_options
            )
            d_comp = None
            _p_local_k = None   # local_marginal not supported in non-separable path

        # --- Determine prior and double-counting marginal for this point ---------
        # The MCP posterior is:
        #   log P(c) = log q_eff(c)  −  n × log p_dc(c)  +  Σ_k log sum_term_k(c)
        #
        # p_dc MUST equal the structural marginal (_mcp_marginal) in every case.
        # It cancels the p_global already embedded in each bivariate table
        # P_biv(i, c | h).  Substituting p_local here introduces a bias factor
        # of (p_global/p_local)^n that grows exponentially with n.
        #
        # Nonstationarity enters only through q_eff (the desired no-data-limit
        # posterior).  Three cases mirror the BMEcatPdf logic:
        #
        #   Case A — external prior AND local_marginal
        #     q_eff(c) = p_local(c) × q_XGB(c) / p_global(c)   (composite)
        #
        #   Case B — local_marginal only
        #     q_eff(c) = p_local(c)
        #
        #   Default — external prior only
        #     q_eff(c) = q_XGB(c)
        #
        # When q_eff = p_dc (default, no external prior, no local_marginal) the
        # formula reduces to the classic (1-n)×log(p_marg) + Σ.

        if _fo is not None:
            # --- First-order mode (MCP analogue) ------------------------------
            # Build q_0 exactly as in BMEcatPdf, then use the *recentered*
            # bivariate tables P_x(j, 0) with margins (q_j, q_0) in the
            # conditional-independence product.  Since each recentered table
            # embeds q_0 on the X_0 axis, the double-counting correction must
            # divide by q_0 (not the global marginal) — no (p_g/p_l)^n
            # blow-up.  Posterior: log P(c) = (1-n) log q_0(c) + Σ_k log Σ_i
            # P_x(i, c | d_k) ps_k(i).
            _src_ck = _fo_q_ck[i, :] if _fo_q_ck is not None else None
            if _src_ck is not None and _fo_use_lm and _p_local_k is not None:
                _odds = np.clip(_src_ck / _mcp_marginal, 1e-15, None)
                _q0 = _p_local_k * _odds
            elif _fo_use_lm and _p_local_k is not None:
                _q0 = _p_local_k.copy()
            elif _src_ck is not None:
                _q0 = _src_ck.copy()
            else:
                _q0 = _mcp_marginal.copy()
            _q0 = np.asarray(_q0, dtype=float).ravel()
            _q0 = np.clip(_q0, 1e-6, None)
            _q0 /= _q0.sum()

            prior_to_use = _q0
            _dc_marg = _q0
        else:
            _q0 = None
            # Double-counting correction: ALWAYS the structural Pmodel marginal
            _dc_marg = _mcp_marginal

            # Raw external prior for this point (or None)
            if _has_ext_array:
                _q_ext = np.asarray(_ext_prior_raw[i, :], dtype=float).ravel()
            elif _has_ext_prior:
                _q_ext = np.asarray(_ext_prior_raw, dtype=float).ravel()
            else:
                _q_ext = None

            # Composite / fallback target q_eff
            if _q_ext is not None and _use_local_marginal and _p_local_k is not None:
                # Case A: q_eff = p_local × (q_XGB / p_global)
                _xgb_odds = _q_ext / _mcp_marginal
                _xgb_odds = np.clip(_xgb_odds, 1e-15, None)
                prior_to_use = _p_local_k * _xgb_odds
            elif _q_ext is None and _use_local_marginal and _p_local_k is not None:
                # Case B: q_eff = p_local
                prior_to_use = _p_local_k.copy()
            elif _q_ext is not None:
                # Default with external prior only
                prior_to_use = _q_ext.copy()
            elif _loop_options.get('use_neighbor_average_as_prior', False) and nsub > 0:
                prior_to_use = np.mean(Zsub, axis=0)
            else:
                # Classic MCP: q_eff = p_marg → (1-n)log(p_marg)+Σ form
                prior_to_use = _mcp_marginal.copy()

            prior_to_use = np.asarray(prior_to_use, dtype=float).ravel()
            prior_to_use = np.clip(prior_to_use, 1e-15, None)
            prior_to_use /= prior_to_use.sum()
        # -------------------------------------------------------------------------

        # --- 4. Handle the "no data" case ---
        if nsub == 0:
            pk[i, :] = prior_to_use
            continue

        # --- START: NUMERICAL STABILITY FIX (LOG-SPACE) ---

        # --- 5a. Normalize the soft data Zsub ---
        row_sums = Zsub.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        Zsub_normalized = Zsub / row_sums

        n = nsub

        # --- 5b. Log prior term (correct MCP formula) ---
        # log P(c) = log q_prior(c) − n × log p_dc(c) + Σ_k log sum_term_k(c)
        # When q_prior = p_dc (default), this reduces to (1-n)×log(p_dc) + Σ.
        log_prior_term = (
            np.log(prior_to_use + 1e-30)
            - n * np.log(_dc_marg + 1e-30)
        )

        # --- 5c. Calculate the log of the likelihood product term ---
        # This is SUM_k [ log( P(X_k | X_0) ) ]
        log_likelihood_sum = np.zeros(nc)

        # Loop over each of the 'n' neighbors
        for k in range(nsub):
            ps_k = Zsub_normalized[k, :]  # Use normalized soft data
            
            # Get the bivariate probability table p(x_k, x_0)
            if _sep_i is not None:
                d_vec = d_comp[k, :]
                comps = _sep_i['components']
                dmodels_s = [c['dmodel'] for c in comps]
                Pmodels_s = [c['Pmodel'] for c in comps]
                p_m = np.asarray(_sep_i['p_marginal'], dtype=float).ravel()
                mode_s = _sep_i.get('mode', 'product')
                biv_k0 = probamodel2bitable_separable(
                    d_vec, dmodels_s, Pmodels_s, p_m, mode=mode_s)
            else:
                d_k = dsub[k]
                biv_k0 = probamodel2bitable(d_k, dmodel, Pmodel)

            # --- First-order mode: IPF-recenter to (q_k, q_0) ---
            # Rows of biv_k0 index the neighbour X_k, columns index X_0.
            # Recentering transports the global odds ratios to the local
            # first-order margins while keeping the constraint set coherent
            # with the (1-n) log q_0 correction above.
            if _fo is not None:
                if _fo_q_cs is not None:
                    _q_k = _fo_q_cs[index[k]]
                else:
                    _q_k = _q0   # local stationarity within the neighbourhood
                biv_k0 = ipf_recenter(biv_k0, _q_k, _q0)

            # sum_term[j] = SUM_i [ P(X_k=i, X_0=j) * P_data(X_k=i) ]
            # This is the term from the original MCP paper for soft data
            # Note: The original code had biv_k0.dot(ps_k), which is a
            # theoretical error (wrong matrix axes). 
            # Biv_k0.T.dot(ps_k) is the correct formulation.
            sum_term = biv_k0.T.dot(ps_k)
            
            # Accumulate the sum of logs
            log_likelihood_sum += np.log(sum_term + 1e-30)
            
        # --- 6. Calculate the unnormalized posterior in log-space ---
        log_numerator = log_prior_term + log_likelihood_sum
        
        # --- 7. Normalize using the Log-Sum-Exp trick ---
        # This converts from log-space back to probabilities
        # without underflowing to zero.
        
        # a. Subtract the max value for numerical stability
        max_log = np.max(log_numerator)
        log_numerator_shifted = log_numerator - max_log
        
        # b. Exponentiate
        numerator_shifted = np.exp(log_numerator_shifted)
        
        # c. Normalize
        sum_num = np.sum(numerator_shifted)
        if sum_num > 1e-30:
            pk[i, :] = numerator_shifted / sum_num
        else:
            # This should no longer happen, but as a fallback:
            pk[i, :] = prior_to_use
        
        # --- END: NUMERICAL STABILITY FIX ---
            
    return pk

# --- START: NEW HYBRID BME/MCP FUNCTION ---

def HBMEcatPdf(ck,cs,ps,dmodel,Pmodel,nsmax,dmax,options = None):
    """
    Performs Hybrid BME (HBME) categorical data prediction.
    
    This function implements a two-step hybrid approach:
    1. **Regional prior (MCP step):** Run `MCPcatPdf` with up to `nsmax_total`
       neighbours, restricted to `dmax_mcp`, to build a spatially varying
       prior that captures the regional trend.
    2. **Local update (BME step):** Run `BMEcatPdf` with the `nsmax` *closest*
       neighbours and `model_structure='full'` to rigorously update the prior,
       correctly accounting for local spatial redundancy.

    Parameters
    ----------
    ck : np.ndarray
        Estimation point coordinates, shape ``(nk, d)``.
    cs : np.ndarray
        Data point coordinates, shape ``(n, d)``.
    ps : np.ndarray
        Soft data (PDFs) at data points, shape ``(n, nc)``.
    dmodel : np.ndarray
        1D array of distances for `Pmodel`, shape ``(nd,)``.
    Pmodel : np.ndarray
        3D array of bivariate probability tables, shape ``(nc, nc, nd)``.
    nsmax : int
        Number of *closest* neighbours for the final BME update step.
    dmax : float
        Maximum search distance for the BME update step (Step 2).
    options : dict, optional
        Dictionary of options:

        - ``'nsmax_total'`` (int): Maximum number of neighbours for the MCP
          prior step (Step 1). Default ``20``.

          .. warning::
             This must be kept **small** (typically 10–25).  At distances
             beyond the covariance correlation range the bivariate table
             factorises to the independence product P(c')·P(c), so each
             out-of-range neighbour contributes ``log P(c)`` to the MCP
             log-posterior.  With n decorrelated neighbours the MCP result
             degenerates to the global marginal distribution **P(c), the
             same at every estimation point**, eliminating all spatial
             variation from the prior and making the final HBME map nearly
             flat.  A good starting value is ``nsmax_total ≈ 2–3 × nsmax``.

        - ``'dmax_mcp'`` (float): Maximum search radius for the MCP prior
          step.  Defaults to ``dmax``.  **Setting this to the covariance
          correlation range is strongly recommended** so that only spatially
          correlated neighbours contribute to the prior.  Ignored when
          ``'auto_dmax_mcp'`` is ``True``.

        - ``'auto_dmax_mcp'`` (bool): If ``True``, ``dmax_mcp`` is
          **determined automatically** from ``dmodel`` and ``Pmodel``
          using :func:`_find_effective_range` with the threshold given
          by ``'auto_dmax_threshold'``.  This ensures the MCP ring radius
          equals the effective decorrelation distance in the bivariate
          model, preventing out-of-range data from contaminating the
          MCP prior.  Default ``True``.

        - ``'auto_dmax_threshold'`` (float): Decorrelation threshold
          for the automatic range detection.  The range is the distance
          at which the normalised total variation between ``Pmodel`` and
          the independence product ``p(c)*p(c')`` drops below this
          fraction.  Default ``0.05`` (5 %).

        - ``'use_annular_mcp'`` (bool): If ``True`` (default), the MCP
          prior is computed using only data in the **ring**
          ``dmax < r <= dmax_mcp``, i.e. data that lie *outside* the local
          BME update zone.  This gives strict data independence between the
          prior step and the likelihood step (no data leakage).  If
          ``False``, MCP uses all data within ``dmax_mcp`` (original
          behaviour, slightly faster but reuses data).

        - ``'global_marginal'`` (np.ndarray, shape ``(nc,)``): Empirical
          category frequencies ``p(c)``.  When provided, the MCP prior
          ``q(c)`` is passed to BMEcatPdf together with ``global_marginal``
          so that the Bayes-factor correction ``q(c)/p(c)`` is applied
          automatically, preventing double-counting of the global
          marginal.  If not provided, ``p(c)`` is estimated automatically
          from ``ps.mean(axis=0)``.

        - All other options (e.g., ``'tol'``, ``'reg'``, ``'estimator'``)
          are forwarded to the final `BMEcatPdf` update step.
        - ``'model_structure'`` is *ignored* for the BME step and forced to
          ``'full'``.

    Returns
    -------
    pk : np.ndarray
        The estimated posterior marginal PDFs, shape ``(nk, nc)``.

    Notes
    -----
    **Choosing nsmax_total and dmax_mcp:**
    Let r be the covariance correlation range.  Set ``dmax_mcp ≈ r`` and
    choose ``nsmax_total`` as the expected number of data points within that
    radius (typically 10–25 for datasets of 50–200 points).  Using too many
    decorrelated neighbours (beyond r) causes the MCP prior to converge to
    the global marginal P(c) at every location, destroying spatial variation.

    **Bayes-factor correction:**
    The Maximum Entropy table solved by `BMEcatPdf` carries an implicit
    ``p(c)`` weight (the global marginal).  Passing the MCP result ``q(c)``
    directly as a prior therefore weights the final posterior by
    ``q(c) · p(c)`` rather than ``q(c)`` alone — a subtle double-counting
    of the baseline frequency.  The Bayes-factor correction divides the
    supplied prior by ``p(c)`` before entering ``producttablepdf``, so the
    net effect is exactly ``q(c)``.  When no prior is given this correction
    is never triggered and the original BME behaviour is preserved.
    """
    
    # --- 1. Setup Options ---
    if options is None:
        options = {}

    cs = np.asarray(cs, dtype=float)
    ck = np.asarray(ck, dtype=float)

    # Detect separable (list) inputs so we can resolve ncat and effective ranges.
    # BMEcatPdf / MCPcatPdf already handle list dmodel/Pmodel via _resolve_pmodel
    # internally, so we just need the metadata here.
    _sep_detect, _, _, _ = _resolve_pmodel(dmodel, Pmodel, dmax, cs, options)
    if _sep_detect is not None:
        _ncat_hbme = int(_sep_detect['components'][0]['Pmodel'].shape[0])
    else:
        _ncat_hbme = int(Pmodel.shape[0])

    ps = _coerce_ps_soft(ps, _ncat_hbme, options)

    # Make a deep copy to avoid modifying the original dict
    bme_options = options.copy()
    mcp_options = options.copy()
    
    # Get the total neighbours for MCP (keep small – see docstring warning)
    nsmax_total = bme_options.pop('nsmax_total', 20)
    # Pop from mcp_options too so it is not forwarded unexpectedly
    mcp_options.pop('nsmax_total', None)

    # dmax for the MCP step: default to dmax but allow user override
    dmax_mcp = bme_options.pop('dmax_mcp', dmax)
    mcp_options.pop('dmax_mcp', None)

    # Auto-determine dmax_mcp from the bivariate model if requested
    _auto_dmax_mcp  = bme_options.pop('auto_dmax_mcp', True)
    mcp_options.pop('auto_dmax_mcp', None)
    _auto_thresh    = bme_options.pop('auto_dmax_threshold', 0.05)
    mcp_options.pop('auto_dmax_threshold', None)
    if _auto_dmax_mcp:
        if _sep_detect is not None:
            dmax_mcp = max(
                _find_effective_range(co['dmodel'], co['Pmodel'], threshold=_auto_thresh)
                for co in _sep_detect['components']
            )
        else:
            dmax_mcp = _find_effective_range(dmodel, Pmodel, threshold=_auto_thresh)
        if options.get('show_progress', 0):
            print(f'[HBMEcatPdf] Auto dmax_mcp = {dmax_mcp:.1f} '
                  f'(threshold={_auto_thresh:.2f})')

    # Fix 4: ensure the MCPcatPdf sub-call respects the dmax_mcp we decided
    # on (whether auto or explicit).  If auto_dmax=True is forwarded from the
    # caller via mcp_options, MCPcatPdf would re-run _find_effective_range and
    # override dmax_mcp.  We disable that here so HBMEcatPdf stays in control
    # of the MCP search radius.
    mcp_options['auto_dmax'] = False

    # Whether to restrict MCP to the annular ring (strict data independence)
    use_annular_mcp = bme_options.pop('use_annular_mcp', True)
    mcp_options.pop('use_annular_mcp', None)

    # Pseudocount for BF correction: prevents zero-count suppression when the
    # annular ring contains no samples of a particular category.
    # bf_pseudocount = 1/nc  →  adds a uniform "1/nc" weight to each category,
    # so no category can be completely zeroed out by the BF ratio.
    # Set to 0 to disable.
    _bf_pseudocount = bme_options.pop('bf_pseudocount', None)  # resolved below
    mcp_options.pop('bf_pseudocount', None)

    # Force the BME step to use the 'full' model structure
    # This is critical for handling redundancy of the 'nsmax' close points
    bme_options['model_structure'] = 'full'
    
    # Set 'show_progress' for sub-functions to 0 to avoid nested printing
    bme_options['show_progress'] = 0
    mcp_options['show_progress'] = 0
    
    # Get high-level progress tracking
    show_progress = options.get('show_progress', 0)

    # --- Determine global marginal p(c) for the Bayes-factor correction ---
    # The BF correction is injected into BMEcatPdf via 'global_marginal'.
    # Estimate p(c) from the soft data if the caller did not provide it.
    _gm = options.get('global_marginal', None)
    if _gm is None:
        _gm = ps.mean(axis=0).astype(float)
    _gm = np.asarray(_gm, dtype=float).ravel()
    _gm = np.clip(_gm, 1e-15, None)
    _gm /= _gm.sum()
    # Inject into the BME step so BMEcatPdf applies q(c)/p(c) automatically.
    bme_options['global_marginal'] = _gm
    # Also remove from mcp_options to keep it clean
    mcp_options.pop('global_marginal', None)
    # ---

    # ── Extract and isolate the per-location XGBoostLSS prior (if any) ──────
    # The outer MCPcatPdf is called one point at a time (ck0, nk=1), so the
    # shape check  _ext_prior_raw.shape == (n_est=1, nc)  evaluates as
    # False for a full (nk_total, nc) array.  The fallback path then calls
    # .ravel() on the whole array, causing a broadcast error when the
    # (nk_total * nc,) vector is divided by the (nc,) marginal.
    # Fix: strip prior_pdf_at_ck from both option dicts immediately and
    # handle the per-point XGB slice ourselves in the inner BME step below.
    _xgb_prior_array = bme_options.pop('prior_pdf_at_ck', None)
    mcp_options.pop('prior_pdf_at_ck', None)
    if _xgb_prior_array is not None:
        _xgb_prior_array = np.asarray(_xgb_prior_array, dtype=float)
        if _xgb_prior_array.ndim < 2:
            _xgb_prior_array = None   # shape (nc,) or scalar – ignore
    # ─────────────────────────────────────────────────────────────────────────

    nk = len(ck)
    ncat = _ncat_hbme
    pk=np.ones((nk,ncat))*np.nan;
    
    for i in range(nk):
        ck0=ck[[i],:]
        
        # --- 2. Step 1: Compute MCP prior -------------------------------------------
        # Option A (default, use_annular_mcp=True):
        #   Use only data in the ring  dmax < r <= dmax_mcp  so the prior
        #   data set and the BME likelihood data set are disjoint → no leakage.
        # Option B (use_annular_mcp=False):
        #   Use all data within dmax_mcp (original behaviour).
        if use_annular_mcp:
            from scipy.spatial.distance import cdist as _cdist
            _dist_all = _cdist(ck0, cs).ravel()
            # dmax may be None for separable auto_dmax; resolve a scalar boundary
            _inner_dmax = dmax
            if _inner_dmax is None:
                if _sep_detect is not None:
                    _thresh0 = options.get('auto_dmax_threshold', 0.05)
                    _inner_dmax = max(
                        _find_effective_range(co['dmodel'], co['Pmodel'],
                                              threshold=_thresh0)
                        for co in _sep_detect['components']
                    )
                else:
                    _inner_dmax = 0.0
            _ring_mask = (_dist_all > _inner_dmax) & (_dist_all <= dmax_mcp)
            cs_mcp = cs[_ring_mask, :]
            ps_mcp = ps[_ring_mask, :]
        else:
            cs_mcp = cs
            ps_mcp = ps

        if cs_mcp.shape[0] == 0:
            # No ring data: fall back to global marginal as prior
            pk_mcp_prior = _gm[np.newaxis, :]
        else:
            pk_mcp_prior = MCPcatPdf(ck0,
                                     cs_mcp,
                                     ps_mcp,
                                     dmodel,
                                     Pmodel,
                                     nsmax_total,
                                     dmax_mcp,
                                     options=mcp_options)
        # -------------------------------------------------------------------------
        
        # --- 3. Step 2: Update prior with local BME ---
        # Apply Laplace (pseudocount) smoothing to the MCP prior before passing
        # it as the BF-correction prior.  This prevents any category from being
        # completely suppressed when the annular ring is empty for that category
        # (which drives BF → 0 and eliminates the category in the final HBME).
        # Default: add 1/nc uniform weight so no entry can be exactly zero.
        _pc_val = _bf_pseudocount if _bf_pseudocount is not None else (1.0 / ncat)
        if _pc_val > 0:
            pk_mcp_smooth = pk_mcp_prior + _pc_val
            pk_mcp_smooth = pk_mcp_smooth / pk_mcp_smooth.sum(axis=1, keepdims=True)
        else:
            pk_mcp_smooth = pk_mcp_prior
        # Set the inner BME prior.
        # For HBME+composite, blend the MCP regional prior with the per-point
        # XGBoostLSS prior: q_inner(c) ∝ q_MCP(c) × q_XGB(c) / p_global(c).
        # For HBME+local (no XGB), the MCP prior is used directly.
        # Because 'global_marginal' is set in bme_options, BMEcatPdf will apply
        # the BF correction q(c)/p_global(c) automatically.
        if _xgb_prior_array is not None:
            _xgb_i = (
                _xgb_prior_array[i, :] if _xgb_prior_array.shape[0] == nk
                else _xgb_prior_array[0, :]
            )
            q_inner = pk_mcp_smooth[0, :] * (_xgb_i / np.clip(_gm, 1e-15, None))
            q_inner = np.clip(q_inner, 1e-15, None)
            q_inner /= q_inner.sum()
            bme_options['prior_pdf_at_ck'] = q_inner.reshape(1, -1)
        else:
            bme_options['prior_pdf_at_ck'] = pk_mcp_smooth
        
        pk_final = BMEcatPdf(ck0, 
                             cs, 
                             ps, 
                             dmodel, 
                             Pmodel, 
                             nsmax,
                             dmax,
                             options=bme_options)
        
        pk[i,:] = pk_final
        
        if show_progress==1:
            print(f'HBME: {i+1}/{nk}')

    return pk
