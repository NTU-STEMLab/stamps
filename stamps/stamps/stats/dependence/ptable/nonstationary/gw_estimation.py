# -*- coding: utf-8 -*-
"""
stamps.stats.dependence.ptable.nonstationary.gw_estimation
===========================================================
GW-Pmodel estimation wrappers: BMEcatPdf_GW, MCPcatPdf_GW, loocv_gw.
"""
import numpy as np
from scipy.spatial.distance import cdist

def BMEcatPdf_GW(ck, cs, ps, gw_pmodel, nsmax, options=None, verbose=False):
    """Nonstationary BMEcatPdf using a Geographically Weighted Pmodel.

    .. note::
        This is a thin wrapper around :func:`~stamps.bme.BMEcatPdf.BMEcatPdf`
        with ``options['gw_pmodel'] = gw_pmodel``.  Prefer calling
        ``BMEcatPdf`` directly with the ``gw_pmodel`` option for a cleaner API::

            pk = BMEcatPdf(ck, cs, ps, None, None, nsmax, None,
                           options={**opts, 'gw_pmodel': gw})

    Parameters
    ----------
    ck : (nk, 3) array
        Estimation coordinates ``[X, Y, Z]``.
    cs : (n, 3) array
        Data coordinates ``[X, Y, Z]``.
    ps : (n, nc) array
        Soft indicators.
    gw_pmodel : GWPmodel
        Fitted nonstationary model container.
    nsmax : int
        Maximum number of neighbours for the inner BMEcatPdf call.
    options : dict or None
        Forwarded to ``BMEcatPdf``.  The ``gw_pmodel`` key is injected
        automatically; ``verbose`` controls per-point progress output via
        ``options['show_progress']``.
    verbose : bool
        Print per-point progress (sets ``show_progress=1`` in options).

    Returns
    -------
    pk : (nk, nc) array
        Posterior class probabilities.
    """
    from .....categorical.estimation import BMEcatPdf

    if options is None:
        options = {}
    opts = dict(options)
    opts['gw_pmodel'] = gw_pmodel
    if verbose:
        opts.setdefault('show_progress', 1)

    return BMEcatPdf(ck, cs, ps, None, None, nsmax, None, options=opts)


# ---------------------------------------------------------------------------
# Covariate Pmodel fitting
# ---------------------------------------------------------------------------



def loocv_gw(
    method_gw_fn,
    coord,
    value,
    station_ids,
    categories,
    region_labels,
    coord_limit_h,
    coord_limit_v=None,
    group_ids=None,
    n_regions=None,
    nsmax=8,
    options=None,
    n_fit=80,
    regularize=True,
    anisotropy=True,
    eval_stations=None,
    scale_alpha=0.2,
    verbose=True,
):
    """Station-level LOOCV for GW-Pmodel estimation.

    Each borehole is held out; the GWPmodel is rebuilt from the training data
    and prediction is made at the held-out locations.

    .. note::
        You may also pass :func:`BMEcatPdf_GW` or :func:`MCPcatPdf_GW` as
        *method_gw_fn*; both are thin wrappers that inject ``gw_pmodel``
        into the options of the underlying stationary estimator.

    Parameters
    ----------
    method_gw_fn : callable
        ``BMEcatPdf_GW`` or ``MCPcatPdf_GW`` (or any callable with the same
        signature ``fn(ck, cs, ps, gw, nsmax, options, verbose)``).
    coord : (n, 3) array
        All data coordinates ``[X, Y, Z]``.
    value : (n, nc) or (n,) array
        Soft indicators or hard labels.
    station_ids : (n,) array
        Group ID per sample.
    categories : array-like
        Ordered category codes.
    region_labels : (n,) array
        Region label per data point.
    coord_limit_h : array
        Horizontal lag bins.
    coord_limit_v : array or None
    group_ids : (n,) array or None
    n_regions : int or None
        Number of regions for ``identify_homogeneous_regions`` per fold.
    nsmax : int
    options : dict or None
    n_fit : int
    regularize, anisotropy : bool
    scale_alpha : float
        Exponent for the data-size weight in ``GWPmodel`` (default 0.2).
    verbose : bool

    Returns
    -------
    result : dict  (same keys as ``loocv_categorical``)
    """
    from .regions import identify_homogeneous_regions
    from .gw_pmodel import fit_regional_pmodels, GWPmodel
    from .....categorical.evaluate import (
        confusion_matrix as _cm, classification_report as _cr)

    coord = np.asarray(coord, dtype=float)
    categories = np.asarray(categories)
    station_ids = np.asarray(station_ids)
    nc = len(categories)

    # Coerce value to soft
    value_arr = np.asarray(value)
    if value_arr.ndim == 1:
        ps_soft = (value_arr[:, None] == categories[None, :]).astype(float)
    else:
        ps_soft = value_arr.astype(float)

    unique_st = np.unique(station_ids)
    n_folds_total = len(unique_st)

    if eval_stations is not None:
        eval_st = np.asarray(eval_stations)
    else:
        eval_st = unique_st
    n_folds = len(eval_st)

    base_opts = dict(options or {})
    base_opts['category_codes'] = list(categories)

    all_post, all_true = [], []

    for k, st in enumerate(eval_st):
        tr = station_ids != st
        te = station_ids == st

        coord_tr = coord[tr]
        ps_tr = ps_soft[tr]
        coord_te = coord[te]

        gids_tr = None
        if group_ids is not None:
            gids_tr = np.asarray(group_ids)[tr]

        # Use the pre-supplied region labels (training subset)
        rl_tr = region_labels[tr]

        # Build region_stats from training data
        unique_rl = np.unique(rl_tr)
        n_reg = len(unique_rl)
        dummy_P = np.zeros((nc, nc, len(coord_limit_h), n_reg))
        dummy_Pstd = np.zeros_like(dummy_P)
        r_stats = {
            'n_centers': np.array([int((rl_tr == lb).sum()) for lb in unique_rl]),
            'centers': [coord_tr[rl_tr == lb] for lb in unique_rl],
            'P_mean': dummy_P,
            'P_std': dummy_Pstd,
        }

        try:
            regions = fit_regional_pmodels(
                coord_tr, ps_tr, rl_tr, r_stats,
                coord_limit_h, coord_limit_v=coord_limit_v,
                group_ids=gids_tr, n_fit=n_fit,
                regularize=regularize, anisotropy=anisotropy,
                verbose=False)
            gw = GWPmodel(regions, n_fit=n_fit, scale_alpha=scale_alpha)

            post = method_gw_fn(
                coord_te, coord_tr, ps_tr, gw, nsmax,
                options=base_opts, verbose=False)
        except Exception as e:
            if verbose:
                import traceback
                traceback.print_exc()
            else:
                import warnings
                warnings.warn(
                    f'loocv_gw fold {k+1} (station={st}) failed: '
                    f'{type(e).__name__}: {e}')
            post = np.full((te.sum(), nc), 1.0 / nc)

        all_post.append(post)
        all_true.append(categories[np.argmax(ps_soft[te], axis=1)])

        if verbose:
            print(f'  Fold {k+1}/{n_folds}  station={st}  n={te.sum()}')

    posterior = np.vstack(all_post)
    y_true = np.concatenate(all_true)
    y_pred = categories[np.argmax(posterior, axis=1)]

    acc = float(np.mean(y_true == y_pred))
    from sklearn.metrics import balanced_accuracy_score
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    log_lik = []
    for j in range(len(y_true)):
        idx = np.where(categories == y_true[j])[0]
        if len(idx) > 0:
            p = max(posterior[j, idx[0]], 1e-15)
            log_lik.append(np.log(p))
    mll = float(np.mean(log_lik)) if log_lik else float('nan')

    brier = float(np.mean(np.sum(
        (posterior - (y_true[:, None] == categories[None, :]).astype(float))**2,
        axis=1)))

    cm = _cm(y_true, y_pred, categories=categories)

    return {
        'posterior': posterior,
        'y_true': y_true,
        'y_pred': y_pred,
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'mean_log_likelihood': mll,
        'brier_score': brier,
        'confusion_matrix': cm,
        'n_folds': n_folds,
        'n_folds_total': n_folds_total,
        'eval_stations': eval_st,
        'categories': categories,
    }


def MCPcatPdf_GW(ck, cs, ps, gw_pmodel, nsmax, options=None, verbose=False):
    """Nonstationary MCPcatPdf using a Geographically Weighted Pmodel.

    .. note::
        This is a thin wrapper around :func:`~stamps.bme.BMEcatPdf.MCPcatPdf`
        with ``options['gw_pmodel'] = gw_pmodel``.  Prefer calling
        ``MCPcatPdf`` directly with the ``gw_pmodel`` option::

            pk = MCPcatPdf(ck, cs, ps, None, None, nsmax, None,
                           options={**opts, 'gw_pmodel': gw})

    Parameters
    ----------
    ck : (nk, 3) array
        Estimation coordinates ``[X, Y, Z]``.
    cs : (n, 3) array
        Data coordinates ``[X, Y, Z]``.
    ps : (n, nc) array
        Soft indicators.
    gw_pmodel : GWPmodel
        Fitted nonstationary model container.
    nsmax : int
        Maximum number of neighbours.
    options : dict or None
        Forwarded to ``MCPcatPdf``; ``gw_pmodel`` key is auto-injected.
    verbose : bool
        Print per-point progress.

    Returns
    -------
    pk : (nk, nc) array
        Posterior class probabilities.
    """
    from .....categorical.estimation import MCPcatPdf

    if options is None:
        options = {}
    opts = dict(options)
    opts['gw_pmodel'] = gw_pmodel
    if verbose:
        opts.setdefault('show_progress', 1)

    return MCPcatPdf(ck, cs, ps, None, None, nsmax, None, options=opts)
