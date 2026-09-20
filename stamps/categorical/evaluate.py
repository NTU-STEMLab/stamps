# -*- coding: utf-8 -*-
"""
stamps.stats.evaluate.categorical
==================================
Evaluation metrics for categorical spatial estimation.

All functions accept either hard class labels (1-D integer arrays) or
probabilistic predictions (2-D float arrays, one row per location, one
column per class).  When probabilistic predictions are provided the MAP
class is derived automatically from ``argmax``.

Public API
----------
confusion_matrix        Row-normalised or raw confusion matrix.
classification_report   Per-class precision, recall, F1 and support.
balanced_accuracy       Macro-averaged recall (equal class weight).
tempered_map            Prevalence-tempered MAP decision rule.
tune_temper_tau         Grid-search the tempering exponent.
brier_score             Mean Brier score for probabilistic predictions.
mean_log_likelihood     Mean log-likelihood for probabilistic predictions.
loocv_categorical       Station-level leave-one-out cross-validation.
"""
from __future__ import annotations

import warnings
import numpy as np
from typing import Sequence, Optional, Union

__all__ = [
    "confusion_matrix",
    "classification_report",
    "balanced_accuracy",
    "tempered_map",
    "tune_temper_tau",
    "brier_score",
    "mean_log_likelihood",
    "select_loocv_stations",
    "loocv_categorical",
]

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
ArrayLike = Union[np.ndarray, list]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_labels(y_true, y_pred_or_prob, categories):
    """Return (y_true_int, y_pred_int, categories) with consistent integer codes."""
    y_true = np.asarray(y_true).ravel()

    # Probabilistic predictions → MAP class
    y_pp = np.asarray(y_pred_or_prob)
    if y_pp.ndim == 2:
        if categories is None:
            raise ValueError(
                "categories must be provided when y_pred is a 2-D probability matrix."
            )
        cats = np.asarray(categories)
        y_pred = cats[np.argmax(y_pp, axis=1)]
    else:
        y_pred = y_pp.ravel()

    if categories is None:
        categories = np.unique(np.concatenate([np.unique(y_true), np.unique(y_pred)]))
    else:
        categories = np.asarray(categories)

    return y_true, y_pred, categories


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def confusion_matrix(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    categories: Optional[ArrayLike] = None,
    normalize: Optional[str] = "true",
    labels: Optional[Sequence[str]] = None,
) -> dict:
    """Compute a confusion matrix for categorical predictions.

    Parameters
    ----------
    y_true : (n,) array-like
        True class labels (integer codes or strings).
    y_pred : (n,) or (n, nc) array-like
        Predicted class labels **or** a 2-D probability matrix (MAP is
        derived via ``argmax``).
    categories : array-like or None
        Ordered list of category codes.  Inferred from the data when
        ``None``.  Required when *y_pred* is a probability matrix.
    normalize : {'true', 'pred', 'all', None}, optional
        Normalisation mode:

        * ``'true'``  — each row sums to 1 (recall per true class).
        * ``'pred'``  — each column sums to 1 (precision per predicted class).
        * ``'all'``   — entire matrix sums to 1.
        * ``None``    — raw counts (default ``'true'``).
    labels : list of str or None
        Human-readable class names for display purposes.  Length must match
        *categories*.

    Returns
    -------
    result : dict with keys
        ``'matrix'``     — (nc, nc) ndarray, the confusion matrix.
        ``'categories'`` — 1-D array of category codes (row/column order).
        ``'labels'``     — list of label strings.
        ``'normalize'``  — the normalisation mode used.
        ``'accuracy'``   — overall accuracy (fraction correct).
        ``'n_samples'``  — total number of predictions.
    """
    y_true, y_pred, categories = _resolve_labels(y_true, y_pred, categories)
    nc = len(categories)
    cat2idx = {c: i for i, c in enumerate(categories)}

    cm = np.zeros((nc, nc), dtype=float)
    for yt, yp in zip(y_true, y_pred):
        i = cat2idx.get(yt)
        j = cat2idx.get(yp)
        if i is not None and j is not None:
            cm[i, j] += 1.0

    n_samples = int(cm.sum())
    accuracy = float(np.diag(cm).sum() / n_samples) if n_samples > 0 else float("nan")

    if normalize == "true":
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm = cm / row_sums
    elif normalize == "pred":
        col_sums = cm.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1.0
        cm = cm / col_sums
    elif normalize == "all":
        if n_samples > 0:
            cm = cm / n_samples
    elif normalize is not None:
        raise ValueError(
            f"normalize must be 'true', 'pred', 'all', or None; got {normalize!r}."
        )

    if labels is None:
        labels = [str(c) for c in categories]

    return {
        "matrix":     cm,
        "categories": categories,
        "labels":     list(labels),
        "normalize":  normalize,
        "accuracy":   accuracy,
        "n_samples":  n_samples,
    }


def plot_confusion_matrix(
    cm_result: dict,
    ax=None,
    cmap: str = "Blues",
    fmt: str = ".2f",
    title: Optional[str] = None,
    figsize: tuple = (5, 4),
):
    """Plot a confusion matrix returned by :func:`confusion_matrix`.

    Parameters
    ----------
    cm_result : dict
        Output of :func:`confusion_matrix`.
    ax : matplotlib Axes or None
        Axes to draw on.  A new figure is created when ``None``.
    cmap : str
        Colour map name.
    fmt : str
        Format string for the cell annotations.
    title : str or None
        Axes title.  Defaults to ``'Confusion matrix (acc=<accuracy>)'``.
    figsize : tuple
        Figure size (width, height) used only when *ax* is ``None``.

    Returns
    -------
    ax : matplotlib Axes
        The axes containing the plot.
    """
    import matplotlib.pyplot as plt

    cm  = cm_result["matrix"]
    lbl = cm_result["labels"]
    nc  = len(lbl)

    if ax is None:
        _fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(cm, vmin=0, vmax=(1.0 if cm_result["normalize"] else cm.max()),
                   cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.04)

    ax.set_xticks(range(nc))
    ax.set_xticklabels(lbl, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(nc))
    ax.set_yticklabels(lbl, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)

    for i in range(nc):
        for j in range(nc):
            val = cm[i, j]
            txt_color = "white" if val > (0.5 if cm_result["normalize"] else cm.max() / 2) else "black"
            ax.text(j, i, format(val, fmt), ha="center", va="center",
                    color=txt_color, fontsize=9)

    if title is None:
        acc = cm_result.get("accuracy", float("nan"))
        title = f"Confusion matrix  (acc = {acc:.3f})"
    ax.set_title(title, fontsize=10)
    return ax


# ---------------------------------------------------------------------------
# Classification report
# ---------------------------------------------------------------------------

def classification_report(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    categories: Optional[ArrayLike] = None,
    labels: Optional[Sequence[str]] = None,
    digits: int = 3,
    print_report: bool = True,
) -> dict:
    """Per-class precision, recall, F1-score and support.

    Parameters
    ----------
    y_true : (n,) array-like
        True class labels.
    y_pred : (n,) or (n, nc) array-like
        Predicted labels or probability matrix.
    categories : array-like or None
        Ordered list of category codes.
    labels : list of str or None
        Human-readable class names.
    digits : int
        Decimal places in the printed table.
    print_report : bool
        If ``True`` (default), print the formatted table to stdout.

    Returns
    -------
    report : dict
        Keys are class labels plus ``'macro avg'`` and ``'weighted avg'``.
        Each value is a dict with keys ``'precision'``, ``'recall'``,
        ``'f1'``, and ``'support'``.
    """
    y_true, y_pred, categories = _resolve_labels(y_true, y_pred, categories)
    nc = len(categories)

    if labels is None:
        labels = [str(c) for c in categories]

    cm_raw = confusion_matrix(y_true, y_pred, categories, normalize=None)["matrix"]

    report = {}
    supports = cm_raw.sum(axis=1)
    tp = np.diag(cm_raw)
    fp = cm_raw.sum(axis=0) - tp
    fn = cm_raw.sum(axis=1) - tp

    for i, lbl in enumerate(labels):
        prec = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0.0
        rec  = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0.0
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        report[lbl] = {
            "precision": float(prec),
            "recall":    float(rec),
            "f1":        float(f1),
            "support":   int(supports[i]),
        }

    total = supports.sum()
    # Macro average (equal class weight)
    macro_p = np.mean([report[l]["precision"] for l in labels])
    macro_r = np.mean([report[l]["recall"]    for l in labels])
    macro_f = np.mean([report[l]["f1"]        for l in labels])
    report["macro avg"] = {
        "precision": float(macro_p),
        "recall":    float(macro_r),
        "f1":        float(macro_f),
        "support":   int(total),
    }

    # Weighted average (weight by support)
    if total > 0:
        w = supports / total
        w_p = float(np.sum([report[l]["precision"] * w[i] for i, l in enumerate(labels)]))
        w_r = float(np.sum([report[l]["recall"]    * w[i] for i, l in enumerate(labels)]))
        w_f = float(np.sum([report[l]["f1"]        * w[i] for i, l in enumerate(labels)]))
    else:
        w_p = w_r = w_f = float("nan")
    report["weighted avg"] = {
        "precision": w_p,
        "recall":    w_r,
        "f1":        w_f,
        "support":   int(total),
    }

    if print_report:
        _print_report(report, labels, digits)

    return report


def _print_report(report, class_labels, digits):
    w = max(len(l) for l in class_labels + ["macro avg", "weighted avg"]) + 2
    d = digits
    header = f"{'':>{w}}  {'precision':>{d+4}}  {'recall':>{d+4}}  {'f1-score':>{d+4}}  {'support':>8}"
    print(header)
    print()
    for lbl in class_labels:
        r = report[lbl]
        print(f"{lbl:>{w}}  {r['precision']:>{d+4}.{d}f}  {r['recall']:>{d+4}.{d}f}  "
              f"{r['f1']:>{d+4}.{d}f}  {r['support']:>8d}")
    print()
    for agg in ["macro avg", "weighted avg"]:
        r = report[agg]
        print(f"{agg:>{w}}  {r['precision']:>{d+4}.{d}f}  {r['recall']:>{d+4}.{d}f}  "
              f"{r['f1']:>{d+4}.{d}f}  {r['support']:>8d}")


# ---------------------------------------------------------------------------
# Scalar metrics
# ---------------------------------------------------------------------------

def balanced_accuracy(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    categories: Optional[ArrayLike] = None,
) -> float:
    """Macro-averaged recall (balanced accuracy).

    Each class contributes equally regardless of its frequency.  A classifier
    that always predicts the majority class achieves ``1/nc``.

    Parameters
    ----------
    y_true : (n,) array-like
        True class labels.
    y_pred : (n,) or (n, nc) array-like
        Predicted labels or probability matrix.
    categories : array-like or None
        Ordered list of category codes.

    Returns
    -------
    bal_acc : float
        Macro-averaged recall in ``[0, 1]``.
    """
    y_true, y_pred, categories = _resolve_labels(y_true, y_pred, categories)
    recalls = []
    for c in categories:
        mask = y_true == c
        if mask.sum() == 0:
            continue
        recalls.append(float((y_pred[mask] == c).mean()))
    return float(np.mean(recalls)) if recalls else float("nan")


# ---------------------------------------------------------------------------
# Decision rules
# ---------------------------------------------------------------------------

def tempered_map(
    posterior: ArrayLike,
    p_ref: ArrayLike,
    tau: float,
    categories: Optional[ArrayLike] = None,
):
    """Prevalence-tempered MAP decision rule.

    Predicts ``argmax_c posterior(c) / p_ref(c)**tau``, discounting each
    class score by its reference prevalence.  ``tau = 0`` is the plain MAP
    (which structurally favours the majority class under imbalance);
    ``tau = 1`` fully divides out the prevalence, approximating a
    balanced-accuracy-oriented (max-likelihood-ratio) decision.  Only the
    decision changes — the posterior probabilities are untouched.

    Parameters
    ----------
    posterior : (n, nc) array-like
        Posterior class probabilities (rows need not be normalised).
    p_ref : (nc,) array-like
        Reference prevalence, e.g. the global class marginal.
    tau : float
        Tempering exponent, typically in ``[0, 1]``.
    categories : (nc,) array-like or None
        Category codes.  When given, returns labels; otherwise returns
        column indices.

    Returns
    -------
    y_pred : (n,) ndarray
        Predicted labels (or column indices when *categories* is None).
    """
    post = np.asarray(posterior, dtype=float)
    p_ref = np.asarray(p_ref, dtype=float).ravel()
    score = post / np.clip(p_ref, 1e-12, None) ** float(tau)
    idx = np.argmax(score, axis=1)
    if categories is None:
        return idx
    return np.asarray(categories)[idx]


def tune_temper_tau(
    posterior: ArrayLike,
    y_true: ArrayLike,
    p_ref: ArrayLike,
    categories: ArrayLike,
    taus: Optional[ArrayLike] = None,
    metric: str = "balanced_accuracy",
):
    """Grid-search the tempering exponent of :func:`tempered_map`.

    Selects the ``tau`` that maximises *metric* on the supplied
    (cross-validated) posteriors.  Since ``tau`` is a single scalar the
    overfitting risk is low, but for strictly unbiased reporting tune it on
    a separate validation split.

    Parameters
    ----------
    posterior : (n, nc) array-like
        Cross-validated posterior probabilities.
    y_true : (n,) array-like
        True labels.
    p_ref : (nc,) array-like
        Reference prevalence used by the decision rule.
    categories : (nc,) array-like
        Category codes matching the posterior columns.
    taus : array-like or None
        Candidate exponents.  Default ``np.linspace(0, 1.5, 31)``.
    metric : {'balanced_accuracy', 'accuracy'}
        Selection criterion.

    Returns
    -------
    tau_best : float
        The selected exponent.
    scores : dict
        ``{tau: (accuracy, balanced_accuracy)}`` for every candidate.
    """
    if taus is None:
        taus = np.linspace(0.0, 1.5, 31)
    y_true = np.asarray(y_true).ravel()
    cats = np.asarray(categories)

    scores = {}
    for tau in np.asarray(taus, dtype=float):
        y_pred = tempered_map(posterior, p_ref, tau, categories=cats)
        acc = float((y_pred == y_true).mean())
        bal = balanced_accuracy(y_true, y_pred, cats)
        scores[float(tau)] = (acc, bal)

    key = 1 if metric == "balanced_accuracy" else 0
    tau_best = max(scores, key=lambda t: scores[t][key])
    return tau_best, scores


def brier_score(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    categories: Optional[ArrayLike] = None,
) -> float:
    """Mean Brier score for probabilistic categorical predictions.

    The Brier score is the mean squared error between the predicted
    probability vector and the one-hot true class vector::

        BS = (1/n) Σ_i Σ_c (q_ic − 1[y_i = c])²

    Lower is better; 0 is perfect.

    Parameters
    ----------
    y_true : (n,) array-like
        True class labels (integer codes).
    y_prob : (n, nc) array-like
        Predicted class probabilities.  Rows must sum to 1.
    categories : array-like or None
        Ordered list of category codes matching the columns of *y_prob*.

    Returns
    -------
    bs : float
        Mean Brier score.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob, dtype=float)
    if y_prob.ndim != 2:
        raise ValueError("y_prob must be a 2-D array of shape (n, nc).")
    if categories is None:
        raise ValueError("categories must be provided when y_prob is a probability matrix.")
    categories = np.asarray(categories)
    nc = len(categories)
    if y_prob.shape[1] != nc:
        raise ValueError(
            f"y_prob has {y_prob.shape[1]} columns but categories has {nc} elements."
        )
    n = len(y_true)
    one_hot = np.zeros_like(y_prob)
    for i, yt in enumerate(y_true):
        idx = np.where(categories == yt)[0]
        if idx.size > 0:
            one_hot[i, idx[0]] = 1.0
    return float(np.mean(np.sum((y_prob - one_hot) ** 2, axis=1)))


def mean_log_likelihood(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    categories: Optional[ArrayLike] = None,
    eps: float = 1e-15,
) -> float:
    """Mean log-likelihood for probabilistic categorical predictions.

    .. math::

        \\text{MLL} = \\frac{1}{n} \\sum_i \\log p(y_i)

    Higher is better; 0 is the maximum (perfect prediction).

    Parameters
    ----------
    y_true : (n,) array-like
        True class labels.
    y_prob : (n, nc) array-like
        Predicted class probabilities.
    categories : array-like or None
        Ordered list of category codes matching the columns of *y_prob*.
    eps : float
        Minimum probability clamp to avoid log(0).

    Returns
    -------
    mll : float
        Mean log-likelihood (≤ 0).
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob, dtype=float)
    if y_prob.ndim != 2:
        raise ValueError("y_prob must be a 2-D array of shape (n, nc).")
    if categories is None:
        raise ValueError("categories must be provided.")
    categories = np.asarray(categories)
    nc = len(categories)
    if y_prob.shape[1] != nc:
        raise ValueError(
            f"y_prob has {y_prob.shape[1]} columns but categories has {nc} elements."
        )
    n = len(y_true)
    log_probs = np.empty(n)
    for i, yt in enumerate(y_true):
        idx = np.where(categories == yt)[0]
        if idx.size > 0:
            log_probs[i] = np.log(max(float(y_prob[i, idx[0]]), eps))
        else:
            log_probs[i] = np.log(eps)
    return float(np.mean(log_probs))


# ---------------------------------------------------------------------------
# Station-level LOOCV — helpers & main function
# ---------------------------------------------------------------------------

def select_loocv_stations(
    coords: np.ndarray,
    station_ids: np.ndarray,
    subsample: Union[int, float, None] = None,
) -> np.ndarray:
    """Select a spatially-systematic subset of stations for LOOCV.

    Call this **once** and pass the result to every :func:`loocv_categorical`
    call via *eval_stations* so that all models are evaluated on exactly the
    same holdout samples.

    Parameters
    ----------
    coords : (n, d) ndarray
        Data coordinates (only the first two columns are used for spatial
        sorting).
    station_ids : (n,) array-like
        Group ID per sample (e.g. borehole name or ``"X_Y"`` string).
    subsample : int, float, or None
        Controls how many stations to keep.

        * ``None`` — return all unique stations (full LOOCV).
        * ``float`` in (0, 1) — fraction of stations to keep (e.g. ``0.5``
          keeps roughly half).
        * ``int`` >= 2 — step size (e.g. ``2`` selects every 2nd station).

    Returns
    -------
    eval_stations : ndarray
        Sorted subset of unique station IDs.
    """
    coords = np.asarray(coords, dtype=float)
    station_ids = np.asarray(station_ids)
    unique_st = np.unique(station_ids)

    if subsample is None:
        return unique_st

    st_xy = np.array(
        [coords[station_ids == s, :2].mean(axis=0) for s in unique_st]
    )
    centroid = st_xy.mean(axis=0)
    angles = np.arctan2(st_xy[:, 1] - centroid[1],
                        st_xy[:, 0] - centroid[0])
    order = np.argsort(angles)
    sorted_st = unique_st[order]

    if isinstance(subsample, float) and 0 < subsample < 1:
        step = max(1, int(round(1.0 / subsample)))
    else:
        step = int(subsample)
        if step < 1:
            step = 1

    return sorted_st[::step]


def loocv_categorical(
    method_fn,
    coords: np.ndarray,
    ps: np.ndarray,
    dmodel,
    Pmodel,
    nsmax: int,
    station_ids: np.ndarray,
    categories: ArrayLike,
    options: Optional[dict] = None,
    dmax: Optional[float] = None,
    xgb_prior_fn=None,
    xgb_features: Optional[np.ndarray] = None,
    xgb_cv_trials: int = 0,
    eval_stations: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> dict:
    """Station-level leave-one-out cross-validation for categorical BME/MCP.

    Each unique station (borehole) is held out in turn.  The remaining data
    are used to predict at the held-out locations, and all metrics are
    computed over the pooled predictions.

    Parameters
    ----------
    method_fn : callable
        Estimation function — ``BMEcatPdf`` or ``MCPcatPdf``.  Must have
        signature ``method_fn(ck, cs, ps, dmodel, Pmodel, nsmax, dmax,
        options=…) → (nk, nc) ndarray``.
    coords : (n, d) ndarray
        All data coordinates (``cs`` in the full estimation).
    ps : (n,) or (n, nc) ndarray
        Hard class labels or soft probability matrix for all data locations.
    dmodel : ndarray or list of ndarray
        Distance array(s) for the Pmodel.
    Pmodel : ndarray or list of ndarray
        Probability table array(s).
    nsmax : int
        Maximum number of neighbours.
    station_ids : (n,) array-like
        Group ID per sample (e.g. borehole name or ``"X_Y"`` string).
        Each unique value defines one LOOCV fold.  Pass
        ``np.arange(n)`` for point-level LOOCV (one fold per sample).
    categories : array-like
        Ordered list of category codes.
    options : dict or None
        Options dict forwarded to *method_fn*.  ``'category_codes'`` is
        injected automatically.
    dmax : float or None
        Maximum search radius forwarded to *method_fn*.  When ``None``
        (default) the estimator uses its own internal default (typically
        unlimited).
    xgb_prior_fn : callable or None
        Optional function with signature
        ``xgb_prior_fn(ck, cs, ps_tr, ck_feat, cs_feat) → (nk, nc) ndarray``
        that generates an XGBoost prior for each fold.  When ``None``, no
        external prior is used (ignores *xgb_features*).
    xgb_features : (n, p) ndarray or None
        Feature matrix aligned with *coords* / *ps*.  Required when
        *xgb_prior_fn* is not ``None``.
    xgb_cv_trials : int
        Optuna trials for XGBoostLSS hyperparameter search within each fold
        (``0`` disables optimisation, default).
    eval_stations : ndarray or None
        Subset of station IDs to use as holdout folds.  When ``None``
        (default), all unique stations are used (full LOOCV).  Obtain a
        spatially-systematic subset from :func:`select_loocv_stations` and
        pass the same array to every model for a fair comparison.
    verbose : bool
        Print fold-level progress.

    Returns
    -------
    result : dict with keys
        ``'posterior'``          — (n, nc) pooled posterior matrix (row order
                                   follows ``unique(station_ids)`` fold order,
                                   then depth order within each fold).
        ``'y_true'``             — (n,) true labels (same row order).
        ``'y_pred'``             — (n,) MAP predicted labels.
        ``'accuracy'``           — overall accuracy.
        ``'balanced_accuracy'``  — macro-averaged recall.
        ``'mean_log_likelihood'``— mean log-likelihood.
        ``'brier_score'``        — mean Brier score.
        ``'confusion_matrix'``   — result dict from :func:`confusion_matrix`.
        ``'report'``             — result dict from :func:`classification_report`.
        ``'n_folds'``            — number of LOOCV folds.
        ``'categories'``         — category codes.
    """
    import warnings as _warnings

    categories = np.asarray(categories)
    station_ids = np.asarray(station_ids)
    ps = np.asarray(ps)
    coords = np.asarray(coords)
    unique_st = np.unique(station_ids)
    n_folds_total = len(unique_st)

    if eval_stations is not None:
        eval_st = np.asarray(eval_stations)
    else:
        eval_st = unique_st
    n_folds = len(eval_st)

    base_opts = dict(options or {})
    base_opts["category_codes"] = list(categories)

    all_post, all_true = [], []

    for k, st in enumerate(eval_st):
        tr = station_ids != st
        te = station_ids == st

        fold_opts = dict(base_opts)

        # Subset covariate values_cs to the training rows for this fold so
        # that the array length matches coords[tr] / ps[tr] inside BMEcatPdf.
        if fold_opts.get('covariate_constraints'):
            fold_opts['covariate_constraints'] = [
                {**lay, 'values_cs': np.asarray(lay['values_cs'])[tr]}
                if lay.get('type') != 'interaction'
                else dict(lay)
                for lay in fold_opts['covariate_constraints']
            ]

        if xgb_prior_fn is not None and xgb_features is not None:
            with _warnings.catch_warnings():
                _warnings.filterwarnings("ignore")
                ext_prior = xgb_prior_fn(
                    coords[te], coords[tr], ps[tr],
                    xgb_features[te], xgb_features[tr],
                    xgb_cv_trials,
                )
            fold_opts["prior_pdf_at_ck"] = ext_prior

        post = method_fn(
            coords[te], coords[tr], ps[tr],
            dmodel, Pmodel, nsmax, dmax,
            options=fold_opts,
        )
        all_post.append(post)
        # Collect hard labels: handle both hard (1-D) and soft (2-D) ps
        if ps.ndim == 2:
            all_true.append(categories[np.argmax(ps[te], axis=1)])
        else:
            all_true.append(ps[te])

        if verbose:
            print(f"  Fold {k + 1}/{n_folds}  station={st}  n={te.sum()}")

    post_all = np.vstack(all_post)
    true_all = np.concatenate(all_true)
    pred_all = categories[np.argmax(post_all, axis=1)]

    acc     = float(np.mean(pred_all == true_all))
    bal_acc = balanced_accuracy(true_all, pred_all, categories)
    mll     = mean_log_likelihood(true_all, post_all, categories)
    bs      = brier_score(true_all, post_all, categories)

    cm_res  = confusion_matrix(true_all, pred_all, categories, normalize="true")
    rep     = classification_report(true_all, pred_all, categories, print_report=False)

    return {
        "posterior":           post_all,
        "y_true":              true_all,
        "y_pred":              pred_all,
        "accuracy":            acc,
        "balanced_accuracy":   bal_acc,
        "mean_log_likelihood": mll,
        "brier_score":         bs,
        "confusion_matrix":    cm_res,
        "report":              rep,
        "n_folds":             n_folds,
        "n_folds_total":       n_folds_total,
        "eval_stations":       eval_st,
        "categories":          categories,
    }


def loocv_categorical_points(
    method_fn,
    coords: np.ndarray,
    ps: np.ndarray,
    dmodel,
    Pmodel,
    nsmax: int,
    dmax: Optional[float],
    categories: ArrayLike,
    options: Optional[dict] = None,
    verbose: bool = False,
) -> dict:
    """Point-level leave-one-out cross-validation for categorical estimators.

    Convenience wrapper around :func:`loocv_categorical` for the common case
    where each sample is its own LOOCV fold (i.e. a unique-location dataset
    with no repeat-station structure).

    Parameters
    ----------
    method_fn : callable
        Estimation function — ``BMEcatPdf``, ``MCPcatPdf``, or
        ``HBMEcatPdf``.
    coords : (n, d) ndarray
        Data coordinates.
    ps : (n, nc) ndarray
        Soft probability matrix (one-hot for hard data).
    dmodel : ndarray
        Distance axis for the Pmodel.
    Pmodel : (nc, nc, nd) ndarray
        Probability table.
    nsmax : int
        Maximum number of neighbours per estimation node.
    dmax : float
        Search radius passed to *method_fn*.
    categories : array-like
        Ordered class codes.
    options : dict or None
        Extra options forwarded to *method_fn*.
    verbose : bool
        Print fold progress (default ``False``).

    Returns
    -------
    result : dict
        Same keys as :func:`loocv_categorical`:
        ``'accuracy'``, ``'mean_log_likelihood'``, ``'brier_score'``,
        ``'balanced_accuracy'``, ``'y_true'``, ``'y_pred'``,
        ``'posterior'``, ``'confusion_matrix'``, ``'report'``,
        ``'n_folds'``, ``'categories'``.

    Examples
    --------
    >>> result = loocv_categorical_points(
    ...     MCPcatPdf, cs, ps, dmodel, Pmodel,
    ...     nsmax=8, dmax=8000, categories=[1, 2, 3, 4])
    >>> print(f"Accuracy: {result['accuracy']:.3f}")
    """
    coords = np.asarray(coords)
    n = coords.shape[0]
    station_ids = np.arange(n)
    return loocv_categorical(
        method_fn,
        coords, ps, dmodel, Pmodel,
        nsmax, station_ids, categories,
        options=options,
        dmax=dmax,
        verbose=verbose,
    )
