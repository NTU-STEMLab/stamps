#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prior PDF estimation utilities for categorical BME workflows.

`BMEcatPrior` mirrors the coordinate-based interface of `BMEcatPdf`: it
accepts estimation coordinates (``ck``), hard-data coordinates (``cs``), and
hard-data observations (``ps``). Feature matrices for machine-learning
priors can be passed separately via ``ck_features`` and ``cs_features``.

Two machine-learning backends are provided:

* **xgboostlss** — XGBoostLSS with a Dirichlet response distribution.  Tuning
  uses the native :func:`xgblss.hyper_opt` Optuna interface.
* **xgboost** — Standard :class:`xgboost.XGBClassifier`.  Tuning uses
  :mod:`spotpy` (SCE-UA, DE, SA, MC, ROPE, or random search).

For either backend a station-level :func:`BMEcatPriorLOOCV` cross-validation
function is provided to evaluate prediction accuracy before using the prior
inside :func:`~stamps.stamps.bme.BMEcatPdf.BMEcatPdf`.

Bug fixes vs. the previous version
-----------------------------------
B1/B2  ``_prior_xgboost`` now accepts a 1-D integer ``ps_labels`` array
       directly (no ``shape[1]`` or ``argmax``  crash).
B3     ``_tune_xgboost_spotpy`` dispatches to the correct spotpy algorithm
       based on ``cv_config["optimization_method"]``.
B4     ``SpotpyHyperparameterSetup.simulation`` uses ``GroupKFold`` keyed on
       ``station_ids`` when provided (prevents data leakage between borehole
       depth samples).
B5/B10 ``_tune_xgboostlss_native`` wraps ``xgblss.hyper_opt()``  (Optuna)
       instead of spotpy; ``n_estimators`` is now tunable.
B6     ``use_label_encoder=False`` removed from ``XGBClassifier`` (parameter
       was dropped in XGBoost ≥ 2.0).
B7     Feature importances are extracted from the final model and returned in
       ``details["feature_importances"]``.
B9     Internal parameter renamed ``ps_matrix`` → ``ps_labels`` throughout.
"""

from __future__ import annotations

import warnings
import multiprocessing
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, StratifiedKFold

# ---------------------------------------------------------------------------
# Optional dependency sentinels
# ---------------------------------------------------------------------------

try:
    from xgboostlss.distributions.Dirichlet import Dirichlet
    from xgboostlss.model import XGBoostLSS

    XGBOOSTLSS_AVAILABLE = True
except ImportError:
    XGBOOSTLSS_AVAILABLE = False

try:
    import xgboost as xgb
    from xgboost import XGBClassifier, XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import spotpy

    SPOTPY_AVAILABLE = True
except ImportError:
    SPOTPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Default parameter dictionaries
# ---------------------------------------------------------------------------

DEFAULT_XGBOOST_PARAMS: Dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3.0,
    "gamma": 0.1,
    "reg_alpha": 0.01,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
}

DEFAULT_XGBOOSTLSS_PARAMS: Dict[str, Any] = {
    "learning_rate": 0.05,
    "max_depth": 3,
    "n_estimators": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3.0,
    "gamma": 0.1,
    "reg_alpha": 0.01,
    "reg_lambda": 1.0,
    "booster": "gbtree",
    "tree_method": "hist",
}

DEFAULT_CV_CONFIG: Dict[str, Any] = {
    # Cross-validation layout
    "n_splits": 5,
    "shuffle": True,
    "random_state": None,
    # Optimization budget
    "n_trials": 50,
    "optimization_method": "sceua",   # "sceua"|"de"|"sa"|"mc"|"rope"|"random"|None
    "max_minutes": 10,                 # max wall time for xgboostlss native tuning
    # spotpy SCE-UA convergence
    "sceua_kstop": 7,
    "sceua_pcento": 0.02,
    "sceua_peps": 0.01,
    "sceua_ngs": None,
    # Misc
    "tuning_cv_splits": None,
    "use_log_scale": True,
    "show_spotpy_output": "auto",
    "random_initial_params": False,
    # XGBoostLSS Dirichlet settings
    "dirichlet_stabilization": "None",
    "dirichlet_response_fn": "softplus",
    "xgboostlss_n_estimators": 30,    # num_boost_round for native LSS tuning
}


# ---------------------------------------------------------------------------
# Internal helper utilities
# ---------------------------------------------------------------------------

def _transform_target_for_dirichlet(
    y_categorical: np.ndarray,
    categories: List[int],
    epsilon: float = 1e-4,
) -> np.ndarray:
    """Convert integer category labels to smoothed compositional data.

    Applies one-hot encoding followed by epsilon-smoothing so that no entry
    equals exactly 0 or 1 (required by the Dirichlet support).

    Parameters
    ----------
    y_categorical : np.ndarray, shape (ns,)
        Integer category labels.
    categories : list of int
        Ordered list of all possible category codes.
    epsilon : float, optional
        Smoothing constant (default 1e-4).  Each row becomes
        ``y * (1 - nc*eps) + eps`` before re-normalisation.

    Returns
    -------
    y_compositional : np.ndarray, shape (ns, nc)
        Row-normalised compositional (simplex) array.

    Raises
    ------
    ValueError
        If ``y_categorical`` is not 1-D.
    """
    if y_categorical.ndim != 1:
        raise ValueError(
            f"y_categorical must be a 1-D array, got shape {y_categorical.shape}"
        )
    nc = len(categories)
    df = pd.DataFrame({"category": y_categorical})
    one_hot = pd.get_dummies(df["category"]).reindex(columns=categories, fill_value=0)
    smoothed = one_hot * (1 - epsilon * nc) + epsilon
    compositional = smoothed.div(smoothed.sum(axis=1), axis=0)
    return compositional.values.astype(float)


def _handle_missing_values(
    cs_features: np.ndarray,
    ck_features: np.ndarray,
    strategy: str = "mean",
) -> Tuple[np.ndarray, np.ndarray]:
    """Impute NaN values in feature matrices.

    Parameters
    ----------
    cs_features : np.ndarray, shape (ns, nf)
        Training feature matrix.
    ck_features : np.ndarray, shape (nk, nf)
        Estimation feature matrix.
    strategy : str, optional
        Imputation strategy passed to :class:`sklearn.impute.SimpleImputer`
        (``"mean"``, ``"median"``, ``"most_frequent"``).  Default ``"mean"``.

    Returns
    -------
    cs_clean : np.ndarray, shape (ns, nf)
    ck_clean : np.ndarray, shape (nk, nf)
    """
    if not (np.isnan(cs_features).any() or np.isnan(ck_features).any()):
        return cs_features, ck_features
    imputer = SimpleImputer(strategy=strategy)
    cs_clean = imputer.fit_transform(cs_features)
    ck_clean = imputer.transform(ck_features)
    return cs_clean, ck_clean


def _validate_inputs(
    ck: np.ndarray,
    cs: np.ndarray,
    ps_labels: np.ndarray,
    ck_feats: np.ndarray,
    cs_feats: np.ndarray,
    categories: List[int],
) -> None:
    """Raise informative errors for common input mistakes."""
    nk, ns = ck.shape[0], cs.shape[0]
    if ck.ndim != 2:
        raise ValueError(f"ck must be 2-D, got shape {ck.shape}")
    if cs.ndim != 2:
        raise ValueError(f"cs must be 2-D, got shape {cs.shape}")
    if cs.shape[1] != ck.shape[1]:
        raise ValueError(
            f"ck and cs must have the same number of spatial dimensions, "
            f"got {ck.shape[1]} vs {cs.shape[1]}"
        )
    if ps_labels.shape[0] != ns:
        raise ValueError(
            f"ps_labels length ({ps_labels.shape[0]}) must equal cs rows ({ns})"
        )
    if ck_feats.shape[0] != nk:
        raise ValueError(
            f"ck_features rows ({ck_feats.shape[0]}) must equal ck rows ({nk})"
        )
    if cs_feats.shape[0] != ns:
        raise ValueError(
            f"cs_features rows ({cs_feats.shape[0]}) must equal cs rows ({ns})"
        )
    if ck_feats.shape[1] != cs_feats.shape[1]:
        raise ValueError(
            f"ck_features and cs_features must have the same number of columns, "
            f"got {ck_feats.shape[1]} vs {cs_feats.shape[1]}"
        )
    unknown = set(np.unique(ps_labels).tolist()) - set(categories)
    if unknown:
        raise ValueError(f"ps_labels contains unknown categories: {unknown}")


# ---------------------------------------------------------------------------
# Estimation backends
# (all accept 1-D integer ps_labels; fix for B1, B2, B9)
# ---------------------------------------------------------------------------

def _prior_xgboostlss(
    ck_features: np.ndarray,
    cs_features: np.ndarray,
    ps_labels: np.ndarray,
    categories: List[int],
    xgb_params: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
    dirichlet_stabilization: str = "None",
    dirichlet_response_fn: str = "softplus",
    return_model: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Any]]:
    """Estimate categorical prior PDFs using XGBoostLSS with Dirichlet output.

    Parameters
    ----------
    ck_features : np.ndarray, shape (nk, nf)
        Feature matrix at estimation locations.
    cs_features : np.ndarray, shape (ns, nf)
        Feature matrix at training (hard-data) locations.
    ps_labels : np.ndarray, shape (ns,)
        Integer category labels at training locations.
    categories : list of int
        Ordered list of all category codes.
    xgb_params : dict, optional
        XGBoost/XGBoostLSS parameters.  Merged with
        :data:`DEFAULT_XGBOOSTLSS_PARAMS`.
    verbose : bool, optional
        Print progress messages.  Default ``False``.
    dirichlet_stabilization : str, optional
        Stabilisation method for the Dirichlet distribution (``"None"``,
        ``"MAD"``, ``"L2"``).  Default ``"None"``.
    dirichlet_response_fn : str, optional
        Response function for Dirichlet parameters (``"softplus"``,
        ``"exp"``).  Default ``"softplus"``.
    return_model : bool, optional
        If ``True``, return ``(prior_matrix, trained_model)`` instead of
        just ``prior_matrix``.

    Returns
    -------
    prior_matrix : np.ndarray, shape (nk, nc)
        Estimated prior probabilities at estimation locations.  Each row
        sums to 1.
    model : XGBoostLSS (only when ``return_model=True``)

    Raises
    ------
    ImportError
        If ``xgboostlss`` is not installed.
    """
    if not XGBOOSTLSS_AVAILABLE:
        raise ImportError(
            "xgboostlss is not available.  Install it with: "
            "pip install xgboostlss"
        )

    nc = len(categories)
    if verbose:
        print(
            f"[XGBoostLSS] Training with {cs_features.shape[0]} → "
            f"{ck_features.shape[0]} locations, {nc} categories."
        )

    # --- target transformation ---
    y_compositional = _transform_target_for_dirichlet(ps_labels, categories)

    # --- parameters ---
    params = DEFAULT_XGBOOSTLSS_PARAMS.copy()
    if xgb_params:
        params.update(xgb_params)
    num_boost_round = int(params.pop("n_estimators", 30))

    # --- build model ---
    xgblss = XGBoostLSS(
        Dirichlet(
            D=nc,
            stabilization=dirichlet_stabilization,
            response_fn=dirichlet_response_fn,
            loss_fn="nll",
        )
    )

    n_cpu = multiprocessing.cpu_count()
    dtrain = xgb.DMatrix(cs_features, label=y_compositional, nthread=n_cpu)
    xgblss.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        verbose_eval=verbose,
    )

    # --- predict ---
    dtest = xgb.DMatrix(ck_features, nthread=n_cpu)
    pred_params = xgblss.predict(dtest, pred_type="parameters")

    alpha_arr = pred_params.values
    alpha_sum = alpha_arr.sum(axis=1, keepdims=True)
    alpha_sum[alpha_sum == 0] = 1.0
    prior_matrix = alpha_arr / alpha_sum

    if verbose:
        print("[XGBoostLSS] Prediction complete.")

    if return_model:
        return prior_matrix, xgblss
    return prior_matrix


def _prior_xgboost(
    ck_features: np.ndarray,
    cs_features: np.ndarray,
    ps_labels: np.ndarray,
    categories: List[int],
    xgb_params: Optional[Dict[str, Any]] = None,
    return_model: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Any]]:
    """Estimate categorical prior PDFs using standard XGBoost classification.

    Parameters
    ----------
    ck_features : np.ndarray, shape (nk, nf)
        Feature matrix at estimation locations.
    cs_features : np.ndarray, shape (ns, nf)
        Feature matrix at training (hard-data) locations.
    ps_labels : np.ndarray, shape (ns,)
        Integer category labels at training locations.  **Must be 1-D.**
    categories : list of int
        Ordered list of all category codes.
    xgb_params : dict, optional
        Parameters merged with :data:`DEFAULT_XGBOOST_PARAMS`.
    return_model : bool, optional
        If ``True``, return ``(prior_matrix, trained_model)``.

    Returns
    -------
    prior_matrix : np.ndarray, shape (nk, nc)
        Row-normalised predicted class probabilities.
    model : XGBClassifier (only when ``return_model=True``)

    Raises
    ------
    ImportError
        If ``xgboost`` is not installed.
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError(
            "xgboost is not available.  Install it with: pip install xgboost"
        )

    # --- impute NaN (XGBoost handles NaN natively; imputation is optional) ---
    cs_clean, ck_clean = _handle_missing_values(cs_features, ck_features)

    # --- derive n_categories from the categories list (fix B1) ---
    nc = len(categories)

    # --- parameters ---
    params = DEFAULT_XGBOOST_PARAMS.copy()
    if xgb_params:
        params.update(xgb_params)
    params.pop("objective", None)
    params.pop("num_class", None)
    params["eval_metric"] = params.get("eval_metric", "mlogloss")
    if nc <= 2:
        params["objective"] = "binary:logistic"
    else:
        params["objective"] = "multi:softprob"
        params["num_class"] = nc

    # note: use_label_encoder removed in XGBoost ≥ 2.0 (fix B6)
    params.pop("use_label_encoder", None)

    # ps_labels is already 1-D int (fix B2 — no argmax needed)
    model = XGBClassifier(**params)
    model.fit(cs_clean, ps_labels)

    raw_probs = model.predict_proba(ck_clean)
    if not np.all(np.isfinite(raw_probs)):
        raw_probs = np.nan_to_num(raw_probs, nan=0.0, posinf=1.0, neginf=0.0)

    # Map model.classes_ → column indices in our canonical categories list
    full = np.zeros((ck_clean.shape[0], nc))
    cat_to_col = {cat: col for col, cat in enumerate(categories)}
    for cls_idx, cls_val in enumerate(model.classes_):
        col = cat_to_col.get(int(cls_val), -1)
        if col >= 0:
            full[:, col] = raw_probs[:, cls_idx]

    row_sums = full.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    prior_matrix = full / row_sums

    if return_model:
        return prior_matrix, model
    return prior_matrix


def _prior_uniform(nk: int, nc: int) -> np.ndarray:
    """Return a uniform prior probability matrix.

    Parameters
    ----------
    nk : int
        Number of estimation locations.
    nc : int
        Number of categories.

    Returns
    -------
    prior_matrix : np.ndarray, shape (nk, nc)
        Every entry equals ``1 / nc``.
    """
    return np.full((nk, nc), 1.0 / nc)


def _extract_feature_importance(
    model: Any,
    feature_labels: Optional[Sequence[str]],
    nf: int,
) -> Dict[str, float]:
    """Extract normalised feature importances from a trained XGBoost model.

    Parameters
    ----------
    model : XGBClassifier | XGBoostLSS | None
        Trained model.
    feature_labels : sequence of str or None
        Human-readable feature names.  Falls back to ``["f0", "f1", ...]``
        when ``None``.
    nf : int
        Number of features (used only when feature_labels is None).

    Returns
    -------
    importance_dict : dict
        Mapping ``{feature_name: importance_weight}``.  Importances are
        normalised to sum to 1.  Returns ``{}`` if the model does not expose
        feature importances.
    """
    if feature_labels is None:
        feature_labels = [f"f{i}" for i in range(nf)]

    try:
        # XGBClassifier / XGBRegressor
        if hasattr(model, "feature_importances_"):
            imp = np.asarray(model.feature_importances_, dtype=float)
        # XGBoostLSS wraps a raw Booster
        elif hasattr(model, "booster") and hasattr(model.booster, "get_score"):
            score_dict = model.booster.get_score(importance_type="gain")
            imp = np.zeros(nf)
            for k, v in score_dict.items():
                # keys are like "f0", "f1" when no feature names set
                idx = int(k[1:]) if k.startswith("f") else -1
                if 0 <= idx < nf:
                    imp[idx] = v
        else:
            return {}
        total = imp.sum()
        if total > 0:
            imp /= total
        labels = list(feature_labels)[:nf]
        # pad if necessary
        labels += [f"f{i}" for i in range(len(labels), nf)]
        return dict(zip(labels, imp.tolist()))
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Hyperparameter tuning — XGBoostLSS native (Optuna) — fix B5/B10
# ---------------------------------------------------------------------------

def _tune_xgboostlss_native(
    cs_features: np.ndarray,
    ps_labels: np.ndarray,
    categories: List[int],
    cv_config: Dict[str, Any],
    verbose: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Tune XGBoostLSS hyperparameters using the native Optuna interface.

    Wraps :meth:`XGBoostLSS.hyper_opt` which is backed by Optuna and handles
    early stopping automatically.

    Parameters
    ----------
    cs_features : np.ndarray, shape (ns, nf)
        Training feature matrix.
    ps_labels : np.ndarray, shape (ns,)
        Integer category labels.
    categories : list of int
        Ordered category codes.
    cv_config : dict
        Cross-validation / tuning configuration.  Relevant keys:

        ``n_trials``
            Number of Optuna trials (default 50).
        ``n_splits``
            Number of CV folds (default 5).
        ``max_minutes``
            Wall-time budget in minutes (default 10).
        ``xgboostlss_n_estimators``
            ``num_boost_round`` for each trial (default 30).
        ``dirichlet_stabilization``
            Dirichlet stabilisation method.
        ``dirichlet_response_fn``
            Dirichlet response function.
    verbose : bool, optional
        Print progress.

    Returns
    -------
    best_params : dict
        Best hyperparameter dictionary (merged with defaults).
    opt_details : dict
        ``{"optimization_succeeded": bool, "n_trials": int,
           "best_score": float}``.
    """
    if not XGBOOSTLSS_AVAILABLE:
        warnings.warn(
            "xgboostlss is not available; skipping native LSS tuning."
        )
        return DEFAULT_XGBOOSTLSS_PARAMS.copy(), {"optimization_succeeded": False}

    nc = len(categories)
    n_trials = cv_config.get("n_trials", 50)
    n_splits = cv_config.get("n_splits", 5)
    max_minutes = cv_config.get("max_minutes", 10)
    num_boost_round = int(cv_config.get("xgboostlss_n_estimators", 30))
    stab = cv_config.get("dirichlet_stabilization", "None")
    resp = cv_config.get("dirichlet_response_fn", "softplus")

    if verbose:
        print(
            f"[XGBoostLSS tuning] {n_trials} Optuna trials, "
            f"{n_splits}-fold CV, max {max_minutes} min."
        )

    y_compositional = _transform_target_for_dirichlet(ps_labels, categories)
    n_cpu = multiprocessing.cpu_count()
    dtrain = xgb.DMatrix(cs_features, label=y_compositional, nthread=n_cpu)

    xgblss_tuner = XGBoostLSS(
        Dirichlet(D=nc, stabilization=stab, response_fn=resp, loss_fn="nll")
    )

    # Optuna search space — standard XGBoostLSS hyper_opt format
    param_dict = {
        "eta": ["float", {"low": 1e-5, "high": 1.0, "log": True}],
        "max_depth": ["int", {"low": 1, "high": 10, "log": False}],
        "gamma": ["float", {"low": 1e-8, "high": 40.0, "log": True}],
        "subsample": ["float", {"low": 0.2, "high": 1.0, "log": False}],
        "colsample_bytree": ["float", {"low": 0.2, "high": 1.0, "log": False}],
        "min_child_weight": ["float", {"low": 1e-8, "high": 500.0, "log": True}],
    }

    try:
        opt_params = xgblss_tuner.hyper_opt(
            param_dict,
            dtrain,
            num_boost_round=num_boost_round,
            nfold=n_splits,
            early_stopping_rounds=max(5, num_boost_round // 5),
            max_minutes=max_minutes,
            n_trials=n_trials,
            silence=not verbose,
            seed=cv_config.get("random_state", 42),
        )
        best_score = float(opt_params.pop("opt_rounds", num_boost_round))
        # merge with defaults
        final_params = DEFAULT_XGBOOSTLSS_PARAMS.copy()
        # rename eta → learning_rate if needed
        if "eta" in opt_params:
            opt_params["learning_rate"] = opt_params.pop("eta")
        final_params.update(opt_params)
        return final_params, {
            "optimization_succeeded": True,
            "n_trials": n_trials,
            "best_score": best_score,
        }
    except Exception as exc:
        warnings.warn(
            f"XGBoostLSS native tuning failed ({exc}); using defaults."
        )
        return DEFAULT_XGBOOSTLSS_PARAMS.copy(), {
            "optimization_succeeded": False,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Hyperparameter tuning — XGBoost via spotpy — fix B3/B4/B6
# ---------------------------------------------------------------------------

class SpotpyHyperparameterSetup:
    """Spotpy setup object for XGBoost hyperparameter optimisation.

    Parameters
    ----------
    X : np.ndarray, shape (ns, nf)
        Training features.
    y : np.ndarray, shape (ns,)
        Integer category labels.
    station_ids : np.ndarray, shape (ns,) or None
        Well / station identifiers used for grouped CV splits.  When
        provided, samples from the same station always stay on the same
        side of the train/test split, preventing depth-sample leakage.
    feature_labels : sequence of str or None
        Human-readable feature names.
    verbose : bool
        Print per-fold progress.
    cv_config : dict
        Cross-validation configuration dictionary.
    method : str
        ``"xgboost"`` (only; LSS now uses native tuning).
    user_search_space : dict or None
        Override ranges for specific hyperparameters.
    random_initial_params : bool
        Randomise initial spotpy parameter values.
    categories : list of int
        Ordered category codes (derived from ``y`` if not provided).
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        station_ids: Optional[np.ndarray],
        feature_labels: Optional[Sequence[str]],
        verbose: bool,
        cv_config: Dict[str, Any],
        method: str,
        user_search_space: Optional[Dict],
        random_initial_params: bool,
        categories: Optional[List[int]] = None,
    ):
        self.X = X
        self.y = y
        self.station_ids = (
            np.asarray(station_ids) if station_ids is not None else None
        )
        self.feature_labels = feature_labels
        self.verbose = verbose
        self.cv_config = cv_config or {}
        self.method = method
        self.user_search_space = user_search_space or {}
        self.random_initial_params = random_initial_params
        self.use_log_scale = self.cv_config.get("use_log_scale", True)
        self.categories = categories if categories is not None else sorted(
            np.unique(y).tolist()
        )

        self.tuning_search_space = self._define_search_space()
        self.params = self._initialize_spotpy_parameters()

        for param_name, properties in self.tuning_search_space.items():
            setattr(
                self,
                f"{param_name}_is_log_scale",
                properties.get("log", False),
            )

    def _define_search_space(self) -> Dict:
        log = self.use_log_scale
        space: Dict[str, Any] = {
            "learning_rate": {"low": 0.001, "high": 0.3, "log": log},
            "max_depth": {"low": 3, "high": 15, "log": False},
            "min_child_weight": {"low": 0.1, "high": 100.0, "log": log},
            "gamma": {"low": 1e-3, "high": 5.0, "log": log},
            "subsample": {"low": 0.5, "high": 1.0, "log": False},
            "colsample_bytree": {"low": 0.8, "high": 1.0, "log": False},
            "n_estimators": {"low": 50, "high": 500, "log": False},
        }
        for param, value in self.user_search_space.items():
            if (
                isinstance(value, dict)
                and "low" in value
                and "high" in value
            ):
                space[param] = value
        return space

    def _initialize_spotpy_parameters(self) -> List:
        params = []
        for name, props in self.tuning_search_space.items():
            low, high = props["low"], props["high"]
            is_log = props.get("log", False)
            if is_log:
                low, high = np.log10(low), np.log10(high)
            opt = (
                np.random.uniform(low, high)
                if self.random_initial_params
                else (low + high) / 2
            )
            params.append(
                spotpy.parameter.Uniform(name, low=low, high=high, optguess=opt)
            )
        return params

    def parameters(self):
        return spotpy.parameter.generate(self.params)

    def simulation(self, vector):
        """Evaluate one set of hyperparameters via CV.

        Uses :class:`sklearn.model_selection.GroupKFold` when
        ``station_ids`` is available (fix B4), falling back to
        :class:`~sklearn.model_selection.StratifiedKFold` otherwise.
        """
        raw = dict(zip([p.name for p in self.params], vector))
        current: Dict[str, Any] = {}
        for name, val in raw.items():
            is_log = getattr(self, f"{name}_is_log_scale", False)
            current[name] = 10 ** val if is_log else val
        for int_param in ("max_depth", "n_estimators"):
            if int_param in current:
                current[int_param] = int(round(current[int_param]))

        n_splits = (
            self.cv_config.get("tuning_cv_splits")
            or self.cv_config.get("n_splits", 3)
        )
        n_cpu = multiprocessing.cpu_count()
        scores: List[float] = []

        # --- choose CV splitter (fix B4) ---
        if self.station_ids is not None:
            n_groups = len(np.unique(self.station_ids))
            n_splits_eff = min(n_splits, n_groups)
            splitter = GroupKFold(n_splits=n_splits_eff)
            split_iter = splitter.split(self.X, self.y, groups=self.station_ids)
        else:
            warnings.warn(
                "station_ids not provided; using StratifiedKFold. "
                "Depth samples from the same borehole may appear on both "
                "sides of the train/test split.",
                UserWarning,
                stacklevel=1,
            )
            splitter = StratifiedKFold(
                n_splits=n_splits,
                shuffle=self.cv_config.get("shuffle", True),
                random_state=self.cv_config.get("random_state", 42),
            )
            split_iter = splitter.split(self.X, self.y)

        for train_idx, test_idx in split_iter:
            X_tr, X_te = self.X[train_idx], self.X[test_idx]
            y_tr, y_te = self.y[train_idx], self.y[test_idx]
            try:
                # note: use_label_encoder removed (fix B6)
                clf_params = {
                    k: v
                    for k, v in current.items()
                    if k != "use_label_encoder"
                }
                nc = len(self.categories)
                clf_params.setdefault(
                    "objective",
                    "binary:logistic" if nc <= 2 else "multi:softprob",
                )
                if nc > 2:
                    clf_params["num_class"] = nc
                clf_params["eval_metric"] = "mlogloss"
                clf_params["n_jobs"] = n_cpu

                model = XGBClassifier(**clf_params)
                model.fit(X_tr, y_tr, verbose=False)
                raw_probs = model.predict_proba(X_te)

                # map to canonical category columns
                cat_to_col = {cat: i for i, cat in enumerate(self.categories)}
                ordered = np.zeros((len(y_te), nc))
                for cls_i, cls_v in enumerate(model.classes_):
                    col = cat_to_col.get(int(cls_v), -1)
                    if col >= 0:
                        ordered[:, col] = raw_probs[:, cls_i]

                # NLL
                y_oh = np.eye(nc)[
                    np.searchsorted(self.categories, y_te)
                ]
                ordered = np.clip(ordered, 1e-15, 1 - 1e-15)
                fold_nll = (
                    -np.sum(y_oh * np.log(ordered)) / len(y_te)
                )
                scores.append(
                    fold_nll if np.isfinite(fold_nll) else 1e10
                )
            except Exception as exc:
                if self.verbose:
                    print(f"[SpotpySetup] CV fold failed: {exc}")
                scores.append(1e10)

        return [float(np.mean(scores)) if scores else 1e10]

    def evaluation(self):
        return None

    def objectivefunction(self, simulation, evaluation):
        """Return NLL score (to minimise)."""
        if isinstance(simulation, (list, tuple)):
            return simulation[0]
        return float(simulation)


_SPOTPY_ALGORITHM_MAP: Dict[str, str] = {
    "sceua": "sceua",
    "de": "de",
    "sa": "sa",
    "mc": "mc",
    "rope": "rope",
    "random": "mc",   # spotpy random search ≈ mc
    "dds": "dds",
}


def _tune_xgboost_spotpy(
    cs_features: np.ndarray,
    ps_labels: np.ndarray,
    categories: List[int],
    station_ids: Optional[np.ndarray],
    feature_labels: Optional[Sequence[str]],
    cv_config: Dict[str, Any],
    verbose: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Tune XGBoost hyperparameters using spotpy.

    Dispatches to the algorithm specified in
    ``cv_config["optimization_method"]`` (fix B3).  Supported values:
    ``"sceua"`` (default), ``"de"``, ``"sa"``, ``"mc"``, ``"rope"``,
    ``"random"``, ``"dds"``.

    Parameters
    ----------
    cs_features : np.ndarray, shape (ns, nf)
    ps_labels : np.ndarray, shape (ns,)
    categories : list of int
    station_ids : np.ndarray or None
    feature_labels : sequence of str or None
    cv_config : dict
    verbose : bool

    Returns
    -------
    best_params : dict
    opt_details : dict
    """
    if not SPOTPY_AVAILABLE:
        warnings.warn("spotpy is not available; skipping optimisation.")
        return DEFAULT_XGBOOST_PARAMS.copy(), {"optimization_succeeded": False}

    n_trials = cv_config.get("n_trials", 0)
    if n_trials <= 0:
        return DEFAULT_XGBOOST_PARAMS.copy(), {"optimization_succeeded": False}

    if verbose:
        print(f"[XGBoost tuning] {n_trials} spotpy trials via "
              f"'{cv_config.get('optimization_method', 'sceua')}'.")

    sid = np.asarray(station_ids) if station_ids is not None else None
    setup = SpotpyHyperparameterSetup(
        X=cs_features,
        y=ps_labels,
        station_ids=sid,
        feature_labels=feature_labels,
        verbose=verbose,
        cv_config=cv_config,
        method="xgboost",
        user_search_space=cv_config.get("search_space"),
        random_initial_params=cv_config.get("random_initial_params", False),
        categories=categories,
    )

    # --- dispatch algorithm (fix B3) ---
    algo_key = str(
        cv_config.get("optimization_method", "sceua")
    ).lower()
    algo_name = _SPOTPY_ALGORITHM_MAP.get(algo_key, "sceua")

    try:
        algo_cls = getattr(spotpy.algorithms, algo_name)
    except AttributeError:
        warnings.warn(
            f"spotpy algorithm '{algo_name}' not found; falling back to sceua."
        )
        algo_cls = spotpy.algorithms.sceua

    # Build kwargs for the chosen algorithm
    algo_kwargs: Dict[str, Any] = {
        "dbname": "hyperopt_stamps",
        "dbformat": "ram",
    }
    if algo_name == "sceua":
        if cv_config.get("sceua_ngs") is not None:
            algo_kwargs["ngs"] = int(cv_config["sceua_ngs"])

    sampler = algo_cls(setup, **algo_kwargs)

    sampler_kwargs: Dict[str, Any] = {}
    if algo_name == "sceua":
        sampler_kwargs.update(
            {
                "kstop": cv_config.get("sceua_kstop", 7),
                "pcento": cv_config.get("sceua_pcento", 0.02),
                "peps": cv_config.get("sceua_peps", 0.01),
            }
        )

    sampler.sample(repetitions=n_trials, **sampler_kwargs)
    results = sampler.getdata()

    try:
        best_set = spotpy.analyser.get_best_parameterset(
            results, maximize=False
        )
    except Exception:
        best_set = None

    if best_set is None:
        warnings.warn("spotpy optimisation returned no valid parameter set.")
        return DEFAULT_XGBOOST_PARAMS.copy(), {"optimization_succeeded": False}

    best_score = float(best_set[1]) if len(best_set) > 1 else np.nan
    raw = dict(zip([p.name for p in setup.params], best_set[0]))
    opt: Dict[str, Any] = {}
    for name, val in raw.items():
        is_log = getattr(setup, f"{name}_is_log_scale", False)
        opt[name] = 10 ** val if is_log else val
    for int_p in ("max_depth", "n_estimators"):
        if int_p in opt:
            opt[int_p] = int(round(opt[int_p]))

    final = DEFAULT_XGBOOST_PARAMS.copy()
    final.update(opt)

    return final, {
        "optimization_succeeded": True,
        "algorithm": algo_name,
        "optimized_params": opt,
        "optimized_score": best_score,
    }


# ---------------------------------------------------------------------------
# Public API — BMEcatPrior
# ---------------------------------------------------------------------------

def BMEcatPrior(
    ck: np.ndarray,
    cs: np.ndarray,
    ps: Union[np.ndarray, Sequence[int]],
    ck_features: Optional[np.ndarray] = None,
    cs_features: Optional[np.ndarray] = None,
    station_ids: Optional[Sequence[Any]] = None,
    method: str = "xgboostlss",
    hyperparams: Optional[Dict[str, Any]] = None,
    cv_config: Optional[Dict[str, Any]] = None,
    categories: Optional[Sequence[int]] = None,
    label_map: Optional[Dict[int, str]] = None,
    feature_labels: Optional[Sequence[str]] = None,
    verbose: bool = False,
    return_details: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
    """Estimate categorical prior PDFs for use in :func:`BMEcatPdf`.

    Trains a machine-learning model on categorical hard data at known
    locations (``cs``, ``ps``) and predicts class probabilities at
    estimation locations (``ck``).  The returned matrix can be passed
    directly to :func:`BMEcatPdf` via its ``external_priors`` argument.

    Parameters
    ----------
    ck : np.ndarray, shape (nk, nd)
        Estimation coordinates (rows = locations, columns = spatial dims).
    cs : np.ndarray, shape (ns, nd)
        Hard-data coordinates.
    ps : np.ndarray, shape (ns,) or (ns, nc)
        Hard-data observations.  Either a 1-D array of integer category
        codes, or a 2-D one-hot / probability matrix.
    ck_features : np.ndarray, shape (nk, nf), optional
        Ancillary feature matrix at estimation locations.  Falls back to
        ``ck`` (spatial coordinates only) when ``None``.
    cs_features : np.ndarray, shape (ns, nf), optional
        Ancillary feature matrix at training locations.  Falls back to
        ``cs`` when ``None``.
    station_ids : sequence, shape (ns,), optional
        Well / station identifiers.  Used to form grouped CV splits so
        depth samples from the same borehole never cross the
        train/test boundary.
    method : {"xgboostlss", "xgboost", "uniform"}, optional
        Prior estimation method (default ``"xgboostlss"``).

        * ``"xgboostlss"`` — Dirichlet-output gradient boosting via
          :mod:`xgboostlss`.  Tuned with native Optuna interface.
        * ``"xgboost"`` — Softmax gradient boosting via
          :class:`xgboost.XGBClassifier`.  Tuned via :mod:`spotpy`.
        * ``"uniform"`` — Flat ``1/nc`` prior (no model, no features).
    hyperparams : dict, optional
        Fixed model hyperparameters.  When provided, **no tuning** is
        performed.
    cv_config : dict, optional
        Cross-validation / tuning configuration.  Merged with
        :data:`DEFAULT_CV_CONFIG`.  Key entries:

        ``n_trials``
            Number of tuning trials (set to ``0`` to skip tuning).
        ``optimization_method``
            Algorithm for XGBoost spotpy tuning
            (``"sceua"``, ``"de"``, ``"sa"``, ``"mc"``, ``"rope"``).
        ``n_splits``
            Number of CV folds.
        ``max_minutes``
            Wall-time cap for XGBoostLSS Optuna tuning.
    categories : sequence of int, optional
        Ordered list of all category codes.  Inferred from ``ps`` when
        ``None``.
    label_map : dict {int: str}, optional
        Human-readable names for each category code, used in ``details``.
    feature_labels : sequence of str, optional
        Human-readable feature names (used for feature-importance output).
    verbose : bool, optional
        Print progress messages.  Default ``False``.
    return_details : bool, optional
        When ``True`` returns a tuple ``(prior_matrix, details)`` where
        ``details`` is a dictionary with keys ``categories``, ``labels``,
        ``method``, ``hyperparameters``, ``optimization``,
        ``feature_importances``, and ``cv_config``.

    Returns
    -------
    prior_matrix : np.ndarray, shape (nk, nc)
        Estimated prior class probabilities.  Every row sums to 1.
    details : dict
        Only returned when ``return_details=True``.

    Raises
    ------
    ValueError
        On inconsistent array shapes or unknown category codes.
    ImportError
        When the required optional library (``xgboost``, ``xgboostlss``)
        is not installed.

    Examples
    --------
    Basic usage with XGBoostLSS:

    >>> prior = BMEcatPrior(ck, cs, ps, ck_features=ck_feat,
    ...                     cs_features=cs_feat, station_ids=sids)
    >>> # Pass to BMEcatPdf:
    >>> result = BMEcatPdf(ck, cs, ps, dmodel, Pmodel, nsmax=6, dmax=8000,
    ...                    external_priors=prior)
    """
    # --- normalise inputs ---
    ck = np.asarray(ck, dtype=float)
    cs = np.asarray(cs, dtype=float)
    ps = np.asarray(ps)

    cs_feats = cs if cs_features is None else np.asarray(cs_features, dtype=float)
    ck_feats = ck if ck_features is None else np.asarray(ck_features, dtype=float)

    if ps.ndim == 1:
        ps_labels = ps.astype(int)
        if categories is None:
            categories = sorted(np.unique(ps_labels).tolist())
    else:
        # 2-D one-hot or probability matrix → derive labels
        ps_labels = np.asarray(categories)[np.argmax(ps, axis=1)].astype(int) \
            if categories is not None \
            else np.argmax(ps, axis=1).astype(int)
        if categories is None:
            categories = list(range(ps.shape[1]))

    categories = list(categories)
    nc = len(categories)

    # --- validate ---
    _validate_inputs(ck, cs, ps_labels, ck_feats, cs_feats, categories)

    # --- label map ---
    if label_map is None:
        label_map = {c: str(c) for c in categories}
    labels = [label_map.get(c, str(c)) for c in categories]

    # --- station ids ---
    sid_arr = (
        np.asarray(station_ids) if station_ids is not None else None
    )

    # --- merged cv_config ---
    merged_cv = {**DEFAULT_CV_CONFIG, **(cv_config or {})}

    details: Dict[str, Any] = {
        "method": method,
        "categories": categories,
        "labels": labels,
        "cv_config": merged_cv,
        "feature_importances": {},
    }

    if verbose:
        print(
            f"[BMEcatPrior] method={method}, nk={ck.shape[0]}, "
            f"ns={cs.shape[0]}, nc={nc}"
        )

    # ------------------------------------------------------------------ #
    # Estimation dispatch
    # ------------------------------------------------------------------ #
    if method == "uniform":
        prior_matrix = _prior_uniform(ck.shape[0], nc)

    elif method in ("xgboost", "xgboostlss"):
        # --- hyperparameter tuning ---
        if hyperparams:
            params = hyperparams
            opt_details: Dict[str, Any] = {}
        else:
            if verbose:
                print(f"[BMEcatPrior] Tuning hyperparameters "
                      f"({merged_cv.get('n_trials', 0)} trials)…")
            if method == "xgboostlss":
                params, opt_details = _tune_xgboostlss_native(
                    cs_feats, ps_labels, categories, merged_cv, verbose
                )
            else:
                params, opt_details = _tune_xgboost_spotpy(
                    cs_feats, ps_labels, categories,
                    sid_arr, feature_labels, merged_cv, verbose
                )

        details["hyperparameters"] = params
        details["optimization"] = opt_details

        # --- train final model ---
        if verbose:
            print(f"[BMEcatPrior] Training final {method} model…")

        if method == "xgboostlss":
            prior_matrix, trained_model = _prior_xgboostlss(
                ck_feats, cs_feats, ps_labels,
                categories=categories,
                xgb_params=params,
                verbose=verbose,
                dirichlet_stabilization=merged_cv.get(
                    "dirichlet_stabilization", "None"
                ),
                dirichlet_response_fn=merged_cv.get(
                    "dirichlet_response_fn", "softplus"
                ),
                return_model=True,
            )
        else:
            prior_matrix, trained_model = _prior_xgboost(
                ck_feats, cs_feats, ps_labels,
                categories=categories,
                xgb_params=params,
                return_model=True,
            )

        # --- feature importance (fix B7) ---
        details["feature_importances"] = _extract_feature_importance(
            trained_model, feature_labels, cs_feats.shape[1]
        )

    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose from "
            f"'xgboostlss', 'xgboost', 'uniform'."
        )

    if return_details:
        return prior_matrix, details
    return prior_matrix


# ---------------------------------------------------------------------------
# Public API — BMEcatPriorLOOCV
# ---------------------------------------------------------------------------

def BMEcatPriorLOOCV(
    cs: np.ndarray,
    ps: Union[np.ndarray, Sequence[int]],
    cs_features: Optional[np.ndarray] = None,
    station_ids: Optional[Sequence[Any]] = None,
    dmodel: Optional[np.ndarray] = None,
    Pmodel: Optional[np.ndarray] = None,
    nsmax: int = 6,
    dmax: float = 10_000.0,
    bme_options: Optional[Dict[str, Any]] = None,
    method: str = "xgboostlss",
    hyperparams: Optional[Dict[str, Any]] = None,
    categories: Optional[Sequence[int]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Station-level leave-one-out cross-validation of the prior model.

    For each station (or individual sample when ``station_ids`` is ``None``),
    the model is trained on all *other* stations and the held-out station's
    prior probabilities are predicted.  If ``dmodel`` and ``Pmodel`` are
    provided the prior is also passed through :func:`BMEcatPdf` to obtain
    the full posterior estimate (not yet implemented — currently returns
    prior-only LOOCV statistics).

    Parameters
    ----------
    cs : np.ndarray, shape (ns, nd)
        Hard-data coordinates.
    ps : np.ndarray, shape (ns,) or (ns, nc)
        Hard-data observations (integer labels or one-hot matrix).
    cs_features : np.ndarray, shape (ns, nf), optional
        Ancillary feature matrix.  Falls back to ``cs`` when ``None``.
    station_ids : sequence, shape (ns,), optional
        Well / station identifiers.  When provided, each station is held
        out in turn (true station-level LOO).  When ``None`` each
        individual sample is held out.
    dmodel : np.ndarray, shape (nd_bins,), optional
        Distance vector for the spatial dependence model.  Required if
        you want posterior BME statistics (not yet implemented).
    Pmodel : np.ndarray, shape (nc, nc, nd_bins), optional
        Transition probability model.  Required alongside ``dmodel``.
    nsmax : int, optional
        Maximum soft data neighbours for BME (default 6).
    dmax : float, optional
        Maximum search radius in metres (default 10 000 m).
    bme_options : dict, optional
        Additional options forwarded to ``BMEcatPdf``.
    method : str, optional
        Prior estimation method (``"xgboostlss"``, ``"xgboost"``,
        ``"uniform"``).  Default ``"xgboostlss"``.
    hyperparams : dict, optional
        Fixed model hyperparameters (no tuning performed when set).
    categories : sequence of int, optional
        Ordered category codes (inferred from ``ps`` if ``None``).
    verbose : bool, optional
        Print per-fold progress.

    Returns
    -------
    results : dict
        Dictionary with keys:

        ``accuracy`` : float
            Fraction of held-out locations correctly classified by the
            MAP category of the predicted prior.
        ``mean_log_likelihood`` : float
            Mean log-probability assigned to the true category (higher
            is better; best possible is 0).
        ``confusion_matrix`` : np.ndarray, shape (nc, nc)
            Row = true class, column = predicted MAP class.
        ``prior_probabilities`` : np.ndarray, shape (ns_unique, nc)
            Concatenated held-out prior predictions.
        ``true_labels`` : np.ndarray, shape (ns_unique,)
            True category codes for the held-out predictions.
        ``fold_details`` : list of dict
            Per-fold accuracy and NLL.

    Notes
    -----
    When ``station_ids`` is provided the number of folds equals the number
    of unique stations.  Each fold trains on ``ns_stations − 1`` stations
    and tests on the held-out station.
    """
    cs = np.asarray(cs, dtype=float)
    ps = np.asarray(ps)
    cs_feats = cs if cs_features is None else np.asarray(cs_features, dtype=float)

    if ps.ndim == 1:
        ps_labels = ps.astype(int)
        if categories is None:
            categories = sorted(np.unique(ps_labels).tolist())
    else:
        if categories is None:
            categories = list(range(ps.shape[1]))
        ps_labels = np.asarray(categories)[np.argmax(ps, axis=1)].astype(int)

    categories = list(categories)
    nc = len(categories)
    cat_arr = np.asarray(categories)

    # Build fold index: unique stations or individual samples
    if station_ids is not None:
        sid_arr = np.asarray(station_ids)
        unique_stations = np.unique(sid_arr)
        folds = [
            (np.where(sid_arr != s)[0], np.where(sid_arr == s)[0])
            for s in unique_stations
        ]
        if verbose:
            print(
                f"[BMEcatPriorLOOCV] Station-level LOO: "
                f"{len(folds)} stations."
            )
    else:
        folds = [(np.delete(np.arange(len(ps_labels)), i), np.array([i]))
                 for i in range(len(ps_labels))]
        if verbose:
            print(
                f"[BMEcatPriorLOOCV] Sample-level LOO: "
                f"{len(folds)} folds."
            )

    all_prior: List[np.ndarray] = []
    all_true: List[np.ndarray] = []
    fold_details: List[Dict[str, Any]] = []

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        if verbose:
            print(
                f"  Fold {fold_i + 1}/{len(folds)}: "
                f"train={len(train_idx)}, test={len(test_idx)}"
            )
        try:
            prior_fold = BMEcatPrior(
                ck=cs[test_idx],
                cs=cs[train_idx],
                ps=ps_labels[train_idx],
                ck_features=cs_feats[test_idx],
                cs_features=cs_feats[train_idx],
                station_ids=(
                    sid_arr[train_idx]
                    if station_ids is not None
                    else None
                ),
                method=method,
                hyperparams=hyperparams,
                cv_config={"n_trials": 0},   # no inner tuning in LOOCV
                categories=categories,
                verbose=False,
                return_details=False,
            )
            true_fold = ps_labels[test_idx]
            all_prior.append(prior_fold)
            all_true.append(true_fold)

            # per-fold stats
            map_pred = cat_arr[np.argmax(prior_fold, axis=1)]
            acc = float(np.mean(map_pred == true_fold))
            eps = 1e-15
            true_col = np.searchsorted(categories, true_fold)
            nll = float(
                -np.mean(
                    np.log(
                        np.clip(
                            prior_fold[np.arange(len(true_fold)), true_col],
                            eps, None,
                        )
                    )
                )
            )
            fold_details.append(
                {"fold": fold_i, "accuracy": acc, "nll": nll,
                 "n_test": len(test_idx)}
            )
        except Exception as exc:
            if verbose:
                print(f"  Fold {fold_i + 1} failed: {exc}")
            fold_details.append({"fold": fold_i, "error": str(exc)})

    if not all_prior:
        warnings.warn("[BMEcatPriorLOOCV] All folds failed.")
        return {
            "accuracy": np.nan,
            "mean_log_likelihood": np.nan,
            "confusion_matrix": np.full((nc, nc), np.nan),
            "prior_probabilities": np.array([]),
            "true_labels": np.array([]),
            "fold_details": fold_details,
        }

    prior_all = np.vstack(all_prior)
    true_all = np.concatenate(all_true)

    map_all = cat_arr[np.argmax(prior_all, axis=1)]
    accuracy = float(np.mean(map_all == true_all))

    eps = 1e-15
    true_cols = np.searchsorted(categories, true_all)
    mll = float(
        np.mean(
            np.log(
                np.clip(
                    prior_all[np.arange(len(true_all)), true_cols],
                    eps, None,
                )
            )
        )
    )

    # confusion matrix
    cm = np.zeros((nc, nc), dtype=int)
    true_ci = np.searchsorted(categories, true_all)
    pred_ci = np.argmax(prior_all, axis=1)
    for t, p in zip(true_ci, pred_ci):
        cm[t, p] += 1

    return {
        "accuracy": accuracy,
        "mean_log_likelihood": mll,
        "confusion_matrix": cm,
        "prior_probabilities": prior_all,
        "true_labels": true_all,
        "fold_details": fold_details,
    }


# ---------------------------------------------------------------------------
# Backward-compatibility shims  (deprecated names)
# ---------------------------------------------------------------------------

def estimate_prior_xgboostlss(
    ck_features: np.ndarray,
    cs_features: np.ndarray,
    ps_matrix: np.ndarray,
    xgb_params: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
    dirichlet_stabilization: str = "None",
    dirichlet_response_fn: str = "softplus",
) -> np.ndarray:
    """Deprecated alias for :func:`_prior_xgboostlss`.

    .. deprecated::
        Use :func:`BMEcatPrior` with ``method="xgboostlss"`` instead.
    """
    warnings.warn(
        "estimate_prior_xgboostlss is deprecated. "
        "Use BMEcatPrior(method='xgboostlss') instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    categories = sorted(np.unique(ps_matrix).tolist())
    return _prior_xgboostlss(
        ck_features=ck_features,
        cs_features=cs_features,
        ps_labels=ps_matrix,
        categories=categories,
        xgb_params=xgb_params,
        verbose=verbose,
        dirichlet_stabilization=dirichlet_stabilization,
        dirichlet_response_fn=dirichlet_response_fn,
    )


def estimate_prior_xgboost(
    ck_features: np.ndarray,
    cs_features: np.ndarray,
    ps_matrix: np.ndarray,
    xgb_params: Optional[Dict[str, Any]] = None,
    convert_to_class_labels: bool = True,
) -> np.ndarray:
    """Deprecated alias for :func:`_prior_xgboost`.

    .. deprecated::
        Use :func:`BMEcatPrior` with ``method="xgboost"`` instead.
    """
    warnings.warn(
        "estimate_prior_xgboost is deprecated. "
        "Use BMEcatPrior(method='xgboost') instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # handle both 1-D labels and 2-D one-hot (legacy callers may pass either)
    if ps_matrix.ndim == 2:
        labels = np.argmax(ps_matrix, axis=1).astype(int)
        cats = list(range(ps_matrix.shape[1]))
    else:
        labels = ps_matrix.astype(int)
        cats = sorted(np.unique(labels).tolist())

    return _prior_xgboost(
        ck_features=ck_features,
        cs_features=cs_features,
        ps_labels=labels,
        categories=cats,
        xgb_params=xgb_params,
    )


def estimate_prior_uniform(
    ck_features: np.ndarray,
    ps_matrix: np.ndarray,
) -> np.ndarray:
    """Deprecated alias for :func:`_prior_uniform`.

    .. deprecated::
        Use :func:`BMEcatPrior` with ``method="uniform"`` instead.
    """
    warnings.warn(
        "estimate_prior_uniform is deprecated. "
        "Use BMEcatPrior(method='uniform') instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    nc = ps_matrix.shape[1] if ps_matrix.ndim == 2 else len(np.unique(ps_matrix))
    return _prior_uniform(ck_features.shape[0], nc)


def estimate_prior(
    ck_features: np.ndarray,
    cs_features: np.ndarray,
    ps_matrix: np.ndarray,
    method: str = "xgboost",
    **kwargs: Any,
) -> np.ndarray:
    """Deprecated generic dispatcher.

    .. deprecated::
        Use :func:`BMEcatPrior` directly.
    """
    warnings.warn(
        "estimate_prior is deprecated. Use BMEcatPrior() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if method == "xgboostlss":
        return estimate_prior_xgboostlss(
            ck_features, cs_features, ps_matrix,
            xgb_params=kwargs.get("xgb_params"),
            verbose=kwargs.get("verbose", False),
        )
    elif method == "xgboost":
        return estimate_prior_xgboost(
            ck_features, cs_features, ps_matrix,
            xgb_params=kwargs.get("xgb_params"),
        )
    elif method == "uniform":
        return estimate_prior_uniform(ck_features, ps_matrix)
    raise ValueError(f"Unknown method: {method!r}")
