# -*- coding: utf-8 -*-
"""
stamps.stats.evaluate
=====================
Evaluation metrics for categorical and probabilistic spatial estimation.

Public symbols
--------------
confusion_matrix            Row-normalised or raw confusion matrix.
classification_report       Per-class precision, recall, F1 and support.
balanced_accuracy           Macro-averaged recall (equal class weight).
brier_score                 Mean Brier score for probabilistic predictions.
mean_log_likelihood         Mean log-likelihood for probabilistic predictions.
loocv_categorical           Station-level LOOCV for BMEcatPdf / MCPcatPdf.
"""
from ...categorical.evaluate import (  # noqa: F401
    confusion_matrix,
    plot_confusion_matrix,
    classification_report,
    balanced_accuracy,
    brier_score,
    mean_log_likelihood,
    select_loocv_stations,
    loocv_categorical,
)

__all__ = [
    "confusion_matrix",
    "plot_confusion_matrix",
    "classification_report",
    "balanced_accuracy",
    "brier_score",
    "mean_log_likelihood",
    "select_loocv_stations",
    "loocv_categorical",
]
