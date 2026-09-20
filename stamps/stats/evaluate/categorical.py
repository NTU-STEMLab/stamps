# -*- coding: utf-8 -*-
"""
Backward-compatibility shim — moved to :mod:`stamps.categorical.evaluate`.
"""
import warnings as _warnings

_warnings.warn(
    "Importing from 'stamps.stats.evaluate.categorical' is deprecated. "
    "Use 'stamps.categorical.evaluate' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from ...categorical.evaluate import *  # noqa: F401,F403
from ...categorical.evaluate import (  # noqa: F401 – explicit names for IDEs
    confusion_matrix,
    plot_confusion_matrix,
    classification_report,
    balanced_accuracy,
    brier_score,
    mean_log_likelihood,
    select_loocv_stations,
    loocv_categorical,
)
