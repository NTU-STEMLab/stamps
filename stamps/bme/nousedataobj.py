# -*- coding: utf-8 -*-
"""No-op progress reporter used when a function is called without the STAR-BME GUI.

Several estimation routines (e.g. ``estimation.kernelsmoothing``) accept an optional
``DataObj`` for GUI progress reporting and fall back to ``NoUseDataObj`` when it is
``None``.
"""


class NoUseDataObj:
    """Implements the DataObj progress interface and does nothing."""

    def __init__(self, text=''):
        self._text = text

    def getProgressText(self):
        return self._text

    def setProgressRange(self, *args, **kwargs):
        pass

    def setCurrentProgress(self, *args, **kwargs):
        pass

    def wasProgressCanceled(self):
        return False
