# -*- coding: utf-8 -*-
"""
stamps.general.optional — placeholders for optional dependencies
================================================================

Modules that need an optional package (matplotlib, scikit-learn, nlopt, …)
import it inside ``try/except ImportError`` and bind a ``MissingDependency``
placeholder on failure.  Importing the module therefore always succeeds;
the first attribute access or call on the placeholder raises an
``ImportError`` that names the package to install.

Example
-------
>>> try:
...     import matplotlib.pyplot as plt
... except ImportError:
...     from stamps.general.optional import MissingDependency
...     plt = MissingDependency('matplotlib')
"""

# pip distribution names where they differ from the import name
_PIP_NAMES = {
    'sklearn': 'scikit-learn',
    'rpy2': 'rpy2',
    'xgboostlss': 'xgboostlss',
}


class MissingDependency(object):
    """Stand-in for an optional package that failed to import."""

    def __init__(self, name, pip_name=None):
        self._name = name
        self._pip = pip_name or _PIP_NAMES.get(name.split('.')[0], name.split('.')[0])

    def _raise(self):
        raise ImportError(
            f"'{self._name}' is required for this function but is not installed: "
            f"pip install {self._pip}"
        )

    def __getattr__(self, item):
        if item.startswith('__'):
            raise AttributeError(item)
        self._raise()

    def __call__(self, *args, **kwargs):
        self._raise()

    def __repr__(self):
        return f"<MissingDependency {self._name}>"

    def __bool__(self):
        return False


def is_available(obj):
    """True if ``obj`` is a real module/object, False if it is a placeholder."""
    return not isinstance(obj, MissingDependency)
