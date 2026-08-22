# -*- coding: utf-8 -*-
"""
stamps.general.read_geoeas
==========================
Parser for the GeoEAS flat-file format.

The GeoEAS format is a simple ASCII tabular format used extensively in
geostatistical software (GSLIB, BMELIB, etc.).  Each file has the structure::

    <title line>
    <number of columns n>
    <column name 1>
    ...
    <column name n>
    <data row 1>
    ...

"""
from __future__ import annotations

import numpy as np

__all__ = ["read_geoeas"]


def read_geoeas(filepath: str):
    """Parse a GeoEAS text file.

    Parameters
    ----------
    filepath : str or path-like
        Path to the ``.txt`` / ``.dat`` file in GeoEAS format.

    Returns
    -------
    title : str
        Title string from the first line.
    col_names : list of str
        Column names (one per column).
    data : np.ndarray, shape (n_rows, n_cols)
        Numeric data block as a float64 array.

    Examples
    --------
    >>> title, col_names, data = read_geoeas('Falmagne.txt')
    >>> print(title)
    Falmagne soil data
    >>> print(col_names)
    ['X', 'Y', 'sand', 'silt', 'clay', 'soil type']
    >>> print(data.shape)
    (118, 6)
    """
    with open(filepath) as fh:
        lines = fh.readlines()
    title = lines[0].strip()
    ncols = int(lines[1].strip())
    col_names = [lines[2 + k].strip() for k in range(ncols)]
    data = np.array(
        [list(map(float, ln.split()))
         for ln in lines[2 + ncols:]
         if ln.strip()]
    )
    return title, col_names, data
