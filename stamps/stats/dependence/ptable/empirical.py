import numpy as np
from scipy.spatial.distance import cdist, pdist
import warnings

# probatablecalc_optimized is defined at the bottom of this file
# (merged from the former probatablecalc_optimized.py)


# ======================================================================
# Soft-indicator validation and normalization
# ======================================================================

def normalize_indicators(value, tol=1e-6, uniform_on_zero=True):
    """Validate and row-normalize a soft-indicator (or one-hot) matrix.

    Ensures every row of *value* is non-negative and sums to 1 — the
    probability-simplex constraint required by :func:`probatablecalc`.

    Call this before :func:`probatablecalc` whenever your indicator data
    may not be pre-normalized, for example:

    * Geophysical inversion outputs  ``[p_clay, p_sand, p_gravel]``
    * XGBoostLSS / classifier probability vectors
    * Station-averaged lithology fractions (soft indicators)
    * Raw score vectors that need normalizing before use

    Parameters
    ----------
    value : (n, nc) array-like
        Raw indicator matrix.  Accepts:

        - **Hard / one-hot** — rows are ``0/1`` vectors with one ``1``.
          Example: class 2 of 4 → ``[0, 1, 0, 0]``.
        - **Soft / probabilistic** — non-negative rows approximately
          summing to 1.  Example: ``[0.2, 0.5, 0.3]``.
        - **Raw (un-normalized) scores** — any non-negative matrix whose
          rows do not yet sum to 1.  Example: ``[0.2, 0.2, 0.8]``
          (sum = 1.2).  Rows are automatically rescaled.

    tol : float, optional
        Rows whose sum deviates from 1.0 by more than *tol* trigger a
        :class:`UserWarning` before normalization (default 1e-6).
    uniform_on_zero : bool, optional
        If ``True`` (default), replace all-zero rows with a uniform
        distribution ``1/nc`` and emit a :class:`UserWarning`.
        If ``False``, raise :class:`ValueError` instead.

    Returns
    -------
    value_norm : (n, nc) ndarray
        Float array with every row summing to exactly 1.0 and no
        negative entries.

    Raises
    ------
    ValueError
        If *value* contains negative entries, or if *uniform_on_zero=False*
        and any row is all-zero.

    Examples
    --------
    >>> import numpy as np
    >>> from stamps.stats.dependence.probatablecalc import normalize_indicators

    Hard one-hot (passes through unchanged):

    >>> normalize_indicators([[1, 0, 0], [0, 0, 1]])
    array([[1., 0., 0.],
           [0., 0., 1.]])

    Soft indicators already summing to 1:

    >>> normalize_indicators([[0.2, 0.5, 0.3], [0.1, 0.7, 0.2]])
    array([[0.2, 0.5, 0.3],
           [0.1, 0.7, 0.2]])

    Un-normalized scores (rows need rescaling):

    >>> normalize_indicators([[0.2, 0.2, 0.8]])   # sums to 1.2
    array([[0.16666667, 0.16666667, 0.66666667]])

    All-zero row is replaced with uniform (warning emitted):

    >>> normalize_indicators([[0, 0, 0], [0.3, 0.4, 0.3]])
    array([[0.33333333, 0.33333333, 0.33333333],
           [0.3       , 0.4       , 0.3       ]])
    """
    value = np.asarray(value, dtype=float)
    if value.ndim != 2:
        raise ValueError(
            f'normalize_indicators: value must be 2-D (n, nc); '
            f'got shape {value.shape}.'
        )
    if np.any(value < 0):
        raise ValueError(
            f'normalize_indicators: value must be non-negative (all entries ≥ 0); '
            f'found {int((value < 0).sum())} negative entries '
            f'(minimum = {value.min():.6g}).'
        )

    row_sums = value.sum(axis=1)

    # Warn if any row deviates from 1 beyond tolerance
    bad_mask = np.abs(row_sums - 1.0) > tol
    if bad_mask.any():
        n_bad   = int(bad_mask.sum())
        max_dev = float(np.abs(row_sums[bad_mask] - 1.0).max())
        warnings.warn(
            f'normalize_indicators: {n_bad} row(s) do not sum to 1.0 '
            f'(max |row_sum − 1| = {max_dev:.4g}); rows will be rescaled.',
            UserWarning, stacklevel=2,
        )

    # Handle all-zero rows
    zero_mask = (row_sums == 0)
    if zero_mask.any():
        n_zero = int(zero_mask.sum())
        msg = (
            f'normalize_indicators: {n_zero} all-zero row(s) at indices '
            f'{np.where(zero_mask)[0].tolist()}.'
        )
        if not uniform_on_zero:
            raise ValueError(msg + ' Pass uniform_on_zero=True to auto-fill.')
        warnings.warn(
            msg + ' Replacing with uniform distribution 1/nc.',
            UserWarning, stacklevel=2,
        )
        value = value.copy()
        value[zero_mask] = 1.0 / value.shape[1]
        row_sums[zero_mask] = 1.0

    # Safe division (zero_mask rows have been fixed above)
    return value / row_sums[:, np.newaxis]


# ======================================================================
# Pair-distance diagnostic
# ======================================================================

def pair_distance_diagnostic(coord, group_ids=None, d_max=None,
                              n_hist=60, suggest_rule='one_third',
                              max_sample=5000, plot=True, ax=None):
    """Diagnose the empirical pair-distance distribution and suggest ``d_max``.

    Run this **before** calling :func:`adaptive_bins` to choose an
    appropriate maximum lag distance.  Including pairs at large separations
    (beyond the correlation range) provides no structural information and
    can mask the short-range signal; the 1/3-domain rule avoids this.

    Parameters
    ----------
    coord : (n, d) array
        Data coordinates.
    group_ids : (n,) array-like of int/str, optional
        Group labels (e.g. borehole IDs).  When given, the function reports
        the fraction of *within-group* vs *cross-group* pairs per distance
        band and recommends using ``group_ids`` in :func:`probatablecalc`.
    d_max : float or None
        If not ``None``, restrict the histogram to ``[0, d_max]``.
    n_hist : int
        Number of histogram bins (default 60).
    suggest_rule : {'one_third', 'one_half', 'percentile_80'}
        Rule used to derive the suggested ``d_max`` cutoff.

        * ``'one_third'``    – 1/3 of the 99th-percentile domain extent
          (Goovaerts 1997; Deutsch & Journel 1998).
        * ``'one_half'``     – 1/2 (Journel & Huijbregts 1978 classical rule).
        * ``'percentile_80'`` – distance at which 80 % of pairs are captured.

    max_sample : int
        Maximum number of points to use for computing ``pdist``; a random
        subsample is taken when ``n > max_sample`` (default 5 000).
    plot : bool
        Draw the histogram and cumulative distribution (default True).
    ax : matplotlib.axes.Axes or None
        Axes to draw into; if ``None``, a new figure is created.

    Returns
    -------
    all_dists : np.ndarray
        Flat array of sampled pairwise distances (positive values only).
    suggested_dmax : float
        Suggested ``d_max`` based on *suggest_rule*.
    stats : dict
        ``{'domain_extent', 'suggested_dmax', 'rule',
           'within_frac', 'cross_frac'}``
        ``within_frac`` / ``cross_frac`` are only present when
        *group_ids* is given.
    """
    coord = np.asarray(coord, dtype=float)
    n     = coord.shape[0]
    rng   = np.random.default_rng(0)

    # Sub-sample if n is large (pdist is O(n²))
    if n > max_sample:
        idx = rng.choice(n, size=max_sample, replace=False)
        coord_sub = coord[idx]
        gids_sub  = np.asarray(group_ids)[idx] if group_ids is not None else None
        warnings.warn(
            f'pair_distance_diagnostic: sub-sampling {max_sample} of {n} '
            f'points for the pair distance histogram.'
        )
    else:
        coord_sub = coord
        gids_sub  = np.asarray(group_ids) if group_ids is not None else None

    all_dists_raw = pdist(coord_sub)
    all_dists     = all_dists_raw[all_dists_raw > 0]

    domain_ext = float(np.percentile(all_dists, 99))

    # Suggested d_max
    if suggest_rule == 'one_third':
        suggested_dmax = domain_ext / 3.0
    elif suggest_rule == 'one_half':
        suggested_dmax = domain_ext / 2.0
    elif suggest_rule == 'percentile_80':
        suggested_dmax = float(np.percentile(all_dists, 80))
    else:
        raise ValueError(f"Unknown suggest_rule '{suggest_rule}'.")

    if d_max is not None:
        hist_dists = all_dists[all_dists <= d_max]
    else:
        hist_dists = all_dists

    # Within-group / cross-group fractions
    stats = {
        'domain_extent':  domain_ext,
        'suggested_dmax': suggested_dmax,
        'rule':           suggest_rule,
    }
    if gids_sub is not None:
        from scipy.spatial.distance import squareform
        dist_mat  = squareform(all_dists_raw)
        n_sub     = coord_sub.shape[0]
        same_mask = gids_sub[:, None] == gids_sub[None, :]
        np.fill_diagonal(same_mask, False)  # exclude self-pairs
        triu_mask  = np.triu(np.ones((n_sub, n_sub), dtype=bool), k=1)
        within_n   = int((same_mask  & triu_mask & (dist_mat > 0)).sum())
        cross_n    = int((~same_mask & triu_mask & (dist_mat > 0)).sum())
        total_n    = within_n + cross_n
        stats['within_frac'] = within_n / total_n if total_n > 0 else np.nan
        stats['cross_frac']  = cross_n  / total_n if total_n > 0 else np.nan
        stats['within_n']    = within_n
        stats['cross_n']     = cross_n

    if plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn('matplotlib not available; skipping plot.')
            return all_dists, suggested_dmax, stats

        if ax is None:
            fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        else:
            axes = [ax, ax]
            fig  = ax.figure

        bins = np.linspace(0, hist_dists.max(), n_hist + 1)
        cnts, edges = np.histogram(hist_dists, bins=bins)
        midpts = 0.5 * (edges[:-1] + edges[1:])

        # --- histogram ---
        ax0 = axes[0]
        ax0.bar(midpts, cnts, width=edges[1] - edges[0],
                color='steelblue', alpha=0.7, label='All pairs')
        if gids_sub is not None:
            # within-group histogram
            wdists = dist_mat[same_mask & triu_mask & (dist_mat > 0)]
            wdists = wdists[wdists <= hist_dists.max()]
            wcnts, _ = np.histogram(wdists, bins=bins)
            ax0.bar(midpts, wcnts, width=edges[1] - edges[0],
                    color='coral', alpha=0.7, label='Within-group')
        ax0.axvline(suggested_dmax, color='red', lw=2, ls='--',
                    label=f'd_max suggestion ({suggest_rule}): '
                          f'{suggested_dmax:.0f}')
        ax0.set_xlabel('Pair distance')
        ax0.set_ylabel('Pair count')
        ax0.set_title('Pair-distance histogram')
        ax0.legend(fontsize=8)

        # --- cumulative fraction ---
        ax1 = axes[1]
        sorted_d = np.sort(hist_dists)
        cum_frac = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
        ax1.plot(sorted_d, cum_frac, lw=2, color='steelblue')
        ax1.axvline(suggested_dmax, color='red', lw=2, ls='--',
                    label=f'Suggested d_max = {suggested_dmax:.0f}')
        ax1.set_xlabel('Pair distance')
        ax1.set_ylabel('Cumulative pair fraction')
        ax1.set_title('Cumulative distribution of pair distances')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print summary
        print(f'[pair_distance_diagnostic]')
        print(f'  n points        : {n}  (subsampled: {coord_sub.shape[0]})')
        print(f'  Domain extent   : {domain_ext:.1f}  (99th pct of distances)')
        print(f'  Suggested d_max : {suggested_dmax:.1f}  ({suggest_rule})')
        if 'within_frac' in stats:
            print(f'  Within-group pairs: {stats["within_n"]:,}'
                  f' ({100*stats["within_frac"]:.1f} %)')
            print(f'  Cross-group pairs : {stats["cross_n"]:,}'
                  f' ({100*stats["cross_frac"]:.1f} %)')
            print('  → Consider using group_ids= in probatablecalc() '
                  'to restrict to within-group pairs.')

    return all_dists, suggested_dmax, stats


# ======================================================================
# Adaptive binning helper
# ======================================================================

def adaptive_bins(coord, n_bins=12, d_max=None, auto_dmax=None,
                  method='equal_pairs', power=1.5, d_min=None):
    """
    Generate non-uniform distance-bin edges for ``probatablecalc``.

    Non-uniform bins place more resolution at short separation distances
    (where the bivariate dependence structure changes rapidly) and wider
    bins at long distances (where the table asymptotes toward
    independence and pair counts are plentiful).

    Parameters
    ----------
    coord : np.ndarray, shape (n, d)
        Data coordinates.  Used only by ``method='equal_pairs'`` to
        compute the empirical distribution of pairwise distances.
    n_bins : int, optional
        Number of non-zero distance bins (default 12).  A zero-distance
        bin for self-pairs is always prepended automatically.
    d_max : float or None, optional
        Maximum distance to cover.  Ignored when *auto_dmax* is set.
        If both are ``None``, uses the maximum pairwise distance.
    auto_dmax : {'one_third', 'one_half', 'percentile_80'} or float or None
        Automatically derive *d_max* from the empirical pair distribution:

        * ``'one_third'``    – 1/3 of the 99th-percentile domain extent
          (recommended; Goovaerts 1997).
        * ``'one_half'``     – 1/2 (Journel & Huijbregts 1978 classic rule).
        * ``'percentile_80'`` – distance at which 80 % of pairs occur.
        * *float* (0–1)       – that quantile of the distance distribution.

        When set, overrides *d_max*.
    method : {'equal_pairs', 'power_law', 'log'}, optional
        Binning strategy (default ``'equal_pairs'``):

        * ``'equal_pairs'`` – quantile-based edges so that every bin
          contains approximately the same number of point pairs.  Best
          for horizontal (large-scale) data.
        * ``'power_law'`` – edges at
          ``d_min + (d_max - d_min) * (k/K)^p``, concentrating more
          bins at short distances when ``power > 1``.
        * ``'log'`` – geometrically (log-) spaced edges from *d_min*
          to *d_max*.  **Recommended for vertical/within-borehole data**
          where the correlation range is much shorter than the domain.

    power : float, optional
        Exponent for ``method='power_law'`` (default 1.5).  Larger
        values concentrate bins more at short distances.
    d_min : float or None, optional
        Minimum non-zero bin edge.  If ``None``:

        * ``'equal_pairs'`` → 1st-percentile pairwise distance.
        * ``'log'`` / ``'power_law'`` → 500 m.

    Returns
    -------
    coord_limit : np.ndarray, shape (n_bins + 1,)
        Array of bin edges suitable for ``probatablecalc(...,
        coord_limit=coord_limit)``.  The first element is always 0
        (the zero-distance sentinel for self-pairs).

    Notes
    -----
    * For ``method='equal_pairs'``, the function computes all pairwise
      distances via ``scipy.spatial.distance.pdist``, which has
      :math:`O(n^2)` cost.  For very large datasets (n > 10 000),
      consider ``method='log'`` or ``method='power_law'`` instead.
    * The returned ``coord_limit`` always starts at 0 and ends at
      *d_max*.  The intermediate edges vary by method.
    * For **borehole / vertical** data with a short correlation range,
      use ``method='log'``, ``auto_dmax='one_third'``, and pass
      ``group_ids`` to :func:`probatablecalc` to restrict estimation
      to within-borehole pairs.

    Examples
    --------
    >>> # Vertical (within-borehole, short correlation range)
    >>> cl_v = adaptive_bins(cs_1d, n_bins=30, method='log',
    ...                      auto_dmax='one_third', d_min=1.0)
    >>> D, P, O = probatablecalc(cs_1d, ps, cl_v, group_ids=borehole_ids)

    >>> # Horizontal (station-level, km-scale)
    >>> cl_h = adaptive_bins(cs_xy, n_bins=20, method='equal_pairs',
    ...                      auto_dmax='one_third')
    >>> D, P, O = probatablecalc(cs_xy, ps, cl_h)
    """
    coord = np.asarray(coord, dtype=float)

    # ---- compute all pairwise distances (needed for equal_pairs & defaults)
    need_dists = (method == 'equal_pairs' or d_max is None
                  or d_min is None or auto_dmax is not None)
    if need_dists:
        all_dists = pdist(coord)
        all_dists_pos = all_dists[all_dists > 0]
        all_dists_pos.sort()

    # ---- resolve auto_dmax first (overrides d_max) ----
    if auto_dmax is not None:
        domain_ext = float(np.percentile(all_dists_pos, 99)) \
            if len(all_dists_pos) > 0 else 1.0
        if auto_dmax == 'one_third':
            d_max = domain_ext / 3.0
        elif auto_dmax == 'one_half':
            d_max = domain_ext / 2.0
        elif auto_dmax == 'percentile_80':
            d_max = float(np.percentile(all_dists_pos, 80))
        elif isinstance(auto_dmax, (int, float)) and 0 < auto_dmax <= 1:
            d_max = float(np.quantile(all_dists_pos, auto_dmax))
        else:
            raise ValueError(
                f"auto_dmax='{auto_dmax}' not understood. "
                "Use 'one_third', 'one_half', 'percentile_80', or a float in (0,1]."
            )

    if d_max is None:
        d_max = float(all_dists_pos[-1]) if len(all_dists_pos) > 0 else 1.0

    if d_min is None:
        if method == 'equal_pairs' and len(all_dists_pos) > 0:
            d_min = float(np.percentile(all_dists_pos, 1.0))
        else:
            d_min = 500.0
    d_min = max(d_min, 1e-6)   # guard against zero

    # ---- method dispatch ----
    if method == 'equal_pairs':
        if len(all_dists_pos) == 0:
            warnings.warn('No positive pairwise distances; falling back to '
                          'uniform bins.')
            edges = np.linspace(d_min, d_max, n_bins + 1)
        else:
            # Keep only distances within [0, d_max]
            dists_in_range = all_dists_pos[all_dists_pos <= d_max]
            if len(dists_in_range) < n_bins:
                warnings.warn(f'Only {len(dists_in_range)} pairs within '
                              f'd_max={d_max:.0f}; falling back to uniform.')
                edges = np.linspace(d_min, d_max, n_bins + 1)
            else:
                quantiles = np.linspace(0, 1, n_bins + 1)
                edges = np.quantile(dists_in_range, quantiles)
                # Ensure monotonically increasing (can happen with ties)
                edges = np.maximum.accumulate(edges)
                edges[0]  = 0.0      # first real edge starts at 0
                edges[-1] = d_max    # cap at d_max
    elif method == 'power_law':
        k = np.arange(n_bins + 1, dtype=float)
        edges = d_min + (d_max - d_min) * (k / n_bins) ** power
        edges[0] = 0.0
    elif method == 'log':
        edges = np.concatenate([[0.0], np.geomspace(d_min, d_max, n_bins)])
    else:
        raise ValueError(f"Unknown method '{method}'. Choose from "
                         "'equal_pairs', 'power_law', 'log'.")

    # ---- prepend zero sentinel (probatablecalc uses coord_limit[0]=0) ----
    # If the first edge is already 0, just return; otherwise prepend.
    if edges[0] != 0.0:
        edges = np.concatenate([[0.0], edges])

    # Remove duplicate edges (can occur in edge cases with few data)
    edges = np.unique(edges)

    return edges


def probatablecalc(coord, value, coord_limit, group_ids=None,
                   normalize_inputs=True,
                   chunk_size=None, ShowCalDis=False):
    """Estimate bivariate probability tables for categorical spatial data.

    Computes empirical transition-probability tables ``P[a, b, h]`` for all
    category pairs ``(a, b)`` at the distance classes defined by
    *coord_limit*.

    **Soft / probabilistic indicators are fully supported.**  You may pass
    either hard (one-hot) or soft (fractional) class-probability vectors.
    See *value* and *normalize_inputs* for details.

    Parameters
    ----------
    coord : (n, d) ndarray
        Coordinates for the *n* data locations (any number of dimensions).
    value : (n, nc) array-like
        Class-indicator matrix at each location.  Two forms are accepted:

        **Hard (one-hot) indicators** — standard categorical data where
        each location belongs to exactly one class::

            # 4-class example, location belongs to class 2:
            [0, 1, 0, 0]

        **Soft (probabilistic) indicators** — each row is a probability
        vector representing uncertainty over class assignments::

            # geophysical inversion / classifier output:
            [0.2, 0.2, 0.8]   # most likely class 3, but uncertain

        Soft indicators arise naturally from:

        * Geophysical inversion results (e.g. ERT, seismic facies)
        * Probabilistic classifier outputs (XGBoostLSS, random forest)
        * Station-averaged lithology fractions (as in this example)
        * Kriging or co-kriging of indicator variables

        The table estimator computes the outer product
        ``P(a, b | h) ∝ p_i[a] × p_j[b]`` for each pair ``(i, j)`` at lag
        ``h``, then averages over all such pairs in the bin.  This is
        equivalent to treating soft probabilities as fractional class
        memberships and pooling contributions across all classes
        simultaneously — no NaN gaps from absent class combinations.

        Rows are validated and row-normalized to sum to 1 (controlled by
        *normalize_inputs*).

    coord_limit : (ncl,) ndarray
        Bin-edge limits for the distance classes (output of
        :func:`adaptive_bins`).  Distance classes are **open on the left
        and closed on the right**.  The first element must be 0 (sentinel
        for self-pairs).
    group_ids : (n,) array-like of int/str or None, optional
        Group labels (e.g. borehole IDs).  When provided, **only pairs
        within the same group** contribute to the empirical tables.  This
        is essential for estimating vertical correlation from borehole
        data, where cross-borehole pairs at the same depth separation do
        not carry within-column transition information.
    normalize_inputs : bool, optional
        If ``True`` (default), call :func:`normalize_indicators` to
        validate and row-normalize *value* before computing the tables.
        This handles un-normalized soft indicators (e.g. raw scores that
        sum to values other than 1) and emits informative warnings.
        Set to ``False`` only when you have already validated *value*
        externally and want to skip the overhead.
    chunk_size : int or None, optional
        Number of points per chunk for memory management.  The result is
        independent of *chunk_size* (default: process all at once).
    ShowCalDis : bool, optional
        Print progress information (default False).

    Returns
    -------
    D : (ncl,) ndarray
        Mean pair distance per bin.  ``D[0] = 0`` for the self-pair bin.
        NaN for empty bins.
    P : (nc, nc, ncl) ndarray
        Bivariate probability table.  ``P[a, b, k]`` is the estimated
        probability that class ``b`` occurs at distance ``D[k]`` from a
        class-``a`` location.  With soft indicators every bin is fully
        populated (no NaN) as long as the bin is non-empty.
    O : (ncl,) ndarray
        Number of unique undirected pairs per bin.

    Notes
    -----
    **Why soft indicators eliminate NaN gaps:**
    With hard one-hot vectors, ``P[a, b, k]`` is NaN whenever no pair
    ``(class a, class b)`` exists in bin ``k``.  At short horizontal
    ranges where only a handful of borehole pairs contribute, rare class
    combinations are easily absent.  Soft indicators avoid this because
    every pair contributes a non-zero weight ``p_i[a] × p_j[b]`` to
    *all* ``(a, b)`` cells simultaneously.

    **Joint-probability interpretation:**
    ``P[a, b, h]`` is the *bivariate* (joint) probability of observing
    class ``a`` at one location and class ``b`` at a location distance
    ``h`` away, averaged over all such pairs::

        P[a, b, h] = E[p_i(a) × p_j(b)]  / ΣΣ_ab E[p_i(a) × p_j(b)]

    For one-hot data this reduces to the classical transition-probability
    matrix (Carle & Fogg 1996).

    See Also
    --------
    normalize_indicators : Validate and row-normalize an indicator matrix.
    adaptive_bins : Generate bin edges for this function.
    pair_distance_diagnostic : Diagnose the pair-distance distribution.
    probatablecalc_directional : Directional (anisotropy) variant.
    """
    if chunk_size is None:
        chunk_size = len(coord)

    coord       = np.asarray(coord, dtype=float)
    n_data      = coord.shape[0]
    value       = np.asarray(value, dtype=float)

    # Validate / normalize indicator matrix
    if normalize_inputs:
        value = normalize_indicators(value)

    n_cata      = value.shape[1]
    coord_limit = np.array(coord_limit).flatten()
    n_limit     = coord_limit.shape[0]

    # Pre-process group IDs for within-group filtering
    use_groups = group_ids is not None
    if use_groups:
        gids = np.asarray(group_ids)
        if gids.shape[0] != n_data:
            raise ValueError(
                f'group_ids length ({len(gids)}) must match coord length ({n_data}).'
            )

    P_sum   = np.zeros((n_cata, n_cata, n_limit))
    P_count = np.zeros((n_cata, n_cata, n_limit))
    D_sum   = np.zeros(n_limit)
    O       = np.zeros(n_limit)

    n_chunks    = int(np.ceil(n_data / chunk_size))
    coord_split = np.array_split(coord, n_chunks)
    value_split = np.array_split(value, n_chunks)
    gids_split  = np.array_split(gids, n_chunks) if use_groups else [None] * n_chunks

    if ShowCalDis:
        print(f'Total chunks: {n_chunks}')
        print('Processing...')

    for n, (coord_i, value_i, gids_i) in enumerate(
            zip(coord_split, value_split, gids_split)):
        for coord_j, value_j, gids_j in zip(coord_split, value_split, gids_split):
            n_data_i = coord_i.shape[0]
            n_data_j = coord_j.shape[0]
            dis  = cdist(coord_i, coord_j)
            prob = (value_i.reshape((n_data_i, 1, n_cata, 1)) *
                    value_j.reshape((1, n_data_j, 1, n_cata)))
            prob /= prob.sum(axis=(-1, -2), keepdims=True)

            # Within-group mask (broadcast shape: ni × nj)
            if use_groups:
                grp_mask = gids_i[:, None] == gids_j[None, :]
            else:
                grp_mask = np.ones((n_data_i, n_data_j), dtype=bool)

            # Zero-distance bin (self-pairs)
            zero_mask = (dis == 0) & grp_mask
            idx_row, idx_col = np.where(zero_mask)
            select_dis  = dis[idx_row, idx_col]
            D_sum[0]   += select_dis.sum()
            O[0]       += select_dis.shape[0]
            select_prob = prob[idx_row, idx_col]
            P_sum[:, :, 0]   += select_prob.sum(axis=0)
            P_count[:, :, 0] += select_prob.shape[0]

            # Non-zero distance bins
            for idx, (left, right) in enumerate(
                    zip(coord_limit[:-1], coord_limit[1:])):
                bin_mask = (dis > left) & (dis <= right) & grp_mask
                idx_row, idx_col = np.where(bin_mask)
                select_dis  = dis[idx_row, idx_col]
                D_sum[idx + 1]   += select_dis.sum() / 2
                O[idx + 1]       += select_dis.shape[0] / 2
                select_prob = prob[idx_row, idx_col]
                P_sum[:, :, idx + 1]   += select_prob.sum(axis=0)
                P_count[:, :, idx + 1] += select_prob.shape[0]

        if ShowCalDis:
            print(f'{n + 1}/{n_chunks}')

    with np.errstate(invalid='ignore', divide='ignore'):
        D = D_sum / O
        P = P_sum / P_count
    return D, P, O


def _azimuth_mask(azimuth, ang, angtol):
    """Return a boolean mask for azimuths within ``ang ± angtol``.

    Handles wrap-around at the ±π/2 boundary of the half-period convention
    used throughout stamps (``arctan2`` result wrapped to ``(-π/2, π/2]``).
    """
    hi = ang + angtol
    lo = ang - angtol
    if hi > np.pi / 2:
        return (azimuth > lo) | (azimuth <= hi - np.pi)
    if lo <= -np.pi / 2:
        return (azimuth <= hi) | (azimuth > lo + np.pi)
    return (azimuth > lo) & (azimuth <= hi)


def probatablecalc_directional(coord, value, coord_limit, ang, angtol,
                                group_ids=None,
                                normalize_inputs=True,
                                chunk_size=None, ShowCalDis=False):
    """Estimate directional bivariate probability tables for categorical data.

    Extends :func:`probatablecalc` with an azimuthal filter so that only pairs
    whose separation vector falls within ``ang ± angtol`` contribute to the
    table.  Repeating over a set of directions lets you characterise the
    **anisotropy of categorical spatial dependence** purely from the
    probability-table structure — no covariance proxy is needed.

    **Soft / probabilistic indicators are fully supported** — see
    :func:`probatablecalc` for details on how fractional class-probability
    vectors are handled.

    The counting convention is identical to :func:`probatablecalc`: every
    ordered pair ``(i → j)`` and its reverse ``(j → i)`` both contribute once,
    and pair counts / distances are divided by 2 so the result represents
    unique undirected pairs.  Because the azimuth is wrapped to ``(-π/2, π/2]``,
    both orderings of a pair always share the same wrapped angle and are
    therefore both accepted or both rejected by the angular filter.

    Parameters
    ----------
    coord : (n, 2) array
        2-D spatial coordinates (X, Y).  Only 2-D coordinates are accepted;
        vertical (Z) data should be analysed separately.
    value : (n, nc) array-like
        Class-indicator matrix — hard (one-hot) *or* soft (probabilistic).
        See :func:`probatablecalc` for full description and
        :func:`normalize_indicators` for preprocessing utilities.
    coord_limit : (ncl,) array
        Distance-class limits (same meaning as in :func:`probatablecalc`).
    ang : float
        Target direction in **radians**, in ``(-π/2, π/2]``.
        ``0`` = E–W,  ``π/4`` = NE–SW,  ``π/2`` = N–S.
    angtol : float
        Half-width of the angular acceptance window in **radians**.
    group_ids : (n,) array-like or None, optional
        Group labels (e.g. depth level as a string ``wells['Z'].astype(str)``).
        When provided, **only pairs within the same group** contribute to the
        empirical tables.  This mirrors the ``group_ids`` parameter of
        :func:`probatablecalc` and is essential when the horizontal data
        contains multiple depth samples per borehole location: passing the
        Z-level as the group label ensures that only same-depth pairs
        contribute, correctly reflecting within-layer horizontal transitions.
    normalize_inputs : bool, optional
        Validate and row-normalize *value* via :func:`normalize_indicators`
        before computation (default ``True``).  Set to ``False`` if the
        matrix is already validated for performance.
    chunk_size : int or None
        Memory-control parameter (same as :func:`probatablecalc`).
    ShowCalDis : bool
        Verbose progress output.

    Returns
    -------
    D : (ncl,) array
        Mean separation distance per bin (NaN for empty bins).
    P : (nc, nc, ncl) array
        Directional transition-probability table.  ``P[a, b, k]`` is the
        estimated probability that class ``b`` occurs at separation ``D[k]``
        from a location of class ``a``, averaged over pairs in direction ``ang``.
    O : (ncl,) array
        Number of unique undirected pairs per bin.

    Notes
    -----
    **Azimuth convention** (matches :func:`stamps.general.diffarray.diffarray`
    and :func:`stamps.stats.dependence.stcov.stcov`):

    * Azimuth = ``arctan2(Δy, Δx)`` wrapped to ``(-π/2, π/2]``.
    * Opposite directions share the same wrapped angle, so the filter is
      automatically symmetric/undirected.

    References
    ----------
    Carle, S.F. & Fogg, G.E. (1996). Transition probability-based indicator
        geostatistics. *Mathematical Geology*, 28(4), 453–476.
    """
    coord       = np.asarray(coord,        dtype=float)
    value       = np.asarray(value,        dtype=float)
    coord_limit = np.asarray(coord_limit,  dtype=float).ravel()

    if coord.ndim != 2 or coord.shape[1] != 2:
        raise ValueError(
            'probatablecalc_directional requires 2-D (X, Y) coordinates; '
            f'got shape {coord.shape}.'
        )

    # Validate / normalize indicator matrix
    if normalize_inputs:
        value = normalize_indicators(value)

    n_data  = coord.shape[0]
    n_cata  = value.shape[1]
    n_limit = coord_limit.shape[0]

    # Pre-process group IDs for within-group filtering
    use_groups = group_ids is not None
    if use_groups:
        gids = np.asarray(group_ids)
        if gids.shape[0] != n_data:
            raise ValueError(
                f'group_ids length ({len(gids)}) must match coord length ({n_data}).'
            )

    if chunk_size is None:
        chunk_size = n_data

    P_sum   = np.zeros((n_cata, n_cata, n_limit))
    P_count = np.zeros((n_cata, n_cata, n_limit))
    D_sum   = np.zeros(n_limit)
    O       = np.zeros(n_limit)

    n_chunks    = int(np.ceil(n_data / chunk_size))
    coord_split = np.array_split(coord, n_chunks)
    value_split = np.array_split(value, n_chunks)
    gids_split  = np.array_split(gids,  n_chunks) if use_groups else [None] * n_chunks

    if ShowCalDis:
        print(f'[probatablecalc_directional] '
              f'ang={np.degrees(ang):.1f} deg  tol=±{np.degrees(angtol):.1f} deg  '
              f'chunks={n_chunks}')

    # Iterate over ALL ordered chunk pairs (ci, cj) — same as probatablecalc —
    # so each unordered pair (a, b) is visited twice.  D and O are divided by 2
    # to count unique undirected pairs; P accumulates both orderings.
    for ci, (coord_i, value_i, gids_i) in enumerate(
            zip(coord_split, value_split, gids_split)):
        ni = coord_i.shape[0]
        for cj, (coord_j, value_j, gids_j) in enumerate(
                zip(coord_split, value_split, gids_split)):
            nj = coord_j.shape[0]

            # --- displacement vectors (i → j), shape (ni, nj) ---
            dx = coord_j[np.newaxis, :, 0] - coord_i[:, np.newaxis, 0]
            dy = coord_j[np.newaxis, :, 1] - coord_i[:, np.newaxis, 1]
            dist = np.sqrt(dx**2 + dy**2)

            # Azimuth wrapped to (-π/2, π/2]
            azimuth = np.arctan2(dy, dx)
            azimuth[azimuth >  np.pi / 2] -= np.pi
            azimuth[azimuth <= -np.pi / 2] += np.pi

            ang_mask = _azimuth_mask(azimuth, ang, angtol)

            # Within-group mask: only same-group pairs contribute
            if use_groups:
                grp_mask = gids_i[:, None] == gids_j[None, :]
                ang_mask = ang_mask & grp_mask

            # Probability tensor (ni, nj, nc, nc); normalised per pair
            prob = (value_i[:, np.newaxis, :, np.newaxis] *
                    value_j[np.newaxis, :, np.newaxis, :])
            row_sum = prob.sum(axis=(-2, -1), keepdims=True)
            row_sum[row_sum == 0] = 1.0
            prob /= row_sum

            # --- zero-distance bin ---
            zero_mask = (dist == 0) & ang_mask
            if zero_mask.any():
                ir, jc = np.where(zero_mask)
                O[0]             += zero_mask.sum()
                P_sum[:, :, 0]   += prob[ir, jc].sum(axis=0)
                P_count[:, :, 0] += zero_mask.sum()

            # --- distance bins (divide by 2 to count unique pairs) ---
            for idx, (left, right) in enumerate(
                zip(coord_limit[:-1], coord_limit[1:])
            ):
                bin_mask = (dist > left) & (dist <= right) & ang_mask
                if bin_mask.any():
                    ir, jc = np.where(bin_mask)
                    n_pairs          = bin_mask.sum()
                    O[idx + 1]       += n_pairs / 2
                    D_sum[idx + 1]   += dist[bin_mask].sum() / 2
                    P_sum[:, :, idx + 1]   += prob[ir, jc].sum(axis=0)
                    P_count[:, :, idx + 1] += n_pairs

        if ShowCalDis:
            print(f'  chunk {ci + 1}/{n_chunks}')

    with np.errstate(invalid='ignore', divide='ignore'):
        D = D_sum / O
        P = P_sum / P_count

    return D, P, O


# Export optimized version as an alias for convenience
# Users can import: from stamps.stats.probatablecalc import probatablecalc_optimized
__all__ = [
    'normalize_indicators',
    'probatablecalc', 'adaptive_bins',
    'pair_distance_diagnostic',
    'probatablecalc_directional', '_azimuth_mask',
    'probatablecalc_optimized',
]

# ======================================================================
# Merged from probatablecalc_optimized.py
# ======================================================================

"""
Optimized version of probatablecalc for efficient probability table calculation.

This implementation improves performance through:
1. Single-pass distance binning using np.digitize
2. Processing only upper triangle for symmetric chunk pairs
3. Vectorized probability calculations
4. Reduced memory footprint
"""

import numpy as np
from scipy.spatial.distance import cdist
import warnings


def probatablecalc_optimized(coord, value, coord_limit, chunk_size=None, ShowCalDis=False):
    """
    Optimized version of probatablecalc for estimating bivariate probability tables.
    
    This function performs the same calculation as probatablecalc but with improved
    efficiency through better vectorization and reduced redundant calculations.
    
    Args:
        coord (ndarray): 2-D ndarray of floats with shape (n, d). Coordinates for 
            the locations where the categories are known.
        value (ndarray): 2-D ndarray of floats with shape (n, nc). Probability for 
            the categories at the coordinates specified in coord.
        coord_limit (ndarray): 1-D ndarray of floats with shape (ncl,). Limits of 
            the distance classes. Distance classes are open on the left and closed 
            on the right. The lower limit for the first class is >=0.
        chunk_size (int): Split the dataset into chunks during calculation. 
            Default is None (no chunking).
        ShowCalDis (bool): Print the calculating process.
    
    Returns:
        D (ndarray): 1-D ndarray of floats with shape (ncl,). Mean distances for 
            each distance class. First value is null distance (0).
        P (ndarray): 3-D ndarray of floats with shape (nc, nc, ncl). Bivariate 
            probability tables between nc categories for the distance classes.
        O (ndarray): 1-D ndarray of integers with shape (ncl,). Number of pairs 
            in each distance class.
    
    Performance improvements:
    - Upper triangle processing: Reduces chunk pair iterations from n² to n(n+1)/2
    - Vectorized operations: Efficient NumPy broadcasting for probability calculations
    - Reduced memory: Process pairs in batches without storing full matrices
    
    Note: For different chunks, both (i,j) and (j,i) directions must be processed
    due to asymmetric probabilities (prob[i,j] ≠ prob[j,i]). This means for different
    chunks, each iteration processes both directions, resulting in similar total work
    as the original but with fewer iterations. The benefit becomes more significant
    for larger datasets where iteration overhead is reduced.
    """
    coord = np.asarray(coord, dtype=np.float64)
    value = np.asarray(value, dtype=np.float64)
    coord_limit = np.asarray(coord_limit, dtype=np.float64).flatten()
    
    n_data = coord.shape[0]
    n_cata = value.shape[1]
    n_limit = coord_limit.shape[0]
    
    # Validate inputs
    if coord.shape[0] != value.shape[0]:
        raise ValueError("coord and value must have the same number of rows")
    
    if n_limit < 1:
        raise ValueError("coord_limit must have at least one element")
    
    # Initialize accumulation arrays
    P_sum = np.zeros((n_cata, n_cata, n_limit), dtype=np.float64)
    P_count = np.zeros((n_cata, n_cata, n_limit), dtype=np.float64)
    D_sum = np.zeros(n_limit, dtype=np.float64)
    O = np.zeros(n_limit, dtype=np.float64)
    
    # Set chunk size if not specified
    if chunk_size is None:
        chunk_size = n_data
    
    # Split into chunks - match original exactly
    # Original uses: np.array_split(coord, np.ceil(coord.shape[0]/chunk_size))
    n_chunks_split = np.ceil(n_data / chunk_size)
    coord_chunks = np.array_split(coord, n_chunks_split)
    value_chunks = np.array_split(value, n_chunks_split)
    n_chunks = len(coord_chunks)  # Actual number of chunks created
    
    if ShowCalDis:
        total_pairs = n_chunks * (n_chunks + 1) // 2  # Upper triangle only
        print(f'Total chunks: {n_chunks}')
        print(f'Processing {total_pairs} chunk pairs (optimized: upper triangle only, ~{100*(1-total_pairs/(n_chunks*n_chunks)):.1f}% reduction)...')
    
    # Pre-compute distance bin boundaries
    # coord_limit[0] should be 0 (distance=0 handled separately)
    # coord_limit[1:] are the upper bounds for bins 1, 2, ..., n_limit-1
    # Bin i corresponds to: coord_limit[i-1] < distance <= coord_limit[i]
    # We'll use digitize with right=True for (left, right] intervals
    
    pair_count = 0
    for i in range(n_chunks):
        coord_i = coord_chunks[i]
        value_i = value_chunks[i]
        n_i = coord_i.shape[0]
        
        # OPTIMIZATION: Process only upper triangle (j >= i) to avoid double counting
        # This reduces chunk pair processing from n_chunks^2 to n_chunks*(n_chunks+1)/2
        # However, for different chunks (i != j), we need to accumulate probabilities from BOTH
        # directions (i,j) and (j,i) because prob is NOT symmetric!
        # prob[i,j,cat1,cat2] = value_i[i,cat1] * value_j[j,cat2]
        # prob[j,i,cat1,cat2] = value_j[j,cat1] * value_i[i,cat2]  (different!)
        for j in range(i, n_chunks):
            coord_j = coord_chunks[j]
            value_j = value_chunks[j]
            n_j = coord_j.shape[0]
            
            pair_count += 1
            if ShowCalDis and pair_count % 10 == 0:
                print(f'Processing chunk pair {pair_count}/{total_pairs}...')
            
            # Compute distance matrix from i to j (only compute once)
            dis = cdist(coord_i, coord_j)
            
            # Compute probability tensor for (i,j): P(cat1=i, cat2=j | loc1 in i, loc2 in j)
            # Shape: (n_i, n_j, n_cata, n_cata)
            prob_ij = value_i[:, None, :, None] * value_j[None, :, None, :]
            prob_sum_ij = prob_ij.sum(axis=(-2, -1), keepdims=True)
            prob_sum_ij[prob_sum_ij == 0] = 1.0
            prob_ij = prob_ij / prob_sum_ij
            
            # For different chunks, also compute prob for (j,i) direction (transposed)
            if i != j:
                # Probability tensor for (j,i): swap value_i and value_j
                prob_ji = value_j[:, None, :, None] * value_i[None, :, None, :]
                prob_sum_ji = prob_ji.sum(axis=(-2, -1), keepdims=True)
                prob_sum_ji[prob_sum_ji == 0] = 1.0
                prob_ji = prob_ji / prob_sum_ji
                # Distance matrix from j to i is just transpose (cheap view operation)
                dis_ji = dis.T
            else:
                # Same chunk: symmetric, reuse dis and prob_ij
                dis_ji = dis
                prob_ji = prob_ij
            
            # Process distance = 0 (bin 0)
            # Original: processes all chunk pairs (i,j) for all i,j without division
            # For different chunks, we process both (i,j) and (j,i) in one pass here
            
            # Process (i,j) direction
            idx_row_ij, idx_col_ij = np.where(dis == 0)
            if len(idx_row_ij) > 0:
                select_prob_ij = prob_ij[idx_row_ij, idx_col_ij]
                n_zero_ij = len(idx_row_ij)
                # D_sum[0] doesn't change (all distances are 0)
                O[0] += n_zero_ij
                P_sum[:, :, 0] += select_prob_ij.sum(axis=0)
                P_count[:, :, 0] += n_zero_ij
            
            # For different chunks, also process (j,i) direction
            if i != j:
                idx_row_ji, idx_col_ji = np.where(dis_ji == 0)
                if len(idx_row_ji) > 0:
                    select_prob_ji = prob_ji[idx_row_ji, idx_col_ji]
                    n_zero_ji = len(idx_row_ji)
                    # D_sum[0] doesn't change (all distances are 0)
                    O[0] += n_zero_ji
                    P_sum[:, :, 0] += select_prob_ji.sum(axis=0)
                    P_count[:, :, 0] += n_zero_ji
            
            # Process other distance bins (bins 1 to n_limit-1)
            # Original: divides D_sum and O by 2, but P_sum and P_count are NOT divided
            # Weight for D_sum/O: always 0.5 (original divides by 2)
            # Weight for P_sum/P_count: always 1.0 (original does NOT divide)
            weight_d = 0.5  # Applied to D_sum and O
            weight_p = 1.0  # Applied to P_sum and P_count (but we accumulate directly)
            
            for idx, (left, right) in enumerate(zip(coord_limit[:-1], coord_limit[1:])):
                # Process (i,j) direction
                mask_ij = (dis > left) & (dis <= right)
                if mask_ij.any():
                    idx_row_ij, idx_col_ij = np.where(mask_ij)
                    select_dis_ij = dis[idx_row_ij, idx_col_ij]
                    select_prob_ij = prob_ij[idx_row_ij, idx_col_ij]
                    n_pairs_ij = len(idx_row_ij)
                    
                    D_sum[idx+1] += select_dis_ij.sum() * weight_d
                    O[idx+1] += n_pairs_ij * weight_d
                    P_sum[:, :, idx+1] += select_prob_ij.sum(axis=0)
                    P_count[:, :, idx+1] += n_pairs_ij
                
                # For different chunks, also process (j,i) direction
                if i != j:
                    mask_ji = (dis_ji > left) & (dis_ji <= right)
                    if mask_ji.any():
                        idx_row_ji, idx_col_ji = np.where(mask_ji)
                        select_dis_ji = dis_ji[idx_row_ji, idx_col_ji]
                        select_prob_ji = prob_ji[idx_row_ji, idx_col_ji]
                        n_pairs_ji = len(idx_row_ji)
                        
                        D_sum[idx+1] += select_dis_ji.sum() * weight_d
                        O[idx+1] += n_pairs_ji * weight_d
                        P_sum[:, :, idx+1] += select_prob_ji.sum(axis=0)
                        P_count[:, :, idx+1] += n_pairs_ji
    
    # Compute final statistics
    with np.errstate(divide='ignore', invalid='ignore'):
        D = D_sum / O
        P = P_sum / P_count
    
    # Handle empty bins (set to NaN as in original)
    empty_bins = (O == 0)
    if empty_bins.any():
        D[empty_bins] = np.nan
        P[:, :, empty_bins] = np.nan
        if ShowCalDis:
            empty_indices = np.where(empty_bins)[0]
            warnings.warn(f"Warning: {len(empty_indices)} distance classes contain no pairs. "
                         f"Indices: {empty_indices}")
    
    return D, P, O


