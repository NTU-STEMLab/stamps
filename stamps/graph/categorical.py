# -*- coding: utf-8 -*-
"""
stamps.graph.categorical
========================
Reusable matplotlib figures for categorical spatial estimation workflows.

All functions return the figure (and axes) so callers can save or further
customise the output.  Every function follows the same convention:

* ``categories``  — 1-D array/list of integer class codes.
* ``labels``      — matching list of human-readable names.
* ``colors``      — matching list of hex colour strings.
* ``ax`` / ``axes`` — optional pre-existing axes; a new figure is created when
  ``None``.
* ``title`` / ``suptitle`` — optional string overrides.

Public API
----------
plot_cross_section          Lithology vs Easting / Northing scatter.
plot_borehole_locations     Plan-view borehole location map (+ optional OSM tile).
plot_depth_slices           Plan-view data maps at multiple depth levels.
plot_pmodel_matrix          Full nc × nc Pmodel transition-probability matrix.
plot_pmodel_self_transitions  Self-transition curves: vertical vs horizontal.
plot_xgb_prior              Per-class XGBoostLSS prior probability maps.
plot_estimation_maps        MAP lithology at each depth level (multiple methods).
plot_posterior_maps         Per-class posterior probability maps at one depth.
plot_loocv_bar              Bar chart of LOOCV accuracy and mean log-likelihood.
plot_categorical_map        2-D pcolormesh MAP-class map with data overlay.
plot_categorical_probmap    2-D pcolormesh maximum-posterior-probability map.
plot_categorical_class_probs  2-D pcolormesh per-category probability maps.
"""
from __future__ import annotations

import warnings
import numpy as np
try:  # optional dependency
    import matplotlib.pyplot as plt
except ImportError:
    from stamps.general.optional import MissingDependency
    plt = MissingDependency('matplotlib.pyplot')
try:  # optional dependency
    import matplotlib.patches as mpatches
except ImportError:
    from stamps.general.optional import MissingDependency
    mpatches = MissingDependency('matplotlib.patches')
try:  # optional dependency
    from matplotlib.colors import ListedColormap
except ImportError:
    from stamps.general.optional import MissingDependency
    ListedColormap = MissingDependency('matplotlib.colors.ListedColormap')

__all__ = [
    "plot_cross_section",
    "plot_borehole_locations",
    "plot_depth_slices",
    "plot_pmodel_matrix",
    "plot_pmodel_self_transitions",
    "plot_xgb_prior",
    "plot_estimation_maps",
    "plot_posterior_maps",
    "plot_loocv",
    "plot_loocv_bar",
    "plot_categorical_map",
    "plot_categorical_probmap",
    "plot_categorical_class_probs",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _legend_patches(categories, labels, colors):
    return [mpatches.Patch(color=c, label=l)
            for c, l, _ in zip(colors, labels, categories)]


def _add_osm(ax, xy, crs_epsg=3826):
    """Try to add an OpenStreetMap basemap via contextily.

    Parameters
    ----------
    ax : Axes
        The axes to annotate.
    xy : (n, 2) array
        Coordinates in *crs_epsg* projection (used to set the extent).
    crs_epsg : int
        EPSG code of the coordinates (default 3826 = TWD97 TM2).

    Returns ``True`` on success, ``False`` if contextily / pyproj are absent.
    """
    try:
        import contextily as ctx
        from pyproj import Transformer

        # Transform bounding box to EPSG:3857 for tile fetching
        tr = Transformer.from_crs(crs_epsg, 3857, always_xy=True)
        pad = 0.05  # 5 % margin
        x0, x1 = xy[:, 0].min(), xy[:, 0].max()
        y0, y1 = xy[:, 1].min(), xy[:, 1].max()
        dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
        x0m, y0m = tr.transform(x0 - dx, y0 - dy)
        x1m, y1m = tr.transform(x1 + dx, y1 + dy)

        # Re-set x/y limits in original CRS then add basemap
        ax.set_xlim(x0 - dx, x1 + dx)
        ax.set_ylim(y0 - dy, y1 + dy)
        ctx.add_basemap(
            ax,
            crs=f"EPSG:{crs_epsg}",
            source=ctx.providers.OpenStreetMap.Mapnik,
            alpha=0.4,
            zoom="auto",
            attribution_size=6,
        )
        return True
    except Exception as exc:
        warnings.warn(
            f"OpenStreetMap basemap not added ({type(exc).__name__}: {exc}). "
            "Install contextily and pyproj to enable basemap tiles.",
            stacklevel=3,
        )
        return False


# ---------------------------------------------------------------------------
# Data overview plots
# ---------------------------------------------------------------------------

def plot_cross_section(
    x, z, litho_codes,
    categories, labels, colors,
    x_label="Easting (m)",
    title="Cross-section: lithology vs easting",
    s=4, alpha=0.6,
    ax=None, figsize=(8, 4),
):
    """Scatter plot of lithology vs a horizontal coordinate (cross-section).

    Parameters
    ----------
    x : (n,) array
        Horizontal coordinate (easting or northing).
    z : (n,) array
        Vertical coordinate (depth; typically negative).
    litho_codes : (n,) array
        Integer class code at each sample.
    categories : list of int
        Ordered class codes.
    labels : list of str
        Class names aligned with *categories*.
    colors : list of str
        Hex colours aligned with *categories*.
    x_label : str
        X-axis label.
    title : str
        Axes title.
    s, alpha : float
        Scatter marker size and transparency.
    ax : Axes or None
        Pre-existing axes; new figure created if ``None``.
    figsize : tuple
        Used only when *ax* is ``None``.

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    for c, lbl, col in zip(categories, labels, colors):
        m = np.asarray(litho_codes) == c
        ax.scatter(np.asarray(x)[m], np.asarray(z)[m],
                   c=col, label=lbl, s=s, alpha=alpha)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Depth Z (m)")
    ax.set_title(title)
    ax.legend(fontsize=8, markerscale=3)
    return fig, ax


def plot_borehole_locations(
    well_xy, station_ids, well_depths=None,
    osm_background=True, crs_epsg=3826,
    title=None,
    ax=None, figsize=(7, 7),
):
    """Plan-view borehole location map, optionally with an OSM basemap.

    Parameters
    ----------
    well_xy : (n_stations, 2) array
        Unique (X, Y) per borehole.
    station_ids : (n_samples,) array
        Station ID per depth sample (used to compute depth-range bubble size).
    well_depths : (n_samples,) array or None
        Z values for all samples.  When provided, bubble size scales with
        per-borehole depth range.  When ``None``, uniform size is used.
    osm_background : bool
        Attempt to add an OpenStreetMap tile background (requires
        *contextily* and *pyproj*).
    crs_epsg : int
        EPSG code for the coordinates (default 3826 = TWD97 TM2).
    title : str or None
        Axes title.  Auto-generated when ``None``.
    ax : Axes or None
    figsize : tuple

    Returns
    -------
    fig, ax
    """
    well_xy = np.asarray(well_xy)
    n_stations = well_xy.shape[0]

    if well_depths is not None and station_ids is not None:
        import pandas as pd
        sids = np.asarray(station_ids)
        zvals = np.asarray(well_depths)
        df = pd.DataFrame({"sid": sids, "z": zvals})
        rng = df.groupby("sid")["z"].apply(lambda v: v.max() - v.min())
        sizes = 10 + rng.values / 2
    else:
        sizes = np.full(n_stations, 30.0)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.scatter(well_xy[:, 0], well_xy[:, 1],
               s=sizes, c="steelblue", edgecolors="k",
               linewidths=0.3, alpha=0.7, zorder=3)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(title or f"Borehole locations (n={n_stations}, size ∝ depth range)")
    ax.set_aspect("equal")

    if osm_background:
        _add_osm(ax, well_xy, crs_epsg=crs_epsg)

    return fig, ax


def plot_depth_slices(
    x, y, z, litho_codes,
    categories, labels, colors,
    z_targets=(-15, -75, -150), dz=5.0,
    osm_background=False, crs_epsg=3826,
    suptitle="Borehole lithology at depth levels",
    figsize=None,
):
    """Plan-view categorical data maps at multiple depth levels.

    Parameters
    ----------
    x, y, z : (n,) arrays
        Coordinates of all depth samples.
    litho_codes : (n,) array
        Integer class code at each sample.
    categories, labels, colors : lists
        Class metadata.
    z_targets : sequence of float
        Depth levels to display.
    dz : float
        Half-window around each target depth.
    osm_background : bool
        Add OSM tile to each subplot (requires contextily + pyproj).
    crs_epsg : int
        Coordinate EPSG code for OSM tiles.
    suptitle : str
    figsize : tuple or None

    Returns
    -------
    fig, axes
    """
    n_z = len(z_targets)
    if figsize is None:
        figsize = (5.5 * n_z, 5)

    fig, axes = plt.subplots(1, n_z, figsize=figsize)
    if n_z == 1:
        axes = [axes]

    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    litho_codes = np.asarray(litho_codes)
    xy = np.column_stack([x, y])

    for ax, zt in zip(axes, z_targets):
        near = np.abs(z - zt) <= dz
        for c, lbl, col in zip(categories, labels, colors):
            m = near & (litho_codes == c)
            ax.scatter(x[m], y[m], c=col, s=30, label=lbl,
                       edgecolors="k", linewidths=0.3, zorder=3)
        ax.set_title(f"Z = {zt:.0f} m  (n = {near.sum()})", fontsize=11)
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.set_aspect("equal")
        if osm_background:
            _add_osm(ax, xy[near], crs_epsg=crs_epsg)

    axes[0].legend(fontsize=7, loc="upper left")
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, y=1.01)
    plt.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Pmodel plots
# ---------------------------------------------------------------------------

def plot_pmodel_matrix(
    D_emp, P_emp, O_emp,
    dmodel=None, Pmodel=None,
    categories=None, labels=None, colors=None,
    x_label="Separation",
    x_scale=1.0,
    suptitle=None,
    ylim=(0, 0.5),
    figsize=None,
    show_independence=True,
    empirical_style="scatter",
):
    """Full nc×nc transition-probability matrix plot.

    Empirical values are shown as bubbles (``empirical_style='scatter'``,
    size ∝ log pair-count) or connected line markers
    (``empirical_style='line'``).  When a fitted model is supplied it is
    overlaid as solid (diagonal) or dashed (off-diagonal) lines.

    Parameters
    ----------
    D_emp : (nbins,) array
        Empirical mean distances per bin.  The first element is treated as
        the lag-0 bin and is excluded from the scatter/line plots (but used
        to compute proportions for the independence baseline).
    P_emp : (nc, nc, nbins) array
        Empirical probability table from :func:`probatablecalc`.
    O_emp : (nbins,) array
        Pair counts per bin.
    dmodel : (nd,) array or None, optional
        Fitted model distance axis.  ``None`` (default) plots empirical data
        only without a model overlay.
    Pmodel : (nc, nc, nd) array or None, optional
        Fitted model probability table.  Ignored when *dmodel* is ``None``.
    categories : list or None, optional
        Integer class codes.  Inferred from ``P_emp.shape[0]`` when ``None``.
    labels : list of str or None, optional
        Human-readable class names.  Defaults to ``['0', '1', ...]``.
    colors : list of str or None, optional
        Per-class colours.  Defaults to ``tab10``.
    x_label : str
        X-axis label on the bottom row.
    x_scale : float
        Divide *D_emp* (and *dmodel*) by this factor for display
        (e.g. ``1000`` to show km when distances are in metres).
    suptitle : str or None
        Figure super-title.  Auto-generated when ``None``.
    ylim : tuple
        Y-axis limits for all subplots.
    figsize : tuple or None
    show_independence : bool
        When ``True`` (default), draw a horizontal dashed line at the
        independence level ``p_i · p_j`` for each subplot, where ``p_i``
        and ``p_j`` are the marginal proportions estimated from lag-0.
    empirical_style : {'scatter', 'line'}
        How to display empirical values when no fitted model is drawn.
        ``'scatter'`` uses bubble markers sized by pair count;
        ``'line'`` uses connected ``'o-'`` markers.

    Returns
    -------
    fig, axes  — (nc, nc) axes grid
    """
    P_emp = np.asarray(P_emp)
    nc = P_emp.shape[0]

    if categories is None:
        categories = list(range(nc))
    if labels is None:
        labels = [str(c) for c in categories]
    if colors is None:
        _cm = plt.colormaps['tab10']
        colors = [_cm(i % 10) for i in range(nc)]

    if figsize is None:
        figsize = (3.5 * nc, 3.0 * nc)

    # Marginal proportions from lag-0 diagonal
    p_marg = np.array([P_emp[i, i, 0] for i in range(nc)])

    valid = ~np.isnan(D_emp[1:])
    D_v   = D_emp[1:][valid] / x_scale
    P_v   = P_emp[:, :, 1:][:, :, valid]
    O_v   = O_emp[1:][valid]

    has_model = (dmodel is not None) and (Pmodel is not None)

    fig, axes = plt.subplots(nc, nc, figsize=figsize,
                             sharex=True, sharey=True)
    for i in range(nc):
        for j in range(nc):
            ax  = axes[i, j]
            col = colors[j] if i != j else colors[i]
            emp = P_v[i, j, :]
            ok  = ~np.isnan(emp)

            if empirical_style == "scatter" or has_model:
                sz = 15 + 40 * (np.log1p(O_v) / np.log1p(O_v.max() + 1e-9))
                ax.scatter(D_v[ok], emp[ok], s=sz[ok], color=col,
                           edgecolors="white", linewidths=0.2,
                           alpha=0.65, zorder=3)
            else:
                ax.plot(D_v[ok], emp[ok], 'o-', color=col,
                        lw=1.5, ms=3, alpha=0.8, zorder=3)

            if has_model:
                ax.plot(np.asarray(dmodel) / x_scale, Pmodel[i, j, :],
                        color=col, lw=1.5,
                        ls="-" if i == j else "--", zorder=4)

            if show_independence:
                ax.axhline(p_marg[i] * p_marg[j], color='gray',
                           ls='--', lw=0.8, alpha=0.7)

            ax.set_ylim(*ylim)
            ax.grid(True, alpha=0.25)
            if i == 0:
                ax.set_title(labels[j], fontsize=10, color=col)
            if j == 0:
                ax.set_ylabel(labels[i], fontsize=9, color=colors[i])
            if i == nc - 1:
                ax.set_xlabel(x_label, fontsize=8)

    if suptitle is None:
        if has_model:
            suptitle = ("Pmodel — transition-probability matrix\n"
                        "(bubbles = empirical, lines = fitted)")
        else:
            suptitle = ("Empirical probability table P(k,l; h)\n"
                        "(dashed = independence baseline)")
    fig.suptitle(suptitle, fontsize=11)
    plt.tight_layout()
    return fig, axes


def plot_pmodel_self_transitions(
    D_v, P_v, O_v, dmodel, Pmodel,
    D_h, P_h, O_h, dmodel_h, Pmodel_h,
    labels, colors,
    x_scale_h=1e3,
    x_unit_h="km",
    suptitle="Vertical vs Horizontal Pmodel — self-transition",
    figsize=(14, 5),
):
    """Diagonal self-transition comparison: vertical (left) vs horizontal (right).

    Parameters
    ----------
    D_v, P_v, O_v : arrays
        Empirical vertical distances, tables (nc,nc,nb), counts (valid bins only).
    dmodel, Pmodel : arrays
        Fitted vertical model.
    D_h, P_h, O_h : arrays
        Empirical horizontal distances, tables (nc,nc,nb), counts (valid bins only).
    dmodel_h, Pmodel_h : arrays
        Fitted horizontal model.
    labels, colors : lists
        Class metadata.
    x_scale_h : float
        Divisor for horizontal distances (default 1000 → km).
    x_unit_h : str
        Unit label after scaling.
    suptitle : str
    figsize : tuple

    Returns
    -------
    fig, axes  — shape (2,)
    """
    nc = len(labels)
    sz_v = 20 + 60 * (np.log1p(O_v) / np.log1p(O_v.max()))
    sz_h = 20 + 60 * (np.log1p(O_h) / np.log1p(O_h.max()))

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for i, (lbl, col) in enumerate(zip(labels, colors)):
        axes[0].scatter(D_v, P_v[i, i, :],
                        s=sz_v, color=col, alpha=0.65, zorder=3)
        axes[0].plot(dmodel, Pmodel[i, i, :], color=col, lw=2, label=lbl)
        axes[1].scatter(D_h / x_scale_h, P_h[i, i, :],
                        s=sz_h, color=col, alpha=0.65, zorder=3)
        axes[1].plot(dmodel_h / x_scale_h, Pmodel_h[i, i, :],
                     color=col, lw=2, label=lbl)

    axes[0].set_xlabel("Vertical separation (m)")
    axes[1].set_xlabel(f"Horizontal separation ({x_unit_h})")
    for ax in axes:
        ax.set_ylabel("P(same class)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[0].set_title("Vertical self-transition")
    axes[1].set_title("Horizontal self-transition")
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    plt.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Prior plot
# ---------------------------------------------------------------------------

def plot_xgb_prior(
    ck, prior_matrix, well_xy,
    z_level, categories, labels,
    dz=1.0,
    cmap="RdYlGn",
    suptitle=None,
    figsize=None,
):
    """Per-class XGBoostLSS prior probability maps at one depth level.

    Parameters
    ----------
    ck : (nk, 3) array
        Estimation grid coordinates (X, Y, Z).
    prior_matrix : (nk, nc) array
        Prior probability matrix from :func:`BMEcatPrior`.
    well_xy : (n_wells, 2) array
        Borehole (X, Y) to overlay as black crosses.
    z_level : float
        Depth slice to display.
    categories, labels : lists
        Class metadata (``len(categories)`` determines number of panels).
    dz : float
        Half-window for depth matching.
    cmap : str
        Colour map for probability values.
    suptitle : str or None
    figsize : tuple or None

    Returns
    -------
    fig, axes
    """
    nc = len(categories)
    if figsize is None:
        figsize = (4.5 * nc, 4)

    ck = np.asarray(ck)
    prior_matrix = np.asarray(prior_matrix)
    well_xy = np.asarray(well_xy)

    mask = np.abs(ck[:, 2] - z_level) < dz
    fig, axes = plt.subplots(1, nc, figsize=figsize)
    if nc == 1:
        axes = [axes]

    for j, (lbl, ax) in enumerate(zip(labels, axes)):
        sc = ax.scatter(ck[mask, 0], ck[mask, 1],
                        c=prior_matrix[mask, j],
                        cmap=cmap, vmin=0, vmax=1, s=20)
        ax.scatter(well_xy[:, 0], well_xy[:, 1],
                   c="k", s=25, marker="x", linewidths=0.8, zorder=4)
        plt.colorbar(sc, ax=ax, fraction=0.04)
        ax.set_title(f"P({lbl})", fontsize=10)
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal")

    if suptitle is None:
        suptitle = f"XGBoostLSS prior at Z = {z_level:.0f} m"
    fig.suptitle(suptitle, fontsize=12, y=1.02)
    plt.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Estimation result plots
# ---------------------------------------------------------------------------

def plot_estimation_maps(
    ck, map_arrays, method_names,
    categories, labels, colors,
    z_levels=None,
    dz=1.0,
    s=18,
    suptitle="MAP lithology — separable 3-D estimation",
    figsize=None,
):
    """Grid of MAP lithology maps: rows = methods, columns = depth levels.

    Parameters
    ----------
    ck : (nk, 3) array
        Estimation grid coordinates (X, Y, Z).
    map_arrays : list of (nk,) arrays
        One MAP class array per method (integer codes).
    method_names : list of str
        Method labels for row annotations.
    categories, labels, colors : lists
        Class metadata.
    z_levels : list of float or None
        Depth levels to display.  Auto-detected from ``ck[:, 2]`` when
        ``None``.
    dz : float
        Half-window for depth matching.
    s : float
        Scatter marker size.
    suptitle : str
    figsize : tuple or None

    Returns
    -------
    fig, axes  — shape (n_methods, n_z)
    """
    ck = np.asarray(ck)
    if z_levels is None:
        z_levels = sorted(np.unique(np.round(ck[:, 2], 0)))
    n_z   = len(z_levels)
    n_met = len(map_arrays)
    cmap  = ListedColormap(colors)

    if figsize is None:
        figsize = (4 * n_z, 4 * n_met)

    fig, axes = plt.subplots(n_met, n_z, figsize=figsize,
                             squeeze=False)

    for row_i, (map_arr, title) in enumerate(zip(map_arrays, method_names)):
        map_arr = np.asarray(map_arr)
        for col_i, zv in enumerate(z_levels):
            ax  = axes[row_i, col_i]
            zm  = np.abs(ck[:, 2] - zv) < dz
            c_idx = [categories.index(int(v)) for v in map_arr[zm]]
            ax.scatter(ck[zm, 0], ck[zm, 1], c=c_idx,
                       cmap=cmap, vmin=-0.5, vmax=len(categories) - 0.5,
                       s=s, edgecolors="none")
            ax.set_title(f"Z = {int(zv)} m", fontsize=9)
            ax.tick_params(labelsize=6)
            if col_i == 0:
                ax.set_ylabel(title, fontsize=8)

    patches = [mpatches.Patch(color=c, label=l)
               for c, l in zip(colors, labels)]
    fig.legend(handles=patches, loc="lower center", ncol=len(categories),
               fontsize=9, bbox_to_anchor=(0.5, -0.03))
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, y=1.01)
    plt.tight_layout()
    return fig, axes


def plot_posterior_maps(
    ck, posterior,
    well_xy,
    z_level, categories, labels,
    dz=1.0,
    cmap="RdYlGn",
    suptitle=None,
    figsize=None,
):
    """Per-class posterior probability maps at one depth level.

    Parameters
    ----------
    ck : (nk, 3) array
        Estimation grid coordinates.
    posterior : (nk, nc) array
        Posterior probability matrix.
    well_xy : (n_wells, 2) array
        Borehole (X, Y) overlay.
    z_level : float
        Depth slice.
    categories, labels : lists
        Class metadata.
    dz : float
        Half-window for depth matching.
    cmap : str
    suptitle : str or None
    figsize : tuple or None

    Returns
    -------
    fig, axes
    """
    nc = len(categories)
    if figsize is None:
        figsize = (4.5 * nc, 4)

    ck  = np.asarray(ck)
    post = np.asarray(posterior)
    well_xy = np.asarray(well_xy)
    mask = np.abs(ck[:, 2] - z_level) < dz

    fig, axes = plt.subplots(1, nc, figsize=figsize)
    if nc == 1:
        axes = [axes]

    for j, (lbl, ax) in enumerate(zip(labels, axes)):
        sc = ax.scatter(ck[mask, 0], ck[mask, 1],
                        c=post[mask, j],
                        cmap=cmap, vmin=0, vmax=1, s=20)
        ax.scatter(well_xy[:, 0], well_xy[:, 1],
                   c="k", s=25, marker="x", linewidths=0.8, zorder=4)
        plt.colorbar(sc, ax=ax, fraction=0.04)
        ax.set_title(f"P({lbl} | data)", fontsize=10)
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal")

    if suptitle is None:
        suptitle = f"Posterior probabilities at Z = {z_level:.0f} m"
    fig.suptitle(suptitle, fontsize=12, y=1.02)
    plt.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# LOOCV results
# ---------------------------------------------------------------------------

def plot_loocv(
    method_names,
    accuracies=None, mean_log_likelihoods=None,
    balanced_accuracies=None,
    metrics=None,
    obs=None, pred_list=None,
    colors=("steelblue", "seagreen", "darkorange"),
    unit="",
    suptitle="Station LOOCV",
    figsize=None,
):
    """Unified LOOCV visualisation: scatter panels and/or metric bar charts.

    Combines two views in one figure:

    * **Scatter panels** (when *obs* and *pred_list* are provided) — one
      panel per method showing observed vs predicted with a 1:1 line,
      RMSE and R² in the title.
    * **Bar panels** — one panel per metric.  Metrics can be supplied via
      the legacy positional arguments (*accuracies*, *mean_log_likelihoods*,
      *balanced_accuracies*) **or** through the generic *metrics* dict
      which works for any metric name (e.g. ``{'RMSE': [...], 'R²': [...]}``.

    Parameters
    ----------
    method_names : list of str
        One label per method.
    accuracies : list of float or None
        Overall accuracy per method (categorical LOOCV).
    mean_log_likelihoods : list of float or None
        Mean log-likelihood per method (categorical LOOCV).
    balanced_accuracies : list of float or None
        Balanced accuracy per method.
    metrics : dict or None
        Generic metric dict ``{name: [val_per_method]}``.  Overrides
        the legacy positional metric arguments when provided.
    obs : (n,) array or None
        Observed values for scatter panels (continuous LOOCV).
    pred_list : list of (n,) arrays or None
        Predicted values, one array per method.
    colors : sequence of str
        Colours cycled across methods.
    unit : str
        Unit label for scatter axis labels (e.g. ``'%'``).
    suptitle : str
    figsize : tuple or None

    Returns
    -------
    fig, axes
    """
    import itertools

    # --- Build metric panels list ---
    if metrics is not None:
        bar_panels = list(metrics.items())
    else:
        bar_panels = []
        if accuracies is not None:
            bar_panels.append(("Accuracy", accuracies))
        if mean_log_likelihoods is not None:
            bar_panels.append(("Mean log-likelihood", mean_log_likelihoods))
        if balanced_accuracies is not None:
            bar_panels.append(("Balanced accuracy", balanced_accuracies))

    n_scatter = len(pred_list) if pred_list is not None else 0
    n_bar     = len(bar_panels)
    n_total   = n_scatter + n_bar
    if n_total == 0:
        raise ValueError("Provide at least one of: obs+pred_list, "
                         "accuracies, metrics")

    bar_colors = list(itertools.islice(
        itertools.cycle(colors), len(method_names)))

    if figsize is None:
        figsize = (5.5 * n_total, 4.5)

    fig, axes = plt.subplots(1, n_total, figsize=figsize)
    if n_total == 1:
        axes = [axes]
    else:
        axes = list(axes)

    # --- Scatter panels ---
    if obs is not None and pred_list is not None:
        obs = np.asarray(obs)
        for idx, (pred, mname) in enumerate(zip(pred_list, method_names)):
            ax = axes[idx]
            pred = np.asarray(pred)
            valid = ~np.isnan(pred)
            obs_v, pred_v = obs[valid], pred[valid]
            rmse = float(np.sqrt(np.mean((pred_v - obs_v) ** 2)))
            ss_res = np.sum((pred_v - obs_v) ** 2)
            ss_tot = np.sum((obs_v - obs_v.mean()) ** 2)
            r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
            ax.scatter(obs_v, pred_v, c=bar_colors[idx], alpha=0.65,
                       edgecolors='k', linewidths=0.3, s=50, zorder=3)
            lim = (min(obs_v.min(), pred_v.min()) - 2,
                   max(obs_v.max(), pred_v.max()) + 2)
            ax.plot(lim, lim, 'k--', lw=1, label='1:1 line')
            ax.set_xlim(lim); ax.set_ylim(lim)
            xlabel_s = f'Observed ({unit})' if unit else 'Observed'
            ylabel_s = f'Predicted ({unit})' if unit else 'Predicted'
            ax.set_xlabel(xlabel_s, fontsize=10)
            ax.set_ylabel(ylabel_s, fontsize=10)
            ax.set_title(f'{mname}\nRMSE={rmse:.3f}  R²={r2:.4f}', fontsize=10)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

    # --- Bar panels ---
    for bi, (metric_name, vals) in enumerate(bar_panels):
        ax = axes[n_scatter + bi]
        bars = ax.bar(method_names, vals, color=bar_colors,
                      edgecolor="k", width=0.5)
        ax.set_title(f"LOOCV {metric_name}")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{v:.3f}", ha="center", fontsize=9)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis='y', alpha=0.35)

    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    plt.tight_layout()
    return fig, axes


def plot_loocv_bar(
    method_names, accuracies, mean_log_likelihoods,
    balanced_accuracies=None,
    colors=("steelblue", "seagreen", "darkorange"),
    suptitle="Station LOOCV",
    figsize=None,
):
    """Bar chart of LOOCV accuracy and mean log-likelihood.

    .. deprecated::
        Use :func:`plot_loocv` instead, which supports scatter panels
        and generic metrics in addition to bar charts.

    This function is a backward-compatible alias for :func:`plot_loocv`.
    """
    return plot_loocv(
        method_names,
        accuracies=accuracies,
        mean_log_likelihoods=mean_log_likelihoods,
        balanced_accuracies=balanced_accuracies,
        colors=colors, suptitle=suptitle, figsize=figsize,
    )


# ---------------------------------------------------------------------------
# 2-D spatial estimation maps (pcolormesh)
# ---------------------------------------------------------------------------

def plot_categorical_map(
    xx, yy,
    map_arrays,
    method_names,
    categories, labels, colors,
    cs_data=None, code_data=None,
    x_scale=1e3, x_unit="km",
    suptitle=None,
    figsize=None,
    ax=None,
):
    """Pcolormesh MAP-class map(s) with optional data overlay.

    Parameters
    ----------
    xx, yy : 2-D ndarrays, shape (nky, nkx)
        Meshgrid of X and Y coordinates (typically from ``np.meshgrid``).
    map_arrays : (nk,) array or list of (nk,) arrays
        MAP class at every grid node (integer codes).  Pass a list to
        produce one subplot per method.
    method_names : str or list of str
        Subplot title(s) aligned with *map_arrays*.
    categories : list of int
        Ordered class codes.
    labels : list of str
        Class names aligned with *categories*.
    colors : list of str
        Hex colours aligned with *categories*.
    cs_data : (n, 2) array or None
        Data coordinates for overlay scatter.  Skipped when ``None``.
    code_data : (n,) int array or None
        True class code at each data location (for coloured scatter).
        Required when *cs_data* is not ``None``.
    x_scale : float
        Divisor applied to all coordinates before plotting (default 1000
        → km).
    x_unit : str
        Axis unit label after scaling.
    suptitle : str or None
        Figure super-title.
    figsize : tuple or None
    ax : Axes or None
        Pre-existing axes.  Used only when a single *map_arrays* is passed.

    Returns
    -------
    fig, axes  — single Axes or 1-D array of Axes
    """
    import matplotlib.colors as mcolors

    single = not isinstance(map_arrays, list)
    if single:
        map_arrays  = [map_arrays]
        method_names = [method_names]

    n = len(map_arrays)
    nky, nkx = xx.shape
    cmap_cat = mcolors.ListedColormap(colors)
    bounds   = [c - 0.5 for c in categories] + [categories[-1] + 0.5]
    norm     = mcolors.BoundaryNorm(bounds, cmap_cat.N)

    if ax is not None and n == 1:
        fig  = ax.figure
        axes = [ax]
    else:
        if figsize is None:
            figsize = (5.5 * n, 5)
        fig, axes_arr = plt.subplots(1, n, figsize=figsize, squeeze=False)
        axes = axes_arr[0].tolist()

    for i, (map_arr, title) in enumerate(zip(map_arrays, method_names)):
        ax_i = axes[i]
        ax_i.pcolormesh(
            xx / x_scale, yy / x_scale,
            np.asarray(map_arr).reshape(nky, nkx),
            cmap=cmap_cat, norm=norm, shading="auto",
        )
        if cs_data is not None and code_data is not None:
            cs_data   = np.asarray(cs_data)
            code_data = np.asarray(code_data)
            for cat, col in zip(categories, colors):
                mask = code_data == cat
                ax_i.scatter(
                    cs_data[mask, 0] / x_scale,
                    cs_data[mask, 1] / x_scale,
                    c=col, s=25, edgecolors="k", linewidths=0.5, zorder=4,
                )
        ax_i.set_title(title, fontsize=9)
        ax_i.set_xlabel(f"Easting ({x_unit})", fontsize=8)
        ax_i.set_ylabel(f"Northing ({x_unit})", fontsize=8)

    patches = [mpatches.Patch(color=c, label=l)
               for c, l in zip(colors, labels)]
    fig.legend(handles=patches, loc="lower center", ncol=len(categories),
               fontsize=8, bbox_to_anchor=(0.5, -0.04))
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    plt.tight_layout()
    return fig, axes[0] if single else axes


def plot_categorical_probmap(
    xx, yy,
    prob_arrays,
    method_names,
    vmin=None, vmax=None,
    cmap="YlOrRd",
    cbar_label="P(MAP class)",
    x_scale=1e3, x_unit="km",
    suptitle=None,
    figsize=None,
    ax=None,
):
    """Pcolormesh maximum-posterior-probability (confidence) map(s).

    Parameters
    ----------
    xx, yy : 2-D ndarrays, shape (nky, nkx)
        Meshgrid coordinates.
    prob_arrays : (nk,) array or list of (nk,) arrays
        Maximum posterior probability at every grid node.
    method_names : str or list of str
        Subplot titles.
    vmin, vmax : float or None
        Colour-scale limits.  Defaults to (min over data, 1.0) when
        ``None``.
    cmap : str
        Matplotlib colour map name.
    cbar_label : str
        Colour-bar label.
    x_scale : float
        Coordinate divisor (default 1000 → km).
    x_unit : str
    suptitle : str or None
    figsize : tuple or None
    ax : Axes or None
        Pre-existing axes (only used for the single-array case).

    Returns
    -------
    fig, axes
    """
    single = not isinstance(prob_arrays, list)
    if single:
        prob_arrays  = [prob_arrays]
        method_names = [method_names]

    n    = len(prob_arrays)
    nky, nkx = xx.shape
    _vmin = vmin if vmin is not None else min(np.asarray(p).min() for p in prob_arrays)
    _vmax = vmax if vmax is not None else 1.0

    if ax is not None and n == 1:
        fig  = ax.figure
        axes = [ax]
    else:
        if figsize is None:
            figsize = (5.5 * n, 4)
        fig, axes_arr = plt.subplots(1, n, figsize=figsize, squeeze=False)
        axes = axes_arr[0].tolist()

    for ax_i, prob_arr, title in zip(axes, prob_arrays, method_names):
        im = ax_i.pcolormesh(
            xx / x_scale, yy / x_scale,
            np.asarray(prob_arr).reshape(nky, nkx),
            cmap=cmap, vmin=_vmin, vmax=_vmax, shading="auto",
        )
        plt.colorbar(im, ax=ax_i, label=cbar_label)
        ax_i.set_title(title, fontsize=9)
        ax_i.set_xlabel(f"Easting ({x_unit})", fontsize=8)
        ax_i.set_ylabel(f"Northing ({x_unit})", fontsize=8)

    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    plt.tight_layout()
    return fig, axes[0] if single else axes


def plot_categorical_class_probs(
    xx, yy,
    posterior,
    categories, labels, colors,
    cs_data=None, code_data=None,
    x_scale=1e3, x_unit="km",
    suptitle=None,
    figsize=None,
):
    """2-D per-category posterior probability maps (one panel per class).

    Parameters
    ----------
    xx, yy : 2-D ndarrays, shape (nky, nkx)
        Meshgrid coordinates.
    posterior : (nk, nc) ndarray
        Full posterior probability matrix.
    categories : list of int
        Ordered class codes.
    labels : list of str
        Class names.
    colors : list of str
        Hex colours aligned with *categories* (used to tint the colormap
        for each panel: white → category colour).
    cs_data : (n, 2) array or None
        Data coordinates for overlay scatter.
    code_data : (n,) int array or None
        True class codes for coloured overlay scatter.
    x_scale : float
        Coordinate divisor (default 1000 → km).
    x_unit : str
    suptitle : str or None
    figsize : tuple or None

    Returns
    -------
    fig, axes  — 2-D axes grid of shape (ceil(nc/2), 2)
    """
    import matplotlib.colors as mcolors

    nc   = len(categories)
    nky, nkx = xx.shape
    cols = 2
    rows = int(np.ceil(nc / cols))

    if figsize is None:
        figsize = (5.5 * cols, 5 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes_flat = np.asarray(axes).ravel()

    post = np.asarray(posterior)
    for idx, (cat, lbl, col) in enumerate(zip(categories, labels, colors)):
        ax_i    = axes_flat[idx]
        prob_cat = post[:, idx].reshape(nky, nkx)
        cmap_i  = mcolors.LinearSegmentedColormap.from_list(
            f"cat_{cat}", ["white", col]
        )
        im = ax_i.pcolormesh(
            xx / x_scale, yy / x_scale, prob_cat,
            cmap=cmap_i, vmin=0, vmax=1, shading="auto",
        )
        plt.colorbar(im, ax=ax_i, label=f"P(Type {cat})")
        if cs_data is not None and code_data is not None:
            cs_arr   = np.asarray(cs_data)
            code_arr = np.asarray(code_data)
            for c2, c2col in zip(categories, colors):
                m = code_arr == c2
                ax_i.scatter(
                    cs_arr[m, 0] / x_scale, cs_arr[m, 1] / x_scale,
                    c=c2col, s=20, edgecolors="k", linewidths=0.4, zorder=4,
                )
        ax_i.set_title(f"P(soil = {cat}: {lbl})", fontsize=9)
        ax_i.set_xlabel(f"Easting ({x_unit})", fontsize=8)
        ax_i.set_ylabel(f"Northing ({x_unit})", fontsize=8)

    # hide unused panels
    for j in range(nc, len(axes_flat)):
        axes_flat[j].set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    plt.tight_layout()
    return fig, axes
