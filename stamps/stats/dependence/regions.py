# -*- coding: utf-8 -*-
"""
stamps.stats.dependence.regions
================================
Unified identification of homogeneous regions for nonstationary BME.

Two complementary strategies are available under a single entry point
:func:`identify_homogeneous_regions`, selected via ``method``:

* ``'pmodel'`` — clusters centers by **similarity of their empirical
  bivariate probability tables** (KL-divergence, Hellinger, or Euclidean).
  Requires pre-computed spatial P-tables from
  :func:`~stamps.stats.dependence.probatablecalc_spatial.probatablecalc_spatial`.

* ``'feature'`` — clusters stations in a joint
  **[coordinate, compositional-data]** feature space.  Works directly on
  raw data without computing P-tables first; supports hierarchical, k-means,
  and DBSCAN back-ends.

Both strategies produce output compatible with
:func:`~stamps.stats.dependence.gw_pmodel.fit_regional_pmodels`.

Public API
----------
identify_homogeneous_regions   Unified dispatcher (method='pmodel'/'feature').
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

__all__ = ["identify_homogeneous_regions"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _merge_small_regions(
    labels: np.ndarray,
    coords: np.ndarray,
    min_samples: int,
) -> np.ndarray:
    """Merge regions with fewer than *min_samples* into the nearest larger."""
    labels = labels.copy()
    unique, counts = np.unique(labels, return_counts=True)
    large = unique[counts >= min_samples]
    small = unique[counts < min_samples]

    if len(large) == 0:
        return labels

    large_centroids = np.array(
        [coords[labels == lbl].mean(axis=0) for lbl in large]
    )

    for s_lbl in small:
        s_centroid = coords[labels == s_lbl].mean(axis=0)
        dists = np.linalg.norm(large_centroids - s_centroid, axis=1)
        nearest = large[np.argmin(dists)]
        labels[labels == s_lbl] = nearest

    return labels


def _relabel_contiguous(labels: np.ndarray) -> np.ndarray:
    """Map arbitrary integer labels to 0, 1, 2, … preserving sorted order."""
    unique = np.unique(labels)
    mapping = {old: new for new, old in enumerate(unique)}
    return np.array([mapping[v] for v in labels], dtype=int)


# ---------------------------------------------------------------------------
# Public unified entry point
# ---------------------------------------------------------------------------

def identify_homogeneous_regions(
    coords: np.ndarray,
    *,
    method: str = "pmodel",
    # ── pmodel branch ───────────────────────────────────────────────────────
    P: Optional[np.ndarray] = None,
    ptable_metric: str = "kl_divergence",
    distance_threshold: Optional[float] = None,
    # ── feature branch ──────────────────────────────────────────────────────
    data_vectors: Optional[np.ndarray] = None,
    cluster_backend: str = "hierarchical",
    weights: Optional[Dict[str, float]] = None,
    standardize: bool = True,
    spatial_connectivity: bool = False,
    connectivity_k: Optional[int] = None,
    connectivity_scale=None,
    eps: Optional[float] = None,
    min_dbscan_samples: int = 3,
    # ── shared ──────────────────────────────────────────────────────────────
    n_regions: Optional[int] = None,
    linkage_method: str = "ward",
    min_samples_per_region: int = 5,
    verbose: bool = False,
    return_extras: bool = False,
) -> Union[
    Tuple[np.ndarray, Dict[str, Any]],
    Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]],
]:
    """Identify homogeneous regions for nonstationary categorical BME.

    Partitions the spatial domain into sub-regions that are approximately
    stationary, so that a separate Pmodel can be fitted per region and
    blended geographically at estimation time (see
    :func:`~stamps.stats.dependence.gw_pmodel.fit_regional_pmodels`).

    Two complementary strategies are available via the ``method`` argument:

    ``method='pmodel'``
        Cluster centers by **similarity of their empirical bivariate
        probability tables**.  Uses a pairwise dissimilarity matrix computed
        from KL-divergence, Hellinger distance, or Euclidean distance on the
        flattened P-tables, then applies hierarchical clustering.

        *Requires:* ``P`` — the 4-D P-table array from
        :func:`probatablecalc_spatial`.

    ``method='feature'``
        Cluster stations in a joint **[coordinate, compositional-data]**
        feature space after z-scoring and optional per-axis weighting.
        Supports hierarchical, k-means, and DBSCAN back-ends, as well as
        spatial-connectivity-constrained clustering.

        *Requires:* ``data_vectors`` — (n, nc) compositional feature matrix
        (e.g. per-station class proportions or transition rates).

    Both strategies share the same output format and are fully compatible
    with :func:`fit_regional_pmodels`.

    Parameters
    ----------
    coords : (n, d) ndarray
        Station (or center) coordinates.
    method : ``{'pmodel', 'feature'}``
        Clustering strategy.  Default ``'pmodel'``.

    P-table branch parameters (``method='pmodel'``)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    P : (nc, nc, ncl, n_centers) ndarray
        Spatial P-tables from :func:`probatablecalc_spatial`.  Required.
    ptable_metric : ``{'kl_divergence', 'euclidean', 'hellinger'}``
        Dissimilarity metric between P-tables.  Default ``'kl_divergence'``.
        Note: ``'ward'`` linkage is automatically switched to ``'average'``
        for KL-divergence (Ward requires Euclidean geometry).
    distance_threshold : float or None
        Cut the dendrogram at this height instead of specifying ``n_regions``.
        One of ``n_regions`` or ``distance_threshold`` must be given for
        ``method='pmodel'``.

    Feature-space branch parameters (``method='feature'``)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    data_vectors : (n, nc) ndarray
        Per-station data feature vectors.  Required.
    cluster_backend : ``{'hierarchical', 'kmeans', 'dbscan'}``
        Clustering algorithm.  Default ``'hierarchical'``.
    weights : dict or None
        Feature-space weighting after z-scoring.  Recognised keys:
        ``'coord'``, ``'data'``, ``'xy'``, ``'z'``, ``'t'``.
        Example: ``{'xy': 1.0, 'z': 0.5, 'data': 2.0}``.
    standardize : bool
        Z-score feature columns before weighting (default True).
    spatial_connectivity : bool
        Enforce spatial k-NN connectivity constraint during hierarchical
        clustering (``AgglomerativeClustering``).
    connectivity_k : int or None
        Number of k-NN neighbours for the connectivity graph.
    connectivity_scale : ``'auto'``, dict, or None
        How coordinates are scaled for the connectivity graph.
    eps : float or None
        DBSCAN neighbourhood radius.  Auto-estimated from k-distance knee
        when ``None``.
    min_dbscan_samples : int
        DBSCAN ``min_samples`` (default 3).

    Shared parameters
    ~~~~~~~~~~~~~~~~~
    n_regions : int or None
        Target number of regions.  Required for ``'feature'`` with
        ``cluster_backend`` in ``{'hierarchical', 'kmeans'}`` and for
        ``method='pmodel'`` unless ``distance_threshold`` is given.
    linkage_method : str
        Linkage criterion for hierarchical clustering (default ``'ward'``).
    min_samples_per_region : int
        Regions with fewer members are merged into the nearest larger region
        by centroid distance (default 5).
    verbose : bool
        Print progress messages.
    return_extras : bool
        If True, return a third dict with diagnostic objects.

    Returns
    -------
    region_labels : (n,) ndarray of int
        Contiguous region label 0, 1, 2, … for each input row.
    region_stats : dict
        Compatible with :func:`fit_regional_pmodels`.  Keys:

        * ``'n_centers'``  — (R,) int array, count per region.
        * ``'centers'``    — list of R (n_k, d) coordinate arrays.
        * ``'centroids'``  — (R, d) region centroid coordinates.
        * ``'P_mean'``     — (nc, nc, ncl, R) mean P-table per region
          (pmodel branch only; NaN placeholder for feature branch).
        * ``'P_std'``      — same shape, std per region.
        * ``'data_mean'``  — (R, nc) mean data vector per region
          (feature branch only; NaN placeholder for pmodel branch).
        * ``'data_std'``   — same shape.

    extras : dict  *(only if return_extras=True)*
        Branch-specific diagnostic objects.
    """
    coords = np.asarray(coords, dtype=float)

    if method == "pmodel":
        return _cluster_pmodel(
            coords,
            P=P,
            ptable_metric=ptable_metric,
            n_regions=n_regions,
            distance_threshold=distance_threshold,
            linkage_method=linkage_method,
            min_samples_per_region=min_samples_per_region,
            verbose=verbose,
            return_extras=return_extras,
        )
    elif method == "feature":
        return _cluster_feature(
            coords,
            data_vectors=data_vectors,
            n_regions=n_regions,
            cluster_backend=cluster_backend,
            linkage_method=linkage_method,
            weights=weights,
            standardize=standardize,
            spatial_connectivity=spatial_connectivity,
            connectivity_k=connectivity_k,
            connectivity_scale=connectivity_scale,
            eps=eps,
            min_dbscan_samples=min_dbscan_samples,
            min_samples_per_region=min_samples_per_region,
            verbose=verbose,
            return_extras=return_extras,
        )
    else:
        raise ValueError(
            f"Unknown method {method!r}. Choose 'pmodel' or 'feature'."
        )


# ---------------------------------------------------------------------------
# P-table clustering branch
# ---------------------------------------------------------------------------

def _cluster_pmodel(
    coords, P, ptable_metric, n_regions, distance_threshold,
    linkage_method, min_samples_per_region, verbose, return_extras,
):
    """Internal: cluster by P-table similarity."""
    if P is None:
        raise ValueError(
            "P (4-D probability table array) is required for method='pmodel'."
        )
    if n_regions is None and distance_threshold is None:
        raise ValueError(
            "Provide either n_regions or distance_threshold for method='pmodel'."
        )

    P = np.asarray(P, dtype=float)
    n_centers = coords.shape[0]
    n_cata, _, n_limit, n_centers_P = P.shape

    if n_centers != n_centers_P:
        raise ValueError(
            f"coords has {n_centers} rows but P has {n_centers_P} centers."
        )

    if verbose:
        print(f"[identify_regions pmodel] {n_centers} centers, "
              f"metric={ptable_metric}, linkage={linkage_method}")

    # ── Pairwise dissimilarity ─────────────────────────────────────────────
    P_flat = P.transpose(3, 0, 1, 2).reshape(n_centers, -1)
    P_flat = np.clip(P_flat, 1e-10, None)

    if ptable_metric == "kl_divergence":
        P_flat /= P_flat.sum(axis=1, keepdims=True)
        dist_mat = np.zeros((n_centers, n_centers), dtype=float)
        for i in range(n_centers):
            for j in range(i + 1, n_centers):
                pi, pj = P_flat[i], P_flat[j]
                kl_ij = float(np.sum(pi * np.log(pi / pj)))
                kl_ji = float(np.sum(pj * np.log(pj / pi)))
                sym   = (kl_ij + kl_ji) / 2.0
                sym   = sym if np.isfinite(sym) else 1e10
                dist_mat[i, j] = dist_mat[j, i] = sym

        # Ward requires Euclidean; switch to average for KL
        if linkage_method == "ward":
            if verbose:
                print("  Switching linkage 'ward' → 'average' (KL is non-Euclidean).")
            linkage_method = "average"

    elif ptable_metric == "euclidean":
        dist_mat = np.zeros((n_centers, n_centers), dtype=float)
        from scipy.spatial.distance import cdist
        dist_mat = cdist(P_flat, P_flat, metric="euclidean")

    elif ptable_metric == "hellinger":
        P_sqrt = np.sqrt(np.clip(P_flat, 0, None))
        from scipy.spatial.distance import cdist
        dist_mat = cdist(P_sqrt, P_sqrt, metric="euclidean")

    else:
        raise ValueError(
            f"ptable_metric must be 'kl_divergence', 'euclidean', or 'hellinger'; "
            f"got {ptable_metric!r}."
        )

    condensed = dist_mat[np.triu_indices(n_centers, k=1)]
    condensed = np.nan_to_num(condensed, nan=1e10, posinf=1e10, neginf=1e10)

    # ── Hierarchical clustering ────────────────────────────────────────────
    Z_link = linkage(condensed, method=linkage_method)

    if n_regions is not None:
        raw_labels = fcluster(Z_link, n_regions, criterion="maxclust")
    else:
        raw_labels = fcluster(Z_link, distance_threshold, criterion="distance")

    # ── Small-region merging and relabeling ────────────────────────────────
    labels = _merge_small_regions(raw_labels, coords, min_samples_per_region)
    labels = _relabel_contiguous(labels)
    n_reg  = int(np.unique(labels).size)

    if verbose:
        print(f"  Final regions: {n_reg}")
        for r in range(n_reg):
            print(f"    Region {r}: {(labels == r).sum()} centers")

    # ── Build region_stats ─────────────────────────────────────────────────
    nc = data_mean_shape = P.shape[0]
    region_stats: Dict[str, Any] = {
        "n_centers": np.zeros(n_reg, dtype=int),
        "centers":   [None] * n_reg,
        "centroids": np.zeros((n_reg, coords.shape[1])),
        "P_mean":    np.zeros((n_cata, n_cata, n_limit, n_reg)),
        "P_std":     np.zeros((n_cata, n_cata, n_limit, n_reg)),
        "data_mean": np.full((n_reg, nc), np.nan),
        "data_std":  np.full((n_reg, nc), np.nan),
    }
    for r in range(n_reg):
        mask = labels == r
        region_stats["n_centers"][r] = int(mask.sum())
        region_stats["centers"][r]   = coords[mask]
        region_stats["centroids"][r] = coords[mask].mean(axis=0)
        Pr = P[:, :, :, mask]
        region_stats["P_mean"][:, :, :, r] = np.nanmean(Pr, axis=3)
        if mask.sum() > 1:
            region_stats["P_std"][:, :, :, r] = np.nanstd(Pr, axis=3)

    extras = {
        "method": "pmodel",
        "ptable_metric": ptable_metric,
        "linkage_matrix": Z_link,
        "distance_matrix": dist_mat,
    }
    if return_extras:
        return labels, region_stats, extras
    return labels, region_stats


# ---------------------------------------------------------------------------
# Feature-space clustering branch
# ---------------------------------------------------------------------------

def _build_connectivity(coords_scaled, n_neighbors):
    from sklearn.neighbors import kneighbors_graph
    G = kneighbors_graph(
        coords_scaled, n_neighbors=n_neighbors,
        mode="connectivity", include_self=False,
    )
    return (G + G.T)  # symmetrize


def _apply_connectivity_scale(coords, scaler, connectivity_scale, d):
    if connectivity_scale is None or connectivity_scale == "auto":
        if scaler is not None:
            return scaler.transform(coords)
        return coords.copy()
    if not isinstance(connectivity_scale, dict):
        raise TypeError(
            "connectivity_scale must be 'auto', None, or a dict."
        )
    is_absolute = any(v > 10 for v in connectivity_scale.values())
    if is_absolute:
        C = coords.copy()
        for key, val in connectivity_scale.items():
            if key == "coord":
                C /= val
            elif key == "xy" and d >= 2:
                C[:, 0] /= val; C[:, 1] /= val
            elif key == "z" and d >= 3:
                C[:, 2] /= val
            elif key == "t" and d >= 4:
                C[:, 3] /= val
        return C
    # Relative weights on z-scored coordinates
    C = scaler.transform(coords) if scaler is not None else coords.copy()
    col_w = np.full(d, connectivity_scale.get("coord", 1.0))
    if "xy" in connectivity_scale and d >= 2:
        col_w[0] = col_w[1] = connectivity_scale["xy"]
    if "z"  in connectivity_scale and d >= 3:
        col_w[2] = connectivity_scale["z"]
    if "t"  in connectivity_scale and d >= 4:
        col_w[3] = connectivity_scale["t"]
    return C * col_w[np.newaxis, :]


def _cluster_feature(
    coords, data_vectors, n_regions, cluster_backend, linkage_method,
    weights, standardize, spatial_connectivity, connectivity_k,
    connectivity_scale, eps, min_dbscan_samples,
    min_samples_per_region, verbose, return_extras,
):
    """Internal: cluster in joint coordinate + data feature space."""
    from sklearn.preprocessing import StandardScaler

    if data_vectors is None:
        raise ValueError(
            "data_vectors is required for method='feature'."
        )

    data_vectors = np.asarray(data_vectors, dtype=float)
    n, d = coords.shape
    nc   = data_vectors.shape[1]

    if data_vectors.shape[0] != n:
        raise ValueError(
            f"coords has {n} rows but data_vectors has {data_vectors.shape[0]}."
        )
    if cluster_backend in ("hierarchical", "kmeans") and n_regions is None:
        raise ValueError(
            f"n_regions is required for cluster_backend='{cluster_backend}'."
        )

    # ── Standardise ───────────────────────────────────────────────────────
    if standardize:
        scaler_c = StandardScaler().fit(coords)
        scaler_d = StandardScaler().fit(data_vectors)
        C = scaler_c.transform(coords)
        D = scaler_d.transform(data_vectors)
    else:
        scaler_c = scaler_d = None
        C = coords.copy()
        D = data_vectors.copy()

    # ── Apply weights ─────────────────────────────────────────────────────
    w = weights or {}
    col_w_c = np.full(d, w.get("coord", 1.0))
    if "xy" in w and d >= 2:
        col_w_c[0] = col_w_c[1] = w["xy"]
    if "z"  in w and d >= 3:
        col_w_c[2] = w["z"]
    if "t"  in w and d >= 4:
        col_w_c[3] = w["t"]
    w_data = w.get("data", 1.0)
    C *= col_w_c[np.newaxis, :]
    D *= w_data
    F  = np.hstack([C, D])

    if verbose:
        print(f"[identify_regions feature] n={n}, d_coord={d}, nc_data={nc}, "
              f"backend={cluster_backend}")
        print(f"  Feature matrix shape: {F.shape}")

    # ── Spatial connectivity ───────────────────────────────────────────────
    connectivity_matrix = None
    if spatial_connectivity and cluster_backend in ("hierarchical", "kmeans"):
        k_conn = connectivity_k or min(n - 1, max(5, (n_regions or 2) * 2))
        k_conn = min(k_conn, n - 1)
        coords_conn = _apply_connectivity_scale(
            coords, scaler_c, connectivity_scale, d
        )
        connectivity_matrix = _build_connectivity(coords_conn, k_conn)
        if verbose:
            print(f"  Spatial connectivity k={k_conn}")

    # ── Cluster ───────────────────────────────────────────────────────────
    extras: Dict[str, Any] = {
        "method": "feature",
        "cluster_backend": cluster_backend,
        "features": F,
        "scaler_coord": scaler_c,
        "scaler_data":  scaler_d,
        "weights_used": {"coord_cols": col_w_c.tolist(), "data": w_data},
    }

    if cluster_backend == "hierarchical":
        if connectivity_matrix is not None:
            from sklearn.cluster import AgglomerativeClustering
            agg = AgglomerativeClustering(
                n_clusters=n_regions,
                connectivity=connectivity_matrix,
                linkage=linkage_method,
            )
            raw_labels = agg.fit_predict(F)
            extras["agglomerative_model"] = agg
        else:
            Z_link = linkage(F, method=linkage_method)
            raw_labels = fcluster(Z_link, n_regions, criterion="maxclust") - 1
            extras["linkage_matrix"] = Z_link
        if verbose:
            print(f"  Hierarchical ({linkage_method}): {n_regions} clusters")

    elif cluster_backend == "kmeans":
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_regions, n_init=10, random_state=42)
        raw_labels = km.fit_predict(F)
        extras["kmeans_model"] = km
        if verbose:
            print(f"  K-means: k={n_regions}, inertia={km.inertia_:.2f}")

    elif cluster_backend == "dbscan":
        from sklearn.cluster import DBSCAN
        from sklearn.neighbors import NearestNeighbors
        if eps is None:
            k_eps = max(min_dbscan_samples, 2)
            nn = NearestNeighbors(n_neighbors=k_eps).fit(F)
            dists, _ = nn.kneighbors(F)
            k_dists = np.sort(dists[:, -1])
            diffs = np.diff(k_dists)
            knee  = int(np.argmax(diffs)) if len(diffs) > 0 else len(k_dists) // 2
            eps   = float(k_dists[min(knee + 1, len(k_dists) - 1)])
            if verbose:
                print(f"  DBSCAN auto-eps={eps:.4f}")
        db = DBSCAN(eps=eps, min_samples=min_dbscan_samples)
        raw_labels = db.fit_predict(F)
        extras["dbscan_model"] = db
        extras["eps_used"] = eps
        noise = raw_labels == -1
        if noise.any():
            core_labels  = raw_labels[~noise]
            core_coords  = coords[~noise]
            for idx in np.where(noise)[0]:
                if len(core_labels) == 0:
                    raw_labels[idx] = 0
                else:
                    raw_labels[idx] = core_labels[
                        np.argmin(np.linalg.norm(core_coords - coords[idx], axis=1))
                    ]
        if verbose:
            print(f"  DBSCAN: {len(np.unique(raw_labels))} clusters")

    else:
        raise ValueError(
            f"Unknown cluster_backend {cluster_backend!r}. "
            "Choose 'hierarchical', 'kmeans', or 'dbscan'."
        )

    # ── Small-region merging and relabeling ────────────────────────────────
    labels = _merge_small_regions(raw_labels, coords, min_samples_per_region)
    labels = _relabel_contiguous(labels)
    n_reg  = int(np.unique(labels).size)

    if verbose:
        print(f"  Final regions: {n_reg}")

    # ── Build region_stats ─────────────────────────────────────────────────
    region_stats: Dict[str, Any] = {
        "n_centers": np.zeros(n_reg, dtype=int),
        "centers":   [None] * n_reg,
        "centroids": np.zeros((n_reg, d)),
        "data_mean": np.zeros((n_reg, nc)),
        "data_std":  np.zeros((n_reg, nc)),
        "P_mean":    np.full((1, 1, 1, n_reg), np.nan),
        "P_std":     np.full((1, 1, 1, n_reg), np.nan),
    }
    for r in range(n_reg):
        mask = labels == r
        region_stats["n_centers"][r] = int(mask.sum())
        region_stats["centers"][r]   = coords[mask]
        region_stats["centroids"][r] = coords[mask].mean(axis=0)
        region_stats["data_mean"][r] = data_vectors[mask].mean(axis=0)
        if mask.sum() > 1:
            region_stats["data_std"][r] = data_vectors[mask].std(axis=0)

    if return_extras:
        return labels, region_stats, extras
    return labels, region_stats
