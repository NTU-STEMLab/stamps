"""
Data-driven identification of homogeneous regions for nonstationary BME.

Clusters stations (or data points) in a joint
``[weighted-coordinates, weighted-data]`` feature space so that the resulting
regions are both **spatially coherent** and **compositionally similar**.

Three clustering back-ends are available:

* ``'hierarchical'`` — agglomerative (Ward linkage by default), returns a
  dendrogram for visual exploration.  Recommended for moderate-size
  station sets (n < ~5 000).
* ``'kmeans'``       — fast, scalable, spherical clusters.
* ``'dbscan'``       — density-based, automatic cluster count, handles
  irregular shapes.  Outliers are labelled ``-1`` and are merged into
  the nearest cluster by centroid distance.

The output is fully compatible with
:func:`~stamps.stats.dependence.gw_pmodel.fit_regional_pmodels`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist
try:  # optional dependency
    from sklearn.preprocessing import StandardScaler
except ImportError:
    from stamps.general.optional import MissingDependency
    StandardScaler = MissingDependency('sklearn.preprocessing.StandardScaler')

__all__ = [
    "identify_homogeneous_regions_data",
]


def _merge_small_regions(
    labels: np.ndarray,
    coords: np.ndarray,
    min_samples: int,
) -> np.ndarray:
    """Merge regions with fewer than *min_samples* into the nearest larger region."""
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
    """Map arbitrary integer labels to 0, 1, 2, ... preserving order."""
    unique = np.unique(labels)
    mapping = {old: new for new, old in enumerate(unique)}
    return np.array([mapping[v] for v in labels])


# ──────────────────────────────────────────────────────────────────────────────
# Main public function
# ──────────────────────────────────────────────────────────────────────────────

def _build_connectivity(
    coords_scaled: np.ndarray,
    n_neighbors: int,
) -> "sparse matrix":
    """Build a symmetric k-nearest-neighbors connectivity graph."""
    from sklearn.neighbors import kneighbors_graph

    conn = kneighbors_graph(
        coords_scaled,
        n_neighbors=n_neighbors,
        mode="connectivity",
        include_self=False,
    )
    return conn + conn.T  # make symmetric


def _apply_connectivity_scale(
    coords: np.ndarray,
    scaler,
    connectivity_scale,
    d: int,
) -> np.ndarray:
    """Return coordinates scaled for the connectivity graph.

    Three modes for *connectivity_scale*:

    * ``'auto'`` — use z-scored coordinates (handles raw-scale differences).
    * *dict* with small values (all <= 10) — treated as **relative weights**
      applied *after* z-scoring (same key convention as *weights*:
      ``'xy'``, ``'z'``, ``'t'``, ``'coord'``).
    * *dict* with any value > 10 — treated as **absolute correlation ranges**;
      raw coordinates are divided by the range so that 1 unit ≈ 1 range.
    """
    if connectivity_scale is None or connectivity_scale == "auto":
        if scaler is not None:
            return scaler.transform(coords)
        return coords.copy()

    if not isinstance(connectivity_scale, dict):
        raise TypeError(
            "connectivity_scale must be 'auto', None, or a dict "
            "with keys like 'xy', 'z', 't', 'coord'."
        )

    is_absolute = any(v > 10 for v in connectivity_scale.values())

    if is_absolute:
        C = coords.copy()
        s_xy = connectivity_scale.get("xy")
        s_z = connectivity_scale.get("z")
        s_t = connectivity_scale.get("t")
        s_coord = connectivity_scale.get("coord")

        if s_coord is not None:
            C /= s_coord

        if s_xy is not None and d >= 2:
            C[:, 0] = coords[:, 0] / s_xy
            C[:, 1] = coords[:, 1] / s_xy
        if s_z is not None and d >= 3:
            C[:, 2] = coords[:, 2] / s_z
        if s_t is not None and d >= 4:
            C[:, 3] = coords[:, 3] / s_t
        return C

    # relative weights on z-scored coordinates
    if scaler is not None:
        C = scaler.transform(coords)
    else:
        C = coords.copy()

    w_default = connectivity_scale.get("coord", 1.0)
    col_w = np.full(d, w_default)
    w_xy = connectivity_scale.get("xy")
    w_z = connectivity_scale.get("z")
    w_t = connectivity_scale.get("t")
    if w_xy is not None and d >= 2:
        col_w[0] = w_xy
        col_w[1] = w_xy
    if w_z is not None and d >= 3:
        col_w[2] = w_z
    if w_t is not None and d >= 4:
        col_w[3] = w_t
    C *= col_w[np.newaxis, :]
    return C


def identify_homogeneous_regions_data(
    coords: np.ndarray,
    data_vectors: np.ndarray,
    n_regions: Optional[int] = None,
    method: str = "hierarchical",
    linkage_method: str = "ward",
    weights: Optional[Dict[str, float]] = None,
    min_samples_per_region: int = 5,
    standardize: bool = True,
    spatial_connectivity: bool = False,
    connectivity_k: Optional[int] = None,
    connectivity_scale=None,
    eps: Optional[float] = None,
    min_dbscan_samples: int = 3,
    return_extras: bool = False,
    verbose: bool = False,
) -> Union[
    Tuple[np.ndarray, Dict[str, Any]],
    Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]],
]:
    """Identify homogeneous regions from combined coordinate + data features.

    Constructs a joint feature matrix

    .. math::

        \\mathbf{f}_i = \\bigl[\\,w_c \\cdot \\tilde{\\mathbf{s}}_i
        \\;,\\; w_d \\cdot \\tilde{\\mathbf{p}}_i\\,\\bigr]

    where :math:`\\tilde{\\cdot}` denotes z-scored values, and clusters
    with the chosen algorithm.

    Parameters
    ----------
    coords : (n, d) array
        Station (or sample) coordinates.  Can be 2-D ``(X, Y)``,
        3-D ``(X, Y, Z)``, or space-time ``(X, Y, Z, T)``; the
        dimensionality is detected automatically.
    data_vectors : (n, nc) array
        Per-station data feature vectors — typically marginal class
        proportions, depth-stratified proportions, or transition rates.
    n_regions : int or None
        Target number of clusters.  **Required** for ``'hierarchical'``
        and ``'kmeans'``; ignored by ``'dbscan'``.
    method : ``{'hierarchical', 'kmeans', 'dbscan'}``
        Clustering algorithm.
    linkage_method : str
        Linkage criterion for ``'hierarchical'`` (e.g. ``'ward'``,
        ``'complete'``, ``'average'``, ``'single'``).  Default ``'ward'``.
    weights : dict or None
        Feature-space weighting.  Recognised keys:

        * ``'coord'``  — scalar weight for *all* coordinate columns (default 1.0)
        * ``'data'``   — scalar weight for *all* data columns (default 1.0)
        * ``'xy'``, ``'z'``, ``'t'`` — per-axis-type overrides (take
          precedence over ``'coord'``).

        Example::

            weights={'xy': 1.0, 'z': 0.5, 'data': 2.0}

    min_samples_per_region : int
        Regions with fewer members are merged into the nearest larger
        region by centroid distance.
    standardize : bool
        Z-score each feature column before weighting (default True).
    spatial_connectivity : bool
        If True, use ``sklearn.cluster.AgglomerativeClustering`` with a
        spatial connectivity constraint (k-nearest-neighbors graph built
        from coordinates only).  This **guarantees** that every station in
        a region is spatially connected to at least one other station in
        the same region.  Only applies to ``method='hierarchical'``.
    connectivity_k : int or None
        Number of spatial neighbours used to build the connectivity graph.
        Defaults to ``min(n - 1, max(5, n_regions * 2))``.
    connectivity_scale : ``'auto'``, dict, or None
        How coordinates are scaled *for the connectivity graph* (separate
        from the clustering feature weights):

        * ``'auto'`` / ``None`` — use z-scored coordinates.
        * *dict* with **relative weights** (values ≤ 10) applied after
          z-scoring, e.g. ``{'xy': 2.0, 'z': 0.5}``.
        * *dict* with **absolute correlation ranges** (any value > 10),
          e.g. ``{'xy': 5000, 'z': 50}``.  Raw coordinates are divided by
          these ranges so that 1 unit ≈ 1 correlation range.
    eps : float or None
        Neighbourhood radius for ``'dbscan'``.  If ``None``, a heuristic
        based on the knee of the k-distance graph is used.
    min_dbscan_samples : int
        ``min_samples`` parameter of DBSCAN (default 3).
    return_extras : bool
        If True, return a third element with diagnostic objects
        (``'linkage_matrix'``, ``'features'``, ``'scaler'``, etc.).
    verbose : bool
        Print progress messages.

    Returns
    -------
    region_labels : (n,) ndarray of int
        Contiguous region label (0, 1, 2, …) per input row.
    region_stats : dict
        Compatible with :func:`fit_regional_pmodels`.  Keys:

        * ``'n_centers'``  — (n_regions,) int array
        * ``'centers'``    — list of (n_k, d) coordinate arrays
        * ``'centroids'``  — (n_regions, d) array of region centroids
        * ``'data_mean'``  — (n_regions, nc) mean data vector per region
        * ``'data_std'``   — (n_regions, nc) std per region
        * ``'P_mean'``     — placeholder (nan) for API compatibility
        * ``'P_std'``      — placeholder (nan) for API compatibility

    extras : dict  *(only if return_extras=True)*
        ``'linkage_matrix'``, ``'features'``, ``'scaler'``,
        ``'method'``, ``'weights_used'``.
    """
    coords = np.asarray(coords, dtype=float)
    data_vectors = np.asarray(data_vectors, dtype=float)
    n, d = coords.shape
    nc = data_vectors.shape[1]

    if data_vectors.shape[0] != n:
        raise ValueError(
            f"coords has {n} rows but data_vectors has {data_vectors.shape[0]}"
        )

    if method in ("hierarchical", "kmeans") and n_regions is None:
        raise ValueError(
            f"n_regions is required for method='{method}'"
        )

    # ── 1. Standardise ────────────────────────────────────────────────────
    if standardize:
        scaler_c = StandardScaler().fit(coords)
        scaler_d = StandardScaler().fit(data_vectors)
        C = scaler_c.transform(coords)
        D = scaler_d.transform(data_vectors)
    else:
        scaler_c = scaler_d = None
        C = coords.copy()
        D = data_vectors.copy()

    # ── 2. Apply weights ──────────────────────────────────────────────────
    w = weights or {}
    w_coord_default = w.get("coord", 1.0)
    w_data = w.get("data", 1.0)

    col_weights_c = np.full(d, w_coord_default)
    w_xy = w.get("xy")
    w_z = w.get("z")
    w_t = w.get("t")

    if w_xy is not None and d >= 2:
        col_weights_c[0] = w_xy
        col_weights_c[1] = w_xy
    if w_z is not None and d >= 3:
        col_weights_c[2] = w_z
    if w_t is not None and d >= 4:
        col_weights_c[3] = w_t

    C *= col_weights_c[np.newaxis, :]
    D *= w_data

    F = np.hstack([C, D])
    weights_used = {"coord_cols": col_weights_c.tolist(), "data": w_data}

    if verbose:
        print(f"[identify_regions_data] n={n}, d_coord={d}, nc_data={nc}")
        print(f"  Feature matrix shape: {F.shape}")
        print(f"  Weights: {weights_used}")

    # ── 2b. Build spatial connectivity (if requested) ─────────────────────
    connectivity_matrix = None
    if spatial_connectivity and method in ("hierarchical", "kmeans"):
        k_conn = connectivity_k or min(n - 1, max(5, n_regions * 2))
        k_conn = min(k_conn, n - 1)
        coords_conn = _apply_connectivity_scale(
            coords, scaler_c, connectivity_scale, d
        )
        connectivity_matrix = _build_connectivity(coords_conn, k_conn)
        if verbose:
            print(f"  Spatial connectivity: k={k_conn}, "
                  f"scale={connectivity_scale or 'auto'}")

    # ── 3. Cluster ────────────────────────────────────────────────────────
    extras: Dict[str, Any] = {
        "features": F,
        "scaler_coord": scaler_c,
        "scaler_data": scaler_d,
        "method": method,
        "weights_used": weights_used,
    }

    if method == "hierarchical":
        if connectivity_matrix is not None:
            from sklearn.cluster import AgglomerativeClustering

            agg = AgglomerativeClustering(
                n_clusters=n_regions,
                connectivity=connectivity_matrix,
                linkage=linkage_method,
            )
            raw_labels = agg.fit_predict(F)
            extras["agglomerative_model"] = agg
            if verbose:
                print(f"  Connectivity-constrained hierarchical "
                      f"({linkage_method}): {n_regions} clusters")
        else:
            Z = linkage(F, method=linkage_method)
            raw_labels = fcluster(Z, n_regions, criterion="maxclust") - 1
            extras["linkage_matrix"] = Z
            if verbose:
                print(f"  Hierarchical ({linkage_method}): "
                      f"requested {n_regions} clusters")

    elif method == "kmeans":
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=n_regions, n_init=10, random_state=42)
        raw_labels = km.fit_predict(F)
        extras["kmeans_model"] = km
        if verbose:
            print(f"  K-means: k={n_regions}, inertia={km.inertia_:.2f}")

    elif method == "dbscan":
        from sklearn.cluster import DBSCAN
        from sklearn.neighbors import NearestNeighbors

        if eps is None:
            k = max(min_dbscan_samples, 2)
            nn = NearestNeighbors(n_neighbors=k)
            nn.fit(F)
            dists, _ = nn.kneighbors(F)
            k_dists = np.sort(dists[:, -1])
            diffs = np.diff(k_dists)
            knee_idx = int(np.argmax(diffs)) if len(diffs) > 0 else len(k_dists) // 2
            eps = float(k_dists[min(knee_idx + 1, len(k_dists) - 1)])
            if verbose:
                print(f"  DBSCAN auto-eps: {eps:.4f} (knee at index {knee_idx})")

        db = DBSCAN(eps=eps, min_samples=min_dbscan_samples)
        raw_labels = db.fit_predict(F)
        extras["dbscan_model"] = db
        extras["eps_used"] = eps

        noise_mask = raw_labels == -1
        if noise_mask.any():
            core_labels = raw_labels[~noise_mask]
            core_coords = coords[~noise_mask]
            if len(core_labels) == 0:
                raw_labels[:] = 0
            else:
                for idx in np.where(noise_mask)[0]:
                    dists = np.linalg.norm(core_coords - coords[idx], axis=1)
                    raw_labels[idx] = core_labels[np.argmin(dists)]
            if verbose:
                print(f"  DBSCAN: {noise_mask.sum()} noise points merged")

        if verbose:
            print(f"  DBSCAN: {len(np.unique(raw_labels))} clusters")

    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose from 'hierarchical', 'kmeans', 'dbscan'."
        )

    # ── 4. Merge small regions ────────────────────────────────────────────
    labels = _merge_small_regions(raw_labels, coords, min_samples_per_region)
    labels = _relabel_contiguous(labels)
    n_regions_final = len(np.unique(labels))

    if verbose:
        print(f"  Final regions: {n_regions_final}")
        for ri in range(n_regions_final):
            print(f"    Region {ri}: {(labels == ri).sum()} stations")

    # ── 5. Build region_stats (compatible with fit_regional_pmodels) ──────
    region_stats: Dict[str, Any] = {
        "n_centers": np.zeros(n_regions_final, dtype=int),
        "centers": [None] * n_regions_final,
        "centroids": np.zeros((n_regions_final, d)),
        "data_mean": np.zeros((n_regions_final, nc)),
        "data_std": np.zeros((n_regions_final, nc)),
        "P_mean": np.full((1, 1, 1, n_regions_final), np.nan),
        "P_std": np.full((1, 1, 1, n_regions_final), np.nan),
    }

    for ri in range(n_regions_final):
        mask = labels == ri
        region_stats["n_centers"][ri] = int(mask.sum())
        region_stats["centers"][ri] = coords[mask]
        region_stats["centroids"][ri] = coords[mask].mean(axis=0)
        region_stats["data_mean"][ri] = data_vectors[mask].mean(axis=0)
        if mask.sum() > 1:
            region_stats["data_std"][ri] = data_vectors[mask].std(axis=0)

    if return_extras:
        return labels, region_stats, extras
    return labels, region_stats
