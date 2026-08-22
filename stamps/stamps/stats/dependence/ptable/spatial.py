"""
Spatial-varying probability table calculation for non-stationary categorical BME.

This module provides functions for:
1. Calculating spatial-varying probability tables (one per location/region)
2. Identifying homogeneous regions based on similarity of probability tables
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import pairwise_distances
import warnings


def probatablecalc_spatial(
    coord, 
    value, 
    coord_limit, 
    center_coords=None,
    max_samples=None,
    anisotropic=False,
    ShowCalDis=False
):
    """
    Calculate spatial-varying probability tables centered at each location.
    
    This function computes a probability table for each center location by considering
    only pairs of points near that center. This allows for non-stationary (spatial-varying)
    probability tables.
    
    **Anisotropic Mode:**
    When anisotropic=True and coord has 3D coordinates (X, Y, Z), the function calculates
    separate probability tables for:
    1. X-Y plane (horizontal): Uses 2D distance sqrt((X_i - X_j)^2 + (Y_i - Y_j)^2)
    2. Z direction (vertical): Uses 1D distance |Z_i - Z_j|
    
    This allows modeling different spatial correlation structures in horizontal vs. vertical
    directions, which is important for geological/stratigraphic data where horizontal and
    vertical spatial relationships can differ significantly.
    
    Args:
        coord (ndarray): 2-D ndarray of floats with shape (n, d). Coordinates for 
            the locations where the categories are known. For 3D data, columns should be [X, Y, Z].
        value (ndarray): 2-D ndarray of floats with shape (n, nc). Probability for 
            the categories at the coordinates specified in coord.
        coord_limit (ndarray): 1-D ndarray of floats with shape (ncl,). Limits of 
            the distance classes for estimating probability tables.
        center_coords (ndarray, optional): 2-D ndarray of floats with shape (n_centers, d).
            Coordinates of center locations where probability tables will be calculated.
            If None, uses all points in coord as centers.
        max_samples (int, optional): Maximum number of pairs to sample for each center.
            If None, uses all pairs within the maximum distance limit.
        anisotropic (bool): If True and coord has 3D coordinates, calculate separate
            probability tables for X-Y plane and Z direction. Default: False.
        ShowCalDis (bool): Print the calculating process.
    
    Returns:
        If anisotropic=False (default):
            center_coords (ndarray): Coordinates of center locations (n_centers, d).
            D (ndarray): 2-D ndarray of floats with shape (n_centers, ncl). Mean distances 
                for each distance class at each center. First column is null distance (0).
            P (ndarray): 4-D ndarray of floats with shape (nc, nc, ncl, n_centers). 
                Spatial-varying bivariate probability tables. P[:,:,lag,center] is the
                probability table for distance class lag at center center.
            O (ndarray): 2-D ndarray of integers with shape (n_centers, ncl). Number of pairs 
                in each distance class at each center.
        
        If anisotropic=True and coord has 3D coordinates:
            center_coords (ndarray): Coordinates of center locations (n_centers, d).
            results_dict (dict): Dictionary with keys:
                - 'xy': Tuple of (D_xy, P_xy, O_xy) for X-Y plane (horizontal) probability tables
                - 'z': Tuple of (D_z, P_z, O_z) for Z direction (vertical) probability tables
                where each tuple follows the same format as the default return values.
                D_xy uses 2D distance sqrt((X_i - X_j)^2 + (Y_i - Y_j)^2)
                D_z uses 1D distance |Z_i - Z_j|
    
    Note:
        For each center location, only pairs within a local neighborhood are considered.
        The neighborhood size is determined by the maximum distance in coord_limit.
        This allows the probability table to vary spatially.
    """
    coord = np.asarray(coord, dtype=np.float64)
    value = np.asarray(value, dtype=np.float64)
    coord_limit = np.asarray(coord_limit, dtype=np.float64).flatten()
    
    n_data = coord.shape[0]
    n_cata = value.shape[1]
    n_limit = coord_limit.shape[0]
    coord_dim = coord.shape[1]
    
    # Validate inputs
    if coord.shape[0] != value.shape[0]:
        raise ValueError("coord and value must have the same number of rows")
    
    if n_limit < 1:
        raise ValueError("coord_limit must have at least one element")
    
    # Check if anisotropic mode is requested and if coordinates are 3D
    if anisotropic and coord_dim != 3:
        if ShowCalDis:
            warnings.warn(f"Anisotropic mode requested but coord has {coord_dim} dimensions (expected 3D). Disabling anisotropic mode.")
        anisotropic = False
    
    # Set center coordinates
    if center_coords is None:
        center_coords = coord.copy()
    else:
        center_coords = np.asarray(center_coords, dtype=np.float64)
    
    n_centers = center_coords.shape[0]
    max_distance = coord_limit[-1]  # Maximum distance to consider for each center
    
    if ShowCalDis:
        if anisotropic:
            print(f'Calculating ANISOTROPIC spatial-varying probability tables for {n_centers} centers...')
            print(f'  - X-Y plane (horizontal) probability tables')
            print(f'  - Z direction (vertical) probability tables')
        else:
            print(f'Calculating spatial-varying probability tables for {n_centers} centers...')
        print(f'Using maximum neighborhood distance: {max_distance:.2f}')
    
    # If anisotropic mode, calculate separate tables for X-Y and Z
    if anisotropic:
        # Calculate X-Y plane probability tables
        if ShowCalDis:
            print('\n=== Calculating X-Y plane (horizontal) probability tables ===')
        center_coords_xy, D_xy, P_xy, O_xy = _probatablecalc_spatial_anisotropic(
            coord, value, coord_limit, center_coords, max_samples, 
            direction='xy', max_distance=max_distance, ShowCalDis=ShowCalDis
        )
        
        # Calculate Z direction probability tables
        if ShowCalDis:
            print('\n=== Calculating Z direction (vertical) probability tables ===')
        center_coords_z, D_z, P_z, O_z = _probatablecalc_spatial_anisotropic(
            coord, value, coord_limit, center_coords, max_samples,
            direction='z', max_distance=max_distance, ShowCalDis=ShowCalDis
        )
        
        # Return dictionary with both results
        results_dict = {
            'xy': (D_xy, P_xy, O_xy),
            'z': (D_z, P_z, O_z)
        }
        return center_coords, results_dict
    
    # Standard (isotropic) mode - continue with existing logic
    # Initialize accumulation arrays for all centers
    P_sum = np.zeros((n_cata, n_cata, n_limit, n_centers), dtype=np.float64)
    P_count = np.zeros((n_cata, n_cata, n_limit, n_centers), dtype=np.float64)
    D_sum = np.zeros((n_limit, n_centers), dtype=np.float64)
    O = np.zeros((n_limit, n_centers), dtype=np.float64)
    
    # MEMORY OPTIMIZATION: Process centers in batches to avoid excessive memory usage
    # This prevents OOM (Out of Memory) errors when processing many centers
    # Use smaller batches for large numbers of centers
    if n_centers > 50:
        batch_size = 5  # Small batches for many centers
    elif n_centers > 20:
        batch_size = 10
    else:
        batch_size = min(10, n_centers)
    
    for batch_start in range(0, n_centers, batch_size):
        batch_end = min(batch_start + batch_size, n_centers)
        batch_centers = center_coords[batch_start:batch_end]
        
        if ShowCalDis:
            print(f'Processing centers {batch_start + 1}-{batch_end}/{n_centers} (batch {batch_start//batch_size + 1})...')
        
        for batch_idx, center_coord in enumerate(batch_centers):
            center_idx = batch_start + batch_idx
            
            # Find points within the neighborhood of this center
            # OPTIMIZATION: Use vectorized distance calculation instead of cdist for single point
            distances_to_center = np.linalg.norm(coord - center_coord, axis=1)
            neighbor_mask = distances_to_center <= max_distance
            neighbor_indices = np.where(neighbor_mask)[0]
            n_neighbors = len(neighbor_indices)
            
            if n_neighbors < 2:
                if ShowCalDis:
                    warnings.warn(f"Center {center_idx} has fewer than 2 neighbors. Skipping.")
                continue
            
            # Extract neighbor data only when needed (save memory)
            neighbor_coord = coord[neighbor_indices]
            neighbor_value = value[neighbor_indices]
            
            # OPTIMIZATION: Only calculate upper triangle (row < col) since probability tables are symmetric
            # This reduces calculations by ~50% (both distance and probability calculations)
            
            # Get upper triangle indices (excluding diagonal) first
            row_indices, col_indices = np.triu_indices(n_neighbors, k=1)
            n_pairs_total = len(row_indices)
            
            # MEMORY OPTIMIZATION: Aggressive sampling for large neighborhoods
            # Automatically reduce max_samples if neighborhood is very large
            effective_max_samples = max_samples
            if max_samples is not None and n_neighbors > 1000:
                # For very large neighborhoods, limit to reasonable number of pairs
                effective_max_samples = min(max_samples, 5000)
            
            # Sample pairs if needed (before computing distances and probabilities)
            if effective_max_samples is not None and n_pairs_total > effective_max_samples:
                sample_indices = np.random.choice(n_pairs_total, size=effective_max_samples, replace=False)
                row_indices = row_indices[sample_indices]
                col_indices = col_indices[sample_indices]
                n_pairs = len(row_indices)
            else:
                n_pairs = n_pairs_total
            
            # Calculate distances only for selected pairs (upper triangle)
            # OPTIMIZATION: Use vectorized distance calculation
            coord_i = neighbor_coord[row_indices]
            coord_j = neighbor_coord[col_indices]
            # Calculate 3D Euclidean distance (for standard isotropic mode)
            pair_distances = np.linalg.norm(coord_i - coord_j, axis=1)
            
            # Compute probability tensor only for selected pairs (upper triangle)
            # Shape: (n_pairs, n_cata, n_cata)
            prob_selected = neighbor_value[row_indices, :, None] * neighbor_value[col_indices, None, :]
            prob_sum = prob_selected.sum(axis=(-2, -1), keepdims=True)
            prob_sum[prob_sum == 0] = 1.0
            prob_selected = prob_selected / prob_sum
            
            # Clear intermediate arrays to free memory
            del coord_i, coord_j, neighbor_coord, neighbor_value
            
            # Process distance = 0 (bin 0) - only upper triangle, so no division needed
            idx_zero = pair_distances == 0
            if idx_zero.any():
                select_prob_zero = prob_selected[idx_zero]
                n_zero = idx_zero.sum()
                
                O[0, center_idx] += n_zero
                P_sum[:, :, 0, center_idx] += select_prob_zero.sum(axis=0)
                P_count[:, :, 0, center_idx] += n_zero
                del select_prob_zero
            
            # Process other distance bins (bins 1 to n_limit-1)
            # Since we only process upper triangle, divide by 2 to match full matrix behavior
            weight = 0.5  # Accounts for processing only upper triangle
            
            for idx, (left, right) in enumerate(zip(coord_limit[:-1], coord_limit[1:])):
                # Find pairs in this distance bin
                mask_bin = (pair_distances > left) & (pair_distances <= right)
                
                if mask_bin.any():
                    select_dis = pair_distances[mask_bin]
                    select_prob = prob_selected[mask_bin]
                    n_pairs_bin = mask_bin.sum()
                    
                    D_sum[idx+1, center_idx] += select_dis.sum() * weight
                    O[idx+1, center_idx] += n_pairs_bin * weight
                    P_sum[:, :, idx+1, center_idx] += select_prob.sum(axis=0)
                    P_count[:, :, idx+1, center_idx] += n_pairs_bin
            
            # Clear arrays after processing to free memory
            del prob_selected, pair_distances, row_indices, col_indices
            
        # Force garbage collection after each batch to free memory
        import gc
        gc.collect()
    
    # Compute final statistics for each center
    with np.errstate(divide='ignore', invalid='ignore'):
        D = D_sum / O
        P = P_sum / P_count
    
    # Handle empty bins
    empty_bins = (O == 0)
    if empty_bins.any():
        D[empty_bins] = np.nan
        for center_idx in range(n_centers):
            for lag_idx in range(n_limit):
                if empty_bins[lag_idx, center_idx]:
                    P[:, :, lag_idx, center_idx] = np.nan
    
    return center_coords, D.T, P, O.T


def _probatablecalc_spatial_anisotropic(
    coord, value, coord_limit, center_coords, max_samples, 
    direction='xy', max_distance=None, ShowCalDis=False
):
    """
    Internal helper function for anisotropic calculation.
    
    Calculates probability tables separately for X-Y plane or Z direction.
    """
    coord = np.asarray(coord, dtype=np.float64)
    value = np.asarray(value, dtype=np.float64)
    coord_limit = np.asarray(coord_limit, dtype=np.float64).flatten()
    
    n_data = coord.shape[0]
    n_cata = value.shape[1]
    n_limit = coord_limit.shape[0]
    n_centers = center_coords.shape[0]
    
    if max_distance is None:
        max_distance = coord_limit[-1]
    
    # Initialize accumulation arrays for all centers
    P_sum = np.zeros((n_cata, n_cata, n_limit, n_centers), dtype=np.float64)
    P_count = np.zeros((n_cata, n_cata, n_limit, n_centers), dtype=np.float64)
    D_sum = np.zeros((n_limit, n_centers), dtype=np.float64)
    O = np.zeros((n_limit, n_centers), dtype=np.float64)
    
    # MEMORY OPTIMIZATION: Process centers in batches
    if n_centers > 50:
        batch_size = 5
    elif n_centers > 20:
        batch_size = 10
    else:
        batch_size = min(10, n_centers)
    
    for batch_start in range(0, n_centers, batch_size):
        batch_end = min(batch_start + batch_size, n_centers)
        batch_centers = center_coords[batch_start:batch_end]
        
        for batch_idx, center_coord in enumerate(batch_centers):
            center_idx = batch_start + batch_idx
            
            # Find points within the neighborhood (using 3D distance for neighborhood search)
            distances_to_center = np.linalg.norm(coord - center_coord, axis=1)
            neighbor_mask = distances_to_center <= max_distance
            neighbor_indices = np.where(neighbor_mask)[0]
            n_neighbors = len(neighbor_indices)
            
            if n_neighbors < 2:
                continue
            
            neighbor_coord = coord[neighbor_indices]
            neighbor_value = value[neighbor_indices]
            
            # Get upper triangle indices
            row_indices, col_indices = np.triu_indices(n_neighbors, k=1)
            n_pairs_total = len(row_indices)
            
            # Sample pairs if needed
            effective_max_samples = max_samples
            if max_samples is not None and n_neighbors > 1000:
                effective_max_samples = min(max_samples, 5000)
            
            if effective_max_samples is not None and n_pairs_total > effective_max_samples:
                sample_indices = np.random.choice(n_pairs_total, size=effective_max_samples, replace=False)
                row_indices = row_indices[sample_indices]
                col_indices = col_indices[sample_indices]
            
            coord_i = neighbor_coord[row_indices]
            coord_j = neighbor_coord[col_indices]
            
            # Calculate distance based on direction
            if direction == 'xy':
                # X-Y plane distance: sqrt((X_i - X_j)^2 + (Y_i - Y_j)^2)
                pair_distances = np.sqrt(
                    (coord_i[:, 0] - coord_j[:, 0])**2 + 
                    (coord_i[:, 1] - coord_j[:, 1])**2
                )
            elif direction == 'z':
                # Z direction distance: |Z_i - Z_j|
                pair_distances = np.abs(coord_i[:, 2] - coord_j[:, 2])
            else:
                raise ValueError(f"Unknown direction: {direction}. Must be 'xy' or 'z'")
            
            # Compute probability tensor
            prob_selected = neighbor_value[row_indices, :, None] * neighbor_value[col_indices, None, :]
            prob_sum = prob_selected.sum(axis=(-2, -1), keepdims=True)
            prob_sum[prob_sum == 0] = 1.0
            prob_selected = prob_selected / prob_sum
            
            # Clear intermediate arrays
            del coord_i, coord_j, neighbor_coord, neighbor_value
            
            # Process distance = 0 (bin 0)
            idx_zero = pair_distances == 0
            if idx_zero.any():
                select_prob_zero = prob_selected[idx_zero]
                n_zero = idx_zero.sum()
                O[0, center_idx] += n_zero
                P_sum[:, :, 0, center_idx] += select_prob_zero.sum(axis=0)
                P_count[:, :, 0, center_idx] += n_zero
                del select_prob_zero
            
            # Process other distance bins
            weight = 0.5
            
            for idx, (left, right) in enumerate(zip(coord_limit[:-1], coord_limit[1:])):
                mask_bin = (pair_distances > left) & (pair_distances <= right)
                
                if mask_bin.any():
                    select_dis = pair_distances[mask_bin]
                    select_prob = prob_selected[mask_bin]
                    n_pairs_bin = mask_bin.sum()
                    
                    D_sum[idx+1, center_idx] += select_dis.sum() * weight
                    O[idx+1, center_idx] += n_pairs_bin * weight
                    P_sum[:, :, idx+1, center_idx] += select_prob.sum(axis=0)
                    P_count[:, :, idx+1, center_idx] += n_pairs_bin
            
            del prob_selected, pair_distances, row_indices, col_indices
        
        # Force garbage collection after each batch
        import gc
        gc.collect()
    
    # Compute final statistics
    with np.errstate(divide='ignore', invalid='ignore'):
        D = D_sum / O
        P = P_sum / P_count
    
    # Handle empty bins
    empty_bins = (O == 0)
    if empty_bins.any():
        D[empty_bins] = np.nan
        for center_idx in range(n_centers):
            for lag_idx in range(n_limit):
                if empty_bins[lag_idx, center_idx]:
                    P[:, :, lag_idx, center_idx] = np.nan
    
    return center_coords, D.T, P, O.T


def identify_homogeneous_regions(
    center_coords,
    P,
    n_regions=None,
    distance_threshold=None,
    linkage_method='ward',
    metric='kl_divergence',
    min_samples_per_region=10,
    ShowCalDis=False
):
    """
    Identify relatively homogeneous regions based on similarity of probability tables.
    
    This function clusters center locations into regions where the probability tables
    are relatively similar (homogeneous). It uses hierarchical clustering based on
    a distance metric between probability tables.
    
    **Clustering Method:**
    The function uses scipy's hierarchical clustering algorithm:
    1. Calculates a pairwise distance matrix between probability tables using the
       specified metric (KL divergence, Euclidean, or Hellinger distance).
    2. Performs hierarchical clustering using scipy.cluster.hierarchy.linkage with
       the specified linkage method ('ward', 'complete', 'average', or 'single').
       Note: 'ward' linkage requires Euclidean distances, so it automatically switches
       to 'average' linkage when using KL divergence metric.
    3. Forms flat clusters using scipy.cluster.hierarchy.fcluster based on either
       the number of regions (n_regions) or a distance threshold (distance_threshold).
    4. Merges regions with fewer than min_samples_per_region samples into the nearest
       larger region based on spatial distance between centroids.
    
    Args:
        center_coords (ndarray): 2-D ndarray of floats with shape (n_centers, d).
            Coordinates of center locations.
        P (ndarray): 4-D ndarray of floats with shape (nc, nc, ncl, n_centers).
            Spatial-varying probability tables from probatablecalc_spatial.
        n_regions (int, optional): Number of homogeneous regions to identify.
            If None, must provide distance_threshold.
        distance_threshold (float, optional): Threshold for hierarchical clustering.
            If None, must provide n_regions.
        linkage_method (str): Linkage method for hierarchical clustering.
            Options: 'ward', 'complete', 'average', 'single'.
            Default: 'ward' (automatically switches to 'average' for non-Euclidean metrics).
        metric (str): Distance metric between probability tables.
            Options: 'kl_divergence' (symmetric KL divergence), 'euclidean', 'hellinger'.
            Default: 'kl_divergence'.
        min_samples_per_region (int): Minimum number of samples per region.
            Regions with fewer samples will be merged with nearest region.
        ShowCalDis (bool): Print the clustering process.
    
    Returns:
        region_labels (ndarray): 1-D array of integers with shape (n_centers,).
            Region label for each center location.
        region_stats (dict): Dictionary with statistics for each region:
            - 'n_centers': Number of centers in each region
            - 'centers': Coordinates of centers in each region
            - 'P_mean': Mean probability table for each region
            - 'P_std': Standard deviation of probability tables in each region
    
    Note:
        The function uses hierarchical clustering to group centers with similar
        probability tables. Regions with too few samples are merged to ensure
        statistical significance.
    """
    center_coords = np.asarray(center_coords, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    
    n_centers = center_coords.shape[0]
    n_cata, n_cata2, n_limit, n_centers_P = P.shape
    
    if n_centers != n_centers_P:
        raise ValueError(f"Number of centers ({n_centers}) doesn't match P shape ({n_centers_P})")
    
    if n_regions is None and distance_threshold is None:
        raise ValueError("Must provide either n_regions or distance_threshold")
    
    if ShowCalDis:
        print(f'Identifying homogeneous regions for {n_centers} centers...')
        print(f'Using metric: {metric}, linkage: {linkage_method}')
    
    # Calculate distance matrix between probability tables
    if ShowCalDis:
        print('Calculating distance matrix between probability tables...')
    
    if metric == 'kl_divergence':
        # Flatten probability tables and calculate KL divergence
        P_flat = P.transpose(3, 0, 1, 2).reshape(n_centers, -1)  # (n_centers, nc*nc*ncl)
        
        # Normalize to ensure valid probabilities
        P_flat = np.clip(P_flat, 1e-10, 1.0)
        P_flat = P_flat / P_flat.sum(axis=1, keepdims=True)
        
        # Calculate pairwise KL divergence
        def kl_divergence(p, q):
            # KL(p||q) = sum(p * log(p/q))
            # p and q are 1D arrays (flattened probability tables)
            p = np.asarray(p).flatten()
            q = np.asarray(q).flatten()
            # Avoid log(0) by clipping
            p = np.clip(p, 1e-10, 1.0)
            q = np.clip(q, 1e-10, 1.0)
            # Normalize to ensure they sum to 1
            p = p / p.sum()
            q = q / q.sum()
            return np.sum(p * np.log(p / q))
        
        distance_matrix = np.zeros((n_centers, n_centers))
        for i in range(n_centers):
            for j in range(i+1, n_centers):
                try:
                    kl_ij = kl_divergence(P_flat[i], P_flat[j])
                    kl_ji = kl_divergence(P_flat[j], P_flat[i])
                    # Handle NaN or inf values
                    if np.isfinite(kl_ij) and np.isfinite(kl_ji):
                        distance_matrix[i, j] = (kl_ij + kl_ji) / 2  # Symmetric KL
                    else:
                        # Use a large value for invalid KL divergences
                        distance_matrix[i, j] = 1e10
                    distance_matrix[j, i] = distance_matrix[i, j]
                except (ValueError, RuntimeWarning):
                    distance_matrix[i, j] = 1e10
                    distance_matrix[j, i] = 1e10
        
        # Replace any remaining NaN or inf values
        distance_matrix = np.nan_to_num(distance_matrix, nan=1e10, posinf=1e10, neginf=1e10)
    
    elif metric == 'euclidean':
        P_flat = P.transpose(3, 0, 1, 2).reshape(n_centers, -1)
        distance_matrix = pairwise_distances(P_flat, metric='euclidean')
    
    elif metric == 'hellinger':
        P_flat = P.transpose(3, 0, 1, 2).reshape(n_centers, -1)
        P_flat = np.sqrt(P_flat)
        distance_matrix = pairwise_distances(P_flat, metric='euclidean')
    
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Perform hierarchical clustering
    if ShowCalDis:
        print('Performing hierarchical clustering...')
    
    # Check for valid distance matrix
    if np.all(np.isnan(distance_matrix)) or np.all(np.isinf(distance_matrix)):
        raise ValueError("Distance matrix contains only invalid values. Cannot perform clustering.")
    
    # Convert to condensed distance matrix for scipy
    condensed_distances = distance_matrix[np.triu_indices(n_centers, k=1)]
    
    # Check for NaN or inf in condensed distances
    if np.any(~np.isfinite(condensed_distances)):
        if ShowCalDis:
            n_invalid = np.sum(~np.isfinite(condensed_distances))
            print(f"Warning: {n_invalid} invalid distance values found. Replacing with large values...")
        condensed_distances = np.nan_to_num(condensed_distances, nan=1e10, posinf=1e10, neginf=1e10)
    
    # Use 'average' or 'complete' linkage instead of 'ward' for non-Euclidean distances
    # 'ward' requires Euclidean distances, so switch to 'average' for KL divergence
    if metric == 'kl_divergence' and linkage_method == 'ward':
        if ShowCalDis:
            print("Switching from 'ward' to 'average' linkage (ward requires Euclidean distances)")
        linkage_method = 'average'
    
    linkage_matrix = linkage(condensed_distances, method=linkage_method)
    
    # Get cluster labels
    if n_regions is not None:
        region_labels = fcluster(linkage_matrix, n_regions, criterion='maxclust')
    else:
        region_labels = fcluster(linkage_matrix, distance_threshold, criterion='distance')
    
    # Merge regions with too few samples
    unique_labels, counts = np.unique(region_labels, return_counts=True)
    small_regions = unique_labels[counts < min_samples_per_region]
    
    if len(small_regions) > 0 and ShowCalDis:
        print(f'Merging {len(small_regions)} regions with < {min_samples_per_region} samples...')
    
    for small_region in small_regions:
        # Find nearest large region
        small_region_mask = region_labels == small_region
        small_region_centers = center_coords[small_region_mask]
        
        # Find nearest large region by distance to centroid
        large_regions = unique_labels[counts >= min_samples_per_region]
        if len(large_regions) == 0:
            continue
        
        # Calculate centroid of small region
        small_centroid = small_region_centers.mean(axis=0)
        
        # Find nearest large region
        min_dist = np.inf
        nearest_region = large_regions[0]
        for large_region in large_regions:
            large_region_mask = region_labels == large_region
            large_region_centers = center_coords[large_region_mask]
            large_centroid = large_region_centers.mean(axis=0)
            dist = np.linalg.norm(small_centroid - large_centroid)
            if dist < min_dist:
                min_dist = dist
                nearest_region = large_region
        
        # Merge small region into nearest large region
        region_labels[small_region_mask] = nearest_region
    
    # Calculate statistics for each region
    if ShowCalDis:
        print('Calculating region statistics...')
    
    unique_labels = np.unique(region_labels)
    n_regions_final = len(unique_labels)
    
    region_stats = {
        'n_centers': np.zeros(n_regions_final, dtype=int),
        'centers': [None] * n_regions_final,
        'P_mean': np.zeros((n_cata, n_cata, n_limit, n_regions_final)),
        'P_std': np.zeros((n_cata, n_cata, n_limit, n_regions_final)),
    }
    
    for region_idx, region_label in enumerate(unique_labels):
        region_mask = region_labels == region_label
        region_centers = center_coords[region_mask]
        region_P = P[:, :, :, region_mask]
        
        n_centers_in_region = region_mask.sum()
        region_stats['n_centers'][region_idx] = n_centers_in_region
        region_stats['centers'][region_idx] = region_centers
        
        # Calculate mean and std only if there are centers in this region
        if n_centers_in_region > 0:
            region_stats['P_mean'][:, :, :, region_idx] = np.nanmean(region_P, axis=3)
            if n_centers_in_region > 1:
                region_stats['P_std'][:, :, :, region_idx] = np.nanstd(region_P, axis=3)
            else:
                # If only one center, std is 0
                region_stats['P_std'][:, :, :, region_idx] = 0.0
        else:
            # Empty region (shouldn't happen after merging, but handle it)
            region_stats['P_mean'][:, :, :, region_idx] = np.nan
            region_stats['P_std'][:, :, :, region_idx] = np.nan
    
    if ShowCalDis:
        print(f'Identified {n_regions_final} homogeneous regions')
        for region_idx in range(n_regions_final):
            print(f'  Region {region_idx}: {region_stats["n_centers"][region_idx]} centers')
    
    return region_labels, region_stats

