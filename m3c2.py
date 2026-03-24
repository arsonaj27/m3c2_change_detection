import numpy as np
from scipy.spatial import cKDTree

def estimate_normals(cloud, core_points, normal_radius, min_neighbors=10):
    """Calculates the surface normal (tilt) at each core point using PCA."""
    tree = cKDTree(cloud)
    normals = np.full((len(core_points), 3), np.nan)
    
    for i, cp in enumerate(core_points):
        idx = tree.query_ball_point(cp, normal_radius)
        if len(idx) < min_neighbors:
            continue
        
        pts = cloud[idx]
        
        # Center the points around the origin for PCA
        centered = pts - np.mean(pts, axis=0)
        
        # Calculate covariance matrix and its eigenvalues/eigenvectors
        cov = np.dot(centered.T, centered) / max(len(pts) - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        
        # The normal is the eigenvector with the smallest eigenvalue (least variance)
        n = eigvecs[:, np.argmin(eigvals)]
        norm = np.linalg.norm(n)
        
        if norm > 0:
            n = n / norm
            # Quick trick: Force the normal to always point "up" (positive Z) 
            # so our distance measurements don't randomly flip signs.
            if n[2] < 0:
                n = -n
            normals[i] = n
            
    return normals


def get_points_in_cylinder(cloud, tree, center, normal, radius, half_length):
    """Acts as a cookie-cutter to extract only points inside the 3D cylinder."""
    # 1. Fast filter: Grab points in a giant sphere first
    search_radius = np.sqrt(radius**2 + half_length**2)
    idx = tree.query_ball_point(center, search_radius)
    if not idx:
        return np.empty(0)
    
    pts = cloud[idx]
    vectors = pts - center
    
    # 2. Calculate distance strictly along the normal vector
    axial_dist = np.dot(vectors, normal)
    
    # 3. Calculate lateral distance away from the center axis (Pythagorean theorem)
    total_dist_sq = np.sum(vectors**2, axis=1)
    radial_dist_sq = total_dist_sq - (axial_dist**2)
    
    # 4. Keep only points that are within the cylinder bounds
    mask = (np.abs(axial_dist) <= half_length) & (radial_dist_sq <= radius**2)
    
    return axial_dist[mask]


def compute_m3c2(cloud_ref, cloud_cmp, core_points, normal_radius, proj_diameter, max_depth, reg_error=0.0):
    """The main engine that drives the M3C2 calculation."""
    # Calculate which way the ground is facing
    normals = estimate_normals(cloud_ref, core_points, normal_radius)
    
    # Build the fast spatial search indices
    tree_ref = cKDTree(cloud_ref)
    tree_cmp = cKDTree(cloud_cmp)
    
    # Prepare empty arrays to hold our results
    k = len(core_points)
    distances = np.full(k, np.nan)
    lod95 = np.full(k, np.nan)
    significant = np.zeros(k, dtype=bool)
    
    cyl_radius = proj_diameter / 2.0
    
    # Loop through every core point to measure the distance
    for i, (cp, n) in enumerate(zip(core_points, normals)):
        if np.any(np.isnan(n)):
            continue
            
        # Extract points falling inside the cylinder for both years
        d_ref = get_points_in_cylinder(cloud_ref, tree_ref, cp, n, cyl_radius, max_depth)
        d_cmp = get_points_in_cylinder(cloud_cmp, tree_cmp, cp, n, cyl_radius, max_depth)
        
        # We need at least 5 points to calculate a meaningful average
        if len(d_ref) < 5 or len(d_cmp) < 5:
            continue
            
        # Find the average position of the points inside the cylinder
        mean_ref = np.mean(d_ref)
        mean_cmp = np.mean(d_cmp)
        
        # Find the roughness (standard deviation) of the points
        std_ref = np.std(d_ref, ddof=1)
        std_cmp = np.std(d_cmp, ddof=1)
        
        # 1. The Raw Distance
        distances[i] = mean_cmp - mean_ref
        
        # 2. The Statistical Confidence (Level of Detection)
        variance_ref = (std_ref**2) / len(d_ref)
        variance_cmp = (std_cmp**2) / len(d_cmp)
        sigma = np.sqrt(variance_ref + variance_cmp + reg_error**2)
        
        lod95[i] = 1.96 * sigma
        
        # 3. Check if the distance is greater than the noise
        significant[i] = np.abs(distances[i]) > lod95[i]
        
    return {
        "distances": distances,
        "normals": normals,
        "lod95": lod95,
        "significant": significant
    }