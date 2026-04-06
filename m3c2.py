import numpy as np
from scipy.spatial import cKDTree

# step 1: estimate normals for each core point
def estimate_normals(cloud, core_points, normal_radius, min_neighbors=10):
    tree = cKDTree(cloud)
    normals = np.full((len(core_points), 3), np.nan)
    
    for i, cp in enumerate(core_points):
        idx = tree.query_ball_point(cp, normal_radius)
        if len(idx) < min_neighbors:
            continue
        
        pts = cloud[idx]
        
        # center the points around the origin for PCA
        centered = pts - np.mean(pts, axis=0)
        
        # calculate covariance matrix and its eigenvalues/eigenvectors
        cov = np.dot(centered.T, centered) / max(len(pts) - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        
        # the normal is the eigenvector with the smallest eigenvalue (least variance)
        n = eigvecs[:, np.argmin(eigvals)]
        norm = np.linalg.norm(n)
        
        if norm > 0:
            n = n / norm
            # ensure the normal is consistently oriented (pointing upwards)
            if n[2] < 0:
                n = -n
            normals[i] = n
            
    return normals

# step 2: extract points within a cylinder defined by the core point and its normal
def get_points_in_cylinder(cloud, tree, center, normal, radius, half_length):
    # collect points within a bounding sphere
    search_radius = np.sqrt(radius**2 + half_length**2)
    idx = tree.query_ball_point(center, search_radius)
    if not idx:
        return np.empty(0)
    
    pts = cloud[idx]
    vectors = pts - center
    
    # calculate distance along the normal vector
    axial_dist = np.dot(vectors, normal)
    
    # calculate lateral distance away from the center axis
    total_dist_sq = np.sum(vectors**2, axis=1)
    radial_dist_sq = total_dist_sq - (axial_dist**2)
    
    # keep only points that are within the cylinder bounds
    mask = (np.abs(axial_dist) <= half_length) & (radial_dist_sq <= radius**2)
    
    return axial_dist[mask]


# step 3: compute M3C2 distances for each core point
def compute_m3c2(cloud_ref, cloud_cmp, core_points, normal_radius, proj_diameter, max_depth, reg_error=0.0):
    # calculate which way the ground is facing
    normals = estimate_normals(cloud_ref, core_points, normal_radius)
    
    # build the fast spatial search indices
    tree_ref = cKDTree(cloud_ref)
    tree_cmp = cKDTree(cloud_cmp)
    
    # prepare empty arrays to hold our results
    k = len(core_points)
    distances = np.full(k, np.nan)
    lod95 = np.full(k, np.nan)
    significant = np.zeros(k, dtype=bool)
    
    cylinder_radius = proj_diameter / 2.0
    
    # Loop through every core point to measure the distance
    for i, (cp, n) in enumerate(zip(core_points, normals)):
        if np.any(np.isnan(n)):
            continue
            
        # extract points falling inside the cylinder for both years
        d_ref = get_points_in_cylinder(cloud_ref, tree_ref, cp, n, cylinder_radius, max_depth)
        d_cmp = get_points_in_cylinder(cloud_cmp, tree_cmp, cp, n, cylinder_radius, max_depth)
        
        # we need at least 5 points to calculate a meaningful average
        if len(d_ref) < 5 or len(d_cmp) < 5:
            continue
            
        # find the average position of the points inside the cylinder
        mean_ref = np.mean(d_ref)
        mean_cmp = np.mean(d_cmp)
        
        # find the roughness (standard deviation) of the points
        std_ref = np.std(d_ref, ddof=1)
        std_cmp = np.std(d_cmp, ddof=1)
        
        # raw distance between the two point clouds at this core point
        distances[i] = mean_cmp - mean_ref
        
        # the statistical noise (LOD95) is based on the standard error of the mean for both sets of points, plus any registration error
        variance_ref = (std_ref**2) / len(d_ref)
        variance_cmp = (std_cmp**2) / len(d_cmp)
        sigma = np.sqrt(variance_ref + variance_cmp)
        
        lod95[i] = 1.96 * sigma + reg_error
        
        # check if the distance is greater than the noise
        significant[i] = np.abs(distances[i]) > lod95[i]
        
    return {
        "distances": distances,
        "normals": normals,
        "lod95": lod95,
        "significant": significant
    }