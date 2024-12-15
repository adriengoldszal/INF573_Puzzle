import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

def homography_unknown_scale(kp1, kp2, matches):
    '''Parameters:
        kp1 (list): Keypoints from the first image.
        kp2 (list): Keypoints from the second image.
        matches (list): Matches between the keypoints.
        Returns:
        H (np.array): Homography matrix.'''
    ## this function requires at leat 3 points
    src_points = np.array([kp1[m.queryIdx].pt for m in matches])
    dst_points = np.array([kp2[m.trainIdx].pt for m in matches])
    n_points = src_points.shape[0]
    matrix, _ = cv2.estimateAffinePartial2D(src_points, dst_points)
    matrix=np.vstack([matrix, [0, 0, 1]])

    return matrix

def show_homography_on_puzzle(img, puzzle_img, H):
    height, width, _ = puzzle_img.shape  # Taille de l'image cible
    warped_piece1 = cv2.warpPerspective(img, H, (width, height))

    # Superposer les images
    result = cv2.addWeighted(puzzle_img, 0.5, warped_piece1, 0.5, 0)
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    plt.imshow(result_rgb)
    plt.show()

def decompose_similarity_homography(H):
    """
    Decomposes a 3x3 homography matrix into rotation angle (in degrees), scaling factor, and translation vector.
    
    Parameters:
    - H: 3x3 homography matrix
    
    Returns:
    - angle_deg: Rotation angle in degrees
    - scale: Scaling factor
    - translation: Translation vector as a (tx, ty) tuple
    """
    # Ensure the matrix is normalized (make H[2, 2] = 1)
    H = H / H[2, 2]
    
    # Extract the rotation and scaling components (first two columns of the matrix)
    R = H[:2, :2]
    t = H[:2, 2]
    
    # Compute the scaling factor (assuming uniform scaling)
    scale = np.linalg.norm(R[:, 0])
    
    # Normalize R to remove scaling and extract pure rotation
    R_normalized = R / scale
    # Compute the rotation angle (in radians)
    theta_rad = np.arctan2(R_normalized[1, 0], R_normalized[0, 0])
    return  scale, theta_rad,t


def transform_points(points, R, t):
    """
    Apply a rigid transformation to points.
    
    Parameters:
        points (np.ndarray): Points to transform of shape (N, 2)
        R (np.ndarray): Rotation matrix of shape (2, 2)
        t (np.ndarray): Translation vector of shape (2,)
        
    Returns:
        np.ndarray: Transformed points of shape (N, 2)
    """
    return (R @ points.T).T + t




############known scale@##############@



import numpy as np

def estimate_rigid_transform_known_scale(src_points, dst_points, scale, 
                                         max_iterations=1000, 
                                         reproj_threshold=3.0, 
                                         confidence=0.99):
    """
    Estimate a rigid transformation (rotation and translation) in 2D given a known scale.
    Uses a RANSAC-like approach for robustness.

    Parameters:
        src_points (np.ndarray): Nx2 array of points from the first image.
        dst_points (np.ndarray): Nx2 array of points from the second image.
        scale (float): Known scale factor to apply to the source points.
        max_iterations (int): Number of RANSAC iterations.
        reproj_threshold (float): Threshold in pixels for considering a point an inlier.
        confidence (float): Desired RANSAC confidence (not strictly implemented, but can be used to stop early).

    Returns:
        best_transform (np.ndarray): 2x3 affine transform matrix [R|t].
                                    The last row [0 0 1] can be appended as needed.
        inliers (list): Indices of inlier matches.
    """
    # Apply known scale to the source points
    scaled_src_points = src_points*scale
    n_points = scaled_src_points.shape[0]
    
    if n_points < 2:
        raise ValueError("At least two points are required to estimate a rigid transform.")

    # A helper function to solve for rotation and translation given all correspondences
    def solve_rigid_transform(scaled_src, dst):
        # Compute centroids
        src_centroid = np.mean(scaled_src, axis=0)
        dst_centroid = np.mean(dst, axis=0)

        # Center the points
        src_centered = scaled_src - src_centroid
        dst_centered = dst - dst_centroid

        # Compute rotation via SVD (Procrustes)
        H = src_centered.T @ dst_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Ensure proper rotation (no reflection)
        if np.linalg.det(R) < 0:
            Vt[1,:] *= -1
            R = Vt.T @ U.T

        # Translation
        t = dst_centroid - R @ src_centroid

        return R, t

    # If we have exactly two points, no need for RANSAC, just solve directly
    if n_points == 2:
        R, t = solve_rigid_transform(scaled_src_points, dst_points)
        transform = np.hstack((R, t.reshape(2, 1)))
        inliers = [0, 1]
        return transform, inliers

    # RANSAC loop
    best_inliers = []
    best_transform = None
    for i in range(max_iterations):
        # Randomly sample 2 distinct points
        idxs = np.random.choice(n_points, 2, replace=False)
        sample_src = scaled_src_points[idxs]
        sample_dst = dst_points[idxs]

        # Estimate model from minimal set
        R, t = solve_rigid_transform(sample_src, sample_dst)
        
        # Apply model to all points
        projected = (R @ scaled_src_points.T).T + t

        # Compute residuals
        residuals = np.linalg.norm(projected - dst_points, axis=1)
        inlier_mask = residuals < reproj_threshold
        inliers = np.nonzero(inlier_mask)[0]

        # Check if this is the best so far
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_transform = np.hstack((R, t.reshape(2, 1)))

            # Early stopping criterion (optional)
            # If we have a very high inlier ratio, we might stop early
            if len(inliers) > n_points * confidence:
                break

    # If we found no inliers at all, fall back to a direct fit without filtering
    if best_transform is None:
        R, t = solve_rigid_transform(scaled_src_points, dst_points)
        best_transform = np.hstack((R, t.reshape(2, 1)))
        best_inliers = np.arange(n_points)

    # Optionally, refine the model using all inliers
    if len(best_inliers) > 2:
        R, t = solve_rigid_transform(scaled_src_points[best_inliers], dst_points[best_inliers])
        best_transform = np.hstack((R, t.reshape(2, 1)))

    return best_transform, best_inliers


def homography_known_scale(kp1, kp2, matches, scale):
    '''
    Parameters:
        kp1 (list): Keypoints from the first image.
        kp2 (list): Keypoints from the second image.
        matches (list): Matches between the keypoints.
        scale (float): Known scale factor to apply to kp1 points.
    
    Returns:
        H (np.array): Homography-like matrix (3x3). Since we are estimating rigid 
                      transform, it will be of the form:
                      
                      [R(2x2) | t(2x1)]
                      [   0   |   1   ]
    '''
    src_points = np.array([kp1[m.queryIdx].pt for m in matches])
    dst_points = np.array([kp2[m.trainIdx].pt for m in matches])

    # Estimate the rigid transform with known scale robustly
    transform, inliers = estimate_rigid_transform_known_scale(src_points, dst_points, scale)
    transform[:2, :2] *= scale

    # Construct a 3x3 matrix to return
    # transform is [R|t], shape (2x3)
    # Append a row [0 0 1]
    H = np.vstack([transform, [0, 0, 1]])
    return H

def homography_matrix(s,theta, t):
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    H = np.eye(3)
    H[:2, :2] = R*s
    H[:2, 2] = t
    return H



