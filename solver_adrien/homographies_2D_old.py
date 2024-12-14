import numpy as np
import cv2
import matplotlib.pyplot as plt



def homography_unknown_scale_naive(src_points, dst_points):
    """
    Estimate scale, rotation, and translation, using SVD to ensure robust rotation recovery.
    
    Parameters:
        src_points (ndarray): n x 2 array of source points.
        dst_points (ndarray): n x 2 array of destination points.
    
    Returns:
        scale (float): Estimated scale factor.
        theta (float): Estimated rotation angle (in radians).
        t (ndarray): 2-element translation vector.
    """
    # Step 1: Compute centroids
    src_centroid = np.mean(src_points, axis=0)
    dst_centroid = np.mean(dst_points, axis=0)

    # Step 2: Center the points
    src_centered = src_points - src_centroid
    dst_centered = dst_points - dst_centroid

    # Filter out zero-length vectors to avoid division by zero
    valid_mask = (np.linalg.norm(src_centered, axis=1) != 0) & \
                 (np.linalg.norm(dst_centered, axis=1) != 0)
    src_centered = src_centered[valid_mask]
    dst_centered = dst_centered[valid_mask]

    # Step 3: Compute scale as the ratio of RMS distances
    src_rms = np.sqrt((src_centered**2).sum())
    dst_rms = np.sqrt((dst_centered**2).sum())
    if src_rms == 0:
        # Degenerate case - no scale can be determined
        scale = 1.0
    else:
        scale = dst_rms / src_rms

    # Step 4: Scale the destination points
    dst_centered_scaled = dst_centered / scale if scale != 0 else dst_centered

    # Step 5: Compute cross-covariance matrix
    H = np.dot(src_centered.T, dst_centered_scaled)

    # Step 6: Extract rotation using SVD
    U, S, Vt = np.linalg.svd(H)
    R = U @ Vt

    # Ensure we have a proper rotation (no reflection)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    # Extract the angle from the rotation matrix
    # R = [[cos(theta), -sin(theta)],
    #      [sin(theta),  cos(theta)]]
    theta = np.arctan2(R[1, 0], R[0, 0])

    # Step 7: Compute translation
    t = dst_centroid - scale * (R @ src_centroid)

    return scale, theta, t



def homography_unknown_scale(kp1, kp2, matches, inlier_ratio=0.7, threshold=10.0, max_iterations=2000, n_batch=4):
    """
    Estimate scale, rotation, and translation using RANSAC.
    
    Parameters:
        kp1 (list): Keypoints from the first image.
        kp2 (list): Keypoints from the second image.
        matches (list): Matches between the keypoints.
        inlier_ratio (float): Minimum fraction of points required as inliers to terminate early.
        threshold (float): Inlier distance threshold.
        max_iterations (int): Maximum number of RANSAC iterations.
        n_batch (int): Number of points to sample per iteration (>=3 recommended).
    
    Returns:
        best_scale (float): Estimated scale.
        best_theta (float): Estimated rotation angle (in radians).
        best_t (ndarray): 2-element translation vector.
        best_inliers (ndarray): Boolean mask of inliers.
    """
    src_points = np.array([kp1[m.queryIdx].pt for m in matches])
    dst_points = np.array([kp2[m.trainIdx].pt for m in matches])
    n_points = src_points.shape[0]

    if n_points < n_batch:
        raise ValueError("Not enough matches to perform RANSAC.")

    best_inliers = np.zeros(n_points, dtype=bool)
    max_inliers = 0
    best_model = None

    for _ in range(max_iterations):
        # Randomly sample n_batch points
        idx = np.random.choice(n_points, n_batch, replace=False)
        src_sample = src_points[idx]
        dst_sample = dst_points[idx]

        # Estimate transformation
        try:
            scale, theta, t = homography_unknown_scale_naive(src_sample, dst_sample)
        except (ValueError, np.linalg.LinAlgError, ZeroDivisionError):
            continue

        # Evaluate the model
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])
        transformed_src = scale * (np.dot(src_points, R.T)) + t
        errors = np.linalg.norm(transformed_src - dst_points, axis=1)
        inliers = errors < threshold

        # Count the inliers
        n_inliers = np.sum(inliers)

        # Update the best model if necessary
        if n_inliers > max_inliers:
            max_inliers = n_inliers
            best_inliers = inliers
            best_model = (scale, theta, t)

        # Early termination if enough inliers
        if max_inliers > n_points * inlier_ratio:
            break

    # Refine using inliers
    if best_model is not None:
        inlier_src = src_points[best_inliers]
        inlier_dst = dst_points[best_inliers]
        # Recompute with the final inliers for better precision
        best_scale, best_theta, best_t = homography_unknown_scale_naive(inlier_src, inlier_dst)
    else:
        raise ValueError("RANSAC failed to find a valid model.")

    return best_scale, best_theta, best_t




import numpy as np

def homography_rigid_transform_robust(src_points, dst_points, max_iterations=1000, threshold=3.0, inlier_ratio=0.5):
    """
    Estimate rigid transformation (scale, rotation, translation) using robust RANSAC.
    
    Parameters:
        src_points (ndarray): n x 2 array of source points.
        dst_points (ndarray): n x 2 array of destination points.
        max_iterations (int): Maximum number of RANSAC iterations.
        threshold (float): Inlier distance threshold (in pixels).
        inlier_ratio (float): Minimum fraction of points required as inliers.
    
    Returns:
        best_scale (float): Estimated scale.
        best_theta (float): Estimated rotation angle (in radians).
        best_t (ndarray): 2-element translation vector.
        inlier_mask (ndarray): Boolean mask of inliers.
    """
    def estimate_rigid_transform(src_sample, dst_sample):
        """Estimate scale, rotation, and translation from 2 sets of points."""
        # Step 1: Compute centroids
        src_centroid = np.mean(src_sample, axis=0)
        dst_centroid = np.mean(dst_sample, axis=0)

        # Step 2: Center the points
        src_centered = src_sample - src_centroid
        dst_centered = dst_sample - dst_centroid

        # Step 3: Compute scale
        scale = np.linalg.norm(dst_centered) / np.linalg.norm(src_centered)

        # Step 4: Scale the destination points
        dst_centered_scaled = dst_centered / scale

        # Step 5: Compute rotation
        H = np.dot(src_centered.T, dst_centered_scaled)
        h11, h12 = H[0, 0], H[0, 1]
        h21, h22 = H[1, 0], H[1, 1]
        theta = np.arctan2(h21 - h12, h11 + h22)

        # Step 6: Compute translation
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])
        t = dst_centroid - scale * np.dot(R, src_centroid)

        return scale, theta, t

    n_points = src_points.shape[0]
    best_inliers = np.zeros(n_points, dtype=bool)
    max_inliers = 0
    best_model = None

    for _ in range(max_iterations):
        # Randomly sample 2 points (minimal for rigid transformation)
        idx = np.random.choice(n_points, 2, replace=False)
        src_sample = src_points[idx]
        dst_sample = dst_points[idx]

        # Estimate rigid transformation
        try:
            scale, theta, t = estimate_rigid_transform(src_sample, dst_sample)
        except:
            continue  # Skip if estimation fails

        # Evaluate the model
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])
        transformed_src = scale * (np.dot(src_points, R.T)) + t
        errors = np.linalg.norm(transformed_src - dst_points, axis=1)
        inliers = errors < threshold

        # Count inliers
        n_inliers = np.sum(inliers)

        # Update the best model
        if n_inliers > max_inliers:
            max_inliers = n_inliers
            best_inliers = inliers
            best_model = (scale, theta, t)

        # Early termination
        if max_inliers > n_points * inlier_ratio:
            break

    # Refine the model using all inliers
    if best_model is not None:
        inlier_src = src_points[best_inliers]
        inlier_dst = dst_points[best_inliers]
        best_scale, best_theta, best_t = estimate_rigid_transform(inlier_src, inlier_dst)
    else:
        raise ValueError("RANSAC failed to find a valid model.")

    return best_scale, best_theta, best_t, best_inliers


def homography_known_scale(kp1, kp2, matches, scale, max_iterations=1000, threshold=5.0, inlier_ratio=0.3):
    """
    Estimate rotation and translation using RANSAC, assuming known scale.
    
    Parameters:
        src_points (ndarray): n x 2 array of source points.
        dst_points (ndarray): n x 2 array of destination points.
        scale (float): Known scale factor.
        max_iterations (int): Maximum number of RANSAC iterations.
        threshold (float): Inlier distance threshold.
        inlier_ratio (float): Minimum fraction of points required as inliers to terminate early.
    
    Returns:
        best_theta (float): Estimated rotation angle (in radians).
        best_t (ndarray): 2-element translation vector.
        best_inliers (ndarray): Boolean mask of inliers.
    """
    src_points = np.array([kp1[m.queryIdx].pt for m in matches])
    dst_points = np.array([kp2[m.trainIdx].pt for m in matches])
    n_points = src_points.shape[0]
    best_inliers = np.zeros(n_points, dtype=bool)
    max_inliers = 0
    best_model = None

    for _ in range(max_iterations):
        # Step 1: Randomly sample 2 points (minimal set for rigid transformation)
        idx = np.random.choice(n_points, 2, replace=False)
        src_sample = src_points[idx]
        dst_sample = dst_points[idx]

        # Step 2: Estimate the transformation using the sampled points
        try:
            theta, t = homography_known_scale_naive(src_sample, dst_sample, scale)
        except:
            continue  # Skip this iteration if estimation fails

        # Step 3: Evaluate the model on all points
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])
        transformed_src = scale * (np.dot(src_points, R.T)) + t
        errors = np.linalg.norm(transformed_src - dst_points, axis=1)
        inliers = errors < threshold

        # Count the inliers
        n_inliers = np.sum(inliers)

        # Step 4: Update the best model if necessary
        if n_inliers > max_inliers:
            max_inliers = n_inliers
            best_inliers = inliers
            best_model = (theta, t)

        # Terminate early if enough inliers are found
        if max_inliers > n_points * inlier_ratio:
            break

    # Refine the model using all inliers
    if best_model is not None:
        inlier_src = src_points[best_inliers]
        inlier_dst = dst_points[best_inliers]
        best_theta, best_t = homography_known_scale_naive(inlier_src, inlier_dst, scale)
    else:
        raise ValueError("RANSAC failed to find a valid model.")

    return best_theta, best_t

def homography_known_scale_naive(src_points, dst_points, scale):
    """
    Estimate rotation and translation given a known scale.
    
    Parameters:
        src_points (ndarray): n x 2 array of source points.
        dst_points (ndarray): n x 2 array of destination points.
        scale (float): Known scale factor.
    
    Returns:
        theta (float): Rotation angle (in radians).
        t (ndarray): 2-element translation vector.
    """
    # Step 1: Compute centroids
    src_centroid = np.mean(src_points, axis=0)
    dst_centroid = np.mean(dst_points, axis=0)

    # Step 2: Center the points
    src_centered = src_points - src_centroid
    dst_centered = dst_points - dst_centroid

    # Step 3: Scale the destination points
    dst_centered_scaled = dst_centered / scale

    # Step 4: Compute cross-covariance matrix
    H = np.dot(src_centered.T, dst_centered_scaled)

    # Step 5: Compute rotation using the structure of the rotation matrix
    h11, h12 = H[0, 0], H[0, 1]
    h21, h22 = H[1, 0], H[1, 1]
    theta = np.arctan2(h21 - h12, h11 + h22)

    # Step 6: Compute translation
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    t = dst_centroid - scale * np.dot(R, src_centroid)

    return theta, t


def homography_matrix(s,theta, t):
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    H = np.eye(3)
    H[:2, :2] = R*s
    H[:2, 2] = t
    return H


def display_matches(img1, kp1, img2, kp2, matches, title="Matches"):
    """
    Display matches between two images using matplotlib and OpenCV.
    
    Parameters:
        img1 (ndarray): First image.
        kp1 (list): Keypoints of the first image.
        img2 (ndarray): Second image.
        kp2 (list): Keypoints of the second image.
        matches (list): List of matches (cv2.DMatch).
        title (str): Title of the plot.
    """
    # Use OpenCV's drawMatches to create a visualization
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the resulting image using matplotlib
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(title)
    plt.show()



def show_homography_on_puzzle(img, puzzle_img, H):
    height, width, _ = puzzle_img.shape  # Taille de l'image cible
    warped_piece1 = cv2.warpPerspective(img, H, (width, height))

    # Superposer les images
    result = cv2.addWeighted(puzzle_img, 0.5, warped_piece1, 0.5, 0)

    plt.imshow(result)
    plt.show()

#########comparaison au livrairies CV2#########

def estimate_similarity_transform(src_pts, dst_pts):
    # Estimate the 2x3 affine matrix that includes rotation, scaling, and translation
    matrix, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    matrix=np.vstack([matrix, [0, 0, 1]])

    return matrix


def estimate_similarity_transform_known_scale(src_pts, dst_pts, scale):
    # Estimate the 2x3 affine matrix that includes rotation and translation
    scaled_src_points = src_pts * scale
    matrix, _ = cv2.estimateAffinePartial2D(scaled_src_points, dst_pts, False)
    matrix[:2, :2] = matrix[:2, :2] *scale
    matrix=np.vstack([matrix, [0, 0, 1]])
    return matrix

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