import numpy as np
import cv2
import matplotlib.pyplot as plt



def homography_unknown_scale_naive(src_points, dst_points):
    """
    Estimate scale, rotation, and translation, leveraging the full structure of the rotation matrix.
    
    Parameters:
        src_points (ndarray): n x 2 array of source points.
        dst_points (ndarray): n x 2 array of destination points.
    
    Returns:
        scale (float): Estimated scale.
        theta (float): Estimated rotation angle (in radians).
        t (ndarray): 2-element translation vector.
    """
    # Step 1: Compute centroids
    src_centroid = np.mean(src_points, axis=0)
    dst_centroid = np.mean(dst_points, axis=0)

    # Step 2: Center the points
    src_centered = src_points - src_centroid
    dst_centered = dst_points - dst_centroid

    # Step 3: Compute scale
    scale = np.linalg.norm(dst_centered) / np.linalg.norm(src_centered)

    # Step 4: Scale the destination points
    dst_centered_scaled = dst_centered / scale

    # Step 5: Compute cross-covariance matrix
    H = np.dot(src_centered.T, dst_centered_scaled)

    # Step 6: Compute rotation using the structure of the rotation matrix
    h11, h12 = H[0, 0], H[0, 1]
    h21, h22 = H[1, 0], H[1, 1]
    theta = np.arctan2(h21 - h12, h11 + h22)

    # Step 7: Compute translation
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    t = dst_centroid - scale * np.dot(R, src_centroid)

    return scale, theta, t



def homography_unknown_scale(kp1, kp2, matches, max_iterations=1000, threshold=1.0, inlier_ratio=0.5):
    """
    Estimate scale, rotation, and translation using RANSAC.
    
    Parameters:
        kp1 (list): List of keypoints from the first image.
        kp2 (list): List of keypoints from the second image.
        m (list): List of matches between the keypoints.
        max_iterations (int): Maximum number of RANSAC iterations.
        threshold (float): Inlier distance threshold.
        inlier_ratio (float): Minimum fraction of points required as inliers to terminate early.
    
    Returns:
        best_scale (float): Estimated scale.
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
        # Step 1: Randomly sample 3 points
        idx = np.random.choice(n_points, 3, replace=False)
        src_sample = src_points[idx]
        dst_sample = dst_points[idx]

        # Step 2: Estimate the transformation using the sampled points
        try:
            scale, theta, t = homography_unknown_scale_naive(src_sample, dst_sample)
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
            best_model = (scale, theta, t)

        # Terminate early if enough inliers are found
        if max_inliers > n_points * inlier_ratio:
            break

    # Refine the model using all inliers
    if best_model is not None:
        inlier_src = src_points[best_inliers]
        inlier_dst = dst_points[best_inliers]
        best_scale, best_theta, best_t = homography_unknown_scale_naive(inlier_src, inlier_dst)
    else:
        raise ValueError("RANSAC failed to find a valid model.")

    return best_scale, best_theta, best_t

import numpy as np

def homography_known_scale(kp1, kp2, matches, scale, max_iterations=1000, threshold=1.0, inlier_ratio=0.5):
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