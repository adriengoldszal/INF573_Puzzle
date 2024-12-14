import numpy as np
import cv2
import matplotlib.pyplot as plt

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


def homography_known_scale(kp1, kp2, matches,scale):
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
    scaled_src_points = src_points * scale
    matrix, _ = cv2.estimateAffinePartial2D(scaled_src_points, dst_points, False)
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

def show_homography_on_puzzle(img, puzzle_img, H):
    height, width, _ = puzzle_img.shape  # Taille de l'image cible
    warped_piece1 = cv2.warpPerspective(img, H, (width, height))

    # Superposer les images
    result = cv2.addWeighted(puzzle_img, 0.5, warped_piece1, 0.5, 0)

    plt.imshow(result)
    plt.show()
