import cv2 as cv2
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.spatial.distance import pdist
import time
def start_camera(url):
    
    """Start the camera stream from IP Webcam."""

    cap = cv2.VideoCapture(url)
    
    if not cap.isOpened():
        print(f"Error: Could not connect to IP Webcam at {url}")
        return None
        
    print(f"Successfully connected to camera at {url}")
    
    return cap

def read_frame(cap):
    """Read a single frame from the camera stream."""
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        return False, None
    return frame


def get_features(img_piece):
    "Prend en argument une image de pièce de puzzle et renvoie un vecteur de caractéristiques et des keypoints"

    sift=cv2.SIFT_create()
    keypoints_full, descriptors_full = sift.detectAndCompute(img_piece, None)
    return keypoints_full, descriptors_full

def load_image_sift_knn(image_path):
    """Load solved puzzle with knn and sift."""
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()
    target_image = cv2.imread(image_path)
    keypoints_full, descriptors_full = sift.detectAndCompute(target_image, None)
    return sift, bf, target_image, keypoints_full, descriptors_full

def extract_pieces(frame, verbose=False):
        """Extract multiple puzzle pieces from the camera frame."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=11,  # Must be odd number
            C=2  # Constant subtracted from mean
        )
        

        # Morphological operations with smaller kernel
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_medium = np.ones((8, 8), np.uint8)
        #kernel_large = np.ones((15, 15), np.uint8)
        
        # Close small holes first
        #morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_large)
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_medium)
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel_small)
        
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(morph)
        
        for contour in contours:
            # Fill each contour
            cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # Now mask has filled pieces without holes
        morph = mask
        
        if verbose :
            plt.figure(figsize=(12, 4))
            plt.subplot(121)
            plt.imshow(binary, cmap='gray')
            plt.title("Adaptive Threshold")
            plt.subplot(122)
            plt.imshow(morph, cmap='gray')
            plt.title("Morphological with filled contours")
            plt.show()
        
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(morph, connectivity=8)
        pieces = []
        
        # Calculate minimum area threshold based on image size
        min_area = frame.shape[0] * frame.shape[1] * 0.005 # 2% of image area
        
        # Skip label 0 as it's background
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            
            # Filter out components that are too small or too large
            if area < min_area :
                continue
            
            # Get the mask for this specific piece
            piece_mask = (labels == i).astype(np.uint8)
            
            # Dilate the mask slightly to include edges
            piece_mask = cv2.dilate(piece_mask, kernel_small)
            
            # Extract the piece with dynamic padding based on piece size
            padding = min(20, min(w, h) // 4)  # Dynamic padding
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(frame.shape[1], x + w + padding)
            y_end = min(frame.shape[0], y + h + padding)
            
            # Extract the piece region
            piece_img = frame[y_start:y_end, x_start:x_end].copy()
            piece_mask = piece_mask[y_start:y_end, x_start:x_end]
            
            # Ensure mask has enough content
            if np.sum(piece_mask) < area * 0.5:  # At least 50% of original area
                continue
            
            # Create white background version for feature matching
            white_bg = np.full_like(piece_img, 255)
            piece_mask_3d = np.stack([piece_mask] * 3, axis=-1)
            piece_on_white = np.where(piece_mask_3d == 1, piece_img, white_bg)
            
            pieces.append({
                'image': piece_img,
                'matching_image': piece_on_white,
                'binary_mask': piece_mask,
                'position': (x_start, y_start),
                'size': (x_end - x_start, y_end - y_start),
                'area': area
            })
        
        print(f"Found {len(pieces)} valid pieces")
        if pieces:
            print("Piece areas:", [p['area'] for p in pieces])
            
        return pieces
        
def show_found_pieces(pieces) :
    n_pieces = len(pieces)
    if n_pieces > 0:
        rows = (n_pieces + 1) // 2  # 2 pieces per row
        plt.figure(figsize=(15, 5*rows))
        
        for i, piece in enumerate(pieces):
            # Original image
            plt.subplot(rows, 4, i*2 + 1)
            plt.imshow(cv2.cvtColor(piece['image'], cv2.COLOR_BGR2RGB))
            plt.title(f"Piece {i}")
            plt.axis('off')
            
            # White background version
            plt.subplot(rows, 4, i*2 + 2)
            plt.imshow(cv2.cvtColor(piece['matching_image'], cv2.COLOR_BGR2RGB))
            plt.title(f"Piece {i} (Matching image used for SIFT)")
            plt.axis('off')
            
            # Print piece information
            print(f"\nPiece {i}:")
            print(f"Position: {piece['position']}")
            print(f"Size: {piece['size']}")
            print(f"Area: {piece['area']}")
        
        plt.tight_layout()
        plt.show()
    else:
        print("No pieces were found in the image")


def get_spatially_consistent_matches(good_matches, keypoints_full, piece_size):
    """
    Find the largest subset of matches where all matched points in the full image 
    are within a distance threshold based on the piece size.
    
    Args:
        good_matches: List of matches that passed the ratio test
        keypoints_full: Keypoints from the full puzzle image
        piece_size: Tuple of (height, width) of the puzzle piece
        
    Returns:
        List of matches that are spatially consistent
    """
    
    
    piece_height, piece_width = piece_size
    max_allowed_distance = max(piece_height, piece_width) 

    if len(good_matches) == 0:
        return []

    # Get points in the full image for all matches
    dst_pts = np.float32([keypoints_full[m.trainIdx].pt for m in good_matches])
    
    # Function to check if a set of points respects the distance constraint
    def is_consistent_set(point_indices):
        points = np.array([dst_pts[i] for i in point_indices])
        if len(points) > 1:
            distances = pdist(points)
            return np.all(distances < max_allowed_distance)
        return True
    
    # Try to find the largest consistent set
    best_set = []
    from itertools import combinations
    # Try different sized subsets, from largest to smallest
    for size in range(len(dst_pts), 0, -1):
        # Check all possible combinations of this size
        for subset_indices in combinations(range(len(dst_pts)), size):
            if is_consistent_set(subset_indices):
                best_set = list(subset_indices)
                # Found the largest consistent set, break both loops
                break
        if best_set:  # If we found a consistent set, stop looking
            break
    
    # Convert the best set of indices back to matches
    return [good_matches[i] for i in best_set]

def calculate_matches(piece, sift, bf, target_image, keypoints_full, descriptors_full, verbose=False):
    """Calculate matches between one piece and the target image."""
    
    # Detect keypoints and compute descriptors
    sift_time = time.time()
    keypoints, descriptors = sift.detectAndCompute(piece['matching_image'], None)
    print(f"Keypoint detection and description took {time.time() - sift_time:.3f} seconds")
    
    # Match descriptors
    knn_matcher_time = time.time()
    matches = bf.knnMatch(descriptors, descriptors_full, k=2)
    print(f"KNN matching took {time.time() - knn_matcher_time:.3f} seconds")
    
    good_matches = []
    for m, _ in matches:
            good_matches.append(m)
            
    good_matches = sorted(good_matches, key=lambda x: x.distance)[:20]
    
    #La spatial consistency crée un bottleneck majeur
    # good_matches = get_spatially_consistent_matches(good_matches, keypoints_full, piece['size'])
    
     
    if verbose :
        
        # Draw matches
        match_img = cv2.drawMatches(
            piece['matching_image'], keypoints,
            target_image, keypoints_full,
            good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            matchesThickness=3,
            matchColor=(0, 255, 0),
        )
        # Display matches 
        print(f"Found {len(good_matches)} good matches")
        plt.figure(figsize=(10, 5))
        plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    
    return piece, good_matches, keypoints
    
def calculate_transform(piece, matches, keypoints_piece, keypoints_full, target_image, verbose=False):
    """Calculate homography transform and apply piece to the puzzle canvas."""
    
    canvas = target_image.copy()
    # Check if we have enough matches
    src_pts = np.float32([keypoints_piece[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_full[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
    if H is not None:
        # Warp piece and its mask
        warped_piece = cv2.warpPerspective(piece['image'], H, 
                                        (canvas.shape[1], canvas.shape[0]))
        warped_mask = cv2.warpPerspective(piece['binary_mask'], H, 
                                        (canvas.shape[1], canvas.shape[0]))
        warped_mask = (warped_mask * 255).astype(np.uint8)
        
        # Convert mask to 3 channels for masking colored image
        warped_mask_3d = cv2.cvtColor(warped_mask, cv2.COLOR_GRAY2BGR)
        
        # Create inverse mask for the canvas
        canvas_mask = cv2.bitwise_not(warped_mask_3d)
        
        # Combine the existing canvas with the new piece
        canvas_masked = cv2.bitwise_and(canvas, canvas_mask)
        piece_masked = cv2.bitwise_and(warped_piece, warped_mask_3d)
        canvas = cv2.add(canvas_masked, piece_masked)
        

        canvas_with_frame = cv2.rectangle(canvas.copy(), 
                                        (0, 0), 
                                        (canvas.shape[1]-1, canvas.shape[0]-1), 
                                        (0, 255, 0), 5)
        
        if verbose :
            # Display assembled puzzle
            plt.figure(figsize=(10, 10))
            plt.imshow(cv2.cvtColor(canvas_with_frame, cv2.COLOR_BGR2RGB))
            plt.title(f"Assembled Puzzle After Adding Piece")
            plt.axis("off")
            plt.show()
    
    return canvas, H
    
    
    
def draw_matches_enhanced(img1, kp1, img2, kp2, matches, color=(0, 255, 0)):
    """
    Draw matches with enhanced visibility, compatible with knnMatch output
    
    Parameters:
    - img1, img2: source images
    - kp1, kp2: keypoints from both images
    - matches: list of matches from knnMatch (takes first match from each pair)
    - color: color of the lines (default: green)
    """
    # Create a new output image that concatenates the two images
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]
    out = np.zeros((max(rows1, rows2), cols1 + cols2, 3), dtype='uint8')
    
    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1]) if len(img1.shape) == 2 else img1
    # Place the second image to the right
    out[:rows2,cols1:] = np.dstack([img2, img2, img2]) if len(img2.shape) == 2 else img2
    
    # For knnMatch, we'll use only the best match (first one) from each pair
    for m in matches:
        if len(m) < 2:  # Skip if we don't have 2 matches
            continue
            
        # Get the matching keypoints for each of the images
        img1_idx = m[0].queryIdx
        img2_idx = m[0].trainIdx

        # Get the coordinates
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        # Calculate match quality (ratio between first and second best match)
        ratio = m[0].distance / m[1].distance
        
        # Color based on ratio (green for good matches, red for potentially poor ones)
        match_color = (0, int(255 * (1 - ratio)), int(255 * ratio)) if color is None else color

        # Draw the match line with increased thickness
        cv2.line(out, (int(x1), int(y1)), 
                (int(x2) + cols1, int(y2)), 
                match_color, 2)
        
        # Draw circles around the keypoints
        cv2.circle(out, (int(x1), int(y1)), 4, match_color, 2)
        cv2.circle(out, (int(x2) + cols1, int(y2)), 4, match_color, 2)

    return out