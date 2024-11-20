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

    sift=cv2.SIFT_create(
                nfeatures=0,        # Keep unlimited features
                nOctaveLayers=5,    # Increase from default 3
                contrastThreshold=0.03,  # Lower to detect more features (default 0.04)
                edgeThreshold=20,    # Increase from default 10
                sigma=2.0           # Increase from default 1.6 for larger features
            )
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
        min_area = frame.shape[0] * frame.shape[1] * 0.001  # 2% of image area
        
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
            plt.title(f"Piece {i} (White BG)")
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
    
    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
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
    
    
    

    