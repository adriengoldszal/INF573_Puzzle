import cv2 as cv2
from skimage import io
import matplotlib.pyplot as plt
from homographies_2D import *
import numpy as np
from scipy.spatial.distance import cdist
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


def filter_spatially_consisten_matches(good_matches, keypoints_full, piece_size):
    """
    Find a large subset of matches where all matched points in the full image 
    are within a distance threshold based on the piece size.
    Calculates all pairwise distances in advance to avoid factorial complexity.
    
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
    
    # Calculate pairwise distances between all points
    distances = cdist(dst_pts, dst_pts)
    
    # Initialize with the point that has the most neighbors within threshold
    valid_neighbors = (distances < max_allowed_distance).sum(axis=1)
    best_start = np.argmax(valid_neighbors)
    
    consistent_indices = {best_start}
    candidates = set(range(len(dst_pts))) - {best_start}
    
    while candidates:
        # Find the point that's consistent with all current points
        best_score = -1
        best_idx = None
        
        for idx in candidates:
            # Check if this point is within threshold of all current points
            if np.all(distances[idx, list(consistent_indices)] < max_allowed_distance):
                # Score is number of additional valid neighbors it would add
                potential_neighbors = set(np.where(distances[idx] < max_allowed_distance)[0])
                score = len(potential_neighbors & candidates)
                if score > best_score:
                    best_score = score
                    best_idx = idx
        
        if best_idx is None:
            break
            
        consistent_indices.add(best_idx)
        candidates.remove(best_idx)
    
    return [good_matches[i] for i in consistent_indices]


def get_mask_with_margin(mask, margin=10):
    """Add a margin around the mask to include more keypoints

    """
    
    kernel = np.ones((margin*2+1, margin*2+1), np.uint8)
    eroded_mask = cv2.erode(mask.astype(np.uint8), kernel)
    
    return eroded_mask

def apply_mask_to_features(keypoints, descriptors,initial_mask, eroded_mask, margin=10):
    """Apply the mask that has a margin to remove noisy keypoitns and descriptors

    """
    height, width = initial_mask.shape
    filtered_keypoints = []
    filtered_descriptors = []
    for i, kp in enumerate(keypoints):
        x, y = map(int, kp.pt)
        if 0 <= y < height and 0 <= x < width and eroded_mask[y, x] > 0:
            filtered_keypoints.append(kp)
            filtered_descriptors.append(descriptors[i])

        return filtered_keypoints, np.array(filtered_descriptors)


def filter_keypoints_by_mask(keypoints, descriptors, mask, margin=10):
    """Reducing the mask to get rid of the edge keypoints that are just noise

    """
    eroded_mask = get_mask_with_margin(mask, margin)
    keypoints_filtered, descriptors_filtered = apply_mask_to_features(keypoints, descriptors, mask, eroded_mask, margin)
    
    return keypoints_filtered, descriptors_filtered


def calculate_keypoints_sift(sift, piece, puzzle, verbose=False):
    
    keypoints, descriptors = sift.detectAndCompute(piece['matching_image'], None)

    keypoints_filtered, descriptors_filtered = filter_keypoints_by_mask(keypoints, descriptors, piece['binary_mask'])
    
    if verbose :
        drawn_keypoints = cv2.drawKeypoints(piece["matching_image"], keypoints_filtered, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.imshow(drawn_keypoints)
        plt.show()
    
    return keypoints_filtered, descriptors_filtered


def calculate_matches(piece, puzzle, keypoints, descriptors, keypoints_full, descriptors_full, verbose=False):
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors, descriptors_full, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 1*n.distance:  # Lowe's ratio test (pas sûr de le garder car features peuvent être proches)
            good_matches.append(m)

    good_matches = sorted(good_matches, key=lambda x: x.distance)[:50]
    #La spatial consistency crée un bottleneck majeur
    good_matches = filter_spatially_consisten_matches(good_matches, keypoints_full, piece['size'])

    print("Matches and their distances:")
    for idx, match in enumerate(good_matches):
        print(f"Match {idx + 1}: Distance = {match.distance:.2f}")
    
    if verbose :
        # Draw matches
        match_img = cv2.drawMatches(
            piece['matching_image'], keypoints,
            puzzle, keypoints_full,
            good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            matchesThickness=3,
            matchColor=(0, 255, 0),
        )
        # Display matches 
        print(f"Found {len(good_matches)} good matches")
        plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
        plt.show()
    
    return good_matches


