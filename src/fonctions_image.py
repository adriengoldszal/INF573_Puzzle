import cv2 as cv2
from skimage import io
import matplotlib.pyplot as plt
from homographies_2D import *
import numpy as np
from scipy.spatial.distance import cdist
from homographies_2D import *

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

def load_puzzle(image_path):
    """Load an image from file."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return image, sift, keypoints, descriptors

def extract_pieces(frame, verbose=False):
        """Extract all the puzzle pieces from the camera frame."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding for edge detection
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=11, 
            C=2  
        )
        

        kernel_small = np.ones((3, 3), np.uint8)
        kernel_medium = np.ones((8, 8), np.uint8)
        
        # Closing holes
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_medium)
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel_small)
        
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(morph)
        
        for contour in contours:
            cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # Fill contours
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
        
        # Get rid of noise
        min_area = frame.shape[0] * frame.shape[1] * 0.005
        
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            
            if area < min_area :
                continue
            
            piece_mask = (labels == i).astype(np.uint8)
            
            padding = min(20, min(w, h) // 4)  
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(frame.shape[1], x + w + padding)
            y_end = min(frame.shape[0], y + h + padding)
            
            piece_img = frame[y_start:y_end, x_start:x_end].copy()
            piece_mask = piece_mask[y_start:y_end, x_start:x_end]
            
            if np.sum(piece_mask) < area * 0.5: 
                continue
            
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
        rows = (n_pieces + 1) // 2 
        plt.figure(figsize=(15, 5*rows))
        
        for i, piece in enumerate(pieces):

            plt.subplot(rows, 4, i*2 + 1)
            plt.imshow(cv2.cvtColor(piece['image'], cv2.COLOR_BGR2RGB))
            plt.title(f"Piece {i}")
            plt.axis('off')
            
            plt.subplot(rows, 4, i*2 + 2)
            plt.imshow(cv2.cvtColor(piece['matching_image'], cv2.COLOR_BGR2RGB))
            plt.title(f"Piece {i} (Matching image used for SIFT)")
            plt.axis('off')
            
            print(f"\nPiece {i}:")
            print(f"Position: {piece['position']}")
            print(f"Size: {piece['size']}")
            print(f"Area: {piece['area']}")
        
        plt.tight_layout()
        plt.show()
    else:
        print("No pieces were found in the image")


def filter_spatially_consistent_matches(good_matches, keypoints_full, piece_size):
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
    
    dst_pts = np.float32([keypoints_full[m.trainIdx].pt for m in good_matches])
    
    distances = cdist(dst_pts, dst_pts)
    
    valid_neighbors = (distances < max_allowed_distance).sum(axis=1)
    best_start = np.argmax(valid_neighbors)
    
    consistent_indices = {best_start}
    candidates = set(range(len(dst_pts))) - {best_start}
    
    while candidates:

        best_score = -1
        best_idx = None
        
        for idx in candidates:
            if np.all(distances[idx, list(consistent_indices)] < max_allowed_distance):
                
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

def apply_mask_to_features(keypoints, descriptors, initial_mask, eroded_mask, margin=10):
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
    keypoints_filtered, descriptors_filtered = apply_mask_to_features(keypoints, descriptors, mask, eroded_mask)
    
    return keypoints_filtered, descriptors_filtered


def calculate_keypoints_sift(sift, piece, verbose=False):
    
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
        if m.distance < 1*n.distance: 
            good_matches.append(m)

    good_matches = sorted(good_matches, key=lambda x: x.distance)[:50]

    good_matches = filter_spatially_consistent_matches(good_matches, keypoints_full, piece['size'])

    if verbose :
        
        match_img = cv2.drawMatches(
            piece['matching_image'], keypoints,
            puzzle, keypoints_full,
            good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            matchesThickness=3,
            matchColor=(0, 255, 0),
        )

        print(f"Found {len(good_matches)} good matches")
        plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
        plt.show()
    
    return good_matches


def calculate_transform(best_piece_matches, best_piece_keypoints, keypoints_full, scale, theta, t) :
    
        if theta is None or t is None or scale is None:
            
            print("Unknown scale")
            H = homography_unknown_scale(best_piece_keypoints, keypoints_full, best_piece_matches)
        else :
            print("Known scale")
            H = homography_known_scale(best_piece_keypoints, keypoints_full, best_piece_matches,scale)

        scale, theta, t = decompose_similarity_homography(H)
        print(f'scale {scale}, theta {theta}, t {t}')
        
        return H, scale, theta, t
            