import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import time
import sys
from scipy.spatial.distance import pdist
from itertools import combinations

from typing import List, Tuple



#la fonction détecte toujours des sous pièces 
def extract_pieces(frame):
        """Extract multiple puzzle pieces from the camera frame."""
    
        # Convert to grayscale and blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=11,
            C=2
        )
        
        # Small closing to connect components
        kernel_small = np.ones((3, 3), np.uint8)
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_small)
        
        # Find and fill holes in each component
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(morph)
        
        for contour in contours:
            # Fill each contour
            cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # Now mask has filled pieces without holes
        morph = mask
        
        cv2.imshow("Adaptive Threshold", binary)
        cv2.imshow("Morphological", morph)
        
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(morph, connectivity=8)
        pieces = []
        
        # Adjust area thresholds for smaller pieces
        min_area = frame.shape[0] * frame.shape[1] * 0.01  # Reduced to 1%
        max_area = frame.shape[0] * frame.shape[1] * 0.25  # Reduced to 25%
        
        # Skip label 0 as it's background
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            
            # Filter out components that are too small or too large
            if area < min_area or area > max_area:
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


def get_match(piece, keyppoints_full, descriptors_full):
    "Prend en argument une image de pièce de puzzle et un vecteur de caractéristiques et renvoie le nom de la pièce"
    keypoints_piece, descriptors_piece = get_features(img_piece)


    bf = cv2.BFMatcher()
    # Match descriptors
    matches =bf.knnMatch(descriptors_piece,descriptors_full, k=2)
    

    # Apply ratio test for better matches
    good_matches = []
    for m,n in matches:
        if m.distance < 0.7 * n.distance:  # Lowe's ratio test
            good_matches.append(m)
    
    # Sort matches by distance and take top 5
    good_matches = sorted(good_matches, key=lambda x: x.distance)[:5]
    return good_matches





# def calculate_transform(img_piece, img_puzzle):
#     "Prend en argument une image de pièce de puzzle et une image de puzzle et renvoie la transformation à appliquer"
#     return transformation