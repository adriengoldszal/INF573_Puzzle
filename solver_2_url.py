import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import time
import sys
from scipy.spatial.distance import pdist
from itert

#
url = "http://10.222.14.104:8080/video" 
class InteractivePuzzleSolver:
    def __init__(self, target_image_path=None):
        """Initialize the puzzle solver with an optional target image."""
        self.target_image = None
        self.visualization = None
        self.cap = None
        self.running = False
        self.captured_frames = []
        
        # Initialize SIFT and matcher
        self.sift = cv2.SIFT_create(
                nfeatures=0,        # Keep unlimited features
                nOctaveLayers=5,    # Increase from default 3
                contrastThreshold=0.03,  # Lower to detect more features (default 0.04)
                edgeThreshold=20,    # Increase from default 10
                sigma=2.0           # Increase from default 1.6 for larger features
            )
        self.bf = cv2.BFMatcher()
        
        # Define window sizes
        self.main_window_size = (1200, 800)  # Larger main window
        self.debug_window_size = (1200, 800)  # Wider debug window
        
        if target_image_path:
            self.set_target_image(target_image_path)
            
    def start_camera(self, use_iriun=False):
        """Initialize the camera capture with format debugging."""
        self.cap = cv2.VideoCapture(url)
        if not self.cap.isOpened():
            print("Error: Unable to access video stream.")
            return
         
    def cleanup(self):
        """Clean up resources."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        for i in range(5):
            cv2.waitKey(1)
    
    def set_target_image(self, image_path):
        """Load and process the target puzzle image."""
        self.target_image = cv2.imread(image_path)
        self.visualization = np.ones(self.main_window_size[::-1] + (3,), dtype=np.uint8) * 255
        
        display_height = int(self.main_window_size[1] * 0.6)
        camera_width = display_height
        target_width = display_height
        
        y_offset = (self.main_window_size[1] - display_height) // 2
        right_padding = 50
        x_target = self.main_window_size[0] - target_width - right_padding
        
        target_resized = cv2.resize(self.target_image, (target_width, display_height))
        self.visualization[y_offset:y_offset+display_height, x_target:x_target+target_width] = target_resized
        
        self.display_height = display_height
        self.camera_width = camera_width
        self.y_offset = y_offset
        self.x_target = x_target
        
        # Precompute target image features
        self.keypoints_full, self.descriptors_full = self.sift.detectAndCompute(self.target_image, None)
        
    def extract_pieces(self, frame):
        """Extract multiple puzzle pieces from the camera frame."""
        try:
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
            
        except Exception as e:
            print(f"Error in extract_pieces: {e}")
            import traceback
            traceback.print_exc()
            return []
            
    def get_spatially_consistent_matches(self, good_matches, keypoints_piece, keypoints_full, piece_size, descriptors_piece, descriptors_full):
        """
        Find the largest subset of matches where all matched points in the full image 
        are within a distance threshold, accounting for potential scale differences between
        captured piece and target puzzle.
        """
        if len(good_matches) < 4:  # Need at least 4 matches for reliable scale estimation
            return [], None
        
        # Get points from both images for matched features
        src_pts = np.float32([keypoints_piece[m.queryIdx].pt for m in good_matches])
        dst_pts = np.float32([keypoints_full[m.trainIdx].pt for m in good_matches])
        
        # Estimate scale difference using the ratio of distances between points
        def estimate_scale():
            # Calculate pairwise distances in both images
            src_distances = pdist(src_pts)
            dst_distances = pdist(dst_pts)
            
            # Use median ratio to be robust to outliers
            ratios = dst_distances / (src_distances + 1e-6)  # Add small epsilon to avoid division by zero
            scale = np.median(ratios)
            return scale
        
        # Get estimated scale between captured piece and target puzzle
        scale = estimate_scale()
        
        # Adjust the maximum allowed distance based on the estimated scale
        piece_height, piece_width = piece_size
        base_max_distance = max(piece_height, piece_width)
        max_allowed_distance = base_max_distance * scale
        
        # Function to check if a set of points respects the scaled distance constraint
        def is_consistent_set(point_indices):
            if len(point_indices) <= 1:
                return True
                
            points = dst_pts[list(point_indices)]  # Convert to list for proper indexing
            distances = pdist(points)
            return np.all(distances < max_allowed_distance)
        
        # Try to find the largest consistent set
        best_set = []
        
        # Start with larger subsets first
        for size in range(len(dst_pts), 3, -1):  # Require at least 4 points for robustness
            for subset_indices in combinations(range(len(dst_pts)), size):
                if is_consistent_set(subset_indices):
                    best_set = list(subset_indices)  # Convert to list
                    # Found the largest consistent set, break both loops
                    break
            if best_set:  # If we found a consistent set, stop looking
                break
        
        # Convert the best set of indices back to matches
        consistent_matches = [good_matches[i] for i in best_set]
        match_points = None
        if best_set:
            match_points = dst_pts[best_set]  # This is where the error was happening
            
        return consistent_matches, match_points

    def save_captured_frames(self, folder_path="screenshots"):
        """Save all captured frames to a specified folder."""
        import os
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        for i, frame in enumerate(self.captured_frames):
            filename = os.path.join(folder_path, f"screenshot_{i}.png")
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")

    def match_pieces(self, pieces):
        """Match multiple detected pieces to the target image and return best match."""
        if not pieces:
            return None, None, None, None
            
        best_match = None
        best_score = 0
        best_matches = None
        best_piece = None
        best_mask = None
        self.keypoints_piece = None  # Reset class attributes
        
        for piece in pieces:
            try:
                # Extract features from the piece
                keypoints_piece, descriptors_piece = self.sift.detectAndCompute(piece['matching_image'], None)
                
                if descriptors_piece is None or len(descriptors_piece) < 2:
                    continue
                    
                # Match descriptors
                matches = self.bf.knnMatch(descriptors_piece, self.descriptors_full, k=2)
                
                if len(matches) < 4:
                    continue

                # Apply ratio test for better matches
                good_matches = []
                for m,n in matches:
                    if m.distance < 0.7 * n.distance:  # Lowe's ratio test
                        good_matches.append(m)
                
                # Sort matches by distance and take top 5
                good_matches = sorted(good_matches, key=lambda x: x.distance)[:5]
                
                h_piece, w_piece = piece['matching_image'].shape[:2]
                scale_piece = 5.0  # Make piece 2x bigger
                piece_resized = cv2.resize(piece['matching_image'], 
                                        (int(w_piece * scale_piece), int(h_piece * scale_piece)))

                matches_img = cv2.drawMatches(piece_resized,  # Use resized piece
                                        keypoints_piece,
                                        self.target_image,
                                        self.keypoints_full,
                                        good_matches, None,
                                        matchColor=(0, 255, 0),
                                        singlePointColor=None,
                                        flags=0,
                                        matchesThickness=3)
                
                # Resize for better visualization
                h, w = matches_img.shape[:2]
                scale = min(1.5, self.debug_window_size[1] / h)
                matches_resized = cv2.resize(matches_img,
                                        (int(w * scale), self.debug_window_size[1]))
                
                cv2.namedWindow(f"Matches for Piece", cv2.WINDOW_NORMAL)
                cv2.imshow(f"Matches for Piece", matches_resized)
                cv2.resizeWindow(f"Matches for Piece",
                            self.debug_window_size[0],
                            self.debug_window_size[1])
                
                print(f"Found {len(good_matches)} good matches")
                
                # Calculate match points for visualization
                if len(good_matches) > 0:
                    dst_pts = np.float32([self.keypoints_full[m.trainIdx].pt for m in good_matches])
                    best_match = np.mean(dst_pts, axis=0)
                    best_matches = good_matches
                    best_piece = piece
                    best_mask = np.ones((len(good_matches), 1), dtype=np.uint8)
                    self.keypoints_piece = keypoints_piece
                    
            except Exception as e:
                print(f"Error matching piece: {e}")
                continue
        
        return best_piece, best_match, best_matches, best_mask
    
    def draw_arrow(self, start_point, end_point, visualization, color=(0, 255, 0)):
        """Draw an arrow from the detected piece to its target location."""
        # Convert coordinates to visualization space
        start_x = int(50 + start_point[0])
        start_y = int(self.y_offset + start_point[1])
        end_x = int(self.x_target + (end_point[0] * self.display_height / self.target_image.shape[1]))
        end_y = int(self.y_offset + (end_point[1] * self.display_height / self.target_image.shape[0]))
        
        # Draw arrow line
        cv2.line(visualization, (start_x, start_y), (end_x, end_y), color, 2)
        
        # Calculate arrow head
        angle = np.arctan2(end_y - start_y, end_x - start_x)
        arrow_length = 20
        arrow_angle = np.pi / 6
        
        # Draw arrow head
        p1_x = int(end_x - arrow_length * np.cos(angle + arrow_angle))
        p1_y = int(end_y - arrow_length * np.sin(angle + arrow_angle))
        p2_x = int(end_x - arrow_length * np.cos(angle - arrow_angle))
        p2_y = int(end_y - arrow_length * np.sin(angle - arrow_angle))
        
        cv2.line(visualization, (end_x, end_y), (p1_x, p1_y), color, 2)
        cv2.line(visualization, (end_x, end_y), (p2_x, p2_y), color, 2)
    
    def run(self, use_iriun=False):
        """Main loop for the interactive puzzle solver."""
        try:
            self.start_camera(use_iriun)
            self.running = True
            last_process_time = time.time()
            current_best_piece = None
            current_match_location = None
            current_good_matches = None
            current_mask = None
            
            while self.running:
                ret, frame = self.cap.read()

                if not ret:
                    break
                    
                # Display camera feed
                frame_resized = cv2.resize(frame, (self.camera_width, self.display_height))
                current_visualization = np.ones(self.main_window_size[::-1] + (3,), dtype=np.uint8) * 255
                
                # Update camera view
                current_visualization[self.y_offset:self.y_offset+self.display_height, 
                                50:50+self.camera_width] = frame_resized
                
                # Update target puzzle view
                target_resized = cv2.resize(self.target_image, (self.camera_width, self.display_height))
                current_visualization[self.y_offset:self.y_offset+self.display_height, 
                                self.x_target:self.x_target+self.camera_width] = target_resized
                
                # Process pieces every few seconds
                current_time = time.time()
                time_remaining = max(0, 7 - (current_time - last_process_time))
                
                # Add timer text
                cv2.putText(current_visualization, f"Next update in: {int(time_remaining)}s",
                        (50, self.y_offset + self.display_height + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
                
                
                if current_time - last_process_time >= 7:
                                    #Save the frame
                    self.captured_frames.append(frame.copy())
                    # Detect pieces
                    pieces = self.extract_pieces(frame_resized)
                    if pieces:
                        # Find best matching piece
                        current_best_piece, current_match_location, current_good_matches, current_mask = self.match_pieces(pieces)
                        
                    last_process_time = current_time
                
                # Always show the current best match if we have one
                if current_best_piece and current_match_location is not None:
                    # Draw visible rectangle around best matching piece
                    x, y = current_best_piece['position']
                    w, h = current_best_piece['size']
                    cv2.rectangle(current_visualization[self.y_offset:self.y_offset+self.display_height,
                                50:50+self.camera_width],
                                (x, y), (x + w, y + h), (0, 0, 255), 3)  # Thicker red rectangle
                    
                    # Draw arrow from piece to target location
                    piece_center = (x + w//2, y + h//2)
                    self.draw_arrow(piece_center, current_match_location, current_visualization, (0, 255, 0))
                    
                    if current_good_matches is not None:
                        cv2.putText(current_visualization,
                                f"Spatially Consistent Matches: {len(current_good_matches)}", 
                                (50, self.y_offset - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                    
                    # Show match status
                    cv2.putText(current_visualization, "Match found!",
                                        (50, self.y_offset + self.display_height + 60),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    if current_good_matches is not None:
                            # Show feature matches in debug window

                            if hasattr(self, 'keypoints_piece') and self.keypoints_piece is not None:
                                matches_img = cv2.drawMatches(current_best_piece['matching_image'],
                                                        self.keypoints_piece,
                                                        self.target_image,
                                                        self.keypoints_full,
                                                        current_good_matches, None,  # Use current_good_matches directly
                                                        matchColor=(0, 255, 0),
                                                        matchesThickness=4,
                                                        flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
                                
                                h, w = matches_img.shape[:2]
                                scale = self.debug_window_size[1] / h
                                matches_resized = cv2.resize(matches_img,
                                                        (int(w * scale), self.debug_window_size[1]))
                                
                                cv2.namedWindow("Spatially Consistent Matches", cv2.WINDOW_NORMAL)
                                cv2.imshow("Spatially Consistent Matches", matches_resized)
                                cv2.resizeWindow("Spatially Consistent Matches",
                                            self.debug_window_size[0],
                                            self.debug_window_size[1])
                
                # Add labels
                cv2.putText(current_visualization, "Camera Feed",
                        (50, self.y_offset - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.putText(current_visualization, "Target Puzzle",
                        (self.x_target, self.y_offset - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                
                # Show main window
                cv2.namedWindow("Puzzle Solver", cv2.WINDOW_NORMAL)
                cv2.imshow("Puzzle Solver", current_visualization)
                cv2.resizeWindow("Puzzle Solver", self.main_window_size[0], self.main_window_size[1])
                
                # Check for exit conditions
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or cv2.getWindowProperty("Puzzle Solver", cv2.WND_PROP_VISIBLE) < 1:
                    self.running = False
                    
        except KeyboardInterrupt:
            print("\nGracefully shutting down...")
        except Exception as e:
            print(f"Unexpected error in run: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()


def main():
    try:
        import argparse
        parser = argparse.ArgumentParser(description='Interactive Puzzle Solver')
        parser.add_argument('--iriun', action='store_true', help='Use Iriun webcam instead of default camera')
        parser.add_argument('--puzzle', type=str, default="yakari.jpg", help='Path to puzzle image')
        args = parser.parse_args()
        
        print(f"Using {'Iriun webcam' if args.iriun else 'default webcam'}")
        solver = InteractivePuzzleSolver(args.puzzle)
        solver.run(args.iriun)

        solver.save_captured_frames()

        
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()