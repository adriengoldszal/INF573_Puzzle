import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import time
import sys

class InteractivePuzzleSolver:
    def __init__(self, target_image_path=None):
        """Initialize the puzzle solver with an optional target image."""
        self.target_image = None
        self.visualization = None
        self.cap = None
        self.running = False
        
        # Initialize SIFT and matcher
        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher()
        
        # Define window sizes
        self.main_window_size = (1200, 800)  # Larger main window
        self.debug_window_size = (1200, 800)  # Wider debug window
        
        if target_image_path:
            self.set_target_image(target_image_path)
            
    def start_camera(self, use_iriun=False):
        """Initialize the camera capture with format debugging."""
        try:
            if use_iriun:
                print("Attempting to connect to Iriun webcam...")
                for index in [1, 2, 3]:
                    print(f"Trying Iriun camera index {index}...")
                    self.cap = cv2.VideoCapture(index)
                    if self.cap.isOpened():
                        # Set and verify camera properties
                        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                        
                        # Read and check frame properties
                        ret, test_frame = self.cap.read()
                        if ret:
                            print("\nCamera properties:")
                            print(f"Resolution: {test_frame.shape}")
                            print(f"Format: {test_frame.dtype}")
                            print(f"Value range: {test_frame.min()} to {test_frame.max()}")
                            print(f"FourCC code: {int(self.cap.get(cv2.CAP_PROP_FOURCC))}")
                            print(f"Backend: {int(self.cap.get(cv2.CAP_PROP_BACKEND))}")
                            
                            # Try to normalize frame format
                            test_frame = cv2.normalize(test_frame, None, 0, 255, cv2.NORM_MINMAX)
                            
                            # Save these properties for later use
                            self.frame_height = test_frame.shape[0]
                            self.frame_width = test_frame.shape[1]
                            return
                            
                raise Exception("Could not connect to Iriun webcam")
            else:
                print("Connecting to default webcam...")
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    raise Exception("Could not open default webcam")
                print("Successfully connected to default webcam")
                
        except Exception as e:
            print(f"Error connecting to camera: {e}")
            raise
         
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
            # Convert to grayscale
            binary = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to binary image
            _, binary = cv2.threshold(binary, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE)
            
            # Debug: show binary image
            cv2.imshow("Binary", binary)
            
            # Find connected components (puzzle pieces)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
            pieces = []
            
            # Skip label 0 as it's background
            for i in range(1, num_labels):
                x, y, w, h, area = stats[i]
                
                # Filter out very small or very large components
                if area < 1000 or area > frame.shape[0] * frame.shape[1] * 0.5:  
                    continue
                
                # Get the mask for this specific piece
                piece_mask = (labels == i).astype(np.uint8)
                
                # Extract the piece with some padding
                padding = 10
                x_start = max(0, x - padding)
                y_start = max(0, y - padding)
                x_end = min(frame.shape[1], x + w + padding)
                y_end = min(frame.shape[0], y + h + padding)
                
                # Extract the piece region
                piece_img = frame[y_start:y_end, x_start:x_end].copy()
                piece_mask = piece_mask[y_start:y_end, x_start:x_end]
                
                # Create white background version for feature matching
                white_bg = np.full_like(piece_img, 255)
                piece_mask_3d = np.stack([piece_mask] * 3, axis=-1)
                piece_on_white = np.where(piece_mask_3d == 1, piece_img, white_bg)
                
                pieces.append({
                    'image': piece_img,  # Store original image
                    'matching_image': piece_on_white,  # Store white background version for matching
                    'binary_mask': piece_mask,
                    'position': (x_start, y_start),
                    'size': (x_end - x_start, y_end - y_start)
                })
            
            print(f"Found {len(pieces)} valid pieces")
            return pieces
            
        except Exception as e:
            print(f"Error in extract_pieces: {e}")
            import traceback
            traceback.print_exc()
            return []
        
    def match_pieces(self, pieces):
        """Match multiple detected pieces to the target image and return best match."""
        if not pieces:
            return None, None, None
            
        best_match = None
        best_score = 0
        best_matches = None
        best_piece = None
        self.keypoints_piece = None  # Reset class attributes
        
        for piece in pieces:
            try:
                # Extract features from the piece
                keypoints_piece, descriptors_piece = self.sift.detectAndCompute(piece['matching_image'], None)
                
                if descriptors_piece is None or len(descriptors_piece) < 2:
                    continue
                    
                # Match descriptors
                matches = self.bf.knnMatch(descriptors_piece, self.descriptors_full, k=2)
                
                # Apply ratio test
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
                
                if len(good_matches) < 4:
                    continue
                    
                # Sort matches by distance
                good_matches = sorted(good_matches, key=lambda x: x.distance)[:20]
                
                # Extract location points
                src_pts = np.float32([keypoints_piece[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([self.keypoints_full[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # Calculate match score based on number of good matches
                score = len(good_matches)
                
                # Check spatial consistency using homography
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if H is None:  # Skip if homography couldn't be computed
                    continue
                    
                spatially_consistent_matches = np.sum(mask)
                
                # Update best match if this piece has better score
                if spatially_consistent_matches > best_score:
                    best_score = spatially_consistent_matches
                    best_match = np.mean(dst_pts.reshape(-1, 2), axis=0)
                    best_matches = good_matches
                    best_piece = piece
                    self.keypoints_piece = keypoints_piece  # Save keypoints for visualization
                    
            except Exception as e:
                print(f"Error matching piece: {e}")
                continue
                
        if best_match is not None:
            print(f"Best match found with {best_score} spatially consistent features")
            
        return best_piece, best_match, best_matches if best_match is not None else None
    
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
                    # Detect pieces
                    pieces = self.extract_pieces(frame_resized)
                    if pieces:
                        # Find best matching piece
                        current_best_piece, current_match_location, current_good_matches = self.match_pieces(pieces)
                        
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
                    
                    # Display confidence score
                    cv2.putText(current_visualization,
                            f"Match Score: {len(current_good_matches)} features",
                            (50, self.y_offset - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Show match status
                    cv2.putText(current_visualization, "Match found!",
                            (50, self.y_offset + self.display_height + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Show feature matches in debug window
                    if hasattr(self, 'keypoints_piece') and self.keypoints_piece is not None:
                        matches_img = cv2.drawMatches(current_best_piece['matching_image'],
                                                self.keypoints_piece,
                                                self.target_image,
                                                self.keypoints_full,
                                                current_good_matches, None,
                                                flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
                        
                        h, w = matches_img.shape[:2]
                        scale = self.debug_window_size[1] / h
                        matches_resized = cv2.resize(matches_img,
                                                (int(w * scale), self.debug_window_size[1]))
                        
                        cv2.namedWindow("Feature Matches", cv2.WINDOW_NORMAL)
                        cv2.imshow("Feature Matches", matches_resized)
                        cv2.resizeWindow("Feature Matches",
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
        parser.add_argument('--puzzle', type=str, default="chateau.jpg", help='Path to puzzle image')
        args = parser.parse_args()
        
        print(f"Using {'Iriun webcam' if args.iriun else 'default webcam'}")
        solver = InteractivePuzzleSolver(args.puzzle)
        solver.run(args.iriun)
        
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()