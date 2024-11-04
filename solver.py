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
        # Create a blank visualization canvas with new size
        self.visualization = np.ones(self.main_window_size[::-1] + (3,), dtype=np.uint8) * 255
        
        # Calculate new sizes for display - make them smaller
        display_height = int(self.main_window_size[1] * 0.6)  # 60% of window height
        camera_width = display_height
        target_width = display_height
        
        # Center the images vertically
        y_offset = (self.main_window_size[1] - display_height) // 2
        
        # Draw target image on the right side with padding
        target_resized = cv2.resize(self.target_image, (target_width, display_height))
        right_padding = 50  # padding from right edge
        x_target = self.main_window_size[0] - target_width - right_padding
        self.visualization[y_offset:y_offset+display_height, x_target:x_target+target_width] = target_resized
        
        # Save these values for use in run method
        self.display_height = display_height
        self.camera_width = camera_width
        self.y_offset = y_offset
        self.x_target = x_target
        
        # Precompute target image features
        self.keypoints_full, self.descriptors_full = self.sift.detectAndCompute(self.target_image, None)
        
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
    
    
    def calculate_transform(self, src_pts, dst_pts):
        """Unused for now : Calculate transformation matrix."""
        # You'll need to implement this based on your original code
        return cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)[0]
    
    def match_piece(self, piece_info):
        """Match the detected piece to the target image using SIFT features."""
        if piece_info is None:
            return None
            
        try:
            # Extract features from the piece
            self.keypoints_piece, descriptors_piece = self.sift.detectAndCompute(piece_info['image'], None)
            print(f"\nFound {len(self.keypoints_piece)} keypoints in piece")
            
            if descriptors_piece is None:
                print("No descriptors found in piece")
                return None
            if len(descriptors_piece) < 2:
                print("Not enough descriptors in piece")
                return None
                
            print(f"Found {len(self.keypoints_full)} keypoints in target puzzle")
            
            try:
                # Match descriptors
                matches = self.bf.knnMatch(descriptors_piece, self.descriptors_full, k=2)
                print(f"Found {len(matches)} initial matches")
                
                # Apply ratio test
                self.good_matches = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        self.good_matches.append(m)
                        
                print(f"After ratio test: {len(self.good_matches)} good matches")
                
                # Need minimum number of matches
                if len(self.good_matches) < 4:
                    print("Not enough good matches to determine location")
                    return None
                    
                # Sort matches by distance and take top 20
                self.good_matches = sorted(self.good_matches, key=lambda x: x.distance)[:20]
                print(f"Top 20 matches distances: {[m.distance for m in self.good_matches[:5]]}")  # Show first 5
                
                # Extract location points
                src_pts = np.float32([self.keypoints_piece[m.queryIdx].pt for m in self.good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([self.keypoints_full[m.trainIdx].pt for m in self.good_matches]).reshape(-1, 1, 2)
                
                # Calculate average destination point as the piece location
                avg_dst = np.mean(dst_pts.reshape(-1, 2), axis=0)
                print(f"Match found at position: {tuple(map(int, avg_dst))}")
                return tuple(map(int, avg_dst))
                
            except Exception as e:
                print(f"Error during matching: {e}")
                return None
                
        except Exception as e:
            print(f"Error in match_piece: {e}")
            return None
    
    def extract_piece(self, frame):
        """Extract a single puzzle piece from the camera frame."""
        try:
            # Convert to grayscale and threshold
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            threshold, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE)
            
            # Find connected components
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            
            # Find the largest component (assumed to be the puzzle piece)
            largest_area = 0
            piece_info = None
            
            for i in range(1, num_labels):  # Skip background (0)
                x, y, w, h, area = stats[i]
                
                if area < 150:  # Skip very small components
                    continue
                    
                if area > largest_area:
                    largest_area = area
                    
                    # Get mask for this piece
                    piece_mask = (labels[y:y+h, x:x+w] == i)
                    
                    # Create properly sized piece image
                    piece = np.full((h, w, 3), 255, dtype=np.uint8)
                    
                    # Get the corresponding region from the frame
                    frame_region = frame[y:y+h, x:x+w]
                    
                    # Apply mask to both the piece and frame region
                    for c in range(3):  # Process each color channel separately
                        piece[:, :, c] = np.where(piece_mask, frame_region[:, :, c], 255)
                    
                    piece_info = {
                        'image': piece,
                        'mask': piece_mask,
                        'position': (x, y),
                        'size': (w, h)
                    }
            
            return piece_info
            
        except Exception as e:
            print(f"Error in extract_piece: {e}")
            return None
    
    def update_visualization(self, piece_info, matched_location):
        """Update the visualization with the new piece location."""
        if piece_info is None or matched_location is None:
            return
            
        try:
            # Draw the detected piece outline
            x, y = piece_info['position']
            w, h = piece_info['size']
            cv2.rectangle(self.visualization[self.y_offset:self.y_offset+self.display_height, 
                        50:50+self.camera_width], 
                        (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            # Scale matched location to visualization size
            vis_x = int(self.x_target + (matched_location[0] * self.display_height / self.target_image.shape[1]))
            vis_y = int(self.y_offset + (matched_location[1] * self.display_height / self.target_image.shape[0]))
            
            # Draw matched location with larger circle and cross
            cv2.circle(self.visualization, (vis_x, vis_y), 8, (0, 255, 0), 2)
            cv2.line(self.visualization, 
                    (vis_x - 10, vis_y), (vis_x + 10, vis_y), 
                    (0, 255, 0), 2)
            cv2.line(self.visualization, 
                    (vis_x, vis_y - 10), (vis_x, vis_y + 10), 
                    (0, 255, 0), 2)
            
            # Create and apply the piece mask overlay
            if y + h <= self.display_height and x + w <= self.camera_width:  # Check if piece is within bounds
                overlay = self.visualization.copy()
                overlay[self.y_offset+y:self.y_offset+y+h, 
                    50+x:50+x+w][piece_info['mask']] = (0, 255, 0)
                cv2.addWeighted(overlay, 0.3, self.visualization, 0.7, 0, self.visualization)
                
        except Exception as e:
            print(f"Error in update_visualization: {e}")
    
    def run(self, use_iriun=False):
        """Main loop for the interactive puzzle solver."""
        try:
            self.start_camera(use_iriun)
            self.running = True
            last_process_time = time.time()
            current_piece_info = None
            current_match_location = None
            
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    break
                    
                # Display camera feed on the left side
                frame_resized = cv2.resize(frame, (self.camera_width, self.display_height))
                x_camera = 50  # left padding
                self.visualization[self.y_offset:self.y_offset+self.display_height, 
                                x_camera:x_camera+self.camera_width] = frame_resized
                
                # Process new piece every 7 seconds
                current_time = time.time()
                time_remaining = max(0, 7 - (current_time - last_process_time))
                
                # Put timer and status below the images
                status_y = self.y_offset + self.display_height + 30
                
                if current_time - last_process_time >= 7:
                    # Clear status area
                    cv2.rectangle(self.visualization, (45, status_y), (400, status_y + 70), (255, 255, 255), -1)
                    
                    current_piece_info = self.extract_piece(frame_resized)
                    if current_piece_info is not None:
                        current_match_location = self.match_piece(current_piece_info)
                        
                        if current_match_location is not None:
                            cv2.putText(self.visualization, "Piece detected and matched!", 
                                    (50, status_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            # Draw feature matches in a separate window
                            matches_img = cv2.drawMatches(current_piece_info['image'], 
                                                    self.keypoints_piece,
                                                    self.target_image, 
                                                    self.keypoints_full,
                                                    self.good_matches, None,
                                                    flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
                            
                            # Resize to fit debug window size
                            h, w = matches_img.shape[:2]
                            scale = self.debug_window_size[1] / h
                            matches_resized = cv2.resize(matches_img, 
                                                    (int(w * scale), self.debug_window_size[1]))
                            
                            cv2.namedWindow("Feature Matches", cv2.WINDOW_NORMAL)
                            cv2.imshow("Feature Matches", matches_resized)
                            cv2.resizeWindow("Feature Matches", 
                                        self.debug_window_size[0], 
                                        self.debug_window_size[1])
                            
                        else:
                            cv2.putText(self.visualization, "Piece found but no match!", 
                                    (50, status_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        cv2.putText(self.visualization, "No piece detected!", 
                                (50, status_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                    last_process_time = current_time
                
                # Always update visualization with current piece info
                if current_piece_info is not None:
                    self.update_visualization(current_piece_info, current_match_location)
                
                # Add labels and instructions
                cv2.putText(self.visualization, "Camera Feed", 
                        (50, self.y_offset - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.putText(self.visualization, "Target Puzzle", 
                        (self.x_target, self.y_offset - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                
                # Clear the timer area
                cv2.rectangle(self.visualization, (45, status_y), (300, status_y + 40), (255, 255, 255), -1)
                
                # Draw the timer text
                cv2.putText(self.visualization, f"Next piece in: {int(time_remaining)}s", 
                    (50, status_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
                # Add quit instruction at bottom
                cv2.putText(self.visualization, "Press 'q' to quit", 
                        (50, self.main_window_size[1]-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
                # Show main window
                cv2.namedWindow("Puzzle Solver", cv2.WINDOW_NORMAL)
                cv2.imshow("Puzzle Solver", self.visualization)
                cv2.resizeWindow("Puzzle Solver", self.main_window_size[0], self.main_window_size[1])
                
                # Check for exit conditions
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or cv2.getWindowProperty("Puzzle Solver", cv2.WND_PROP_VISIBLE) < 1:
                    self.running = False
                    
        except KeyboardInterrupt:
            print("\nGracefully shutting down...")
        except Exception as e:
            print(f"Unexpected error in run: {e}")
        finally:
            cv2.destroyAllWindows()
            self.cleanup()

def main():
    try:
        # Get camera choice from command line argument
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
        for i in range(5):
            cv2.waitKey(1)


if __name__ == "__main__":
    main()