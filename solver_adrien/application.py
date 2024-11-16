import cv2
import time
import numpy as np
from fonctions_image import *

def run_realtime_view(url, puzzle_image_path, update_interval, verbose):
    # Initialize camera and puzzle image
    cap = start_camera(url)
    sift, bf, target_image, keypoints_full, descriptors_full = load_image_sift_knn(puzzle_image_path)
    last_update = 0
    canvas = np.zeros_like(target_image)  # Initialize canvas
    bbox = None
    
    while True:
        # Get the current frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from camera.")
            break
        
        current_time = time.time()
        
        # Update the puzzle assembly every `update_interval` seconds
        if current_time - last_update > update_interval:
            print("Processing frame for puzzle assembly...")
            last_update = current_time
            
            # Extract pieces and process them
            canvas, bbox, best_piece = update_puzzle(frame.copy(), sift, bf, target_image, keypoints_full, descriptors_full, verbose)
            
        if bbox is not None:
           x, y, w, h = bbox
           # Draw the rectangle on the frame
           cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
        # Create the display frame regardless of update interval
        height = 500
        frame_resized = cv2.resize(frame, (int(height * frame.shape[1] / frame.shape[0]), height))
        canvas_resized = cv2.resize(canvas, (int(height * canvas.shape[1] / canvas.shape[0]), height))
        
        # Adjust widths if necessary to make them match exactly
        width = min(frame_resized.shape[1], canvas_resized.shape[1])
        frame_resized = frame_resized[:, :width]
        canvas_resized = canvas_resized[:, :width]
        
            
        # Calculate remaining time for next update
        remaining_time = max(0, update_interval - (current_time - last_update))
        
        # Display the "next match in X secs..." text
        text = f"Next match in {remaining_time:.1f} secs..."
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        color = (0, 255, 0)  # Green color for the text
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = frame_resized.shape[1] - text_size[0] - 10  # Right-aligned
        text_y = 30  # Position from the top
        
        # Add text on the frame
        cv2.putText(frame_resized, text, (text_x, text_y), font, font_scale, color, thickness)
        
        combined_view = np.hstack((frame_resized, canvas_resized))
        
        # Display the combined view
        cv2.imshow("Real-time Puzzle Assembly", combined_view)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()



def update_puzzle(frame, sift, bf, target_image, keypoints_full, descriptors_full, verbose):
    
    extract_start = time.time()
    pieces = extract_pieces(frame)
    print(f"Extracting pieces took {time.time() - extract_start:.3f} seconds")
    
    find_best_piece_start = time.time()    
    best_piece, best_piece_keypoints, best_piece_matches, bbox = find_best_piece(pieces, sift, bf, target_image, keypoints_full, descriptors_full, verbose)
    print(f"Finding best piece took {time.time() - find_best_piece_start:.3f} seconds")
    
    transform_start = time.time()
    canvas, H = calculate_transform(best_piece, best_piece_matches, best_piece_keypoints, keypoints_full, target_image, verbose)
    print(f"Calculating transform took {time.time() - transform_start:.3f} seconds")
    
    return canvas, bbox, best_piece


def find_best_piece(pieces, sift, bf, target_image, keypoints_full, descriptors_full, verbose=False):
    best_piece = None
    best_piece_keypoints = None
    best_piece_matches = None
    max_matches = 0
    
    for i, piece in enumerate(pieces):
        match_start = time.time()
        piece, good_matches, keypoints_piece = calculate_matches(
            piece, sift, bf, target_image, keypoints_full, descriptors_full, verbose
        )
        if len(good_matches) > max_matches:
            max_matches = len(good_matches)
            best_piece = piece
            best_piece_keypoints = keypoints_piece
            best_piece_matches = good_matches
        print(f"Piece {i+1} matching took {time.time() - match_start:.3f} seconds ({len(good_matches)} matches)")

    bbox = best_piece['position'] + best_piece['size']
    
    return best_piece, best_piece_keypoints, best_piece_matches, bbox