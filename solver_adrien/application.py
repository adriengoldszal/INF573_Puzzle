import cv2
import time
import numpy as np
from fonctions_image import *
from homographies_2D import *

def run_realtime_view(url, puzzle_image_path, update_interval, verbose):
    # Initialize camera and puzzle image
    cap = start_camera(url)
    target_image, sift, keypoints_full, descriptors_full = load_puzzle(puzzle_image_path)
    last_update = 0
    canvas = np.zeros_like(target_image)  # Initialize canvas
    bbox = None
    
    # Initialize variables for homography calculation
    scale = None
    theta = None
    t = None
    
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
            H, scale, theta, t, bbox, best_piece = update_puzzle(frame.copy(), sift, target_image, keypoints_full, descriptors_full, scale, theta, t, verbose)

            show_homography_on_puzzle(best_piece['matching_image'], target_image, H)
            
        canvas = update_canvas(H, canvas, best_piece)
        
        if bbox is not None:
           x, y, w, h = bbox
           # Draw the rectangle on the frame
           cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
        combined_view = create_fullscreen_display(frame, canvas, update_interval, last_update)
        
        # Display the combined view
        cv2.imshow("Real-time Puzzle Assembly", combined_view)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()


def update_puzzle(frame, sift, target_image, keypoints_full, descriptors_full, scale, theta, t, verbose):
    
    extract_start = time.time()
    pieces = extract_pieces(frame)
    print(f"Extracting pieces took {time.time() - extract_start:.3f} seconds")
    
    find_best_piece_start = time.time()    
    best_piece, best_piece_keypoints, best_piece_matches, bbox = find_best_piece(pieces, sift, target_image, keypoints_full, descriptors_full, verbose)
    print(f"Finding best piece took {time.time() - find_best_piece_start:.3f} seconds")
    
    transform_start = time.time()
    H, scale, theta, t = calculate_transform(best_piece_matches, best_piece_keypoints, keypoints_full, scale, theta, t)
    print(f"Calculating transform took {time.time() - transform_start:.3f} seconds")
    
    return H, scale, theta, t, bbox, best_piece


def find_best_piece(pieces, sift, target_image, keypoints_full, descriptors_full, verbose=False):
    best_piece = None
    best_piece_keypoints = None
    best_piece_matches = None
    max_matches = 0
    
    for i, piece in enumerate(pieces):
        match_start = time.time()
        keypoints_piece, descriptors_piece = calculate_keypoints_sift(sift, piece)
        good_matches = calculate_matches(piece, target_image, keypoints_piece, descriptors_piece, keypoints_full, descriptors_full)
        if len(good_matches) > max_matches:
            max_matches = len(good_matches)
            best_piece = piece
            best_piece_keypoints = keypoints_piece
            best_piece_matches = good_matches
        print(f"Piece {i+1} matching took {time.time() - match_start:.3f} seconds ({len(good_matches)} matches)")

    bbox = best_piece['position'] + best_piece['size']
    
    return best_piece, best_piece_keypoints, best_piece_matches, bbox

def update_canvas(H, canvas, piece):
    
    print("H type:", type(H))
    print("H shape:", H.shape if isinstance(H, np.ndarray) else "not ndarray")
    print("H contents:", H)
    
    H = np.float32(H)
    # Create mask and remove existing content
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
    
    return canvas_with_frame
    
    
def create_fullscreen_display(frame, canvas, update_interval, last_update):

    screen = cv2.namedWindow("Real-time Puzzle Assembly", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Real-time Puzzle Assembly", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    screen_width = cv2.getWindowImageRect("Real-time Puzzle Assembly")[2]
    screen_height = cv2.getWindowImageRect("Real-time Puzzle Assembly")[3]
    
    # Calculate aspect ratios
    frame_aspect = frame.shape[1] / frame.shape[0]
    canvas_aspect = canvas.shape[1] / canvas.shape[0]
    
    # Target height will be half of screen height to accommodate both images
    target_height = screen_height // 2
    
    # Calculate widths based on aspect ratio
    frame_width = int(target_height * frame_aspect)
    canvas_width = int(target_height * canvas_aspect)
    
    # Resize images while maintaining aspect ratio
    frame_resized = cv2.resize(frame, (frame_width, target_height))
    canvas_resized = cv2.resize(canvas, (canvas_width, target_height))
    
    # Create a white background image of screen size
    background = np.full((screen_height, screen_width, 3), 255, dtype=np.uint8)
    
    # Calculate positions to center the images
    frame_x = (screen_width - (frame_width + canvas_width)) // 3
    canvas_x = frame_x * 2 + frame_width
    y_offset = (screen_height - target_height) // 2
    
    # Place images on white background
    background[y_offset:y_offset+target_height, frame_x:frame_x+frame_width] = frame_resized
    background[y_offset:y_offset+target_height, canvas_x:canvas_x+canvas_width] = canvas_resized
    
    # Add the countdown text
    current_time = time.time()
    remaining_time = max(0, update_interval - (current_time - last_update))
    text = f"Next match in {remaining_time:.1f} secs..."
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    color = (0, 255, 0)
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = screen_width - text_size[0] - 20
    text_y = 40
    
    cv2.putText(background, text, (text_x, text_y), font, font_scale, color, thickness)
    
    escape_text = "Press 'q' to quit"
    escape_size = cv2.getTextSize(escape_text, font, font_scale, thickness)[0]
    escape_x = screen_width - escape_size[0] - 20
    escape_y = screen_height - 40
    
    cv2.putText(background, escape_text, (escape_x, escape_y), font, font_scale, color, thickness)
    
    return background