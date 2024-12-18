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
    canvas = np.zeros_like(target_image)  
    cumulative_mask = np.zeros((canvas.shape[0], canvas.shape[1]), dtype=np.uint8)
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
            print(f'Updating puzzle with current scale {scale}, theta {theta}, t {t}')
            H, scale, theta, t, bbox, best_piece, piece_found = update_puzzle(frame.copy(), cumulative_mask, sift, target_image, keypoints_full, descriptors_full, scale, theta, t, verbose)

        if piece_found:
            canvas, cumulative_mask = update_canvas(H, canvas, best_piece, cumulative_mask)
        
        if bbox is not None:
           x, y, w, h = bbox
           # Draw the rectangle on the frame
           cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
        combined_view = create_fullscreen_display(frame, canvas, update_interval, last_update, piece_found)
        
        # Display the combined view
        cv2.imshow("Real-time Puzzle Assembly", combined_view)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def update_puzzle(frame, cumulative_mask, sift, target_image, keypoints_full, descriptors_full, scale, theta, t, verbose):
    
    extract_start = time.time()
    pieces = extract_pieces(frame)
    print(f"Extracting pieces took {time.time() - extract_start:.3f} seconds")
    
    find_best_piece_start = time.time()    
    sorted_pieces = find_best_pieces_sorted(pieces, sift, target_image,cumulative_mask, keypoints_full, descriptors_full, verbose)
    print(f"Matching and sorting pieces took {time.time() - find_best_piece_start:.3f} seconds")
    
    transform_start = time.time()
    print(f"Finding piece above threshold with scale {scale}, theta {theta}, t {t}")
    result = find_first_piece_above_threshold(sorted_pieces, target_image, keypoints_full, scale, theta, t)
    print(f"Calculating transform took {time.time() - transform_start:.3f} seconds")
    
    if result :
        best_piece = result['piece']
        bbox = result['bbox']
        H = result['transform']
        scale, theta, t = decompose_similarity_homography(H)
        show_transform_zncc(best_piece, target_image, H)
        piece_found = True
        return H, scale, theta, t, bbox, best_piece, piece_found
    
    piece_found = False
    return None, scale, theta, t, None, None, piece_found
    


def find_best_pieces_sorted(pieces, sift, target_image, cumulative_mask, keypoints_full, descriptors_full, verbose=False):
    piece_matches = []
    
    for i, piece in enumerate(pieces):
        match_start = time.time()
        keypoints_piece, descriptors_piece = calculate_keypoints_sift(sift, piece)
        good_matches = calculate_matches(piece, target_image, keypoints_piece, 
                                      descriptors_piece, keypoints_full, descriptors_full)
        
        if len(good_matches) > 1:
        
            piece_matches.append({
                'piece': piece,
                'keypoints': keypoints_piece,
                'matches': good_matches,
                'num_matches': len(good_matches)
            })
        
            if verbose:
                print(f"Piece {i}: {len(good_matches)} matches, "
                    f"time: {time.time() - match_start:.2f}s")
    
    # Sort pieces by number of matches in descending order
    sorted_pieces = sorted(piece_matches, 
                         key=lambda x: x['num_matches'], 
                         reverse=True)
    
    return sorted_pieces

def find_first_piece_above_threshold(sorted_pieces, target_image, keypoints_full, scale, theta, t, zncc_threshold=0.3):
    transform_start = time.time()
    best_zncc = float('-inf')
    best_result = None
    
    for piece_info in sorted_pieces:
        piece = piece_info['piece']
        piece_keypoints = piece_info['keypoints']
        piece_matches = piece_info['matches']
        
        temp_H, _, _, _ = calculate_transform(piece_matches, piece_keypoints, 
                                               keypoints_full, scale, theta, t)
        
        height, width, _ = target_image.shape
        warped_piece = cv2.warpPerspective(piece['image'], temp_H, (width, height))
        warped_mask = cv2.warpPerspective(piece['binary_mask'], temp_H, (width, height))
        
        warped_mask = warped_mask > 0
        
        warped_region = warped_piece[warped_mask]
        puzzle_region = target_image[warped_mask]
        
        zncc_value = calculate_zncc(warped_region, puzzle_region)
        print(f"ZNCC value: {zncc_value}")
        
        # Store result for first piece
        if best_result is None:
            best_result = {
                'piece': piece,
                'keypoints': piece_keypoints,
                'matches': piece_matches,
                'bbox': piece['position'] + piece['size'],
                'transform': temp_H,
                'zncc': zncc_value
            }
        
        # If ZNCC is above threshold, return this piece
        if zncc_value > zncc_threshold:
            print(f"Found piece above threshold. ZNCC: {zncc_value}")
            print(f"Total transform calculation time: {time.time() - transform_start:.3f} seconds")
            
            return {
                'piece': piece,
                'keypoints': piece_keypoints,
                'matches': piece_matches,
                'bbox': piece['position'] + piece['size'],
                'transform': temp_H,
                'zncc': zncc_value
            }
    
    print("No pieces found above ZNCC threshold")
    return None

def update_canvas(H, canvas, piece, cumulative_mask):
    
    H = np.float32(H)

    warped_piece = cv2.warpPerspective(piece['image'], H, 
                                    (canvas.shape[1], canvas.shape[0]))
    warped_mask = cv2.warpPerspective(piece['binary_mask'], H, 
                                    (canvas.shape[1], canvas.shape[0]))
    warped_mask = (warped_mask * 255).astype(np.uint8)
    
    cumulative_mask = cv2.bitwise_or(cumulative_mask, warped_mask)

    warped_mask_3d = cv2.cvtColor(warped_mask, cv2.COLOR_GRAY2BGR)
    
    canvas_mask = cv2.bitwise_not(warped_mask_3d)
    
    canvas_masked = cv2.bitwise_and(canvas, canvas_mask)
    piece_masked = cv2.bitwise_and(warped_piece, warped_mask_3d)
    canvas = cv2.add(canvas_masked, piece_masked)
    

    canvas_with_frame = cv2.rectangle(canvas.copy(), 
                                    (0, 0), 
                                    (canvas.shape[1]-1, canvas.shape[0]-1), 
                                    (0, 255, 0), 5)
    
    return canvas_with_frame, cumulative_mask
    
    
def create_fullscreen_display(frame, canvas, update_interval, last_update, piece_found):

    screen_width, screen_height = 1920, 1080
    
    cv2.namedWindow("Real-time Puzzle Assembly", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Real-time Puzzle Assembly", screen_width, screen_height)
    
    frame_aspect = frame.shape[1] / frame.shape[0]
    canvas_aspect = canvas.shape[1] / canvas.shape[0]
    
    target_height = screen_height // 2
    
    frame_width = int(target_height * frame_aspect)
    canvas_width = int(target_height * canvas_aspect)

    frame_resized = cv2.resize(frame, (frame_width, target_height))
    canvas_resized = cv2.resize(canvas, (canvas_width, target_height))
    
    background = np.full((screen_height, screen_width, 3), 255, dtype=np.uint8)
    
    frame_x = (screen_width - (frame_width + canvas_width)) // 3
    canvas_x = frame_x * 2 + frame_width
    y_offset = (screen_height - target_height) // 2
    
    background[y_offset:y_offset+target_height, frame_x:frame_x+frame_width] = frame_resized
    background[y_offset:y_offset+target_height, canvas_x:canvas_x+canvas_width] = canvas_resized
    
    current_time = time.time()
    remaining_time = max(0, update_interval - (current_time - last_update))
    if piece_found:
        text = f"Piece found! Updating puzzle in {remaining_time:.1f} secs..."
    else :
        text = f"No piece found, retrying in {remaining_time:.1f} secs..."
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