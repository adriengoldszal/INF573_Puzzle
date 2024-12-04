from fonctions_image import *

# Variables
url = "http://192.168.30.13:8080/video"
puzzle_image_path = "nos_puzzles/yakari.jpg"
verbose = True

# Chargement
cap = start_camera(url)
sift, bf, target_image, keypoints_full, descriptors_full = load_image_sift_knn(puzzle_image_path)

frame = read_frame(cap)
print(frame.shape)
cv2.imwrite("frame.jpg", frame)

# Extraction des pièces
pieces = extract_pieces(frame, verbose)
show_found_pieces(pieces)

for piece in pieces:
    keypoints_piece, descriptors_piece, keypoints_full, descriptors_full = calculate_keypoints_sift(piece, target_image)
    good_matches = calculate_matches(piece, target_image, keypoints_piece, descriptors_piece, keypoints_full, descriptors_full)
    
    
    if len(good_matches) > 4 :
        canvas = calculate_transform(piece, good_matches, keypoints_piece, keypoints_full, target_image, byhand=False,verbose=True)
        

