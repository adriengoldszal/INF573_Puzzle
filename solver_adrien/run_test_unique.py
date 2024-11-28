from fonctions_image import *

# Variables
url = "http://10.220.14.33:8080/video"
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
    piece, good_matches, keypoints_piece = calculate_matches(piece, sift, bf, target_image, keypoints_full, descriptors_full, verbose)
    if len(good_matches) > 4 :
        canvas = calculate_transform(piece, good_matches, keypoints_piece, keypoints_full, target_image, verbose)

