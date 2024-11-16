from fonctions_image import *

# Variables
url = "http://192.168.1.15:8080/video"
puzzle_image_path = "nos_puzzles/fete.jpg"


# Chargement
cap = start_camera(url)
sift, bf, target_image, keypoints_full, descriptors_full = load_image_sift_knn(puzzle_image_path)

frame = read_frame(cap)
print(frame.shape)

# Extraction des pièces
pieces = extract_pieces(frame)
show_found_pieces(pieces)

for piece in pieces:
    piece, good_matches, keypoints_piece = calculate_matches(piece, sift, bf, target_image, keypoints_full, descriptors_full)
    canvas = calculate_transform(piece, good_matches, keypoints_piece, keypoints_full, target_image)

