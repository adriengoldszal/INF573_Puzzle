from functions_for_notebooks import *

# Variables
iriun_index = 1
puzzle_image_path = "nos_puzzles/chateau.jpg"


# Chargement
cap = start_camera(iriun_index)
sift, bf, target_image, keypoints_full, descriptors_full = load_image_sift_knn(puzzle_image_path)

""" A décommenter si on veut tester en direct
# Lecture d'une image
_, frame = read_frame(cap)
print(frame.shape)
"""

frame = cv2.imread("screenshots/screenshot_4.png") # Commenter si on veut tester en direct
# Extraction des pièces
pieces = extract_pieces(frame)
show_found_pieces(pieces)

calculate_matches(pieces, sift, bf, target_image, keypoints_full, descriptors_full)

