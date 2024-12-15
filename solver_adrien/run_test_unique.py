from fonctions_image import *

# Variables
#url = "http://192.168.225.205:8080/video"
url = "http://10.220.14.33:8080/video"
puzzle_image_path = "nos_puzzles/chateau.jpg"
verbose = True

#Transform :
scale = None
theta = None
t = None
# Chargement
cap = start_camera(url)
target_image, sift, keypoints_full, descriptors_full = load_puzzle(puzzle_image_path)

frame = read_frame(cap)
print(frame.shape)
cv2.imwrite("frame.jpg", frame)

# Extraction des pièces
pieces = extract_pieces(frame, verbose)
show_found_pieces(pieces)

for piece in pieces:
    keypoints_piece, descriptors_piece = calculate_keypoints_sift(sift, piece, verbose=True)
    good_matches = calculate_matches(piece, target_image, keypoints_piece, descriptors_piece, keypoints_full, descriptors_full, verbose=True)
    
    
    if len(good_matches) > 4 :
        H, scale, theta, t = calculate_transform(good_matches, keypoints_piece, keypoints_full,scale, theta, t)
        show_transform_zncc(piece, target_image, H)