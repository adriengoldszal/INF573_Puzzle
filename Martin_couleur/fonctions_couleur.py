import sys
import os
sys.path.append('/Users/martindrieux/Documents/GitHub/INF573_Puzzle/solver_adrien')  # Je n'arrive mas un fair un chemin relaitf

from fonctions_image import *  # Importer toutes les fonctions du fichier
import cv2
import matplotlib.pyplot as plt
import numpy as np


def compute_log_polar_histogram(image, bins=30):
    """
    Compute a log-polar histogram for an image to ensure rotation and scaling robustness.
    
    Parameters:
        image (numpy.ndarray): Input image (assumed to be in RGB or HSV format).
        bins (int): Number of bins for the histogram.
        
    Returns:
        histogram (numpy.ndarray): Log-polar histogram normalized to sum to 1.
    """
    # Convert to HSV for better illumination invariance
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Extract the hue channel (or you can modify this for multi-channel histograms)
    hue_channel = hsv_image[:, :, 0]
    
    # Compute the histogram
    center = (hue_channel.shape[1] // 2, hue_channel.shape[0] // 2)
    log_polar_img = cv2.logPolar(
        hue_channel.astype(np.float32),
        center,
        40,  # Magnitude scaling factor for log-polar transformation
        cv2.INTER_LINEAR
    )
    
    # Create histogram bins
    histogram, _ = np.histogram(
        log_polar_img.flatten(),
        bins=bins,
        range=(0, 255),
        density=True  # Normalize the histogram
    )
    
    return histogram

def compute_emd_similarity(hist1, hist2):
    """
    Compute the Earth Mover's Distance (EMD) between two histograms.
    
    Parameters:
        hist1 (numpy.ndarray): First histogram.
        hist2 (numpy.ndarray): Second histogram.
        
    Returns:
        emd (float): Earth Mover's Distance.
    """
    # Convert histograms to cumulative distributions
    cdf1 = np.cumsum(hist1)
    cdf2 = np.cumsum(hist2)
    
    # Calculate the Earth Mover's Distance
    emd = np.sum(np.abs(cdf1 - cdf2))
    return emd


def display_image_grid(image_grid, title="Image Grid"):
    """
    Displays a 2D grid of images in their correct spatial arrangement.
    
    Parameters:
        image_grid (list of lists): A 2D list of images (NumPy arrays) to be displayed.
        title (str): Title for the entire grid display.
    """
    num_rows = len(image_grid)
    num_cols = len(image_grid[0]) if num_rows > 0 else 0

    # Create a subplot for the grid
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
    fig.suptitle(title, fontsize=16)

    for i, row in enumerate(image_grid):
        for j, image in enumerate(row):
            ax = axes[i, j] if num_rows > 1 else axes[j]  # Handle single row/column grids
            if image is not None:
                # Convert image from BGR (OpenCV format) to RGB for Matplotlib
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                ax.imshow(image_rgb)
            ax.axis("off")  # Hide axes for cleaner display

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust spacing to fit the title
    plt.show()




def split_image_to_grid(image, num_vertical, num_horizontal):
    """
    Splits an image into equal-sized rectangles and returns a 2D list of the parts.

    Parameters:
        image (numpy.ndarray): Input image as a NumPy array.
        num_vertical (int): Number of vertical pieces (rows).
        num_horizontal (int): Number of horizontal pieces (columns).
        
    Returns:
        grid (list of lists): 2D list where each element is a sub-image (NumPy array).
    """
    if image is None:
        raise ValueError("The provided image is None. Please provide a valid image.")

    # Get image dimensions
    height, width, _ = image.shape

    # Calculate the size of each piece
    piece_height = height // num_vertical
    piece_width = width // num_horizontal

    # Initialize the grid
    grid = []

    # Split the image
    for i in range(num_vertical):
        row = []
        for j in range(num_horizontal):
            # Calculate the coordinates of the current piece
            y_start = i * piece_height
            y_end = (i + 1) * piece_height
            x_start = j * piece_width
            x_end = (j + 1) * piece_width

            # Crop the piece
            piece = image[y_start:y_end, x_start:x_end]
            row.append(piece)
        grid.append(row)

    return grid

def compute_similarity_matrix(grid, piece):
    """
    Compute the similarity matrix between a grid of images and a target image.

    Parameters:
        grid (list of lists): 2D list of images (NumPy arrays).
        piece (numpy.ndarray): Target image as a NumPy array.

    Returns:
        similarity_matrix (numpy.ndarray): Matrix of similarities between the target image and each grid element.
    """
    # Initialize the similarity matrix
    num_rows = len(grid)
    num_cols = len(grid[0])
    similarity_matrix = np.zeros((num_rows, num_cols))
    hist_im=compute_log_polar_histogram(piece)
    # Compute the similarity between the target image and each grid element

    for i in range(num_rows):
        for j in range(num_cols):
            hist_ij=compute_log_polar_histogram(grid[i][j])
            similarity_matrix[i, j] = compute_emd_similarity(hist_im, hist_ij)

    return similarity_matrix

def display_heatmap(matrix, title="Heatmap", cmap="viridis"):
    """
    Affiche une heatmap d'une matrice donnée.

    Paramètres :
        matrix (numpy.ndarray) : La matrice à afficher.
        title (str) : Le titre de la heatmap.
        cmap (str) : Le colormap utilisé pour la heatmap (par défaut : 'viridis').
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap=cmap, aspect='auto')
    plt.colorbar(label="Intensity")  # Ajoute une barre de couleur
    plt.title(title)
    plt.xlabel("Colonnes")
    plt.ylabel("Lignes")
    plt.show()

##########################


def create_descriptor(img, mask=None, square_size=20, crop_to_mask=True):
    """
    Create a color-based descriptor for an image with an optional mask.

    Parameters:
        img (numpy array): BGR image.
        mask (numpy array, optional): Binary mask of the region of interest (can be None).
        square_size (int): Size of the square blocks in pixels.
        crop_to_mask (bool): Whether to crop to the bounding box of the mask.

    Returns:
        descriptor (list): List of dictionaries with block features.
    """
    # Convert the image to LAB color space for better color matching
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    if mask is not None and crop_to_mask:
        # Get bounding box around the mask
        y_indices, x_indices = np.where(mask > 0)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        x_min, x_max = np.min(x_indices), np.max(x_indices)

        # Crop image and mask to the bounding box
        img_lab = img_lab[y_min:y_max+1, x_min:x_max+1]
        mask = mask[y_min:y_max+1, x_min:x_max+1]
    else:
        y_min, x_min = 0, 0
        mask = np.ones(img.shape[:2], dtype=np.uint8) * 255 if mask is None else mask

    # Get the height and width of the image region
    h, w = img_lab.shape[:2]

    descriptor = []

    # Calculate the number of blocks based on the square size
    blocks_y = h // square_size
    blocks_x = w // square_size

    for i in range(blocks_y):
        for j in range(blocks_x):
            # Define block coordinates
            y0, y1 = i * square_size, (i + 1) * square_size
            x0, x1 = j * square_size, (j + 1) * square_size

            # Ensure the block is within image bounds
            if y1 > h:
                y1 = h
            if x1 > w:
                x1 = w

            # Extract the patch and corresponding mask
            patch = img_lab[y0:y1, x0:x1]
            patch_mask = mask[y0:y1, x0:x1]

            # Only consider patches where the mask is non-zero
            if np.count_nonzero(patch_mask) == 0:
                continue

            # Extract valid pixels within the mask
            valid_pixels = patch[patch_mask > 0]

            # Compute mean color (L, A, B channels)
            mean_color = np.mean(valid_pixels, axis=0)

            # Append block feature to the descriptor
            descriptor.append({
                'mean_color': mean_color,
                'block_coords': (i, j),
                'bounding_box': (y0 + y_min, y1 + y_min, x0 + x_min, x1 + x_min)
            })

    return descriptor


def compute_descriptor_distance(desc1, desc2):
    """
    Compute the distance between two descriptors by comparing compatible blocks.

    Parameters:
        desc1 (list): First descriptor.
        desc2 (list): Second descriptor.

    Returns:
        float: The total distance between compatible blocks.
    """
    distance = 0
    count = 0

    # Create a set of block coordinates for each descriptor
    coords1 = set(block['block_coords'] for block in desc1)
    coords2 = set(block['block_coords'] for block in desc2)

    # Find common coordinates
    common_coords = coords1.intersection(coords2)

    # Compare mean colors for blocks with common coordinates
    for block1 in desc1:
        if block1['block_coords'] in common_coords:
            for block2 in desc2:
                if block1['block_coords'] == block2['block_coords']:
                    dist = np.linalg.norm(block1['mean_color'] - block2['mean_color'])
                    distance += dist
                    count += 1

    # Return average distance if there are common blocks
    return distance / count if count > 0 else float('inf')


def search_puzzle_for_piece(puzzle_img, puzzle_mask, piece_img, piece_mask, puzzle_square_size=20, piece_square_size=20, top_n=5):
    """
    Search for the top non-overlapping matches for a piece within the puzzle and create a simplified distance map.

    Parameters:
        puzzle_img (numpy array): BGR image of the whole puzzle.
        puzzle_mask (numpy array): Binary mask for the puzzle.
        piece_img (numpy array): BGR image of the puzzle piece.
        piece_mask (numpy array): Binary mask for the puzzle piece.
        puzzle_square_size (int): Square size for the puzzle descriptor.
        piece_square_size (int): Square size for the piece descriptor.
        top_n (int): Number of top matches to return.

    Returns:
        tuple: 
            - List of tuples containing match positions (top-left corners) and their distances.
            - Distance map (2D numpy array) representing the distance at each valid position.
    """
    # Create descriptors for the puzzle and the piece
    piece_descriptor = create_descriptor(piece_img, piece_mask, piece_square_size, crop_to_mask=True)
    h_piece = max(block['bounding_box'][1] for block in piece_descriptor) - min(block['bounding_box'][0] for block in piece_descriptor)
    w_piece = max(block['bounding_box'][3] for block in piece_descriptor) - min(block['bounding_box'][2] for block in piece_descriptor)

    # Calculate the number of positions in the y and x directions
    y_steps = (puzzle_img.shape[0] - h_piece) // puzzle_square_size + 1
    x_steps = (puzzle_img.shape[1] - w_piece) // puzzle_square_size + 1

    # Initialize a smaller distance map
    distance_map = np.full((y_steps, x_steps), np.inf)

    # List to store all matches
    matches = []

    # Slide the piece over the puzzle
    for yi in range(y_steps):
        for xi in range(x_steps):
            y = yi * puzzle_square_size
            x = xi * puzzle_square_size

            # Define a sliding window region
            window = puzzle_img[y:y + h_piece, x:x + w_piece]
            window_mask = puzzle_mask[y:y + h_piece, x:x + w_piece] if puzzle_mask is not None else None

            # Create a descriptor for the window
            window_descriptor = create_descriptor(window, window_mask, piece_square_size, crop_to_mask=False)

            # Compute distance between the piece descriptor and the window descriptor
            distance = compute_descriptor_distance(piece_descriptor, window_descriptor)

            # Update the distance map
            distance_map[yi, xi] = distance

            # Store the position and distance
            matches.append(((x, y), distance))

    # Sort matches by distance
    matches.sort(key=lambda x: x[1])

    # Select the top N non-overlapping matches
    selected_matches = []
    for pos, dist in matches:
        x, y = pos
        overlap = False

        # Check for overlap with already selected matches
        for sel_x, sel_y in [match[0] for match in selected_matches]:
            if (abs(x - sel_x) < w_piece) and (abs(y - sel_y) < h_piece):
                overlap = True
                break

        if not overlap:
            selected_matches.append((pos, dist))

        if len(selected_matches) == top_n:
            break

    return selected_matches, distance_map

