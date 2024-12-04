import sys
import os
sys.path.append('/Users/martindrieux/Documents/GitHub/INF573_Puzzle/solver_adrien')  # Je n'arrive mas un fair un chemin relaitf

from fonctions_image import *  # Importer toutes les fonctions du fichier



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