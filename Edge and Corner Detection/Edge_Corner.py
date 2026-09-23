from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def compute_structure_tensor(image, sigma=2):
    gray_image = image.convert("L")
    image_array = np.array(gray_image, dtype=np.float64)

    # Compute image gradients
    Ix = np.zeros_like(image_array)
    Iy = np.zeros_like(image_array)

    Ix[:, 1:-1] = (image_array[:, 2:] - image_array[:, :-2])/2
    Iy[1:-1, :] = (image_array[2:, :] - image_array[:-2, :])/2

    # Compute the components of the structure tensor
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    IxIy = Ix * Iy

    # Gaussian-weighted summation over the local neighborhood
    A11 = gaussian_filter(Ix2, sigma=sigma)
    A22 = gaussian_filter(Iy2, sigma=sigma)
    A12 = gaussian_filter(IxIy, sigma=sigma)

    # Store the structure tensor at every pixel
    structure_tensor = np.zeros(
        (image_array.shape[0], image_array.shape[1], 2, 2)
    )

    structure_tensor[:, :, 0, 0] = A11
    structure_tensor[:, :, 0, 1] = A12
    structure_tensor[:, :, 1, 0] = A12
    structure_tensor[:, :, 1, 1] = A22

    # Compute eigenvalues and eigenvectors at every pixel
    eigenvalues, eigenvectors = np.linalg.eigh(structure_tensor)

    return structure_tensor, eigenvalues, eigenvectors


def detect_corners(structure_tensor, eigenvalues, type, k=0.08):
    
    # Harris-Stephens
    if type == "harris-stephens":

        A11 = structure_tensor[:, :, 0, 0]
        A12 = structure_tensor[:, :, 0, 1]
        A22 = structure_tensor[:, :, 1, 1]

        # Compute determinant and trace
        determinant = A11 * A22 - A12 * A12
        trace = A11 + A22

        # Harris corner-ness measure
        score = determinant - k * (trace * trace)

    # Shi-Tomasi
    elif type == "shi-tomasi":

        # since np.linalg.eigh gives eigenvalues in ascending order
        score = eigenvalues[:, :, 0]

    else:
        raise ValueError("type must be 'harris-stephens' or 'shi-tomasi'")

    return score

def non_max_suppression(score, window_size=3):
    output = score.copy()

    half_window = window_size // 2

    for i in range(half_window, score.shape[0] - half_window):
        for j in range(half_window, score.shape[1] - half_window):

            neighborhood = score[
                i-half_window:i+half_window+1,
                j-half_window:j+half_window+1
            ]

            if score[i, j] < np.max(neighborhood):
                output[i, j] = 0

    return output

def threshold_image(image, threshold):
    binary = np.zeros_like(image, dtype=np.uint8)
    binary[image > threshold] = 255

    return binary
