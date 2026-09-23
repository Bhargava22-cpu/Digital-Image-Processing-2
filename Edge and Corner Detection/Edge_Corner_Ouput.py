from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from Edge_Corner import (
    compute_structure_tensor,
    detect_corners,
    non_max_suppression,
    threshold_image
)


def normalize_image(image):
    image_min = np.min(image)
    image_max = np.max(image)

    if image_max == image_min:
        return np.zeros(image.shape, dtype=np.uint8)

    image = (image - image_min) / (image_max - image_min)
    image = image * 255

    return image.astype(np.uint8)


def save_results(image, sigma, k,
                 harris_threshold, shi_tomasi_threshold,
                 edge_threshold, nms_window):

    original_image = np.array(image)

    # Compute structure tensor and eigenvalues
    structure_tensor, eigenvalues, eigenvectors = compute_structure_tensor(
        image,
        sigma=sigma
    )

    # Compute Harris score
    harris_score = detect_corners(
        structure_tensor,
        eigenvalues,
        "harris-stephens",
        k=k
    )

    # Compute Shi-Tomasi score
    shi_tomasi_score = detect_corners(
        structure_tensor,
        eigenvalues,
        "shi-tomasi"
    )

    # Get eigenvalue images
    largest_eigenvalue = eigenvalues[:, :, 1]
    second_largest_eigenvalue = eigenvalues[:, :, 0]

    # Apply non-maximum suppression
    harris_nms = non_max_suppression(
        harris_score,
        nms_window
    )

    shi_tomasi_nms = non_max_suppression(
        shi_tomasi_score,
        nms_window
    )

    # Create binary corner images
    harris_binary = threshold_image(
        harris_nms,
        harris_threshold
    )

    shi_tomasi_binary = threshold_image(
        shi_tomasi_nms,
        shi_tomasi_threshold
    )

    # Draw Harris corners in red
    harris_corner_image = original_image.copy()

    for y, x in zip(*np.where(harris_binary == 255)):
        harris_corner_image[y-1:y+2, x-1:x+2] = [255, 0, 0]

    # Draw Shi-Tomasi corners in red
    shi_tomasi_corner_image = original_image.copy()

    for y, x in zip(*np.where(shi_tomasi_binary == 255)):
        shi_tomasi_corner_image[y-1:y+2, x-1:x+2] = [255, 0, 0]

    # Harris edge-ness
    harris_edge_score = -harris_score

    harris_edge_nms = non_max_suppression(
        harris_edge_score,
        nms_window
    )

    # Create binary edge image
    harris_edge_binary = threshold_image(
        harris_edge_nms,
        edge_threshold
    )

   # Draw Harris edges in green
    harris_edge_image = original_image.copy()

    for y, x in zip(*np.where(harris_edge_binary == 255)):
        harris_edge_image[y-1:y+2, x-1:x+2] = [0, 255, 0]

    output_folder = (
        "/Users/bhargavaaddepalli/Desktop/CS663/"
        "assignment_2_Filtering/Edge and Corner/"
        "Edge_Corner Output Images"
    )

    image_name = "paithani"

    # Save largest eigenvalue
    Image.fromarray(
        normalize_image(largest_eigenvalue)
    ).save(
        output_folder + "/" + image_name + "_02_Largest_Eigenvalue.png"
    )

    # Save second-largest eigenvalue
    Image.fromarray(
        normalize_image(second_largest_eigenvalue)
    ).save(
        output_folder + "/" + image_name + "_03_Second_Largest_Eigenvalue.png"
    )

    # Save Harris corner-ness
    Image.fromarray(
        normalize_image(harris_score)
    ).save(
        output_folder + "/" + image_name + "_04_Harris_Cornerness.png"
    )

    # Save Harris after NMS
    Image.fromarray(
        normalize_image(harris_nms)
    ).save(
        output_folder + "/" + image_name + "_05_Harris_After_NMS.png"
    )

    # Save Shi-Tomasi after NMS
    Image.fromarray(
        normalize_image(shi_tomasi_nms)
    ).save(
        output_folder + "/" + image_name + "_06_Shi_Tomasi_After_NMS.png"
    )

    # Save binary corner outputs
    Image.fromarray(harris_binary).save(
        output_folder + "/" + image_name + "_07_Harris_Binary_Corners.png"
    )

    Image.fromarray(shi_tomasi_binary).save(
        output_folder + "/" + image_name + "_08_Shi_Tomasi_Binary_Corners.png"
    )

    # Save corners drawn on input image
    Image.fromarray(harris_corner_image).save(
        output_folder + "/" + image_name + "_09_Harris_Corners.png"
    )

    Image.fromarray(shi_tomasi_corner_image).save(
        output_folder + "/" + image_name + "_10_Shi_Tomasi_Corners.png"
    )

    # Save Harris edge-ness after NMS
    Image.fromarray(
        normalize_image(harris_edge_nms)
    ).save(
        output_folder + "/" + image_name + "_11_Harris_Edgeness_After_NMS.png"
    )

    # Save binary edges
    Image.fromarray(harris_edge_binary).save(
        output_folder + "/" + image_name + "_12_Harris_Binary_Edges.png"
    )

    # Save edges drawn on input image
    Image.fromarray(harris_edge_image).save(
        output_folder + "/" + image_name + "_13_Harris_Edges.png"
    )


image = Image.open(
    "/Users/bhargavaaddepalli/Desktop/CS663/"
    "assignment_2_Filtering/data/corner/paithaniCorner.png"
)


# Set parameters here
sigma = 2.5

k = 0.08

harris_threshold = 5000
shi_tomasi_threshold = 75

edge_threshold = 1500

nms_window = 5

"""
Params for paithani

sigma = 2.5

k = 0.08

harris_threshold = 5000
shi_tomasi_threshold = 75

edge_threshold = 1500

nms_window = 5
"""

"""
Params for nandadevi

sigma = 2.5

k = 0.08

harris_threshold = 1.5e6
shi_tomasi_threshold = 1400
edge_threshold = 100

nms_window = 5
"""

"""
Params for warli

sigma = 2.5

k = 0.08

harris_threshold = 2e6
shi_tomasi_threshold = 1600
edge_threshold = 1000

nms_window = 5

"""

# Run the complete experiment
save_results(
    image,
    sigma,
    k,
    harris_threshold,
    shi_tomasi_threshold,
    edge_threshold,
    nms_window
)
