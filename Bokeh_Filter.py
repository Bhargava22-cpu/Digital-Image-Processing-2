from PIL import Image
import numpy as np
from scipy.ndimage import convolve, distance_transform_edt


def create_disc_kernel(diameter):
    radius = diameter // 2

    x = np.arange(-radius, radius + 1).reshape(-1, 1)
    y = np.arange(-radius, radius + 1)

    kernel = ((y*y + x*x) <= radius*radius).astype(np.float64)
    kernel = kernel / np.sum(kernel)

    return kernel


def bokeh_filter(image, foreground_mask, diameter):

    image = np.array(image, dtype=np.float64)
    foreground_mask = foreground_mask.astype(bool)

    background_mask = ~foreground_mask

    kernel = create_disc_kernel(diameter)
    radius = diameter // 2

    # Normal convolution for pixels far from boundaries
    blurred = np.zeros_like(image)

    for c in range(3):
        blurred[:, :, c] = convolve(
            image[:, :, c],
            kernel,
            mode="constant",
            cval=0
        )

    # Distance from foreground boundary
    background_distance = distance_transform_edt(background_mask)

    h, w = background_mask.shape

    x, y = np.indices((h, w))

    image_boundary_distance = np.minimum.reduce([
        x,
        y,
        h - 1 - x,
        w - 1 - y
    ])

    # Pixels where the complete filter fits inside background and image
    safe_pixels = (
        background_mask &
        (background_distance > radius) &
        (image_boundary_distance > radius)
    )

    # Pixels requiring cropped and renormalized filtering
    boundary_pixels = background_mask & ~safe_pixels

    for x, y in zip(*np.where(boundary_pixels)):

        x1 = max(0, x - radius)
        x2 = min(h, x + radius + 1)

        y1 = max(0, y - radius)
        y2 = min(w, y + radius + 1)

        kernel_x1 = x1 - (x - radius)
        kernel_x2 = kernel_x1 + (x2 - x1)

        kernel_y1 = y1 - (y - radius)
        kernel_y2 = kernel_y1 + (y2 - y1)

        local_kernel = kernel[
            kernel_x1:kernel_x2,
            kernel_y1:kernel_y2
        ].copy()

        local_mask = background_mask[x1:x2, y1:y2]

        local_kernel *= local_mask

        kernel_sum = np.sum(local_kernel)

        if kernel_sum > 0:

            local_kernel /= kernel_sum

            for c in range(3):
                blurred[x, y, c] = np.sum(
                    image[x1:x2, y1:y2, c] * local_kernel
                )

    # Keep foreground unchanged
    output = image.copy()
    output[background_mask] = blurred[background_mask]

    return output.astype(np.uint8)


def process_bokeh(image_path, mask_path, output_path, diameter):

    image = Image.open(image_path).convert("RGB")
    foreground_mask = np.array(
        Image.open(mask_path).convert("L")
    )

    foreground_mask = foreground_mask > 127

    output = bokeh_filter(
        image,
        foreground_mask,
        diameter
    )

    Image.fromarray(output).save(output_path)


image_path = (
    "/Users/bhargavaaddepalli/Desktop/CS663/"
    "assignment_2_Filtering/data/bokeh/marigold.png"
)

mask_path = (
    "/Users/bhargavaaddepalli/Desktop/CS663/"
    "assignment_2_Filtering/Bokeh/Bokeh Output Images/mask_marigold.png"
)


output_folder = (
    "/Users/bhargavaaddepalli/Desktop/CS663/"
    "assignment_2_Filtering/Bokeh/Bokeh Output Images"
)

process_bokeh(
    image_path,
    mask_path,
    output_folder + "/marigold_bokeh_50.png",
    50
)

process_bokeh(
    image_path,
    mask_path,
    output_folder + "/marigold_bokeh_100.png",
    100
)