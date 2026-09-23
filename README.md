# Digital Image Processing — Assignment 2

Implementation of fundamental digital image filtering, template matching, edge detection, corner detection, and background blurring techniques using Python.

## Contents

### 1. Image Sharpening

- **Unsharp Masking** — Sharpens images by subtracting a Gaussian-blurred version from the original image and adding the resulting high-frequency mask back to the image.
- **Multi-Level Sharpening** — Generates two sharpened versions using different sharpening scales and compares them with the original image.
- **Intensity Clipping** — Clips the sharpened image intensities to the intensity range of the input image.

The implementation applies Gaussian smoothing, computes the unsharp mask, and scales the mask before adding it to the original image.

### 2. Template Matching

- **Normalized Cross-Correlation (NCC)** — Searches for a template within a scene image independently for each RGB color channel.
- **Multi-Scale Template Matching** — Computes NCC results for template sizes of `41×41`, `51×51`, and `61×61` on a scene image reduced by a factor of 5.
- **Masked NCC** — Excludes the white background surrounding the red ring in the template when computing the correlation.
- **Large-Scale Masked Matching** — Computes masked NCC results for template sizes of `201×201`, `251×251`, and `301×301`.

The NCC implementation computes cross-correlation and normalizes it using the RMS values of the template and corresponding image patches.

### 3. Canny Edge Detection

- **Grayscale Conversion** — Converts RGB images to grayscale using luminance weights.
- **Gaussian Smoothing** — Reduces image noise before computing gradients.
- **Gradient Computation** — Computes horizontal and vertical gradients using Sobel operators.
- **Gradient Magnitude and Direction** — Calculates the magnitude and orientation of image gradients.
- **Non-Maximum Suppression** — Retains local gradient maxima along the gradient direction.
- **Double Thresholding** — Classifies pixels as strong or weak edges using high and low thresholds.
- **Edge Tracking by Hysteresis** — Keeps weak edges that are connected to strong edge pixels.

The implementation produces binary edge maps as well as visualizations with the detected edges overlaid on the original image. 

### 4. Edge and Corner Detection

- **Structure Tensor** — Computes image gradients and the Gaussian-weighted structure tensor at every pixel.
- **Eigenvalue Computation** — Computes the eigenvalues and eigenvectors of the structure tensor.
- **Harris-Stephens Corner Detection** — Computes the Harris cornerness measure using the determinant and trace of the structure tensor.
- **Shi-Tomasi Corner Detection** — Uses the smaller eigenvalue of the structure tensor as the cornerness measure.
- **Non-Maximum Suppression** — Retains local maxima in the cornerness response.
- **Thresholding** — Converts the cornerness and edge responses into binary detections.
- **Harris Edge Detection** — Uses the negative Harris response to obtain an edge-ness measure.
- **Corner and Edge Visualization** — Draws detected corners and edges on the original image.

The structure tensor is constructed from the local image gradients and Gaussian-weighted gradient products, followed by per-pixel eigenvalue decomposition.

The Harris-Stephens and Shi-Tomasi responses are then computed from the structure tensor and its eigenvalues, followed by non-maximum suppression and thresholding.

### 5. Bokeh Effect

- **Foreground/Background Masking** — Separates the foreground object from the background using a manually specified mask.
- **Disc-Shaped Filter** — Creates a normalized circular filter kernel.
- **Background Blurring** — Applies the disc-shaped filter only to background pixels while preserving the foreground.
- **Boundary Handling** — Crops the filter near image and object boundaries and renormalizes the cropped kernel.
- **Distance Transform** — Identifies pixels sufficiently far from boundaries so that they can be processed efficiently using standard convolution.
- **Multiple Blur Levels** — Generates Bokeh effects using filter diameters of `50` and `100` pixels.

The Bokeh implementation uses a normalized disc-shaped kernel and handles boundary pixels separately using cropped and renormalized kernels. 

## Repository Structure

```text
Digital-Image-Processing-2/
│
├── Sharpening/
│   └── Unsharp_masking.py
│
├── Template_Matching/
│   └── Template_matching.py
│
├── Canny/
│   └── Canny_edge_detection.py
│
├── Edge and Corner/
│   ├── Edge_Corner.py
│   ├── Edge_Corner_Ouput.py
│   └── Edge_Corner Output Images/
│
├── Bokeh/
│   ├── Bokeh_Filter.py
│   └── Bokeh Output Images/
│
├── data/
│   ├── sharpen/
│   ├── templateMatch/
│   ├── edge/
│   ├── corner/
│   └── bokeh/
│
├── report.pdf
├── requirements.txt
└── README.md
```

> The exact folder names can be adjusted to match the organization of the local repository.

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- Pillow
- SciPy

Install the required libraries using:

```bash
pip install -r requirements.txt
```

Alternatively:

```bash
pip install numpy matplotlib pillow scipy
```

## Running the Code

Each component is implemented as a separate Python script.

Navigate to the corresponding folder and run the desired script.

For example:

```bash
cd Sharpening
python Unsharp_masking.py
```

For Canny edge detection:

```bash
cd Canny
python Canny_edge_detection.py
```

For template matching:

```bash
cd Template_Matching
python Template_matching.py
```

For edge and corner detection:

```bash
cd "Edge and Corner"
python Edge_Corner.py
```

For Bokeh filtering:

```bash
cd Bokeh
python Bokeh_Filter.py
```

The input images for the different experiments are provided in the corresponding `data/` subdirectories. The assignment specification identifies separate input sets for sharpening, template matching, Canny edge detection, edge/corner detection, and Bokeh filtering.    

## Output

The implementations generate visualizations and intermediate results for each experiment, including:

- Sharpened images using different sharpening levels
- NCC response images for different template sizes and color channels
- Binary Canny edge maps and edge overlays
- Structure tensor eigenvalue images
- Harris-Stephens and Shi-Tomasi cornerness maps
- Non-maximum suppressed responses
- Binary corner and edge detections
- Corner and edge overlays
- Bokeh-filtered images with different filter diameters

The output visualizations correspond to the required displays specified in the assignment.    
