"""
Preprocessing Module
Handles noise removal and image preparation.
Techniques: Gaussian blur, Median blur, Bilateral filter, CLAHE enhancement.
"""

import cv2
import numpy as np


def preprocess_image(image, method="gaussian", kernel_size=5, apply_clahe=True):
    """
    Apply noise removal and optional contrast enhancement.

    Parameters
    ----------
    image : np.ndarray
        Input BGR image.
    method : str
        Filtering method: 'gaussian', 'median', or 'bilateral'.
    kernel_size : int
        Kernel size for filtering (must be odd).
    apply_clahe : bool
        Whether to apply CLAHE contrast enhancement.

    Returns
    -------
    dict with keys:
        'gray'      : Grayscale version
        'denoised'  : After noise removal
        'enhanced'  : After CLAHE (if applied), else same as denoised
    """
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply noise removal
    if method == "gaussian":
        denoised = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    elif method == "median":
        denoised = cv2.medianBlur(gray, kernel_size)
    elif method == "bilateral":
        denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    else:
        denoised = gray.copy()

    # Apply CLAHE for contrast enhancement
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
    else:
        enhanced = denoised.copy()

    return {
        "gray": gray,
        "denoised": denoised,
        "enhanced": enhanced,
    }
