"""
Edge Detection Module
Handles edge detection using various methods.
Techniques: Canny, Sobel, Laplacian.
"""

import cv2
import numpy as np


def detect_edges(gray_image, method="canny", canny_low=50, canny_high=150):
    """
    Apply edge detection.

    Parameters
    ----------
    gray_image : np.ndarray
        Grayscale or binary input image.
    method : str
        'canny', 'sobel', or 'laplacian'.
    canny_low : int
        Lower threshold for Canny.
    canny_high : int
        Upper threshold for Canny.

    Returns
    -------
    np.ndarray
        Edge map (binary image).
    """
    if method == "canny":
        edges = cv2.Canny(gray_image, canny_low, canny_high)

    elif method == "sobel":
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        edges = np.uint8(magnitude / magnitude.max() * 255)
        _, edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)

    elif method == "laplacian":
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        edges = np.uint8(np.abs(laplacian) / np.abs(laplacian).max() * 255)
        _, edges = cv2.threshold(edges, 30, 255, cv2.THRESH_BINARY)

    else:
        edges = cv2.Canny(gray_image, canny_low, canny_high)

    return edges
