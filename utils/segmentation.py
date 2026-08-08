"""
Segmentation Module — Extended
Handles thresholding, morphology, GrabCut, K-Means color clustering,
and Watershed segmentation.
"""

import cv2
import numpy as np


def segment_image(
    gray_image,
    threshold_method="otsu",
    adaptive_block_size=11,
    adaptive_c=2,
    manual_threshold=127,
    morph_operation="close",
    morph_kernel_size=5,
    morph_iterations=2,
    invert=False,
):
    """Basic thresholding + morphology pipeline."""
    if adaptive_block_size % 2 == 0:
        adaptive_block_size += 1
    if adaptive_block_size < 3:
        adaptive_block_size = 3

    if threshold_method == "otsu":
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif threshold_method == "adaptive_mean":
        binary = cv2.adaptiveThreshold(
            gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, adaptive_block_size, adaptive_c,
        )
    elif threshold_method == "adaptive_gaussian":
        binary = cv2.adaptiveThreshold(
            gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, adaptive_block_size, adaptive_c,
        )
    elif threshold_method == "manual":
        _, binary = cv2.threshold(gray_image, manual_threshold, 255, cv2.THRESH_BINARY)
    else:
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if invert:
        binary = cv2.bitwise_not(binary)

    thresholded = binary.copy()

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
    )
    if morph_operation == "open":
        morphed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)
    elif morph_operation == "close":
        morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)
    elif morph_operation == "dilate":
        morphed = cv2.dilate(binary, kernel, iterations=morph_iterations)
    elif morph_operation == "erode":
        morphed = cv2.erode(binary, kernel, iterations=morph_iterations)
    elif morph_operation == "open_close":
        morphed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)
        morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)
    else:
        morphed = binary.copy()

    return {"thresholded": thresholded, "morphed": morphed}


def grabcut_segment(image, iterations=5, margin_ratio=0.02):
    """
    GrabCut foreground extraction — works on complex backgrounds.
    Automatically initializes a rectangle slightly inset from image borders.
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), np.uint8)

    mx = max(int(w * margin_ratio), 5)
    my = max(int(h * margin_ratio), 5)
    rect = (mx, my, w - 2 * mx, h - 2 * my)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return np.ones((h, w), np.uint8) * 255

    # Foreground = definite fg + probable fg
    fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return fg_mask


def kmeans_segment(image, k=3, target_cluster="darkest"):
    """
    K-Means color clustering segmentation.
    Clusters the image into k color groups and returns a binary mask
    for the target cluster(s).

    Parameters
    ----------
    image : np.ndarray (BGR)
    k : int — number of clusters
    target_cluster : str — 'darkest', 'brightest', or 'largest'

    Returns
    -------
    binary mask, colored_clusters visualization
    """
    # Convert to LAB for better perceptual clustering
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    h, w = lab.shape[:2]
    pixels = lab.reshape(-1, 3).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)

    labels = labels.flatten()
    centers = centers.astype(np.uint8)

    # Build colored visualization
    colored = centers[labels].reshape(h, w, 3)
    colored_bgr = cv2.cvtColor(colored, cv2.COLOR_LAB2BGR)

    # Pick target cluster
    if target_cluster == "darkest":
        # L channel is index 0 in LAB
        target_idx = np.argmin(centers[:, 0])
    elif target_cluster == "brightest":
        target_idx = np.argmax(centers[:, 0])
    elif target_cluster == "largest":
        # Find the cluster that is NOT the largest (largest = background)
        counts = np.bincount(labels, minlength=k)
        # Exclude the largest cluster (background), pick the next largest
        sorted_idx = np.argsort(counts)[::-1]
        if len(sorted_idx) > 1:
            target_idx = sorted_idx[1]  # second largest = likely foreground
        else:
            target_idx = sorted_idx[0]
    else:
        target_idx = 0

    mask = np.where(labels == target_idx, 255, 0).astype(np.uint8).reshape(h, w)

    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return mask, colored_bgr


def watershed_segment(image):
    """
    Marker-based Watershed segmentation.
    Uses distance transform to find sure foreground markers.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Noise removal
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background (dilated)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Sure foreground (distance transform)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    # Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Markers
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Watershed
    markers = cv2.watershed(image, markers)

    # Create mask: markers > 1 are objects (1 is background)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    mask[markers > 1] = 255

    # Boundary visualization
    boundary_viz = image.copy()
    boundary_viz[markers == -1] = [0, 0, 255]

    return mask, dist_transform, boundary_viz


def hsv_segment(image, lower_h=0, upper_h=180, lower_s=0, upper_s=255, lower_v=0, upper_v=255):
    """
    HSV color-space segmentation.
    Useful for objects with distinct colors (leaves, fruits, wounds).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([lower_h, lower_s, lower_v])
    upper = np.array([upper_h, upper_s, upper_v])
    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return mask
