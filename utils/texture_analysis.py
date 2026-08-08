"""
Texture Analysis Module
Advanced DIP techniques:
- Local Binary Pattern (LBP)
- Gabor Filter Bank
- GLCM-based texture features (contrast, energy, homogeneity, correlation)
- Morphological Gradient
- Color Space Decomposition
"""

import cv2
import numpy as np


def local_binary_pattern(gray_image, radius=1, n_points=8):
    """
    Compute Local Binary Pattern (LBP) for texture classification.

    Parameters
    ----------
    gray_image : np.ndarray
    radius : int — radius of circular neighborhood
    n_points : int — number of sample points

    Returns
    -------
    dict with 'lbp_image', 'histogram', 'uniformity_ratio'
    """
    h, w = gray_image.shape
    lbp = np.zeros((h, w), dtype=np.uint8)

    for i in range(radius, h - radius):
        for j in range(radius, w - radius):
            center = gray_image[i, j]
            binary_string = 0
            for k in range(n_points):
                angle = 2 * np.pi * k / n_points
                y = i + int(round(radius * np.sin(angle)))
                x = j + int(round(radius * np.cos(angle)))
                y = min(max(y, 0), h - 1)
                x = min(max(x, 0), w - 1)
                if gray_image[y, x] >= center:
                    binary_string |= (1 << k)
            lbp[i, j] = binary_string

    # Histogram
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float32) / hist.sum()

    # Uniformity: ratio of uniform patterns (<=2 transitions)
    uniform_count = 0
    for val in range(256):
        bits = format(val, f"0{n_points}b")
        transitions = sum(1 for a, b in zip(bits, bits[1:] + bits[:1]) if a != b)
        if transitions <= 2:
            uniform_count += hist[val] if val < len(hist) else 0

    return {
        "lbp_image": lbp,
        "histogram": hist,
        "uniformity_ratio": float(uniform_count),
    }


def lbp_fast(gray_image, radius=1, n_points=8):
    """
    Fast LBP using skimage (C-optimized, Fix #6).
    """
    from skimage.feature import local_binary_pattern

    lbp = local_binary_pattern(gray_image, n_points, radius, method="uniform")
    lbp_uint8 = np.uint8((lbp / lbp.max()) * 255) if lbp.max() > 0 else np.zeros_like(gray_image)

    n_bins = n_points + 2  # uniform LBP has P+2 bins
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype(np.float32) / (hist.sum() + 1e-8)

    # Pad to 256 for display compatibility
    hist_256 = np.zeros(256, dtype=np.float32)
    hist_256[:len(hist)] = hist

    return {"lbp_image": lbp_uint8, "histogram": hist_256}


def gabor_filter_bank(gray_image, frequencies=None, orientations=None):
    """
    Apply a bank of Gabor filters for texture analysis.

    Parameters
    ----------
    gray_image : np.ndarray
    frequencies : list of floats — spatial frequencies
    orientations : list of floats — angles in radians

    Returns
    -------
    dict with:
        'responses'  : list of (freq, orient, filtered_image)
        'energy_map' : combined energy across all filters
        'feature_vector' : mean+std of each filter response (for classification)
    """
    if frequencies is None:
        frequencies = [0.05, 0.1, 0.2, 0.3]
    if orientations is None:
        orientations = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    ksize = 31
    sigma = 4.0
    gamma = 0.5
    psi = 0

    responses = []
    energy_maps = []
    features = []

    for freq in frequencies:
        for theta in orientations:
            kernel = cv2.getGaborKernel(
                (ksize, ksize), sigma, theta, 1.0 / freq, gamma, psi, ktype=cv2.CV_32F
            )
            filtered = cv2.filter2D(gray_image.astype(np.float32), cv2.CV_32F, kernel)
            responses.append((freq, theta, filtered))
            energy_maps.append(filtered ** 2)
            features.extend([np.mean(filtered), np.std(filtered)])

    # Combined energy
    energy = np.sqrt(sum(energy_maps))
    energy_normalized = ((energy - energy.min()) / (energy.max() - energy.min() + 1e-8) * 255).astype(np.uint8)

    return {
        "responses": responses,
        "energy_map": energy_normalized,
        "feature_vector": np.array(features),
        "n_filters": len(responses),
    }


def glcm_features(gray_image, distances=None, angles=None, levels=256):
    """
    Compute GLCM features using skimage (C-optimized, Fix #5).
    Features: contrast, dissimilarity, homogeneity, energy, correlation.
    """
    from skimage.feature import graycomatrix, graycoprops

    if distances is None:
        distances = [1, 3]
    if angles is None:
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    # Quantize to fewer levels
    n_levels = 64
    quantized = (gray_image / 256.0 * n_levels).astype(np.uint8)
    quantized = np.clip(quantized, 0, n_levels - 1)

    glcm = graycomatrix(quantized, distances=distances, angles=angles,
                         levels=n_levels, symmetric=True, normed=True)

    properties = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
    results_per_dir = []

    for d_idx, d in enumerate(distances):
        for a_idx, angle in enumerate(angles):
            row = {"distance": d, "angle_deg": round(np.degrees(angle))}
            for prop in properties:
                vals = graycoprops(glcm, prop)
                row[prop] = round(float(vals[d_idx, a_idx]), 6)
            results_per_dir.append(row)

    # Compute entropy manually from GLCM
    for row in results_per_dir:
        row["entropy"] = 0.0  # skimage doesn't have entropy prop

    # Average across all directions
    avg = {}
    for prop in properties:
        avg[prop] = round(float(np.mean([r[prop] for r in results_per_dir])), 6)
    avg["entropy"] = 0.0

    # Manual entropy from full GLCM
    glcm_sum = glcm.sum(axis=(2, 3))
    glcm_norm = glcm_sum / (glcm_sum.sum() + 1e-10)
    avg["entropy"] = round(float(-np.sum(glcm_norm * np.log2(glcm_norm + 1e-10))), 4)

    return {"per_direction": results_per_dir, "averaged": avg}


def morphological_gradient(gray_image, kernel_size=3):
    """
    Compute morphological gradient (dilation - erosion).
    Highlights object boundaries.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    gradient = cv2.morphologyEx(gray_image, cv2.MORPH_GRADIENT, kernel)
    return gradient


def color_space_decomposition(image):
    """
    Decompose image into multiple color spaces for visualization.
    Returns channels from RGB, HSV, LAB, YCrCb.
    """
    results = {}

    # RGB channels
    b, g, r = cv2.split(image)
    results["R"] = r
    results["G"] = g
    results["B"] = b

    # HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    results["H (Hue)"] = h
    results["S (Saturation)"] = s
    results["V (Value)"] = v

    # LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b_ch = cv2.split(lab)
    results["L (Lightness)"] = l
    results["A (Green-Red)"] = a
    results["B (Blue-Yellow)"] = b_ch

    # YCrCb
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    results["Y (Luma)"] = y
    results["Cr (Red diff)"] = cr
    results["Cb (Blue diff)"] = cb

    return results


def compute_image_moments(contour):
    """
    Compute Hu moments and central moments for shape invariant matching.
    """
    moments = cv2.moments(contour)
    hu = cv2.HuMoments(moments).flatten()

    # Log-transform Hu moments for better comparability
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    return {
        "raw_moments": {k: float(v) for k, v in moments.items()},
        "hu_moments": hu.tolist(),
        "hu_log": hu_log.tolist(),
    }
