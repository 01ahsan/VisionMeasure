"""
Advanced Analysis Module
Additional DIP/ML techniques for comprehensive image analysis:
- FFT frequency domain analysis
- Color histogram analysis
- Hough circle & line detection
- ORB feature detection
- Blur/quality assessment
- Contour shape classification
"""

import cv2
import numpy as np
from collections import Counter


def fft_analysis(gray_image):
    """
    Frequency domain analysis using 2D FFT.
    Returns magnitude spectrum and phase spectrum.
    """
    f = np.fft.fft2(gray_image.astype(np.float32))
    fshift = np.fft.fftshift(f)

    magnitude = 20 * np.log(np.abs(fshift) + 1)
    magnitude = np.uint8(magnitude / magnitude.max() * 255)

    phase = np.angle(fshift)
    phase_normalized = np.uint8((phase - phase.min()) / (phase.max() - phase.min() + 1e-8) * 255)

    # Compute dominant frequency (energy distribution)
    h, w = gray_image.shape
    cy, cx = h // 2, w // 2
    mag_float = np.abs(fshift)
    total_energy = np.sum(mag_float)

    # Low frequency ratio (center 10%)
    r = min(h, w) // 10
    y, x = np.ogrid[:h, :w]
    center_mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
    low_freq_energy = np.sum(mag_float[center_mask])
    low_freq_ratio = low_freq_energy / (total_energy + 1e-8)

    return {
        "magnitude": magnitude,
        "phase": phase_normalized,
        "low_freq_ratio": low_freq_ratio,
        "is_blurry": low_freq_ratio > 0.85,
    }


def color_histogram_analysis(image):
    """
    Compute and analyze color histograms in BGR and HSV.
    Returns histogram images and statistics.
    """
    # BGR histograms
    colors = ("b", "g", "r")
    color_names = ("Blue", "Green", "Red")
    histograms = {}

    for i, (col, name) in enumerate(zip(colors, color_names)):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        histograms[name] = hist.flatten()

    # HSV analysis
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256]).flatten()

    # Dominant hue
    dominant_hue = np.argmax(h_hist)
    hue_to_color = {
        (0, 10): "Red", (10, 25): "Orange", (25, 35): "Yellow",
        (35, 85): "Green", (85, 130): "Blue", (130, 170): "Purple",
        (170, 180): "Red",
    }
    dominant_color = "Unknown"
    for (low, high), color in hue_to_color.items():
        if low <= dominant_hue < high:
            dominant_color = color
            break

    # Saturation analysis
    mean_saturation = np.mean(hsv[:, :, 1])
    mean_value = np.mean(hsv[:, :, 2])

    return {
        "bgr_histograms": histograms,
        "h_hist": h_hist,
        "s_hist": s_hist,
        "v_hist": v_hist,
        "dominant_hue": dominant_hue,
        "dominant_color": dominant_color,
        "mean_saturation": mean_saturation,
        "mean_brightness": mean_value,
    }


def hough_circle_detection(gray_image, dp=1.2, min_dist=50, param1=100, param2=40, min_radius=10, max_radius=0):
    """
    Detect circles using Hough Circle Transform.
    """
    blurred = cv2.medianBlur(gray_image, 5)

    if max_radius == 0:
        max_radius = min(gray_image.shape) // 3

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist,
        param1=param1, param2=param2,
        minRadius=min_radius, maxRadius=max_radius,
    )

    results = []
    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        for c in circles:
            results.append({
                "center": (int(c[0]), int(c[1])),
                "radius": int(c[2]),
                "diameter_px": int(c[2]) * 2,
                "area_px": np.pi * int(c[2]) ** 2,
            })

    return results


def hough_line_detection(gray_image, threshold=80, min_length=50, max_gap=10):
    """
    Detect lines using Probabilistic Hough Line Transform.
    """
    edges = cv2.Canny(gray_image, 50, 150)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=threshold, minLineLength=min_length, maxLineGap=max_gap,
    )

    results = []
    if lines is not None:
        for line in lines:
            try:
                coords = line[0] if len(line.shape) > 1 else line
                x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                results.append({
                    "start": (x1, y1),
                    "end": (x2, y2),
                    "length_px": length,
                    "angle_deg": angle,
                })
            except (IndexError, ValueError, TypeError):
                continue

    return results


def orb_feature_detection(gray_image, max_features=500):
    """
    Detect ORB keypoints and descriptors.
    Returns keypoints, descriptors, and visualization.
    """
    orb = cv2.ORB_create(nfeatures=max_features)
    keypoints, descriptors = orb.detectAndCompute(gray_image, None)

    # Classify keypoint distribution
    h, w = gray_image.shape
    quadrants = {"TL": 0, "TR": 0, "BL": 0, "BR": 0}
    for kp in keypoints:
        x, y = kp.pt
        q = ("T" if y < h / 2 else "B") + ("L" if x < w / 2 else "R")
        quadrants[q] += 1

    return {
        "keypoints": keypoints,
        "descriptors": descriptors,
        "count": len(keypoints),
        "distribution": quadrants,
    }


def blur_detection(gray_image):
    """
    Assess image blur using Laplacian variance method.
    Lower variance = more blur.
    """
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    variance = laplacian.var()
    mean_abs = np.mean(np.abs(laplacian))

    # Thresholds (heuristic)
    if variance < 50:
        quality = "Very Blurry"
        score = 1
    elif variance < 200:
        quality = "Slightly Blurry"
        score = 2
    elif variance < 500:
        quality = "Acceptable"
        score = 3
    elif variance < 1500:
        quality = "Good"
        score = 4
    else:
        quality = "Sharp"
        score = 5

    return {
        "laplacian_variance": variance,
        "mean_gradient": mean_abs,
        "quality": quality,
        "score": score,
    }


def classify_contour_shape(contour):
    """
    Classify a contour's shape using polygon approximation and Hu moments.
    """
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return "Unknown", 0

    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    vertices = len(approx)
    area = cv2.contourArea(contour)
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

    # Hu moments for shape matching
    moments = cv2.moments(contour)
    hu = cv2.HuMoments(moments).flatten()

    if vertices == 3:
        shape = "Triangle"
    elif vertices == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect = w / float(h) if h > 0 else 0
        if 0.85 <= aspect <= 1.15:
            shape = "Square"
        else:
            shape = "Rectangle"
    elif vertices == 5:
        shape = "Pentagon"
    elif vertices == 6:
        shape = "Hexagon"
    elif circularity > 0.80:
        shape = "Circle"
    elif circularity > 0.60:
        shape = "Ellipse"
    else:
        shape = f"Polygon ({vertices} vertices)"

    return shape, circularity


def noise_estimation(gray_image):
    """
    Estimate image noise level using median absolute deviation.
    """
    # Sigma estimation using MAD of Laplacian
    h, w = gray_image.shape
    # Use a high-pass filter
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])
    filtered = cv2.filter2D(gray_image.astype(np.float64), -1, kernel)
    sigma = np.median(np.abs(filtered)) * 1.4826 / np.sqrt(6)

    if sigma < 3:
        level = "Very Low"
    elif sigma < 8:
        level = "Low"
    elif sigma < 15:
        level = "Moderate"
    elif sigma < 30:
        level = "High"
    else:
        level = "Very High"

    return {"sigma": sigma, "level": level}


def draw_hough_circles(image, circles):
    """Draw detected circles on image."""
    result = image.copy()
    for c in circles:
        cv2.circle(result, c["center"], c["radius"], (0, 255, 0), 2)
        cv2.circle(result, c["center"], 3, (0, 0, 255), -1)
        label = f"r={c['radius']}px"
        cv2.putText(result, label, (c["center"][0] + 5, c["center"][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    return result


def draw_hough_lines(image, lines):
    """Draw detected lines on image."""
    result = image.copy()
    for l in lines:
        cv2.line(result, l["start"], l["end"], (0, 255, 0), 2)
    return result


def draw_orb_keypoints(image, keypoints):
    """Draw ORB keypoints on image."""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    result = cv2.drawKeypoints(
        image, keypoints, None, color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    return result
