"""
Calibration Module — v2
=======================
Completely redesigned reference/calibration system:

1. ANY object can be a reference — user picks which detected object, enters its real size
2. Manual pixel-per-cm entry for users who already know their scale
3. Known-distance mode: user gives two pixel coordinates + real distance
4. Auto-detect reference by smallest/most-circular object (smart guess)
5. ArUco marker detection
6. Multiple reference objects for average calibration
7. Unit support: mm, cm, inches, m

No longer limited to coins or specific shapes.
"""

import cv2
import numpy as np

# ──────────────────────────────────────────────
# UNIT SYSTEM
# ──────────────────────────────────────────────
UNIT_MULTIPLIERS = {
    "mm": 0.1,       # 1mm = 0.1cm
    "cm": 1.0,
    "inches": 2.54,  # 1 inch = 2.54cm
    "m": 100.0,      # 1m = 100cm
}

UNIT_LABELS = {
    "mm": ("mm", "mm²"),
    "cm": ("cm", "cm²"),
    "inches": ("in", "in²"),
    "m": ("m", "m²"),
}

# Quick presets — just a convenience, not the only option
QUICK_PRESETS = {
    "— Select preset (optional) —": None,
    "Credit/Debit Card (85.6mm width)": {"size_cm": 8.56, "desc": "Standard bank card width"},
    "A4 Paper Short Side (210mm)": {"size_cm": 21.0, "desc": "Standard A4 width"},
    "US Letter Short Side (215.9mm)": {"size_cm": 21.59, "desc": "US Letter width"},
    "Standard Ruler (30cm)": {"size_cm": 30.0, "desc": "30cm ruler full length"},
    "US Quarter (24.26mm ∅)": {"size_cm": 2.426, "desc": "Quarter dollar diameter"},
    "Euro 1€ (23.25mm ∅)": {"size_cm": 2.325, "desc": "1 Euro coin diameter"},
    "BDT 5 Taka (25mm ∅)": {"size_cm": 2.5, "desc": "5 Taka coin diameter"},
    "Golf Ball (42.67mm ∅)": {"size_cm": 4.267, "desc": "Standard golf ball diameter"},
    "Tennis Ball (67mm ∅)": {"size_cm": 6.7, "desc": "Standard tennis ball diameter"},
    "Standard Pen Length (14cm)": {"size_cm": 14.0, "desc": "Typical ballpoint pen"},
    "USB-A Plug Width (12mm)": {"size_cm": 1.2, "desc": "USB Type-A connector width"},
    "SD Card Width (24mm)": {"size_cm": 2.4, "desc": "Standard SD card width"},
    "AA Battery Length (50.5mm)": {"size_cm": 5.05, "desc": "AA battery length"},
}


def calibrate_pixel_ratio(reference_size_px, reference_size_cm):
    """Calculate pixels per centimeter from a reference object."""
    if reference_size_cm <= 0 or reference_size_px <= 0:
        return None
    return reference_size_px / reference_size_cm


def calibrate_from_two_points(p1, p2, real_distance_cm):
    """
    Calibrate from two known points and their real-world distance.
    p1, p2: (x, y) pixel coordinates
    real_distance_cm: distance between them in cm
    """
    pixel_dist = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
    if pixel_dist <= 0 or real_distance_cm <= 0:
        return None
    return pixel_dist / real_distance_cm


def calibrate_from_object_selection(contour, known_dimension_cm, dimension_type="width"):
    """
    Calibrate using any detected contour that the user identifies.
    
    dimension_type: 'width', 'height', 'diameter', 'perimeter'
    """
    if contour is None or known_dimension_cm <= 0:
        return None

    min_rect = cv2.minAreaRect(contour)
    rect_w = max(min_rect[1])
    rect_h = min(min_rect[1])

    if dimension_type == "width":
        px_size = rect_w
    elif dimension_type == "height":
        px_size = rect_h
    elif dimension_type == "diameter":
        _, radius = cv2.minEnclosingCircle(contour)
        px_size = radius * 2
    elif dimension_type == "diagonal":
        px_size = np.sqrt(rect_w ** 2 + rect_h ** 2)
    elif dimension_type == "perimeter":
        px_size = cv2.arcLength(contour, True)
    else:
        px_size = rect_w

    return calibrate_pixel_ratio(px_size, known_dimension_cm)


def calibrate_from_multiple_references(references):
    """
    Average calibration from multiple reference objects.
    references: list of (pixel_size, real_size_cm) tuples
    """
    ratios = []
    for px, cm in references:
        r = calibrate_pixel_ratio(px, cm)
        if r:
            ratios.append(r)
    if not ratios:
        return None
    return float(np.median(ratios))  # Median is more robust than mean


def auto_pick_reference(contours, image_area):
    """
    Smart reference detection — works with ANY shape.
    Picks the object most likely to be a reference based on:
    - Small-to-medium size relative to image
    - Regular shape (high circularity or rectangularity)
    - Not at extreme edges

    Returns list of candidates sorted by confidence, not limited to circles.
    """
    if not contours:
        return []

    h_img = int(np.sqrt(image_area * 3 / 4))
    w_img = int(image_area / h_img) if h_img > 0 else 100

    candidates = []
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area < 200:
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)
        area_ratio = area / image_area

        # Any regular shape scores well
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        n_vertices = len(approx)

        # Regularity: circles, rectangles, or any shape with good fill ratio
        hull = cv2.convexHull(c)
        solidity = area / (cv2.contourArea(hull) + 1e-8)

        min_rect = cv2.minAreaRect(c)
        rect_area = min_rect[1][0] * min_rect[1][1] + 1e-8
        extent = area / rect_area

        # Shape type detection
        if circularity > 0.8:
            shape = "circular"
            shape_score = circularity
        elif n_vertices == 4 and extent > 0.8:
            shape = "rectangular"
            shape_score = extent
        else:
            shape = "other"
            shape_score = solidity * 0.7

        # Size scoring — references are typically 1-15% of image area
        if 0.003 < area_ratio < 0.15:
            size_score = 1.0
        elif 0.001 < area_ratio < 0.25:
            size_score = 0.6
        else:
            size_score = 0.2

        # Edge penalty
        x, y, bw, bh = cv2.boundingRect(c)
        edge_margin = min(x, y, w_img - x - bw, h_img - y - bh)
        edge_score = min(edge_margin / 20.0, 1.0)

        confidence = shape_score * 0.4 + size_score * 0.4 + edge_score * 0.2
        confidence = min(confidence, 1.0)

        # Get measurable size
        rect_w = max(min_rect[1])
        rect_h = min(min_rect[1])
        (cx, cy), radius = cv2.minEnclosingCircle(c)

        candidates.append({
            "index": i,
            "contour": c,
            "center": (int(min_rect[0][0]), int(min_rect[0][1])),
            "width_px": rect_w,
            "height_px": rect_h,
            "diameter_px": radius * 2,
            "size_px": rect_w,  # Default calibration dimension
            "area_px": area,
            "shape": shape,
            "shape_score": round(shape_score, 3),
            "confidence": round(confidence, 3),
            "circularity": round(circularity, 3),
            "solidity": round(solidity, 3),
        })

    candidates.sort(key=lambda x: x["confidence"], reverse=True)
    return candidates


# Keep the old function signature for backward compat
def detect_reference_object(binary_image, ref_type="circle"):
    """Backward-compatible wrapper."""
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = binary_image.shape[0] * binary_image.shape[1]
    candidates = auto_pick_reference(contours, image_area)
    if not candidates:
        return None
    best = candidates[0]
    best["all_candidates"] = candidates[:5]
    best["type"] = best["shape"]
    return best


def apply_calibration(pixel_value, pixel_per_cm):
    """Convert a pixel measurement to centimeters."""
    if pixel_per_cm is None or pixel_per_cm <= 0:
        return None
    return pixel_value / pixel_per_cm


def convert_units(value_cm, target_unit):
    """Convert from cm to target unit."""
    mult = UNIT_MULTIPLIERS.get(target_unit, 1.0)
    return value_cm / mult * mult  # identity, but explicit


def detect_aruco_markers(image):
    """Detect ArUco markers for automatic calibration."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dictionaries = [
        cv2.aruco.DICT_4X4_50,
        cv2.aruco.DICT_5X5_50,
        cv2.aruco.DICT_6X6_50,
    ]
    for dict_type in dictionaries:
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, rejected = detector.detectMarkers(gray)
        if ids is not None and len(ids) > 0:
            markers = []
            for i, corner in enumerate(corners):
                pts = corner[0]
                side_lengths = [np.linalg.norm(pts[j] - pts[(j + 1) % 4]) for j in range(4)]
                avg_side = np.mean(side_lengths)
                markers.append({
                    "id": int(ids[i][0]),
                    "corners": pts,
                    "size_px": avg_side,
                    "center": np.mean(pts, axis=0).astype(int),
                })
            return markers
    return None


def get_perspective_transform(image, src_points):
    """Apply perspective correction given 4 source points."""
    src_points = np.float32(src_points)
    width_top = np.linalg.norm(src_points[1] - src_points[0])
    width_bottom = np.linalg.norm(src_points[2] - src_points[3])
    width = int(max(width_top, width_bottom))
    height_left = np.linalg.norm(src_points[3] - src_points[0])
    height_right = np.linalg.norm(src_points[2] - src_points[1])
    height = int(max(height_left, height_right))
    dst_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    corrected = cv2.warpPerspective(image, matrix, (width, height))
    return corrected
