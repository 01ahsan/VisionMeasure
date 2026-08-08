"""
Contour Analysis Module
Handles contour extraction, measurement, and object property computation.
Techniques: Contour detection, bounding box, min-area rectangle, connected components.
"""

import cv2
import numpy as np


def find_and_measure_contours(
    binary_image, original_image, pixel_per_cm=None, min_area_ratio=0.001, max_objects=20
):
    """
    Find contours and compute measurements for each detected object.

    Parameters
    ----------
    binary_image : np.ndarray
        Binary (segmented) image.
    original_image : np.ndarray
        Original BGR image for drawing.
    pixel_per_cm : float or None
        Pixels per centimeter. If None, measurements are in pixels.
    min_area_ratio : float
        Minimum contour area as fraction of image area to filter noise.
    max_objects : int
        Maximum number of objects to measure.

    Returns
    -------
    list[dict]
        Each dict contains:
            'contour'       : np.ndarray of contour points
            'area_px'       : Area in pixels
            'perimeter_px'  : Perimeter in pixels
            'width_px'      : Width of min-area bounding rect in pixels
            'height_px'     : Height of min-area bounding rect in pixels
            'bbox'          : (x, y, w, h) upright bounding box
            'min_rect'      : ((cx,cy), (w,h), angle) rotated rectangle
            'centroid'      : (cx, cy) center point
            'circularity'   : How circular the object is (1.0 = perfect circle)
            'aspect_ratio'  : width / height
            'area_cm2'      : Area in cm² (if calibrated)
            'perimeter_cm'  : Perimeter in cm (if calibrated)
            'width_cm'      : Width in cm (if calibrated)
            'height_cm'     : Height in cm (if calibrated)
    """
    image_area = binary_image.shape[0] * binary_image.shape[1]
    min_area = image_area * min_area_ratio

    # Find contours
    contours, hierarchy = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter by minimum area and sort by area (largest first)
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)
    valid_contours = valid_contours[:max_objects]

    measurements = []

    for contour in valid_contours:
        # Basic measurements in pixels
        area_px = cv2.contourArea(contour)
        perimeter_px = cv2.arcLength(contour, True)

        # Upright bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # Minimum area rotated rectangle
        min_rect = cv2.minAreaRect(contour)
        rect_w = min_rect[1][0]
        rect_h = min_rect[1][1]

        # Ensure width >= height for consistency
        width_px = max(rect_w, rect_h)
        height_px = min(rect_w, rect_h)

        # Centroid
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2

        # Circularity
        if perimeter_px > 0:
            circularity = 4 * np.pi * area_px / (perimeter_px**2)
        else:
            circularity = 0

        # Aspect ratio
        aspect_ratio = width_px / height_px if height_px > 0 else 0

        measurement = {
            "contour": contour,
            "area_px": area_px,
            "perimeter_px": perimeter_px,
            "width_px": width_px,
            "height_px": height_px,
            "bbox": (x, y, w, h),
            "min_rect": min_rect,
            "centroid": (cx, cy),
            "circularity": circularity,
            "aspect_ratio": aspect_ratio,
        }

        # Convert to real-world units if calibrated
        if pixel_per_cm is not None and pixel_per_cm > 0:
            measurement["width_cm"] = width_px / pixel_per_cm
            measurement["height_cm"] = height_px / pixel_per_cm
            measurement["area_cm2"] = area_px / (pixel_per_cm**2)
            measurement["perimeter_cm"] = perimeter_px / pixel_per_cm
        else:
            measurement["width_cm"] = None
            measurement["height_cm"] = None
            measurement["area_cm2"] = None
            measurement["perimeter_cm"] = None

        measurements.append(measurement)

    return measurements


def detect_reference_object(binary_image, ref_type="circle"):
    """
    Auto-detect reference object with confidence scoring.
    Returns the best candidate AND a list of all candidates for user confirmation.

    Returns
    -------
    dict or None
        Best reference with 'contour', 'center', 'size_px', 'confidence',
        'type', 'all_candidates' (list of alternatives).
    """
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    image_area = binary_image.shape[0] * binary_image.shape[1]
    candidates = []

    if ref_type == "circle":
        for c in contours:
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0 or area < 300:
                continue

            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < 0.55:
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(c)
            enclosing_area = np.pi * radius ** 2
            fill_ratio = area / (enclosing_area + 1e-8)
            area_ratio = area / image_area

            # Confidence: circularity matters most, then fill ratio,
            # penalize very large objects (probably not a reference)
            confidence = circularity * 0.5 + fill_ratio * 0.3
            if area_ratio > 0.3:
                confidence *= 0.3  # too large to be a reference
            elif area_ratio > 0.15:
                confidence *= 0.6
            # Small-to-medium objects are more likely references
            if 0.005 < area_ratio < 0.08:
                confidence *= 1.2

            confidence = min(confidence, 1.0)

            candidates.append({
                "contour": c,
                "center": (int(cx), int(cy)),
                "size_px": radius * 2,
                "type": "circle",
                "circularity": circularity,
                "fill_ratio": fill_ratio,
                "area_ratio": area_ratio,
                "confidence": round(confidence, 3),
            })

    elif ref_type == "rectangle":
        for c in contours:
            area = cv2.contourArea(c)
            if area < 300:
                continue
            perimeter = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)

            if len(approx) < 4 or len(approx) > 6:
                continue

            min_rect = cv2.minAreaRect(c)
            rect_area = min_rect[1][0] * min_rect[1][1]
            fill_ratio = area / (rect_area + 1e-8)
            area_ratio = area / image_area

            # Rectangularity score
            rectangularity = fill_ratio if len(approx) == 4 else fill_ratio * 0.8

            confidence = rectangularity * 0.6 + (1.0 if len(approx) == 4 else 0.5) * 0.4
            if area_ratio > 0.3:
                confidence *= 0.3
            elif area_ratio > 0.15:
                confidence *= 0.6
            if 0.005 < area_ratio < 0.1:
                confidence *= 1.15

            confidence = min(confidence, 1.0)
            w = max(min_rect[1])

            candidates.append({
                "contour": c,
                "center": (int(min_rect[0][0]), int(min_rect[0][1])),
                "size_px": w,
                "type": "rectangle",
                "fill_ratio": fill_ratio,
                "area_ratio": area_ratio,
                "confidence": round(confidence, 3),
            })

    if not candidates:
        return None

    # Sort by confidence
    candidates.sort(key=lambda x: x["confidence"], reverse=True)
    best = candidates[0]
    best["all_candidates"] = candidates[:5]  # top 5 for user review

    return best
