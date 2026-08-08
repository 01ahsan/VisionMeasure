"""
Wound & Anomaly Segmentation Module
Detects color anomalies (wounds, defects, lesions) within an object surface.

Approach:
1. Convert to multiple color spaces (LAB, HSV, YCrCb)
2. Isolate red/dark regions that differ from surrounding skin/surface
3. Combine multiple color channels for robust detection
4. Apply morphological cleanup

This is fundamentally different from object-vs-background segmentation.
Here we're finding anomalies WITHIN a surface.
"""

import cv2
import numpy as np


def segment_wound(image, sensitivity="medium"):
    """
    Detect wound/lesion regions using multi-channel color analysis.

    Wounds are typically characterized by:
    - Higher redness (high a* in LAB, high Cr in YCrCb)
    - Lower brightness than surrounding skin
    - Different saturation than normal skin

    Parameters
    ----------
    image : np.ndarray (BGR)
    sensitivity : str — 'low', 'medium', 'high'

    Returns
    -------
    dict with:
        'wound_mask'    : binary mask of wound region
        'skin_mask'     : binary mask of detected skin
        'overlay'       : visualization with wound highlighted
        'wound_area_px' : wound area in pixels
        'skin_area_px'  : total skin area in pixels
        'wound_ratio'   : wound_area / skin_area
        'channel_masks' : dict of individual channel contributions
    """
    h, w = image.shape[:2]

    # Convert to color spaces
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    l_ch, a_ch, b_ch = cv2.split(lab)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    y_ch, cr_ch, cb_ch = cv2.split(ycrcb)

    # Sensitivity thresholds
    sens_map = {
        "low": {"a_thresh": 145, "cr_thresh": 155, "sat_thresh": 80, "dark_thresh": 0.7},
        "medium": {"a_thresh": 135, "cr_thresh": 145, "sat_thresh": 60, "dark_thresh": 0.8},
        "high": {"a_thresh": 125, "cr_thresh": 135, "sat_thresh": 40, "dark_thresh": 0.85},
    }
    params = sens_map.get(sensitivity, sens_map["medium"])

    # ── Step 1: Detect skin region (to limit wound search) ──
    # Skin detection using YCrCb ranges (well-established method)
    skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
    # Also include broader range for various skin tones
    skin_mask2 = cv2.inRange(hsv, np.array([0, 20, 70]), np.array([35, 255, 255]))
    skin_combined = cv2.bitwise_or(skin_mask, skin_mask2)

    # Clean up skin mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    skin_combined = cv2.morphologyEx(skin_combined, cv2.MORPH_CLOSE, kernel, iterations=3)
    skin_combined = cv2.morphologyEx(skin_combined, cv2.MORPH_OPEN, kernel, iterations=1)

    # If skin detection fails (< 10% of image), use the whole image
    skin_area = np.sum(skin_combined > 0)
    if skin_area < 0.1 * h * w:
        skin_combined = np.ones((h, w), dtype=np.uint8) * 255

    # ── Step 2: Detect wound-like regions using multiple cues ──

    # Cue 1: High redness in LAB a* channel (a* > threshold = red)
    a_mean = np.mean(a_ch[skin_combined > 0]) if np.any(skin_combined > 0) else np.mean(a_ch)
    a_std = np.std(a_ch[skin_combined > 0]) if np.any(skin_combined > 0) else np.std(a_ch)
    # Wound pixels have significantly higher a* than surrounding skin
    a_threshold = max(a_mean + 0.8 * a_std, params["a_thresh"])
    redness_mask = (a_ch > a_threshold).astype(np.uint8) * 255

    # Cue 2: High Cr in YCrCb (red chrominance)
    cr_mean = np.mean(cr_ch[skin_combined > 0]) if np.any(skin_combined > 0) else np.mean(cr_ch)
    cr_std = np.std(cr_ch[skin_combined > 0]) if np.any(skin_combined > 0) else np.std(cr_ch)
    cr_threshold = max(cr_mean + 0.8 * cr_std, params["cr_thresh"])
    cr_mask = (cr_ch > cr_threshold).astype(np.uint8) * 255

    # Cue 3: Darker than surrounding skin (wounds are often darker)
    l_mean = np.mean(l_ch[skin_combined > 0]) if np.any(skin_combined > 0) else np.mean(l_ch)
    darkness_threshold = l_mean * params["dark_thresh"]
    dark_mask = (l_ch < darkness_threshold).astype(np.uint8) * 255

    # Cue 4: Higher saturation than normal skin
    s_mean = np.mean(s_ch[skin_combined > 0]) if np.any(skin_combined > 0) else np.mean(s_ch)
    s_std = np.std(s_ch[skin_combined > 0]) if np.any(skin_combined > 0) else np.std(s_ch)
    sat_mask = (s_ch > max(s_mean + 0.5 * s_std, params["sat_thresh"])).astype(np.uint8) * 255

    # ── Step 3: Combine cues ──
    # Wound = (red OR dark) AND within skin region
    # Use weighted combination: redness is strongest signal
    combined = np.zeros((h, w), dtype=np.float32)
    combined += redness_mask.astype(np.float32) * 0.35
    combined += cr_mask.astype(np.float32) * 0.30
    combined += dark_mask.astype(np.float32) * 0.20
    combined += sat_mask.astype(np.float32) * 0.15

    # Threshold the weighted combination
    wound_raw = (combined > 127).astype(np.uint8) * 255

    # Only within skin region
    wound_mask = cv2.bitwise_and(wound_raw, skin_combined)

    # ── Step 4: Morphological cleanup ──
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_med = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    # Remove noise
    wound_mask = cv2.morphologyEx(wound_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    # Fill holes
    wound_mask = cv2.morphologyEx(wound_mask, cv2.MORPH_CLOSE, kernel_med, iterations=2)

    # Remove very small regions (noise)
    contours, _ = cv2.findContours(wound_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_wound_area = h * w * 0.001  # at least 0.1% of image
    wound_mask_clean = np.zeros_like(wound_mask)
    for c in contours:
        if cv2.contourArea(c) > min_wound_area:
            cv2.drawContours(wound_mask_clean, [c], -1, 255, -1)

    wound_mask = wound_mask_clean

    # ── Step 5: Create overlay visualization ──
    overlay = image.copy()
    # Red highlight on wound
    wound_highlight = np.zeros_like(image)
    wound_highlight[:, :, 2] = 255  # Red channel
    overlay = np.where(
        wound_mask[:, :, np.newaxis] > 0,
        cv2.addWeighted(overlay, 0.5, wound_highlight, 0.5, 0),
        overlay
    )
    # Draw wound contour
    contours, _ = cv2.findContours(wound_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)

    # ── Stats ──
    wound_area = np.sum(wound_mask > 0)
    skin_area = np.sum(skin_combined > 0)
    wound_ratio = wound_area / (skin_area + 1e-8)

    return {
        "wound_mask": wound_mask,
        "skin_mask": skin_combined,
        "overlay": overlay,
        "wound_area_px": int(wound_area),
        "skin_area_px": int(skin_area),
        "wound_ratio": round(float(wound_ratio), 4),
        "wound_contours": contours,
        "channel_masks": {
            "Redness (LAB a*)": redness_mask,
            "Red Chroma (YCrCb Cr)": cr_mask,
            "Darkness (LAB L*)": dark_mask,
            "Saturation (HSV S)": sat_mask,
        },
    }


def segment_color_anomaly(image, target_color="red", sensitivity="medium"):
    """
    General color anomaly detection — for fruits (bruises), leaves (disease spots),
    machine parts (rust), etc.

    Parameters
    ----------
    image : np.ndarray (BGR)
    target_color : str — 'red', 'brown', 'yellow', 'dark', 'green'
    sensitivity : str — 'low', 'medium', 'high'

    Returns
    -------
    dict with 'mask', 'overlay', 'area_px', 'contours'
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    sens_factor = {"low": 0.7, "medium": 1.0, "high": 1.3}.get(sensitivity, 1.0)

    if target_color == "red":
        # Red wraps around in HSV
        mask1 = cv2.inRange(hsv, np.array([0, int(70*sens_factor), 50]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([170, int(70*sens_factor), 50]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(mask1, mask2)
    elif target_color == "brown":
        mask = cv2.inRange(hsv, np.array([10, int(50*sens_factor), 30]), np.array([30, 255, 180]))
    elif target_color == "yellow":
        mask = cv2.inRange(hsv, np.array([20, int(80*sens_factor), 80]), np.array([40, 255, 255]))
    elif target_color == "dark":
        l_ch = lab[:, :, 0]
        threshold = np.mean(l_ch) * (0.6 / sens_factor)
        mask = (l_ch < threshold).astype(np.uint8) * 255
    elif target_color == "green":
        mask = cv2.inRange(hsv, np.array([35, int(50*sens_factor), 50]), np.array([85, 255, 255]))
    else:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Overlay
    overlay = image.copy()
    overlay[mask > 0] = cv2.addWeighted(
        overlay[mask > 0], 0.5,
        np.full_like(overlay[mask > 0], [0, 0, 255]), 0.5, 0
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    return {
        "mask": mask,
        "overlay": overlay,
        "area_px": int(np.sum(mask > 0)),
        "contours": contours,
    }


def measure_wound_from_contours(contours, pixel_per_cm=None):
    """
    Measure wound dimensions from detected wound contours.
    If multiple contours, treats them as one wound region.
    """
    if not contours:
        return None

    # Merge all wound contours into one
    all_points = np.vstack(contours)

    # Overall bounding
    min_rect = cv2.minAreaRect(all_points)
    rect_w = max(min_rect[1])
    rect_h = min(min_rect[1])

    total_area = sum(cv2.contourArea(c) for c in contours)
    total_perimeter = sum(cv2.arcLength(c, True) for c in contours)

    # Centroid
    moments = cv2.moments(all_points)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
    else:
        cx, cy = int(min_rect[0][0]), int(min_rect[0][1])

    result = {
        "width_px": rect_w,
        "height_px": rect_h,
        "area_px": total_area,
        "perimeter_px": total_perimeter,
        "centroid": (cx, cy),
        "bounding_rect": min_rect,
        "n_regions": len(contours),
    }

    if pixel_per_cm and pixel_per_cm > 0:
        result["width_cm"] = rect_w / pixel_per_cm
        result["height_cm"] = rect_h / pixel_per_cm
        result["area_cm2"] = total_area / (pixel_per_cm ** 2)
        result["perimeter_cm"] = total_perimeter / pixel_per_cm

    return result
