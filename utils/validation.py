"""
Validation & Accuracy Module
- Generates synthetic test images with known ground-truth dimensions
- Computes measurement error statistics
- Detects perspective distortion from reference object shape
- Provides measurement uncertainty estimates
"""

import cv2
import numpy as np


def generate_test_image(objects, ref_diameter_px=100, image_size=(800, 1200), bg_color=240, noise_std=5):
    """
    Generate a synthetic test image with known object dimensions.

    Parameters
    ----------
    objects : list of dict
        Each dict: {'type': 'rect'|'circle'|'ellipse', 'x': int, 'y': int,
                     'width': int, 'height': int, 'color': int}
    ref_diameter_px : int
        Diameter of the reference circle in pixels.
    image_size : tuple (height, width)
    bg_color : int (0-255)
    noise_std : float

    Returns
    -------
    image : np.ndarray (BGR)
    ground_truth : list of dict with true pixel dimensions
    ref_info : dict with reference location and size
    """
    h, w = image_size
    img = np.full((h, w, 3), bg_color, dtype=np.uint8)

    # Add noise
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    ground_truth = []

    # Draw reference circle
    ref_cx, ref_cy = 100, h - 100
    ref_radius = ref_diameter_px // 2
    cv2.circle(img, (ref_cx, ref_cy), ref_radius, (40, 40, 40), -1)
    ref_info = {"center": (ref_cx, ref_cy), "diameter_px": ref_diameter_px}

    # Draw objects
    for obj in objects:
        color = (obj.get("color", 50),) * 3

        if obj["type"] == "rect":
            x, y = obj["x"], obj["y"]
            ow, oh = obj["width"], obj["height"]
            cv2.rectangle(img, (x, y), (x + ow, y + oh), color, -1)
            ground_truth.append({
                "type": "rect",
                "width_px": ow,
                "height_px": oh,
                "area_px": ow * oh,
                "perimeter_px": 2 * (ow + oh),
            })

        elif obj["type"] == "circle":
            cx, cy = obj["x"], obj["y"]
            r = obj["width"] // 2
            cv2.circle(img, (cx, cy), r, color, -1)
            ground_truth.append({
                "type": "circle",
                "width_px": obj["width"],
                "height_px": obj["width"],
                "area_px": np.pi * r ** 2,
                "perimeter_px": 2 * np.pi * r,
            })

        elif obj["type"] == "ellipse":
            cx, cy = obj["x"], obj["y"]
            ax1, ax2 = obj["width"] // 2, obj["height"] // 2
            cv2.ellipse(img, (cx, cy), (ax1, ax2), 0, 0, 360, color, -1)
            ground_truth.append({
                "type": "ellipse",
                "width_px": obj["width"],
                "height_px": obj["height"],
                "area_px": np.pi * ax1 * ax2,
                "perimeter_px": np.pi * (3 * (ax1 + ax2) - np.sqrt((3 * ax1 + ax2) * (ax1 + 3 * ax2))),
            })

    return img, ground_truth, ref_info


def run_accuracy_test(process_fn, calibrate_fn, n_tests=10):
    """
    Run automated accuracy tests with synthetic images.

    Parameters
    ----------
    process_fn : callable(image) -> list of measurements
    calibrate_fn : callable(image, ref_size_cm) -> pixel_per_cm
    n_tests : int

    Returns
    -------
    dict with:
        'results'       : list of per-image results
        'mean_error_pct': average percentage error across all measurements
        'width_error'   : mean absolute width error
        'height_error'  : mean absolute height error
        'area_error'    : mean absolute area error
        'summary'       : human-readable summary
    """
    np.random.seed(42)
    all_errors = {"width": [], "height": [], "area": []}
    results = []

    for i in range(n_tests):
        # Random objects
        n_obj = np.random.randint(1, 5)
        objects = []
        for j in range(n_obj):
            obj_type = np.random.choice(["rect", "circle", "ellipse"])
            ow = np.random.randint(60, 200)
            oh = ow if obj_type == "circle" else np.random.randint(40, 200)
            ox = np.random.randint(200, 900)
            oy = np.random.randint(50, 500)
            objects.append({"type": obj_type, "x": ox, "y": oy, "width": ow, "height": oh, "color": np.random.randint(20, 80)})

        ref_diam = np.random.randint(60, 120)
        noise = np.random.uniform(0, 8)

        img, gt, ref_info = generate_test_image(objects, ref_diameter_px=ref_diam, noise_std=noise)

        try:
            measured = process_fn(img)

            # Match measured to ground truth by proximity/size
            matched = _match_measurements(gt, measured)

            test_result = {"test_id": i, "n_objects_gt": len(gt), "n_detected": len(measured), "matches": []}

            for gt_obj, meas_obj in matched:
                if meas_obj is None:
                    test_result["matches"].append({"status": "missed", "gt": gt_obj})
                    continue

                w_err = abs(meas_obj["width_px"] - gt_obj["width_px"]) / gt_obj["width_px"] * 100
                h_err = abs(meas_obj["height_px"] - gt_obj["height_px"]) / gt_obj["height_px"] * 100
                a_err = abs(meas_obj["area_px"] - gt_obj["area_px"]) / gt_obj["area_px"] * 100

                all_errors["width"].append(w_err)
                all_errors["height"].append(h_err)
                all_errors["area"].append(a_err)

                test_result["matches"].append({
                    "status": "matched",
                    "width_error_pct": round(w_err, 2),
                    "height_error_pct": round(h_err, 2),
                    "area_error_pct": round(a_err, 2),
                })

            results.append(test_result)

        except Exception as e:
            results.append({"test_id": i, "error": str(e)[:100]})

    # Aggregate
    summary = {}
    for key in ["width", "height", "area"]:
        if all_errors[key]:
            arr = np.array(all_errors[key])
            summary[f"{key}_mean_error_pct"] = round(float(np.mean(arr)), 2)
            summary[f"{key}_median_error_pct"] = round(float(np.median(arr)), 2)
            summary[f"{key}_std_error_pct"] = round(float(np.std(arr)), 2)
            summary[f"{key}_max_error_pct"] = round(float(np.max(arr)), 2)

    overall_errors = all_errors["width"] + all_errors["height"] + all_errors["area"]
    mean_error = float(np.mean(overall_errors)) if overall_errors else 100.0

    return {
        "results": results,
        "mean_error_pct": round(mean_error, 2),
        "error_stats": summary,
        "n_tests": n_tests,
        "n_matched": len(all_errors["width"]),
        "summary": f"Tested {n_tests} images. Mean measurement error: ±{mean_error:.1f}%",
    }


def _match_measurements(ground_truth, measured):
    """Match measured objects to ground truth by area similarity."""
    matches = []
    used = set()

    for gt in ground_truth:
        best_idx = None
        best_diff = float("inf")
        for i, m in enumerate(measured):
            if i in used:
                continue
            diff = abs(m["area_px"] - gt["area_px"]) / (gt["area_px"] + 1e-8)
            if diff < best_diff and diff < 0.5:  # within 50% area match
                best_diff = diff
                best_idx = i

        if best_idx is not None:
            used.add(best_idx)
            matches.append((gt, measured[best_idx]))
        else:
            matches.append((gt, None))

    return matches


def estimate_measurement_uncertainty(contour, pixel_per_cm=None):
    """
    Estimate measurement uncertainty based on contour smoothness
    and pixel resolution.

    Returns uncertainty in the same unit as the measurement (px or cm).
    """
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)

    if perimeter == 0:
        return {"width": 0, "height": 0, "area": 0, "confidence_pct": 0}

    # Contour smoothness: compare actual perimeter with convex hull perimeter
    hull = cv2.convexHull(contour)
    hull_perimeter = cv2.arcLength(hull, True)
    smoothness = hull_perimeter / (perimeter + 1e-8)  # 1.0 = perfectly smooth

    # Pixel resolution uncertainty: ±1 pixel on each boundary edge
    min_rect = cv2.minAreaRect(contour)
    w = max(min_rect[1])
    h_val = min(min_rect[1])

    # Base uncertainty is ±2 pixels (1 pixel each side)
    px_uncertainty = 2.0

    # Scale by contour irregularity
    irregularity_factor = max(1.0, 1.0 / (smoothness + 1e-8))
    total_px_uncertainty = px_uncertainty * irregularity_factor

    # Convert to real units if calibrated
    if pixel_per_cm and pixel_per_cm > 0:
        cm_uncertainty = total_px_uncertainty / pixel_per_cm
        area_uncertainty = (2 * w * total_px_uncertainty + 2 * h_val * total_px_uncertainty) / (pixel_per_cm ** 2)
        confidence = min(smoothness * 100, 99)
        return {
            "width_cm": round(cm_uncertainty, 4),
            "height_cm": round(cm_uncertainty, 4),
            "area_cm2": round(area_uncertainty, 4),
            "confidence_pct": round(confidence, 1),
        }
    else:
        confidence = min(smoothness * 100, 99)
        return {
            "width_px": round(total_px_uncertainty, 2),
            "height_px": round(total_px_uncertainty, 2),
            "area_px": round(2 * w * total_px_uncertainty + 2 * h_val * total_px_uncertainty, 2),
            "confidence_pct": round(confidence, 1),
        }


def detect_perspective_distortion(ref_contour, ref_type="circle"):
    """
    Detect if the image has perspective distortion by analyzing
    the reference object's shape.
    A circle viewed at an angle becomes an ellipse.
    Returns distortion info and a warning level.
    """
    if ref_contour is None or len(ref_contour) < 5:
        return {"distorted": False, "warning": "", "angle_estimate_deg": 0}

    if ref_type == "circle":
        # Fit ellipse to the reference contour
        try:
            ellipse = cv2.fitEllipse(ref_contour)
            (cx, cy), (major, minor), angle = ellipse

            if minor == 0:
                return {"distorted": False, "warning": "", "angle_estimate_deg": 0}

            # Eccentricity: 1.0 = perfect circle
            aspect = minor / major
            eccentricity = np.sqrt(1 - aspect ** 2) if aspect < 1 else 0

            if aspect > 0.92:
                return {
                    "distorted": False,
                    "warning": "",
                    "angle_estimate_deg": 0,
                    "aspect_ratio": round(aspect, 3),
                }
            elif aspect > 0.80:
                est_angle = np.degrees(np.arccos(aspect))
                return {
                    "distorted": True,
                    "warning": f"⚠️ Mild perspective distortion detected (tilt ~{est_angle:.0f}°). Measurements may have ±{(1-aspect)*100:.0f}% error.",
                    "angle_estimate_deg": round(est_angle, 1),
                    "aspect_ratio": round(aspect, 3),
                    "severity": "mild",
                }
            else:
                est_angle = np.degrees(np.arccos(aspect))
                return {
                    "distorted": True,
                    "warning": f"🚨 Significant perspective distortion (tilt ~{est_angle:.0f}°). Retake photo from directly above for accurate measurements.",
                    "angle_estimate_deg": round(est_angle, 1),
                    "aspect_ratio": round(aspect, 3),
                    "severity": "severe",
                }
        except cv2.error:
            return {"distorted": False, "warning": "", "angle_estimate_deg": 0}

    return {"distorted": False, "warning": "", "angle_estimate_deg": 0}
