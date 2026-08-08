"""
Smart Features Module — Unique Capabilities
=============================================
Features that make VisionMeasure genuinely useful and stand out:

1. Size Grading & Sorting — auto-classify objects into size categories
2. Dimensional Compliance Checker — flag objects outside tolerance
3. Object Comparison — side-by-side analysis of any two objects
4. Click-to-Isolate — flood fill based single object isolation
5. Size Distribution Analysis — statistical analysis of object populations
6. Measurement Report Generator — structured export-ready summary
7. Object Cropper — extract individual objects as separate images
8. Uniformity Score — how consistent are the objects in size/shape
9. Nearest-Neighbor Distance — spacing analysis between objects
10. Convex Hull Analysis — convexity defects, solidity, etc.
"""

import cv2
import numpy as np
from scipy import ndimage


# ──────────────────────────────────────────────
# 1. SIZE GRADING & SORTING
# ──────────────────────────────────────────────
def grade_objects_by_size(measurements, n_grades=4, custom_thresholds=None, grade_by="area"):
    """
    Automatically sort detected objects into size grades.

    Default grades: XS, S, M, L, XL (or custom labels).
    Uses percentile-based thresholds or user-defined thresholds.

    Parameters
    ----------
    measurements : list of dict
    n_grades : int (3, 4, or 5)
    custom_thresholds : list of float or None
        If provided, must have n_grades-1 threshold values in the measurement unit.
    grade_by : str — 'area', 'width', 'height', 'perimeter'

    Returns
    -------
    dict with graded_objects, grade_summary, grade_distribution
    """
    if not measurements:
        return {"graded_objects": [], "grade_summary": {}, "grade_distribution": {}}

    # Get the dimension to grade by
    cm_key = f"{grade_by}_cm2" if grade_by == "area" else f"{grade_by}_cm"
    px_key = f"{grade_by}_px" if grade_by != "area" else "area_px"

    values = []
    for m in measurements:
        v = m.get(cm_key) or m.get(px_key, 0)
        values.append(float(v) if v else 0)

    values = np.array(values)

    # Define grade labels
    if n_grades == 3:
        labels = ["Small", "Medium", "Large"]
    elif n_grades == 4:
        labels = ["Small", "Medium", "Large", "X-Large"]
    elif n_grades == 5:
        labels = ["X-Small", "Small", "Medium", "Large", "X-Large"]
    else:
        labels = [f"Grade {i+1}" for i in range(n_grades)]

    # Compute thresholds
    if custom_thresholds:
        thresholds = sorted(custom_thresholds)
    else:
        # Percentile-based
        percentiles = np.linspace(0, 100, n_grades + 1)[1:-1]
        thresholds = [float(np.percentile(values, p)) for p in percentiles]

    # Assign grades
    graded = []
    for i, (m, v) in enumerate(zip(measurements, values)):
        grade_idx = 0
        for t in thresholds:
            if v > t:
                grade_idx += 1
        grade = labels[min(grade_idx, len(labels) - 1)]
        graded.append({
            "index": i,
            "grade": grade,
            "value": round(v, 3),
        })

    # Summary
    grade_dist = {}
    for g in graded:
        grade_dist[g["grade"]] = grade_dist.get(g["grade"], 0) + 1

    grade_summary = {}
    for label in labels:
        grade_vals = [g["value"] for g in graded if g["grade"] == label]
        if grade_vals:
            grade_summary[label] = {
                "count": len(grade_vals),
                "min": round(min(grade_vals), 3),
                "max": round(max(grade_vals), 3),
                "mean": round(float(np.mean(grade_vals)), 3),
            }

    return {
        "graded_objects": graded,
        "grade_summary": grade_summary,
        "grade_distribution": grade_dist,
        "thresholds": [round(t, 3) for t in thresholds],
        "grade_by": grade_by,
    }


# ──────────────────────────────────────────────
# 2. DIMENSIONAL COMPLIANCE CHECKER
# ──────────────────────────────────────────────
def check_compliance(measurements, specs):
    """
    Check if detected objects meet dimensional specifications.

    Parameters
    ----------
    measurements : list of dict
    specs : dict with optional keys:
        'min_width', 'max_width', 'min_height', 'max_height',
        'min_area', 'max_area', 'min_circularity', 'max_aspect_ratio'
        All in the measurement's unit (cm if calibrated, px if not).

    Returns
    -------
    dict with results per object and overall pass/fail stats
    """
    results = []
    n_pass = 0
    n_fail = 0
    failure_reasons = {}

    for i, m in enumerate(measurements):
        is_calibrated = m.get("width_cm") is not None
        w = m.get("width_cm") if is_calibrated else m.get("width_px", 0)
        h = m.get("height_cm") if is_calibrated else m.get("height_px", 0)
        a = m.get("area_cm2") if is_calibrated else m.get("area_px", 0)
        circ = m.get("circularity", 0)
        aspect = m.get("aspect_ratio", 1)

        violations = []

        if specs.get("min_width") is not None and w < specs["min_width"]:
            violations.append(f"Width {w:.2f} < min {specs['min_width']:.2f}")
        if specs.get("max_width") is not None and w > specs["max_width"]:
            violations.append(f"Width {w:.2f} > max {specs['max_width']:.2f}")
        if specs.get("min_height") is not None and h < specs["min_height"]:
            violations.append(f"Height {h:.2f} < min {specs['min_height']:.2f}")
        if specs.get("max_height") is not None and h > specs["max_height"]:
            violations.append(f"Height {h:.2f} > max {specs['max_height']:.2f}")
        if specs.get("min_area") is not None and a < specs["min_area"]:
            violations.append(f"Area {a:.2f} < min {specs['min_area']:.2f}")
        if specs.get("max_area") is not None and a > specs["max_area"]:
            violations.append(f"Area {a:.2f} > max {specs['max_area']:.2f}")
        if specs.get("min_circularity") is not None and circ < specs["min_circularity"]:
            violations.append(f"Circularity {circ:.3f} < min {specs['min_circularity']:.3f}")
        if specs.get("max_aspect_ratio") is not None and aspect > specs["max_aspect_ratio"]:
            violations.append(f"Aspect {aspect:.2f} > max {specs['max_aspect_ratio']:.2f}")

        passed = len(violations) == 0
        if passed:
            n_pass += 1
        else:
            n_fail += 1
            for v in violations:
                reason_key = v.split(" ")[0]  # e.g. "Width"
                failure_reasons[reason_key] = failure_reasons.get(reason_key, 0) + 1

        results.append({
            "index": i,
            "passed": passed,
            "violations": violations,
        })

    return {
        "results": results,
        "total": len(measurements),
        "passed": n_pass,
        "failed": n_fail,
        "pass_rate": round(n_pass / max(len(measurements), 1) * 100, 1),
        "failure_reasons": failure_reasons,
    }


def draw_compliance_overlay(image, measurements, compliance_results):
    """Draw pass/fail overlay on the image."""
    result = image.copy()
    for m, cr in zip(measurements, compliance_results["results"]):
        color = (0, 200, 0) if cr["passed"] else (0, 0, 255)
        label = "✓ PASS" if cr["passed"] else "✗ FAIL"
        cv2.drawContours(result, [m["contour"]], -1, color, 3)
        cx, cy = m["centroid"]
        cv2.putText(result, label, (cx - 30, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if not cr["passed"]:
            for j, v in enumerate(cr["violations"][:2]):
                cv2.putText(result, v[:40], (cx - 30, cy + 15 + j * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 255), 1)
    return result


# ──────────────────────────────────────────────
# 3. OBJECT COMPARISON
# ──────────────────────────────────────────────
def compare_objects(m1, m2, pixel_per_cm=None):
    """
    Detailed side-by-side comparison of two measured objects.
    """
    is_cal = m1.get("width_cm") is not None

    def _get(m, key_cm, key_px):
        return m.get(key_cm) if is_cal and m.get(key_cm) is not None else m.get(key_px, 0)

    w1 = _get(m1, "width_cm", "width_px")
    w2 = _get(m2, "width_cm", "width_px")
    h1 = _get(m1, "height_cm", "height_px")
    h2 = _get(m2, "height_cm", "height_px")
    a1 = _get(m1, "area_cm2", "area_px")
    a2 = _get(m2, "area_cm2", "area_px")
    p1 = _get(m1, "perimeter_cm", "perimeter_px")
    p2 = _get(m2, "perimeter_cm", "perimeter_px")

    unit = "cm" if is_cal else "px"

    def _diff(v1, v2):
        if v1 == 0 and v2 == 0:
            return 0
        return round(abs(v1 - v2) / max(abs(v1), abs(v2)) * 100, 1)

    return {
        "unit": unit,
        "comparison": [
            {"metric": f"Width ({unit})", "obj_1": round(w1, 3), "obj_2": round(w2, 3),
             "diff": round(abs(w1 - w2), 3), "diff_pct": _diff(w1, w2)},
            {"metric": f"Height ({unit})", "obj_1": round(h1, 3), "obj_2": round(h2, 3),
             "diff": round(abs(h1 - h2), 3), "diff_pct": _diff(h1, h2)},
            {"metric": f"Area ({unit}²)", "obj_1": round(a1, 3), "obj_2": round(a2, 3),
             "diff": round(abs(a1 - a2), 3), "diff_pct": _diff(a1, a2)},
            {"metric": f"Perimeter ({unit})", "obj_1": round(p1, 3), "obj_2": round(p2, 3),
             "diff": round(abs(p1 - p2), 3), "diff_pct": _diff(p1, p2)},
            {"metric": "Circularity", "obj_1": round(m1.get("circularity", 0), 3),
             "obj_2": round(m2.get("circularity", 0), 3),
             "diff": round(abs(m1.get("circularity", 0) - m2.get("circularity", 0)), 3),
             "diff_pct": _diff(m1.get("circularity", 0), m2.get("circularity", 0))},
            {"metric": "Aspect Ratio", "obj_1": round(m1.get("aspect_ratio", 0), 3),
             "obj_2": round(m2.get("aspect_ratio", 0), 3),
             "diff": round(abs(m1.get("aspect_ratio", 0) - m2.get("aspect_ratio", 0)), 3),
             "diff_pct": _diff(m1.get("aspect_ratio", 0), m2.get("aspect_ratio", 0))},
        ],
    }


# ──────────────────────────────────────────────
# 4. CLICK-TO-ISOLATE (Flood Fill)
# ──────────────────────────────────────────────
def isolate_object_at_point(image, seed_point, tolerance=30):
    """
    Isolate a single object using flood fill from a seed point.
    Works like a magic wand — selects connected region similar to the seed.

    Parameters
    ----------
    image : np.ndarray (BGR)
    seed_point : (x, y)
    tolerance : int — color similarity threshold (0-255)

    Returns
    -------
    dict with mask, cropped_object, measurements
    """
    h, w = image.shape[:2]
    x, y = int(seed_point[0]), int(seed_point[1])
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))

    # Flood fill on a copy
    mask = np.zeros((h + 2, w + 2), np.uint8)
    flood_image = image.copy()

    lo_diff = (tolerance, tolerance, tolerance)
    hi_diff = (tolerance, tolerance, tolerance)

    cv2.floodFill(flood_image, mask, (x, y), (255, 0, 255),
                  loDiff=lo_diff, upDiff=hi_diff,
                  flags=cv2.FLOODFILL_MASK_ONLY | (255 << 8))

    # Extract the mask (remove the 1-pixel border added by floodFill)
    object_mask = mask[1:-1, 1:-1]

    # Cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contour
    contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Pick the contour containing the seed point
    best = max(contours, key=cv2.contourArea)

    # Crop
    bx, by, bw, bh = cv2.boundingRect(best)
    padding = 10
    x1 = max(0, bx - padding)
    y1 = max(0, by - padding)
    x2 = min(w, bx + bw + padding)
    y2 = min(h, by + bh + padding)
    cropped = image[y1:y2, x1:x2].copy()

    # Highlight
    overlay = image.copy()
    cv2.drawContours(overlay, [best], -1, (0, 255, 0), 3)

    return {
        "mask": object_mask,
        "contour": best,
        "cropped": cropped,
        "overlay": overlay,
        "bbox": (bx, by, bw, bh),
        "area_px": cv2.contourArea(best),
    }


# ──────────────────────────────────────────────
# 5. SIZE DISTRIBUTION ANALYSIS
# ──────────────────────────────────────────────
def analyze_size_distribution(measurements, metric="area"):
    """
    Statistical analysis of object size population.
    Useful for quality control, agricultural sorting, lab analysis.
    """
    cm_key = f"{metric}_cm2" if metric == "area" else f"{metric}_cm"
    px_key = f"{metric}_px" if metric != "area" else "area_px"

    values = []
    for m in measurements:
        v = m.get(cm_key) or m.get(px_key, 0)
        if v and v > 0:
            values.append(float(v))

    if len(values) < 2:
        return {"status": "need_more_data", "n": len(values)}

    values = np.array(values)

    from scipy import stats as sp_stats

    # Basic stats
    result = {
        "n": len(values),
        "mean": round(float(np.mean(values)), 4),
        "median": round(float(np.median(values)), 4),
        "std": round(float(np.std(values)), 4),
        "cv": round(float(np.std(values) / np.mean(values) * 100), 1),  # Coefficient of variation
        "min": round(float(np.min(values)), 4),
        "max": round(float(np.max(values)), 4),
        "range": round(float(np.ptp(values)), 4),
        "iqr": round(float(np.percentile(values, 75) - np.percentile(values, 25)), 4),
        "skewness": round(float(sp_stats.skew(values)), 3),
        "kurtosis": round(float(sp_stats.kurtosis(values)), 3),
        "percentiles": {
            "5th": round(float(np.percentile(values, 5)), 4),
            "25th": round(float(np.percentile(values, 25)), 4),
            "50th": round(float(np.percentile(values, 50)), 4),
            "75th": round(float(np.percentile(values, 75)), 4),
            "95th": round(float(np.percentile(values, 95)), 4),
        },
        "values": values.tolist(),
        "status": "ok",
    }

    # Normality test
    if len(values) >= 8:
        stat, p_value = sp_stats.shapiro(values)
        result["normality_test"] = {
            "statistic": round(float(stat), 4),
            "p_value": round(float(p_value), 4),
            "is_normal": p_value > 0.05,
        }

    # Uniformity classification
    cv = result["cv"]
    if cv < 5:
        result["uniformity"] = "Highly Uniform"
    elif cv < 10:
        result["uniformity"] = "Uniform"
    elif cv < 20:
        result["uniformity"] = "Moderate Variation"
    elif cv < 35:
        result["uniformity"] = "High Variation"
    else:
        result["uniformity"] = "Very High Variation"

    return result


# ──────────────────────────────────────────────
# 6. OBJECT CROPPER
# ──────────────────────────────────────────────
def crop_individual_objects(image, measurements, padding=15):
    """
    Extract each detected object as a separate cropped image.
    """
    crops = []
    h, w = image.shape[:2]

    for i, m in enumerate(measurements):
        bx, by, bw, bh = m["bbox"]
        x1 = max(0, bx - padding)
        y1 = max(0, by - padding)
        x2 = min(w, bx + bw + padding)
        y2 = min(h, by + bh + padding)

        cropped = image[y1:y2, x1:x2].copy()

        # Also create a masked version (object only, transparent background)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [m["contour"]], -1, 255, -1)
        masked_crop = cv2.bitwise_and(image, image, mask=mask)
        masked_crop = masked_crop[y1:y2, x1:x2]

        crops.append({
            "index": i,
            "cropped": cropped,
            "masked": masked_crop,
            "size": (x2 - x1, y2 - y1),
        })

    return crops


# ──────────────────────────────────────────────
# 7. NEAREST-NEIGHBOR SPACING ANALYSIS
# ──────────────────────────────────────────────
def analyze_spacing(measurements):
    """
    Compute nearest-neighbor distances between all objects.
    Useful for analyzing distribution patterns (regular, clustered, random).
    """
    if len(measurements) < 2:
        return {"status": "need_more_objects"}

    centroids = np.array([m["centroid"] for m in measurements], dtype=np.float64)
    n = len(centroids)

    # All pairwise distances
    from scipy.spatial.distance import cdist
    distances = cdist(centroids, centroids)
    np.fill_diagonal(distances, np.inf)

    # Nearest neighbor for each object
    nn_distances = np.min(distances, axis=1)
    nn_indices = np.argmin(distances, axis=1)

    mean_nn = float(np.mean(nn_distances))
    std_nn = float(np.std(nn_distances))

    # Clark-Evans index: R = observed_mean_nn / expected_mean_nn
    # Expected for random distribution = 0.5 * sqrt(A / n)
    # where A is the area (approximate from bounding box of all centroids)
    x_range = np.ptp(centroids[:, 0])
    y_range = np.ptp(centroids[:, 1])
    area = max((x_range + 50) * (y_range + 50), 1)
    expected_nn = 0.5 * np.sqrt(area / n)
    clark_evans_r = mean_nn / (expected_nn + 1e-8)

    if clark_evans_r < 0.5:
        pattern = "Clustered"
    elif clark_evans_r < 1.2:
        pattern = "Random"
    else:
        pattern = "Regular/Dispersed"

    return {
        "nn_distances": nn_distances.tolist(),
        "nn_indices": nn_indices.tolist(),
        "mean_nn_distance": round(mean_nn, 2),
        "std_nn_distance": round(std_nn, 2),
        "clark_evans_r": round(float(clark_evans_r), 3),
        "pattern": pattern,
        "status": "ok",
    }


# ──────────────────────────────────────────────
# 8. CONVEX HULL ANALYSIS
# ──────────────────────────────────────────────
def convex_hull_analysis(contour):
    """
    Detailed convexity analysis of a contour.
    """
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour, returnPoints=True)
    hull_area = cv2.contourArea(hull)
    hull_perimeter = cv2.arcLength(hull, True)
    perimeter = cv2.arcLength(contour, True)

    solidity = area / (hull_area + 1e-8)
    convexity = hull_perimeter / (perimeter + 1e-8)

    # Convexity defects
    hull_indices = cv2.convexHull(contour, returnPoints=False)
    n_defects = 0
    max_defect_depth = 0

    try:
        if len(hull_indices) > 2 and len(contour) > 3:
            defects = cv2.convexityDefects(contour, hull_indices)
            if defects is not None:
                n_defects = len(defects)
                max_defect_depth = float(np.max(defects[:, 0, 3]) / 256.0)
    except cv2.error:
        pass

    return {
        "solidity": round(solidity, 4),
        "convexity": round(convexity, 4),
        "hull_area": round(hull_area, 1),
        "n_defects": n_defects,
        "max_defect_depth": round(max_defect_depth, 1),
        "hull_contour": hull,
    }


# ──────────────────────────────────────────────
# 9. MEASUREMENT REPORT DATA
# ──────────────────────────────────────────────
def generate_report_data(measurements, image_info, calibration_info=None):
    """
    Generate structured data for a measurement report.
    """
    is_cal = measurements and measurements[0].get("width_cm") is not None
    unit = "cm" if is_cal else "px"
    unit_sq = "cm²" if is_cal else "px²"

    objects = []
    for i, m in enumerate(measurements):
        w = m.get("width_cm") if is_cal else m.get("width_px", 0)
        h = m.get("height_cm") if is_cal else m.get("height_px", 0)
        a = m.get("area_cm2") if is_cal else m.get("area_px", 0)
        p = m.get("perimeter_cm") if is_cal else m.get("perimeter_px", 0)
        objects.append({
            "id": i + 1,
            f"width_{unit}": round(w, 3) if w else 0,
            f"height_{unit}": round(h, 3) if h else 0,
            f"area_{unit_sq}": round(a, 3) if a else 0,
            f"perimeter_{unit}": round(p, 3) if p else 0,
            "circularity": round(m.get("circularity", 0), 3),
            "aspect_ratio": round(m.get("aspect_ratio", 0), 3),
        })

    # Summary statistics
    if measurements:
        areas = [o[f"area_{unit_sq}"] for o in objects]
        widths = [o[f"width_{unit}"] for o in objects]
        summary = {
            "total_objects": len(objects),
            "calibrated": is_cal,
            "unit": unit,
            f"total_area_{unit_sq}": round(sum(areas), 3),
            f"mean_area_{unit_sq}": round(float(np.mean(areas)), 3),
            f"mean_width_{unit}": round(float(np.mean(widths)), 3),
            f"std_width_{unit}": round(float(np.std(widths)), 3),
        }
    else:
        summary = {"total_objects": 0, "calibrated": is_cal, "unit": unit}

    return {
        "image_info": image_info,
        "calibration": calibration_info or {"status": "not calibrated"},
        "summary": summary,
        "objects": objects,
    }
