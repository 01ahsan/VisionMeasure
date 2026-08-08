"""
Smart Processing Module — Robust Version
Tries EVERY segmentation strategy (threshold, GrabCut, K-Means, Watershed, HSV)
and picks the one that produces the best object detections.
Designed to handle ANY image type, size, and complexity.
"""

import cv2
import numpy as np

from .preprocessing import preprocess_image
from .segmentation import (
    segment_image, grabcut_segment, kmeans_segment,
    watershed_segment, hsv_segment,
)
from .contour_analysis import find_and_measure_contours


def analyze_image_characteristics(image):
    """Analyze image properties to guide strategy selection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    brightness = np.mean(gray)
    contrast = np.std(gray)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Background analysis via corners
    corners = [
        gray[0:h // 8, 0:w // 8],
        gray[0:h // 8, -w // 8:],
        gray[-h // 8:, 0:w // 8],
        gray[-h // 8:, -w // 8:],
    ]
    corner_mean = np.mean([np.mean(c) for c in corners])
    center_mean = np.mean(gray[h // 4: 3 * h // 4, w // 4: 3 * w // 4])
    light_bg = corner_mean > center_mean

    # Color richness
    mean_saturation = np.mean(hsv[:, :, 1])
    is_colorful = mean_saturation > 60

    # Texture complexity
    is_complex = laplacian_var > 500

    return {
        "brightness": brightness,
        "contrast": contrast,
        "noise_level": laplacian_var,
        "light_background": light_bg,
        "suggested_invert": not light_bg,
        "is_low_contrast": contrast < 40,
        "is_noisy": laplacian_var > 1000,
        "is_dark": brightness < 80,
        "is_bright": brightness > 200,
        "is_colorful": is_colorful,
        "is_complex": is_complex,
        "mean_saturation": mean_saturation,
        "resolution": (w, h),
    }


def _score_segmentation(binary_mask, image, min_area_ratio=0.003):
    """
    Score a binary segmentation result.
    Higher = better (more likely to contain real objects, not noise).
    """
    h, w = binary_mask.shape
    image_area = h * w

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) > image_area * min_area_ratio]

    n = len(valid)
    if n == 0:
        return 0.0, 0

    # Penalize: 0 objects (useless), too many (noise)
    if n > 20:
        count_score = 0.2
    elif n > 10:
        count_score = 0.5
    else:
        count_score = min(n / 5.0, 1.0)

    # Reward reasonable total foreground coverage (not too little, not the whole image)
    fg_ratio = np.sum(binary_mask > 0) / image_area
    if fg_ratio < 0.005:
        coverage_score = 0.1
    elif fg_ratio > 0.9:
        coverage_score = 0.1  # probably inverted or failed
    elif 0.01 <= fg_ratio <= 0.7:
        coverage_score = 1.0
    else:
        coverage_score = 0.5

    # Reward objects with good shapes (not tiny fragments)
    shape_scores = []
    for c in valid[:10]:
        area = cv2.contourArea(c)
        perim = cv2.arcLength(c, True)
        if perim > 0:
            circ = 4 * np.pi * area / (perim ** 2)
            shape_scores.append(min(circ * 1.5, 1.0))
    avg_shape = np.mean(shape_scores) if shape_scores else 0.3

    # Reward objects that aren't at the very edge (edge = often border artifacts)
    edge_penalty = 0
    for c in valid[:10]:
        x, y, cw, ch = cv2.boundingRect(c)
        if x <= 2 or y <= 2 or x + cw >= w - 2 or y + ch >= h - 2:
            # Check if this contour is huge (>50% of image) — likely border
            if cv2.contourArea(c) > image_area * 0.4:
                edge_penalty += 0.3

    score = count_score * 0.3 + coverage_score * 0.3 + avg_shape * 0.3 + 0.1
    score = max(0, score - edge_penalty)

    return score, n


def auto_process(image, min_area_ratio=0.003):
    """
    Try segmentation strategies with early termination + downsampling.
    Stops searching if a strategy scores >= EARLY_STOP_THRESHOLD.
    Uses downsampled image for strategy selection, full-res for final measurement.
    """
    EARLY_STOP_THRESHOLD = 0.88  # Fix #4: stop early if good enough

    chars = analyze_image_characteristics(image)
    h, w = image.shape[:2]

    # Fix #4: Downsample for strategy testing (full-res only for winner)
    MAX_TEST_DIM = 600
    test_image = image
    test_scale = 1.0
    if max(h, w) > MAX_TEST_DIM:
        test_scale = MAX_TEST_DIM / max(h, w)
        test_image = cv2.resize(image, (int(w * test_scale), int(h * test_scale)))

    # Preprocess once for threshold-based methods
    prep_results = {}
    for filt in ["gaussian", "median", "bilateral"]:
        prep_results[filt] = preprocess_image(test_image, method=filt, kernel_size=5, apply_clahe=True)

    candidates = []
    early_stopped = False

    # ── STRATEGY GROUP 1: Threshold-based ──
    threshold_configs = [
        ("Otsu", "otsu", "close", False),
        ("Otsu Inverted", "otsu", "close", True),
        ("Adaptive Gaussian", "adaptive_gaussian", "close", False),
        ("Adaptive Gaussian Inv", "adaptive_gaussian", "close", True),
        ("Adaptive Mean", "adaptive_mean", "open_close", False),
        ("Otsu + Open", "otsu", "open", False),
        ("Otsu + Open Inv", "otsu", "open", True),
    ]

    for name, thresh, morph, inv in threshold_configs:
        if early_stopped:
            break
        for filt_name, prep in prep_results.items():
            try:
                seg = segment_image(
                    prep["enhanced"], threshold_method=thresh,
                    morph_operation=morph, morph_kernel_size=5,
                    morph_iterations=2, invert=inv,
                )
                score, n_obj = _score_segmentation(seg["morphed"], test_image, min_area_ratio)
                candidates.append({
                    "name": f"{name} ({filt_name})",
                    "type": "threshold",
                    "mask": seg["morphed"],
                    "prep": prep,
                    "seg": seg,
                    "score": score,
                    "n_objects": n_obj,
                    "params": {"filter": filt_name, "threshold": thresh, "morph": morph, "invert": inv},
                })
                if score >= EARLY_STOP_THRESHOLD:
                    early_stopped = True
                    break
            except Exception:
                continue

    # ── STRATEGY GROUP 2: GrabCut ──
    if not early_stopped:
        try:
            gc_mask = grabcut_segment(test_image, iterations=5)
            score, n_obj = _score_segmentation(gc_mask, test_image, min_area_ratio)
            if chars["is_complex"]:
                score *= 1.15
            candidates.append({
                "name": "GrabCut",
                "type": "grabcut",
                "mask": gc_mask,
                "prep": prep_results["gaussian"],
                "seg": {"thresholded": gc_mask, "morphed": gc_mask},
                "score": score,
                "n_objects": n_obj,
                "params": {"type": "grabcut"},
            })
            if score >= EARLY_STOP_THRESHOLD:
                early_stopped = True
        except Exception:
            pass

    # ── STRATEGY GROUP 3: K-Means ──
    # Already using test_image which is downsampled
    km_image = test_image

    if not early_stopped:
        for k in [2, 3, 4]:
            if early_stopped:
                break
            for target in ["darkest", "largest"]:
                try:
                    km_mask, km_viz = kmeans_segment(km_image, k=k, target_cluster=target)
                    score, n_obj = _score_segmentation(km_mask, test_image, min_area_ratio)
                    if chars["is_colorful"]:
                        score *= 1.1
                    candidates.append({
                        "name": f"K-Means (k={k}, {target})",
                        "type": "kmeans",
                        "mask": km_mask,
                        "prep": prep_results["gaussian"],
                        "seg": {"thresholded": km_mask, "morphed": km_mask},
                        "score": score,
                        "n_objects": n_obj,
                        "params": {"type": "kmeans", "k": k, "target": target},
                    })
                    if score >= EARLY_STOP_THRESHOLD:
                        early_stopped = True
                        break
                except Exception:
                    continue

    # ── STRATEGY GROUP 4: Watershed ──
    if not early_stopped:
        try:
            ws_mask, ws_dist, ws_viz = watershed_segment(test_image)
            score, n_obj = _score_segmentation(ws_mask, test_image, min_area_ratio)
            candidates.append({
                "name": "Watershed",
                "type": "watershed",
                "mask": ws_mask,
                "prep": prep_results["gaussian"],
                "seg": {"thresholded": ws_mask, "morphed": ws_mask},
                "score": score,
                "n_objects": n_obj,
                "params": {"type": "watershed"},
            })
        except Exception:
            pass

    # ── STRATEGY GROUP 5: ML-Optimized Threshold (Bayesian) ──
    if not early_stopped:
        try:
            from .ml_analysis import bayesian_threshold_optimize
            test_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
            test_gray = cv2.GaussianBlur(test_gray, (5, 5), 0)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            test_gray = clahe.apply(test_gray)

            opt = bayesian_threshold_optimize(test_gray, n_iterations=20)
            _, binary = cv2.threshold(test_gray, opt["threshold"], 255, cv2.THRESH_BINARY)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opt["morph_kernel"], opt["morph_kernel"]))
            morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=opt["morph_iterations"])
            morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, k, iterations=max(1, opt["morph_iterations"] - 1))

            score, n_obj = _score_segmentation(morphed, test_image, min_area_ratio)
            score *= 1.05  # Slight bonus for ML-optimized parameters
            candidates.append({
                "name": f"ML-Optimized (t={opt['threshold']}, k={opt['morph_kernel']})",
                "type": "ml_optimized",
                "mask": morphed,
                "prep": prep_results["gaussian"],
                "seg": {"thresholded": binary, "morphed": morphed},
                "score": score,
                "n_objects": n_obj,
                "params": {"type": "ml_optimized", "threshold": opt["threshold"],
                           "morph_kernel": opt["morph_kernel"],
                           "morph_iterations": opt["morph_iterations"]},
            })
        except Exception:
            pass

    # ── Pick the best ──
    if not candidates:
        prep = preprocess_image(image)
        seg = segment_image(prep["enhanced"])
        return {
            "prep_result": prep,
            "seg_result": seg,
            "measurements": [],
            "strategy_name": "Fallback",
            "characteristics": chars,
            "params_used": {},
            "all_strategies": [],
        }

    candidates.sort(key=lambda x: x["score"], reverse=True)
    best = candidates[0]

    # ── Fix #4: Re-run winning strategy at FULL resolution ──
    if test_scale != 1.0:
        full_prep = preprocess_image(image, method=best.get("params", {}).get("filter", "gaussian"), kernel_size=5, apply_clahe=True)

        if best["type"] == "threshold":
            p = best.get("params", {})
            full_seg = segment_image(
                full_prep["enhanced"],
                threshold_method=p.get("threshold", "otsu"),
                morph_operation=p.get("morph", "close"),
                morph_kernel_size=5, morph_iterations=2,
                invert=p.get("invert", False),
            )
            best_mask = full_seg["morphed"]
            best["prep"] = full_prep
            best["seg"] = full_seg
        elif best["type"] == "grabcut":
            best_mask = grabcut_segment(image, iterations=5)
            best["prep"] = full_prep
            best["seg"] = {"thresholded": best_mask, "morphed": best_mask}
        elif best["type"] == "kmeans":
            p = best.get("params", {})
            best_mask, _ = kmeans_segment(image, k=p.get("k", 3), target_cluster=p.get("target", "darkest"))
            best["prep"] = full_prep
            best["seg"] = {"thresholded": best_mask, "morphed": best_mask}
        elif best["type"] == "watershed":
            best_mask, _, _ = watershed_segment(image)
            best["prep"] = full_prep
            best["seg"] = {"thresholded": best_mask, "morphed": best_mask}
        elif best["type"] == "ml_optimized":
            p = best.get("params", {})
            gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray_full = cv2.GaussianBlur(gray_full, (5, 5), 0)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray_full = clahe.apply(gray_full)
            _, binary = cv2.threshold(gray_full, p.get("threshold", 127), 255, cv2.THRESH_BINARY)
            mk = p.get("morph_kernel", 5)
            mi = p.get("morph_iterations", 2)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))
            best_mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=mi)
            best_mask = cv2.morphologyEx(best_mask, cv2.MORPH_OPEN, k, iterations=max(1, mi - 1))
            best["prep"] = full_prep
            best["seg"] = {"thresholded": binary, "morphed": best_mask}
        else:
            best_mask = best["mask"]
    else:
        best_mask = best["mask"]

    # Final measurement at full resolution
    measurements = find_and_measure_contours(
        best_mask if test_scale != 1.0 else best["mask"],
        image, pixel_per_cm=None, min_area_ratio=min_area_ratio
    )

    # Top 5 strategies summary
    top_strategies = [
        {"name": c["name"], "score": round(c["score"], 3), "objects": c["n_objects"]}
        for c in candidates[:5]
    ]

    return {
        "prep_result": best["prep"],
        "seg_result": best["seg"],
        "measurements": measurements,
        "strategy_name": best["name"],
        "strategy_type": best["type"],
        "characteristics": chars,
        "params_used": {"type": best["type"]},
        "all_strategies": top_strategies,
        "best_mask": best["mask"],
    }
