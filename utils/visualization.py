"""
Visualization Module
Handles drawing measurements on images and creating pipeline visualizations.
"""

import cv2
import numpy as np


def draw_measurements(image, measurements, show_labels=True, color=(0, 255, 0), thickness=2):
    """
    Draw contours, bounding boxes, and measurement labels on the image.

    Parameters
    ----------
    image : np.ndarray
        Original BGR image (will be copied).
    measurements : list[dict]
        Output from find_and_measure_contours.
    show_labels : bool
        Whether to draw text labels.
    color : tuple
        BGR color for drawing.
    thickness : int
        Line thickness.

    Returns
    -------
    np.ndarray
        Annotated image.
    """
    result = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    label_color = (255, 255, 255)
    bg_color = (0, 0, 0)

    for i, m in enumerate(measurements):
        # Draw contour
        cv2.drawContours(result, [m["contour"]], -1, color, thickness)

        # Draw minimum area rectangle
        box = cv2.boxPoints(m["min_rect"])
        box = np.intp(box)
        cv2.drawContours(result, [box], 0, (0, 165, 255), 2)

        # Draw centroid
        cx, cy = m["centroid"]
        cv2.circle(result, (cx, cy), 5, (0, 0, 255), -1)

        if show_labels:
            # Build label text
            if m["width_cm"] is not None:
                label_w = f"W:{m['width_cm']:.2f}cm"
                label_h = f"H:{m['height_cm']:.2f}cm"
                label_a = f"A:{m['area_cm2']:.2f}cm2"
                label_p = f"P:{m['perimeter_cm']:.2f}cm"
            else:
                label_w = f"W:{m['width_px']:.0f}px"
                label_h = f"H:{m['height_px']:.0f}px"
                label_a = f"A:{m['area_px']:.0f}px2"
                label_p = f"P:{m['perimeter_px']:.0f}px"

            obj_label = f"Obj {i + 1}"

            # Position labels near centroid
            labels = [obj_label, label_w, label_h, label_a, label_p]
            for j, text in enumerate(labels):
                y_pos = cy - 50 + j * 18
                x_pos = cx + 10

                # Background rectangle for readability
                (tw, th), _ = cv2.getTextSize(text, font, font_scale, 1)
                cv2.rectangle(
                    result,
                    (x_pos - 2, y_pos - th - 2),
                    (x_pos + tw + 2, y_pos + 2),
                    bg_color,
                    -1,
                )
                cv2.putText(result, text, (x_pos, y_pos), font, font_scale, label_color, 1)

    return result


def draw_reference_highlight(image, ref_info, ref_size_cm):
    """
    Highlight the detected reference object.

    Parameters
    ----------
    image : np.ndarray
        Image to draw on (will be copied).
    ref_info : dict
        Reference object info from detect_reference_object.
    ref_size_cm : float
        Known size in cm.

    Returns
    -------
    np.ndarray
        Image with reference highlighted.
    """
    result = image.copy()

    if ref_info is None:
        return result

    cv2.drawContours(result, [ref_info["contour"]], -1, (255, 0, 255), 3)

    cx, cy = ref_info["center"]
    label = f"REF: {ref_size_cm:.2f}cm"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(label, font, 0.6, 2)
    cv2.rectangle(
        result,
        (cx - tw // 2 - 4, cy - 30 - th),
        (cx + tw // 2 + 4, cy - 26),
        (255, 0, 255),
        -1,
    )
    cv2.putText(
        result, label, (cx - tw // 2, cy - 30), font, 0.6, (255, 255, 255), 2
    )

    return result


def create_pipeline_visualization(stages):
    """
    Create a side-by-side grid of pipeline stages.

    Parameters
    ----------
    stages : list of (str, np.ndarray)
        Each tuple is (stage_name, image).

    Returns
    -------
    np.ndarray
        Combined visualization image.
    """
    processed = []
    target_h = 300

    for name, img in stages:
        # Convert grayscale to BGR for consistent display
        if len(img.shape) == 2:
            display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            display = img.copy()

        # Resize to uniform height
        h, w = display.shape[:2]
        scale = target_h / h
        new_w = int(w * scale)
        display = cv2.resize(display, (new_w, target_h))

        # Add label bar at top
        label_bar = np.zeros((30, new_w, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(label_bar, name, (5, 22), font, 0.55, (255, 255, 255), 1)
        display = np.vstack([label_bar, display])

        processed.append(display)

    # Arrange in grid (2 per row)
    rows = []
    for i in range(0, len(processed), 2):
        row_imgs = processed[i : i + 2]
        if len(row_imgs) == 1:
            # Pad with black image
            h, w = row_imgs[0].shape[:2]
            row_imgs.append(np.zeros_like(row_imgs[0]))
        # Make same height
        max_h = max(im.shape[0] for im in row_imgs)
        padded = []
        for im in row_imgs:
            if im.shape[0] < max_h:
                pad = np.zeros((max_h - im.shape[0], im.shape[1], 3), dtype=np.uint8)
                im = np.vstack([im, pad])
            padded.append(im)
        # Make same width
        max_w = max(im.shape[1] for im in padded)
        final = []
        for im in padded:
            if im.shape[1] < max_w:
                pad = np.zeros((im.shape[0], max_w - im.shape[1], 3), dtype=np.uint8)
                im = np.hstack([im, pad])
            final.append(im)
        rows.append(np.hstack(final))

    # Stack rows
    max_w = max(r.shape[1] for r in rows)
    final_rows = []
    for r in rows:
        if r.shape[1] < max_w:
            pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)
            r = np.hstack([r, pad])
        final_rows.append(r)

    return np.vstack(final_rows)
