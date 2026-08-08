"""
Batch Processing Module
Handles:
- Zip file extraction with nested folder traversal
- Bulk image upload validation
- Corrupted/non-image file detection
- Parallel-style batch measurement pipeline
"""

import os
import cv2
import zipfile
import tempfile
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime


# Supported image extensions
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif",
    ".webp", ".jp2", ".pbm", ".pgm", ".ppm",
}

# Max image dimension for processing
MAX_DIM = 2000


def validate_image(filepath):
    """
    Validate if a file is a readable image.

    Returns
    -------
    dict with keys:
        'valid'     : bool
        'path'      : str
        'filename'  : str
        'reason'    : str (if invalid)
        'width'     : int (if valid)
        'height'    : int (if valid)
        'channels'  : int (if valid)
        'filesize'  : int (bytes)
    """
    filename = os.path.basename(filepath)
    result = {
        "valid": False,
        "path": filepath,
        "filename": filename,
        "reason": "",
        "relative_path": "",
    }

    # Check existence
    if not os.path.isfile(filepath):
        result["reason"] = "File not found"
        return result

    # Check file size
    filesize = os.path.getsize(filepath)
    result["filesize"] = filesize

    if filesize == 0:
        result["reason"] = "Empty file (0 bytes)"
        return result

    if filesize > 100 * 1024 * 1024:  # 100MB limit per image
        result["reason"] = f"File too large ({filesize / 1024 / 1024:.1f}MB > 100MB limit)"
        return result

    # Check extension
    ext = Path(filepath).suffix.lower()
    if ext not in IMAGE_EXTENSIONS:
        result["reason"] = f"Unsupported format: {ext}"
        return result

    # Try reading with OpenCV
    try:
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if img is None:
            result["reason"] = "Corrupted or unreadable image"
            return result

        h, w = img.shape[:2]
        channels = img.shape[2] if len(img.shape) == 3 else 1

        if h < 10 or w < 10:
            result["reason"] = f"Image too small ({w}×{h})"
            return result

        result["valid"] = True
        result["width"] = w
        result["height"] = h
        result["channels"] = channels
        return result

    except Exception as e:
        result["reason"] = f"Read error: {str(e)[:80]}"
        return result


def extract_zip(zip_file_path, extract_to=None):
    """
    Extract a zip file, handling nested folders.

    Returns
    -------
    dict with:
        'extract_dir' : str — root of extracted content
        'all_files'   : list[str] — all file paths found
        'image_files' : list[dict] — validated image files
        'skipped'     : list[dict] — non-image or corrupted files
        'folder_structure' : dict — nested folder tree
        'total_files' : int
        'total_images': int
    """
    if extract_to is None:
        extract_to = tempfile.mkdtemp(prefix="visionmeasure_")

    # Extract
    try:
        with zipfile.ZipFile(zip_file_path, "r") as zf:
            # Security: check for zip bombs and path traversal
            total_size = sum(info.file_size for info in zf.infolist())
            if total_size > 2 * 1024 * 1024 * 1024:  # 2GB limit
                return {
                    "error": "Zip file too large (extracted > 2GB)",
                    "extract_dir": extract_to,
                }

            for info in zf.infolist():
                # Prevent path traversal
                if info.filename.startswith("/") or ".." in info.filename:
                    continue
                zf.extract(info, extract_to)
    except zipfile.BadZipFile:
        return {"error": "Invalid or corrupted zip file", "extract_dir": extract_to}
    except Exception as e:
        return {"error": f"Extraction failed: {str(e)[:100]}", "extract_dir": extract_to}

    # Recursively find all files
    all_files = []
    for root, dirs, files in os.walk(extract_to):
        # Skip hidden directories and __MACOSX
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__MACOSX"]
        for f in files:
            if not f.startswith("."):
                full_path = os.path.join(root, f)
                all_files.append(full_path)

    # Validate each file
    image_files = []
    skipped = []

    for fpath in all_files:
        rel_path = os.path.relpath(fpath, extract_to)
        validation = validate_image(fpath)
        validation["relative_path"] = rel_path

        if validation["valid"]:
            image_files.append(validation)
        else:
            skipped.append(validation)

    # Build folder structure
    folder_tree = _build_folder_tree(extract_to, all_files)

    return {
        "extract_dir": extract_to,
        "all_files": all_files,
        "image_files": image_files,
        "skipped": skipped,
        "folder_structure": folder_tree,
        "total_files": len(all_files),
        "total_images": len(image_files),
    }


def _build_folder_tree(root, files):
    """Build a nested dict representing the folder structure."""
    tree = {"_files": [], "_dirs": {}}
    for fpath in files:
        rel = os.path.relpath(fpath, root)
        parts = Path(rel).parts
        current = tree
        for part in parts[:-1]:
            if part not in current["_dirs"]:
                current["_dirs"][part] = {"_files": [], "_dirs": {}}
            current = current["_dirs"][part]
        current["_files"].append(parts[-1])
    return tree


def validate_bulk_images(file_list):
    """
    Validate a list of uploaded file objects (from Streamlit file_uploader).

    Parameters
    ----------
    file_list : list of UploadedFile objects

    Returns
    -------
    dict with:
        'valid_images'  : list[dict] with 'file', 'filename', 'width', 'height'
        'skipped'       : list[dict] with 'filename', 'reason'
    """
    valid = []
    skipped = []

    for uploaded_file in file_list:
        filename = uploaded_file.name

        try:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            uploaded_file.seek(0)  # Reset for potential re-read

            if len(file_bytes) == 0:
                skipped.append({"filename": filename, "reason": "Empty file"})
                continue

            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None:
                skipped.append({"filename": filename, "reason": "Corrupted or unsupported format"})
                continue

            h, w = img.shape[:2]
            if h < 10 or w < 10:
                skipped.append({"filename": filename, "reason": f"Too small ({w}×{h})"})
                continue

            # Resize if needed
            if max(h, w) > MAX_DIM:
                scale = MAX_DIM / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
                h, w = img.shape[:2]

            valid.append({
                "file": uploaded_file,
                "image": img,
                "filename": filename,
                "width": w,
                "height": h,
                "filesize": len(file_bytes),
            })

        except Exception as e:
            skipped.append({"filename": filename, "reason": f"Error: {str(e)[:60]}"})

    return {"valid_images": valid, "skipped": skipped}


def process_batch(images, process_fn, progress_callback=None):
    """
    Process a batch of images through a given function.

    Parameters
    ----------
    images : list[dict]
        Each dict has 'image' (np.ndarray), 'filename', etc.
    process_fn : callable
        Function that takes an image (np.ndarray) and returns a result dict.
    progress_callback : callable or None
        Called with (current_index, total, filename) for progress updates.

    Returns
    -------
    list[dict]
        Each dict has: 'filename', 'success', 'result' or 'error', 'processing_time'
    """
    results = []
    total = len(images)

    for i, img_info in enumerate(images):
        if progress_callback:
            progress_callback(i, total, img_info["filename"])

        start = datetime.now()

        try:
            result = process_fn(img_info["image"])
            elapsed = (datetime.now() - start).total_seconds()

            results.append({
                "index": i,
                "filename": img_info["filename"],
                "width": img_info.get("width", 0),
                "height": img_info.get("height", 0),
                "success": True,
                "result": result,
                "processing_time": elapsed,
            })
        except Exception as e:
            elapsed = (datetime.now() - start).total_seconds()
            results.append({
                "index": i,
                "filename": img_info["filename"],
                "width": img_info.get("width", 0),
                "height": img_info.get("height", 0),
                "success": False,
                "error": str(e)[:200],
                "processing_time": elapsed,
            })

    return results


def cleanup_temp_dir(dir_path):
    """Safely remove temporary extraction directory."""
    try:
        if dir_path and os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
    except Exception:
        pass
