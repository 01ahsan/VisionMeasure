"""
Unit Tests for VisionMeasure
Run: pytest tests/ -v
"""

import pytest
import numpy as np
import cv2
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ──────────────────────────────────────────────
# FIXTURES
# ──────────────────────────────────────────────

@pytest.fixture
def simple_image():
    """White background with dark rectangle and circle."""
    img = np.full((500, 700, 3), 240, dtype=np.uint8)
    cv2.rectangle(img, (200, 100), (500, 350), (30, 30, 30), -1)
    cv2.circle(img, (100, 400), 40, (30, 30, 30), -1)
    return img

@pytest.fixture
def gray_image(simple_image):
    return cv2.cvtColor(simple_image, cv2.COLOR_BGR2GRAY)

@pytest.fixture
def noisy_image():
    img = np.random.randint(180, 240, (400, 600, 3), dtype=np.uint8)
    cv2.circle(img, (200, 200), 80, (30, 30, 30), -1)
    cv2.rectangle(img, (350, 100), (550, 300), (30, 30, 30), -1)
    return img


# ──────────────────────────────────────────────
# PREPROCESSING TESTS
# ──────────────────────────────────────────────

class TestPreprocessing:
    def test_gaussian_filter(self, simple_image):
        from utils.preprocessing import preprocess_image
        result = preprocess_image(simple_image, method="gaussian")
        assert "gray" in result
        assert "denoised" in result
        assert "enhanced" in result
        assert result["gray"].shape == simple_image.shape[:2]

    def test_median_filter(self, simple_image):
        from utils.preprocessing import preprocess_image
        result = preprocess_image(simple_image, method="median")
        assert result["denoised"].dtype == np.uint8

    def test_bilateral_filter(self, simple_image):
        from utils.preprocessing import preprocess_image
        result = preprocess_image(simple_image, method="bilateral")
        assert result["denoised"] is not None

    def test_clahe_enhancement(self, simple_image):
        from utils.preprocessing import preprocess_image
        with_clahe = preprocess_image(simple_image, apply_clahe=True)
        without_clahe = preprocess_image(simple_image, apply_clahe=False)
        assert not np.array_equal(with_clahe["enhanced"], without_clahe["enhanced"])

    def test_even_kernel_corrected(self, simple_image):
        from utils.preprocessing import preprocess_image
        result = preprocess_image(simple_image, kernel_size=4)  # even → should auto-correct
        assert result["denoised"] is not None


# ──────────────────────────────────────────────
# SEGMENTATION TESTS
# ──────────────────────────────────────────────

class TestSegmentation:
    def test_otsu_threshold(self, gray_image):
        from utils.segmentation import segment_image
        result = segment_image(gray_image, threshold_method="otsu")
        assert set(np.unique(result["thresholded"])).issubset({0, 255})

    def test_adaptive_threshold(self, gray_image):
        from utils.segmentation import segment_image
        result = segment_image(gray_image, threshold_method="adaptive_gaussian")
        assert result["morphed"] is not None

    def test_morphology_close(self, gray_image):
        from utils.segmentation import segment_image
        result = segment_image(gray_image, morph_operation="close")
        assert result["morphed"].shape == gray_image.shape

    def test_invert(self, gray_image):
        from utils.segmentation import segment_image
        normal = segment_image(gray_image, invert=False)
        inverted = segment_image(gray_image, invert=True)
        assert not np.array_equal(normal["thresholded"], inverted["thresholded"])

    def test_grabcut(self, simple_image):
        from utils.segmentation import grabcut_segment
        mask = grabcut_segment(simple_image, iterations=3)
        assert mask.shape == simple_image.shape[:2]
        assert mask.dtype == np.uint8

    def test_kmeans(self, simple_image):
        from utils.segmentation import kmeans_segment
        small = cv2.resize(simple_image, (200, 150))
        mask, viz = kmeans_segment(small, k=2)
        assert mask.shape == small.shape[:2]
        assert viz.shape == small.shape

    def test_watershed(self, simple_image):
        from utils.segmentation import watershed_segment
        mask, dist, viz = watershed_segment(simple_image)
        assert mask.shape == simple_image.shape[:2]


# ──────────────────────────────────────────────
# EDGE DETECTION TESTS
# ──────────────────────────────────────────────

class TestEdgeDetection:
    def test_canny(self, gray_image):
        from utils.edge_detection import detect_edges
        edges = detect_edges(gray_image, method="canny")
        assert edges.shape == gray_image.shape
        assert edges.dtype == np.uint8

    def test_sobel(self, gray_image):
        from utils.edge_detection import detect_edges
        edges = detect_edges(gray_image, method="sobel")
        assert edges is not None

    def test_laplacian(self, gray_image):
        from utils.edge_detection import detect_edges
        edges = detect_edges(gray_image, method="laplacian")
        assert edges is not None


# ──────────────────────────────────────────────
# CONTOUR ANALYSIS TESTS
# ──────────────────────────────────────────────

class TestContourAnalysis:
    def test_find_contours(self, simple_image):
        from utils.contour_analysis import find_and_measure_contours
        from utils.segmentation import segment_image
        from utils.preprocessing import preprocess_image
        prep = preprocess_image(simple_image)
        seg = segment_image(prep["enhanced"])
        measurements = find_and_measure_contours(seg["morphed"], simple_image)
        assert len(measurements) >= 1
        assert "area_px" in measurements[0]
        assert "width_px" in measurements[0]
        assert "circularity" in measurements[0]

    def test_calibrated_measurements(self, simple_image):
        from utils.contour_analysis import find_and_measure_contours
        from utils.segmentation import segment_image
        from utils.preprocessing import preprocess_image
        prep = preprocess_image(simple_image)
        seg = segment_image(prep["enhanced"])
        measurements = find_and_measure_contours(seg["morphed"], simple_image, pixel_per_cm=50.0)
        assert measurements[0]["width_cm"] is not None
        assert measurements[0]["area_cm2"] is not None

    def test_reference_detection(self, simple_image):
        from utils.contour_analysis import detect_reference_object
        from utils.segmentation import segment_image
        from utils.preprocessing import preprocess_image
        prep = preprocess_image(simple_image)
        seg = segment_image(prep["enhanced"])
        ref = detect_reference_object(seg["morphed"], ref_type="circle")
        # Should find the circle and include confidence
        if ref:
            assert "confidence" in ref
            assert "all_candidates" in ref

    def test_min_area_filter(self, simple_image):
        from utils.contour_analysis import find_and_measure_contours
        from utils.segmentation import segment_image
        from utils.preprocessing import preprocess_image
        prep = preprocess_image(simple_image)
        seg = segment_image(prep["enhanced"])
        strict = find_and_measure_contours(seg["morphed"], simple_image, min_area_ratio=0.1)
        lenient = find_and_measure_contours(seg["morphed"], simple_image, min_area_ratio=0.001)
        assert len(strict) <= len(lenient)


# ──────────────────────────────────────────────
# CALIBRATION TESTS
# ──────────────────────────────────────────────

class TestCalibration:
    def test_pixel_ratio(self):
        from utils.calibration import calibrate_pixel_ratio
        ratio = calibrate_pixel_ratio(100, 2.5)
        assert ratio == 40.0

    def test_zero_size(self):
        from utils.calibration import calibrate_pixel_ratio
        assert calibrate_pixel_ratio(0, 2.5) is None
        assert calibrate_pixel_ratio(100, 0) is None

    def test_reference_sizes(self):
        from utils.calibration import REFERENCE_SIZES_CM
        assert "US Quarter (24.26mm)" in REFERENCE_SIZES_CM
        assert REFERENCE_SIZES_CM["US Quarter (24.26mm)"] == 2.426


# ──────────────────────────────────────────────
# SMART PROCESS TESTS
# ──────────────────────────────────────────────

class TestSmartProcess:
    def test_auto_process(self, simple_image):
        from utils.smart_process import auto_process
        result = auto_process(simple_image)
        assert "strategy_name" in result
        assert "measurements" in result
        assert "characteristics" in result
        assert len(result["measurements"]) >= 1

    def test_auto_process_noisy(self, noisy_image):
        from utils.smart_process import auto_process
        result = auto_process(noisy_image)
        assert len(result["measurements"]) >= 1

    def test_characteristics(self, simple_image):
        from utils.smart_process import analyze_image_characteristics
        chars = analyze_image_characteristics(simple_image)
        assert "brightness" in chars
        assert "contrast" in chars
        assert 0 <= chars["brightness"] <= 255


# ──────────────────────────────────────────────
# ADVANCED ANALYSIS TESTS
# ──────────────────────────────────────────────

class TestAdvancedAnalysis:
    def test_fft(self, gray_image):
        from utils.advanced_analysis import fft_analysis
        result = fft_analysis(gray_image)
        assert "magnitude" in result
        assert "phase" in result
        assert 0 <= result["low_freq_ratio"] <= 1

    def test_histogram(self, simple_image):
        from utils.advanced_analysis import color_histogram_analysis
        result = color_histogram_analysis(simple_image)
        assert "dominant_color" in result
        assert result["dominant_hue"] >= 0

    def test_hough_circles(self, gray_image):
        from utils.advanced_analysis import hough_circle_detection
        circles = hough_circle_detection(gray_image)
        assert isinstance(circles, list)

    def test_orb_features(self, gray_image):
        from utils.advanced_analysis import orb_feature_detection
        result = orb_feature_detection(gray_image)
        assert result["count"] >= 0
        assert "distribution" in result

    def test_blur_detection(self, gray_image):
        from utils.advanced_analysis import blur_detection
        result = blur_detection(gray_image)
        assert result["score"] >= 1
        assert result["score"] <= 5

    def test_noise_estimation(self, gray_image):
        from utils.advanced_analysis import noise_estimation
        result = noise_estimation(gray_image)
        assert result["sigma"] >= 0

    def test_shape_classification(self):
        from utils.advanced_analysis import classify_contour_shape
        # Rectangle contour
        rect = np.array([[[0,0]], [[100,0]], [[100,50]], [[0,50]]])
        shape, circ = classify_contour_shape(rect)
        assert shape == "Rectangle"


# ──────────────────────────────────────────────
# TEXTURE ANALYSIS TESTS
# ──────────────────────────────────────────────

class TestTextureAnalysis:
    def test_lbp(self, gray_image):
        from utils.texture_analysis import lbp_fast
        small = cv2.resize(gray_image, (200, 150))
        result = lbp_fast(small)
        assert "lbp_image" in result
        assert result["lbp_image"].shape == small.shape

    def test_gabor(self, gray_image):
        from utils.texture_analysis import gabor_filter_bank
        small = cv2.resize(gray_image, (200, 150))
        result = gabor_filter_bank(small)
        assert result["n_filters"] == 16
        assert result["energy_map"].shape == small.shape

    def test_glcm(self, gray_image):
        from utils.texture_analysis import glcm_features
        small = cv2.resize(gray_image, (100, 100))
        result = glcm_features(small)
        assert "contrast" in result["averaged"]
        assert "homogeneity" in result["averaged"]

    def test_morphological_gradient(self, gray_image):
        from utils.texture_analysis import morphological_gradient
        mg = morphological_gradient(gray_image)
        assert mg.shape == gray_image.shape

    def test_color_spaces(self, simple_image):
        from utils.texture_analysis import color_space_decomposition
        cs = color_space_decomposition(simple_image)
        assert len(cs) == 12  # 3 each for RGB, HSV, LAB, YCrCb


# ──────────────────────────────────────────────
# VALIDATION TESTS
# ──────────────────────────────────────────────

class TestValidation:
    def test_synthetic_image_generation(self):
        from utils.validation import generate_test_image
        objects = [{"type": "rect", "x": 200, "y": 100, "width": 150, "height": 100, "color": 40}]
        img, gt, ref = generate_test_image(objects)
        assert img.shape == (800, 1200, 3)
        assert len(gt) == 1
        assert gt[0]["width_px"] == 150

    def test_perspective_detection_circle(self):
        from utils.validation import detect_perspective_distortion
        # Perfect circle contour
        angles = np.linspace(0, 2*np.pi, 50)
        contour = np.array([[[int(100 + 50*np.cos(a)), int(100 + 50*np.sin(a))]] for a in angles], dtype=np.int32)
        result = detect_perspective_distortion(contour, "circle")
        assert not result["distorted"]

    def test_uncertainty_estimation(self):
        from utils.validation import estimate_measurement_uncertainty
        contour = np.array([[[0,0]], [[100,0]], [[100,80]], [[0,80]]], dtype=np.int32)
        result = estimate_measurement_uncertainty(contour)
        assert "confidence_pct" in result
        assert result["confidence_pct"] > 0


# ──────────────────────────────────────────────
# AUTH TESTS
# ──────────────────────────────────────────────

class TestAuth:
    def setup_method(self):
        """Use a temp database for each test."""
        import utils.auth as auth_mod
        auth_mod.DB_PATH = "/tmp/test_visionmeasure.db"
        auth_mod.init_db()

    def teardown_method(self):
        try:
            os.remove("/tmp/test_visionmeasure.db")
        except:
            pass

    def test_signup_success(self):
        from utils.auth import signup
        ok, msg = signup("testuser", "test@test.com", "password123")
        assert ok
        assert "created" in msg.lower()

    def test_signup_duplicate_username(self):
        from utils.auth import signup
        signup("dupuser", "dup1@test.com", "password123")
        ok, msg = signup("dupuser", "dup2@test.com", "password123")
        assert not ok
        assert "taken" in msg.lower()

    def test_signup_short_password(self):
        from utils.auth import signup
        ok, msg = signup("shortpw", "short@test.com", "abc")
        assert not ok

    def test_login_success(self):
        from utils.auth import signup, login
        signup("logintest", "login@test.com", "password123")
        ok, result = login("logintest", "password123")
        assert ok
        assert result["username"] == "logintest"

    def test_login_wrong_password(self):
        from utils.auth import signup, login
        signup("wrongpw", "wrong@test.com", "password123")
        ok, msg = login("wrongpw", "wrongpassword")
        assert not ok

    def test_login_by_email(self):
        from utils.auth import signup, login
        signup("emaillogin", "email@test.com", "password123")
        ok, result = login("email@test.com", "password123")
        assert ok


# ──────────────────────────────────────────────
# BATCH PROCESSOR TESTS
# ──────────────────────────────────────────────

class TestBatchProcessor:
    def test_validate_image_valid(self, tmp_path):
        from utils.batch_processor import validate_image
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        path = str(tmp_path / "test.png")
        cv2.imwrite(path, img)
        result = validate_image(path)
        assert result["valid"]
        assert result["width"] == 100

    def test_validate_image_nonexistent(self):
        from utils.batch_processor import validate_image
        result = validate_image("/nonexistent/path.jpg")
        assert not result["valid"]

    def test_validate_image_too_small(self, tmp_path):
        from utils.batch_processor import validate_image
        img = np.full((5, 5, 3), 128, dtype=np.uint8)
        path = str(tmp_path / "tiny.png")
        cv2.imwrite(path, img)
        result = validate_image(path)
        assert not result["valid"]
