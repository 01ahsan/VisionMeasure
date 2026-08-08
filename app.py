"""
VisionMeasure: Automatic Object Dimension and Area Estimation from Images
Professional Streamlit Application with Authentication & Smart Processing
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import io
import os
import uuid
import time
import re
from datetime import datetime

from utils.preprocessing import preprocess_image
from utils.segmentation import segment_image
from utils.edge_detection import detect_edges
from utils.contour_analysis import find_and_measure_contours, detect_reference_object
from utils.calibration import calibrate_pixel_ratio, QUICK_PRESETS, auto_pick_reference, calibrate_from_object_selection
from utils.visualization import draw_measurements, draw_reference_highlight
from utils.smart_process import auto_process, analyze_image_characteristics
from utils.segmentation import grabcut_segment, kmeans_segment, watershed_segment
from utils.advanced_analysis import (
    fft_analysis, color_histogram_analysis, hough_circle_detection,
    hough_line_detection, orb_feature_detection, blur_detection,
    classify_contour_shape, noise_estimation,
    draw_hough_circles, draw_hough_lines, draw_orb_keypoints,
)
from utils.auth import signup, login, get_guest_usage_count, record_guest_usage, save_analysis, get_analysis_history, get_user_analysis_count
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
GUEST_LIMIT = 3
APP_VERSION = "2.0.0"

st.set_page_config(
    page_title="VisionMeasure — Object Measurement Tool",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
# SESSION STATE INIT
# ──────────────────────────────────────────────
if "user" not in st.session_state:
    st.session_state.user = None
if "guest_id" not in st.session_state:
    st.session_state.guest_id = str(uuid.uuid4())
if "page" not in st.session_state:
    st.session_state.page = "home"
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

# ──────────────────────────────────────────────
# GLOBAL CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global */
    .stApp { font-family: 'Inter', sans-serif; }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Hero Section */
    .hero-container {
        text-align: center;
        padding: 60px 20px 40px;
        max-width: 900px;
        margin: 0 auto;
    }
    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 20px;
        text-transform: uppercase;
    }
    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 16px;
        background: linear-gradient(135deg, #ffffff, #a5b4fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.15rem;
        color: #94a3b8;
        line-height: 1.6;
        max-width: 650px;
        margin: 0 auto 36px;
    }

    /* Feature Cards */
    .features-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 20px;
        max-width: 1000px;
        margin: 40px auto;
        padding: 0 20px;
    }
    .feature-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 16px;
        padding: 28px 24px;
        text-align: left;
        transition: all 0.3s ease;
    }
    .feature-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
    }
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 14px;
        display: block;
    }
    .feature-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 8px;
    }
    .feature-desc {
        font-size: 0.88rem;
        color: #94a3b8;
        line-height: 1.5;
    }

    /* How it works */
    .steps-container {
        display: flex;
        justify-content: center;
        gap: 40px;
        max-width: 900px;
        margin: 30px auto;
        padding: 0 20px;
        flex-wrap: wrap;
    }
    .step-item {
        text-align: center;
        flex: 1;
        min-width: 180px;
        position: relative;
    }
    .step-number {
        width: 48px;
        height: 48px;
        background: linear-gradient(135deg, #6366f1, #4f46e5);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 1.2rem;
        color: white;
        margin: 0 auto 12px;
    }
    .step-title {
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 6px;
        font-size: 0.95rem;
    }
    .step-desc {
        font-size: 0.82rem;
        color: #94a3b8;
        line-height: 1.4;
    }

    /* Tech Pipeline */
    .pipeline-bar {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 6px;
        flex-wrap: wrap;
        max-width: 1000px;
        margin: 20px auto;
        padding: 0 20px;
    }
    .pipeline-step {
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: #a5b4fc;
        padding: 6px 14px;
        border-radius: 8px;
        font-size: 0.78rem;
        font-weight: 500;
    }
    .pipeline-arrow {
        color: #475569;
        font-size: 0.9rem;
    }

    /* Use Cases */
    .usecases-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 14px;
        max-width: 800px;
        margin: 20px auto;
    }
    .usecase-chip {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(148, 163, 184, 0.1);
        padding: 12px 16px;
        border-radius: 10px;
        text-align: center;
        font-size: 0.85rem;
        color: #cbd5e1;
    }
    .usecase-chip span {
        display: block;
        font-size: 1.4rem;
        margin-bottom: 4px;
    }

    /* Auth Card */
    .auth-card {
        max-width: 440px;
        margin: 40px auto;
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 20px;
        padding: 40px 36px;
    }
    .auth-title {
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 6px;
    }
    .auth-subtitle {
        text-align: center;
        font-size: 0.88rem;
        color: #94a3b8;
        margin-bottom: 28px;
    }

    /* Stats Bar */
    .stats-bar {
        display: flex;
        justify-content: center;
        gap: 50px;
        padding: 30px 20px;
        max-width: 700px;
        margin: 0 auto;
    }
    .stat-item {
        text-align: center;
    }
    .stat-value {
        font-size: 2rem;
        font-weight: 800;
        color: #6366f1;
    }
    .stat-label {
        font-size: 0.82rem;
        color: #94a3b8;
        margin-top: 4px;
    }

    /* Navbar */
    .navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 16px 30px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.1);
        margin-bottom: 10px;
    }
    .nav-brand {
        font-size: 1.3rem;
        font-weight: 700;
        color: #e2e8f0;
    }
    .nav-brand span {
        color: #6366f1;
    }

    /* Result Card */
    .result-metric {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 14px;
        padding: 22px;
        text-align: center;
    }
    .result-metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #6366f1;
    }
    .result-metric-label {
        font-size: 0.82rem;
        color: #94a3b8;
        margin-top: 4px;
    }

    /* Guest Banner */
    .guest-banner {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(139, 92, 246, 0.15));
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 16px 24px;
        text-align: center;
        margin: 16px 0;
    }
    .guest-banner p {
        margin: 0;
        color: #c7d2fe;
        font-size: 0.9rem;
    }

    /* Section headers */
    .section-header {
        text-align: center;
        margin: 50px 0 10px;
    }
    .section-header h2 {
        font-size: 1.8rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 8px;
    }
    .section-header p {
        color: #94a3b8;
        font-size: 0.95rem;
    }

    /* Mode selector */
    .mode-card {
        background: rgba(30, 41, 59, 0.5);
        border: 2px solid rgba(148, 163, 184, 0.1);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .mode-card:hover {
        border-color: rgba(99, 102, 241, 0.5);
    }
    .mode-icon { font-size: 2.2rem; margin-bottom: 10px; display: block; }
    .mode-name { font-weight: 600; color: #e2e8f0; font-size: 1rem; margin-bottom: 4px; }
    .mode-desc { font-size: 0.8rem; color: #94a3b8; }

    /* Footer */
    .app-footer {
        text-align: center;
        padding: 30px 20px;
        border-top: 1px solid rgba(148, 163, 184, 0.1);
        margin-top: 50px;
        color: #64748b;
        font-size: 0.82rem;
    }

    /* Responsive */
    @media (max-width: 768px) {
        .hero-title { font-size: 2rem; }
        .features-grid { grid-template-columns: 1fr; }
        .usecases-grid { grid-template-columns: repeat(2, 1fr); }
        .steps-container { gap: 20px; }
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# NAVBAR
# ──────────────────────────────────────────────
def render_navbar():
    c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
    with c1:
        st.markdown("### 📐 Vision**Measure**")
    with c2:
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()
    with c3:
        if st.session_state.user:
            if st.button("📊 Analyze", use_container_width=True):
                st.session_state.page = "analyze"
                st.rerun()
        else:
            if st.button("📊 Try Free", use_container_width=True):
                st.session_state.page = "analyze"
                st.rerun()
    with c4:
        if st.session_state.user:
            if st.button(f"👤 {st.session_state.user['username']}", use_container_width=True):
                st.session_state.page = "dashboard"
                st.rerun()
        else:
            if st.button("🔑 Sign In", use_container_width=True):
                st.session_state.page = "auth"
                st.rerun()
    st.markdown("---")


# ──────────────────────────────────────────────
# HOME PAGE
# ──────────────────────────────────────────────
def render_home():
    # Hero
    st.markdown("""
    <div class="hero-container">
        <div class="hero-badge">🔬 Digital Image Processing</div>
        <div class="hero-title">Measure Any Object<br>From a Single Photo</div>
        <div class="hero-subtitle">
            Upload an image with a reference object of known size.
            VisionMeasure automatically detects objects and estimates their
            real-world dimensions — length, width, area, and perimeter — using
            classical image processing techniques.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # CTA Buttons
    cta1, cta2, cta3 = st.columns([1, 1, 1])
    with cta1:
        if st.button("🚀 Start Measuring — Free", use_container_width=True, type="primary"):
            st.session_state.page = "analyze"
            st.rerun()
    with cta2:
        if st.button("📝 Create Free Account", use_container_width=True):
            st.session_state.page = "auth"
            st.rerun()
    with cta3:
        if st.button("📖 How It Works ↓", use_container_width=True):
            pass  # scrolls naturally

    # Pipeline visualization
    st.markdown("""
    <div class="section-header"><h2>Processing Pipeline</h2>
    <p>Every image goes through a complete DIP pipeline — fully visible at each stage</p></div>
    <div class="pipeline-bar">
        <div class="pipeline-step">📷 Input</div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step">🔇 Noise Removal</div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step">⬛ Thresholding</div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step">🔵 Morphology</div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step">📐 Edge Detection</div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step">🔲 Contour Extraction</div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step">📏 Measurement</div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step">🤖 ML Analysis</div>
    </div>
    """, unsafe_allow_html=True)

    # Features
    st.markdown("""
    <div class="section-header"><h2>Features</h2></div>
    <div class="features-grid">
        <div class="feature-card">
            <span class="feature-icon">🎯</span>
            <div class="feature-title">Smart Auto-Detection</div>
            <div class="feature-desc">Automatically analyzes image characteristics and selects the best processing strategy — handles varying lighting, backgrounds, and noise levels.</div>
        </div>
        <div class="feature-card">
            <span class="feature-icon">📏</span>
            <div class="feature-title">Real-World Calibration</div>
            <div class="feature-desc">Place a coin, card, or any reference of known size. The system calibrates pixel-to-centimeter ratio for accurate physical measurements.</div>
        </div>
        <div class="feature-card">
            <span class="feature-icon">🔬</span>
            <div class="feature-title">Full Pipeline Visibility</div>
            <div class="feature-desc">See every processing stage: grayscale, denoised, thresholded, morphology, edges, contours. Perfect for understanding and debugging DIP techniques.</div>
        </div>
        <div class="feature-card">
            <span class="feature-icon">🖼️</span>
            <div class="feature-title">Any Image, Any Size</div>
            <div class="feature-desc">Supports JPG, PNG, BMP, TIFF, WebP — from phone photos to high-res DSLR images. Auto-resizes for fast processing without quality loss.</div>
        </div>
        <div class="feature-card">
            <span class="feature-icon">📊</span>
            <div class="feature-title">Multi-Object Detection</div>
            <div class="feature-desc">Measures all objects in frame simultaneously. Export results as CSV or download annotated images with measurement overlays.</div>
        </div>
        <div class="feature-card">
            <span class="feature-icon">⚙️</span>
            <div class="feature-title">Manual Override</div>
            <div class="feature-desc">Auto mode not perfect? Switch to manual and fine-tune every parameter — threshold, morphology kernel, edge sensitivity — in real time.</div>
        </div>
        <div class="feature-card">
            <span class="feature-icon">🤖</span>
            <div class="feature-title">ML-Powered Analysis</div>
            <div class="feature-desc">SVM shape classification, BRISQUE image quality assessment, Isolation Forest anomaly detection, GMM color segmentation, PCA texture learning, and Bayesian parameter optimization.</div>
        </div>
        <div class="feature-card">
            <span class="feature-icon">🧩</span>
            <div class="feature-title">Superpixel & HOG</div>
            <div class="feature-desc">SLIC superpixel segmentation, HOG feature descriptors, DBSCAN spatial clustering, and Random Forest confidence scoring for every measurement.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # How it works
    st.markdown("""
    <div class="section-header"><h2>How It Works</h2></div>
    <div class="steps-container">
        <div class="step-item">
            <div class="step-number">1</div>
            <div class="step-title">Prepare Scene</div>
            <div class="step-desc">Place objects on a plain surface with a coin or card as reference</div>
        </div>
        <div class="step-item">
            <div class="step-number">2</div>
            <div class="step-title">Take Photo</div>
            <div class="step-desc">Shoot from directly above with even lighting</div>
        </div>
        <div class="step-item">
            <div class="step-number">3</div>
            <div class="step-title">Upload & Select</div>
            <div class="step-desc">Upload image, pick your reference object type and size</div>
        </div>
        <div class="step-item">
            <div class="step-number">4</div>
            <div class="step-title">Get Results</div>
            <div class="step-desc">View measurements, pipeline stages, and export data</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Use cases
    st.markdown("""
    <div class="section-header"><h2>Applications</h2>
    <p>From lab work to industry — measure anything that fits in a photo</p></div>
    <div class="usecases-grid">
        <div class="usecase-chip"><span>🌿</span>Leaf Area Measurement</div>
        <div class="usecase-chip"><span>🍎</span>Fruit & Produce Sizing</div>
        <div class="usecase-chip"><span>🩹</span>Wound Area Estimation</div>
        <div class="usecase-chip"><span>⚙️</span>Machine Part Inspection</div>
        <div class="usecase-chip"><span>🛒</span>E-Commerce Product Photos</div>
        <div class="usecase-chip"><span>🔬</span>Lab Sample Analysis</div>
    </div>
    """, unsafe_allow_html=True)

    # Guest vs Account
    st.markdown("""
    <div class="section-header"><h2>Pricing</h2></div>
    """, unsafe_allow_html=True)

    p1, p2 = st.columns(2)
    with p1:
        st.markdown("""
        <div class="feature-card" style="text-align:center;">
            <span class="feature-icon">🆓</span>
            <div class="feature-title">Guest</div>
            <div class="feature-desc" style="margin-top:10px;">
                ✓ 3 free analyses<br>
                ✓ Auto & manual modes<br>
                ✓ CSV & image export<br>
                ✗ No history<br>
                ✗ No batch processing
            </div>
        </div>
        """, unsafe_allow_html=True)
    with p2:
        st.markdown("""
        <div class="feature-card" style="text-align:center; border-color: rgba(99, 102, 241, 0.4);">
            <span class="feature-icon">⭐</span>
            <div class="feature-title">Free Account</div>
            <div class="feature-desc" style="margin-top:10px;">
                ✓ Unlimited analyses<br>
                ✓ Auto & manual modes<br>
                ✓ CSV & image export<br>
                ✓ Analysis history & dashboard<br>
                ✓ Batch processing
            </div>
        </div>
        """, unsafe_allow_html=True)

    # DIP Techniques
    st.markdown("""
    <div class="section-header"><h2>DIP Techniques Used</h2>
    <p>A comprehensive set of classical image processing methods</p></div>
    """, unsafe_allow_html=True)

    t1, t2, t3, t4 = st.columns(4)
    with t1:
        st.markdown("""
        **Preprocessing**
        - Gaussian filtering
        - Median filtering
        - Bilateral filtering
        - CLAHE enhancement

        **Segmentation**
        - Otsu thresholding
        - Adaptive (mean/Gaussian)
        - GrabCut / K-Means / Watershed
        - HSV color segmentation
        - GMM probabilistic clustering
        - SLIC superpixels
        """)
    with t2:
        st.markdown("""
        **Morphological Operations**
        - Erosion / Dilation
        - Opening / Closing
        - Morphological gradient
        - Open + Close combined

        **Edge & Feature Detection**
        - Canny / Sobel / Laplacian
        - Hough circles & lines
        - ORB keypoints
        - HOG descriptors
        """)
    with t3:
        st.markdown("""
        **Analysis & Measurement**
        - Contour detection (external)
        - Min-area bounding rectangle
        - Circularity & aspect ratio
        - Pixel-to-cm calibration
        - ArUco marker detection
        - Perspective correction
        - FFT frequency analysis
        - LBP / Gabor / GLCM texture
        - Color space decomposition
        """)
    with t4:
        st.markdown("""
        **Machine Learning**
        - SVM shape classifier (Hu+geo)
        - BRISQUE image quality (NR-IQA)
        - Isolation Forest anomalies
        - Bayesian threshold optimizer
        - KNN reference matching
        - PCA texture feature learning
        - Random Forest confidence
        - DBSCAN spatial clustering
        - Scene classification
        - Cosine similarity matrix
        """)

    # Footer
    st.markdown("""
    <div class="app-footer">
        <b>VisionMeasure v""" + APP_VERSION + """</b> — Digital Image Processing Lab Project<br>
        Built with OpenCV · Streamlit · Python<br><br>
        <a href="https://github.com/YOUR_USERNAME/VisionMeasure" target="_blank">GitHub</a>
        &nbsp;·&nbsp;
        <a href="https://scholar.google.com/citations?user=LlKQVegAAAAJ&hl=en" target="_blank">Google Scholar</a>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# AUTH PAGE
# ──────────────────────────────────────────────
def render_auth():
    st.markdown('<div class="auth-card">', unsafe_allow_html=True)

    # Social login buttons
    st.markdown("""
    <div style="text-align:center; margin-bottom: 24px;">
        <div style="font-size:1.6rem; font-weight:700; color:#e2e8f0; margin-bottom:6px;">Get Started</div>
        <div style="font-size:0.88rem; color:#94a3b8; margin-bottom:20px;">Sign in or create an account</div>
    </div>
    """, unsafe_allow_html=True)

    gc1, gc2 = st.columns(2)
    with gc1:
        st.markdown("""
        <a href="#" style="display:flex;align-items:center;justify-content:center;gap:8px;
            background:#1e293b;border:1px solid rgba(148,163,184,0.2);border-radius:10px;
            padding:12px;text-decoration:none;color:#e2e8f0;font-size:0.9rem;font-weight:500;
            transition:all 0.2s ease;" onmouseover="this.style.borderColor='rgba(99,102,241,0.5)'"
            onmouseout="this.style.borderColor='rgba(148,163,184,0.2)'">
            <svg width="18" height="18" viewBox="0 0 24 24"><path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z"/><path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/><path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/><path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/></svg>
            Google
        </a>
        """, unsafe_allow_html=True)
    with gc2:
        st.markdown("""
        <a href="#" style="display:flex;align-items:center;justify-content:center;gap:8px;
            background:#1e293b;border:1px solid rgba(148,163,184,0.2);border-radius:10px;
            padding:12px;text-decoration:none;color:#e2e8f0;font-size:0.9rem;font-weight:500;
            transition:all 0.2s ease;" onmouseover="this.style.borderColor='rgba(99,102,241,0.5)'"
            onmouseout="this.style.borderColor='rgba(148,163,184,0.2)'">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="#e2e8f0"><path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"/></svg>
            GitHub
        </a>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin:20px 0;">
        <div style="flex:1;height:1px;background:rgba(148,163,184,0.2);"></div>
        <div style="color:#64748b;font-size:0.8rem;">or continue with email</div>
        <div style="flex:1;height:1px;background:rgba(148,163,184,0.2);"></div>
    </div>
    """, unsafe_allow_html=True)

    # Determine which tab to show
    if "auth_tab" not in st.session_state:
        st.session_state.auth_tab = "login"

    tab_login, tab_signup = st.tabs(["🔑 Sign In", "📝 Create Account"])

    with tab_login:
        # Show success message if user just registered
        if st.session_state.get("signup_success"):
            st.success(f"✅ Account **{st.session_state.get('signup_username', '')}** created! Sign in below.")
            st.session_state.signup_success = False

        login_user = st.text_input("Username or Email", key="login_user",
                                    value=st.session_state.get("signup_username", ""))
        login_pass = st.text_input("Password", type="password", key="login_pass")

        if st.button("Sign In", use_container_width=True, type="primary", key="btn_login"):
            if not login_user or not login_pass:
                st.warning("Please enter both username/email and password.")
            else:
                success, result = login(login_user, login_pass)
                if success:
                    st.session_state.user = result
                    st.session_state.page = "analyze"
                    st.success(f"Welcome back, {result['username']}!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("Invalid username/email or password.")

    with tab_signup:
        s_fullname = st.text_input("Full Name *", key="s_name")
        s_email = st.text_input("Email *", key="s_email")
        s_username = st.text_input("Username *", key="s_user", help="Must be 3-20 characters, letters, numbers, underscores only")
        s_pass = st.text_input("Password *", type="password", key="s_pass", help="Min 8 characters with at least 1 letter and 1 number")
        s_pass2 = st.text_input("Confirm Password *", type="password", key="s_pass2")

        # Real-time validation feedback
        validation_errors = []
        if s_username:
            if len(s_username) < 3 or len(s_username) > 20:
                validation_errors.append("Username must be 3–20 characters")
            elif not re.match(r'^[a-zA-Z0-9_]+$', s_username):
                validation_errors.append("Username: letters, numbers, underscores only")
        if s_email:
            if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', s_email):
                validation_errors.append("Enter a valid email address")
        if s_pass:
            pw_issues = _validate_password(s_pass)
            validation_errors.extend(pw_issues)
        if s_pass and s_pass2 and s_pass != s_pass2:
            validation_errors.append("Passwords do not match")

        # Show validation inline
        if validation_errors:
            for err in validation_errors:
                st.caption(f"⚠️ {err}")

        if st.button("Create Account", use_container_width=True, type="primary", key="btn_signup"):
            if not all([s_fullname, s_email, s_username, s_pass, s_pass2]):
                st.warning("Please fill in all required fields.")
            elif validation_errors:
                st.error("Please fix the errors above.")
            else:
                success, msg = signup(s_username, s_email, s_pass, s_fullname, "")
                if success:
                    st.session_state.signup_success = True
                    st.session_state.signup_username = s_username
                    st.success("✅ Account created successfully! Please sign in with your credentials.")
                    st.balloons()
                    time.sleep(1.0)
                    # Force page reload — user will see login tab with success message
                    st.session_state.page = "auth"
                    st.rerun()
                else:
                    st.error(msg)

    st.markdown('</div>', unsafe_allow_html=True)


def _validate_password(password):
    """Validate password strength. Returns list of error messages."""
    errors = []
    if len(password) < 8:
        errors.append("Password must be at least 8 characters")
    if not re.search(r'[a-zA-Z]', password):
        errors.append("Password must contain at least one letter")
    if not re.search(r'[0-9]', password):
        errors.append("Password must contain at least one number")
    if len(password) > 0 and password.isspace():
        errors.append("Password cannot be only spaces")
    return errors


# ──────────────────────────────────────────────
# ANALYZE PAGE — Single + Batch + Sessions
# ──────────────────────────────────────────────
def render_analyze():
    from utils.batch_processor import validate_bulk_images, extract_zip, validate_image, cleanup_temp_dir
    from utils.session_manager import (
        create_session, save_session_result, update_session_stats,
        get_user_sessions, get_session_detail, get_session_statistics, delete_session,
    )

    is_guest = st.session_state.user is None

    if is_guest:
        used = get_guest_usage_count(st.session_state.guest_id)
        remaining = max(0, GUEST_LIMIT - used)
        st.markdown(f"""<div class="guest-banner"><p>👋 <b>Guest Mode</b> — {remaining}/{GUEST_LIMIT} free analyses. Create account for unlimited + batch + sessions.</p></div>""", unsafe_allow_html=True)
        if remaining <= 0:
            st.error("Guest limit reached.")
            if st.button("Create Free Account", type="primary"): st.session_state.page = "auth"; st.rerun()
            return

    if is_guest:
        tabs = st.tabs(["📷 Single Image"])
        with tabs[0]: _render_single_analysis(is_guest)
    else:
        tabs = st.tabs(["📷 Single Image", "📦 Batch / Zip Upload", "📁 Sessions"])
        with tabs[0]: _render_single_analysis(is_guest)
        with tabs[1]: _render_batch_analysis()
        with tabs[2]: _render_sessions()


def _render_single_analysis(is_guest):
    st.markdown("### 📷 Upload Image")
    st.caption("Supports JPG, JPEG, PNG, BMP, TIFF, WebP — any resolution, any size")
    uc, cc = st.columns([2, 1])
    with uc: uploaded = st.file_uploader("Drop image", type=["jpg","jpeg","png","bmp","tiff","tif","webp"], key="s_up")
    with cc: camera = st.camera_input("📸 Camera", key="s_cam")
    src = uploaded or camera
    if not src:
        st.info("Upload an image with objects and a reference item on a plain background.")
        return
    fb = np.asarray(bytearray(src.read()), dtype=np.uint8)
    original = cv2.imdecode(fb, cv2.IMREAD_COLOR)
    if original is None:
        st.error("Cannot read image — file may be corrupted.")
        return
    h, w = original.shape[:2]
    if max(h, w) > 2000:
        sc = 2000 / max(h, w)
        original = cv2.resize(original, (int(w*sc), int(h*sc)))
        h, w = original.shape[:2]
    st.image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), caption=f"Original — {w}×{h}", use_container_width=True)
    _run_full_analysis(original, getattr(src, "name", "capture.jpg"), is_guest)


def _render_batch_analysis():
    from utils.batch_processor import validate_bulk_images, extract_zip
    from utils.session_manager import create_session, save_session_result, update_session_stats
    from utils.advanced_analysis import blur_detection as bd, classify_contour_shape

    st.markdown("### 📦 Batch Processing")
    st.caption("Upload multiple images or a zip file with nested folders. Corrupted/non-image files are automatically identified and skipped.")

    session_name = st.text_input("Session Name", value=f"Batch {datetime.now().strftime('%Y-%m-%d %H:%M')}", key="bn")
    session_desc = st.text_input("Description (optional)", key="bd", placeholder="e.g. Leaf samples from Field A")

    upload_mode = st.radio("Upload mode", ["📁 Multiple Images", "📦 Zip File"], horizontal=True, key="bm")

    images_to_process = []
    skipped_files = []

    if "📁" in upload_mode:
        files = st.file_uploader("Upload images", type=["jpg","jpeg","png","bmp","tiff","tif","webp"], accept_multiple_files=True, key="bf")
        if files:
            r = validate_bulk_images(files)
            images_to_process = r["valid_images"]
            skipped_files = r["skipped"]
    else:
        zf = st.file_uploader("Upload zip", type=["zip"], key="bz")
        if zf:
            with st.spinner("Extracting zip..."):
                import tempfile
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
                tmp.write(zf.read()); tmp.close()
                r = extract_zip(tmp.name)
                if "error" in r:
                    st.error(r["error"])
                else:
                    with st.expander(f"📁 Archive: {r['total_files']} files found"):
                        _show_tree(r["folder_structure"], "")
                    st.info(f"✅ {r['total_images']} valid images | ⚠️ {len(r['skipped'])} skipped")
                    for img_info in r["image_files"]:
                        img = cv2.imread(img_info["path"], cv2.IMREAD_COLOR)
                        if img is not None:
                            h, w = img.shape[:2]
                            if max(h,w) > 2000:
                                sc = 2000/max(h,w); img = cv2.resize(img,(int(w*sc),int(h*sc))); h,w = img.shape[:2]
                            images_to_process.append({"image":img,"filename":img_info["filename"],"relative_path":img_info.get("relative_path",""),"width":w,"height":h,"filesize":img_info.get("filesize",0)})
                    skipped_files = [{"filename":s["filename"],"reason":s["reason"]} for s in r["skipped"]]
                os.remove(tmp.name)

    if not images_to_process: return

    if skipped_files:
        with st.expander(f"⚠️ {len(skipped_files)} skipped files"):
            for s in skipped_files: st.caption(f"❌ **{s['filename']}** — {s['reason']}")

    st.success(f"**{len(images_to_process)} images** ready")

    # Reference
    st.markdown("#### 📏 Batch Reference")
    rc1, rc2 = st.columns(2)
    with rc1:
        batch_cal = st.selectbox("Calibration", ["No calibration (pixels)", "Known reference size", "Manual px/cm"], key="bcal")
    with rc2:
        brc = None
        batch_ppcm = None
        if "Known reference" in batch_cal:
            preset = st.selectbox("Preset", list(QUICK_PRESETS.keys()), key="brn")
            preset_val = QUICK_PRESETS[preset]
            brc = st.number_input("Reference size (cm)", 0.1, 100.0,
                                   preset_val["size_cm"] if preset_val else 2.5, key="brc")
        elif "Manual" in batch_cal:
            batch_ppcm = st.number_input("Pixels per cm", 1.0, 10000.0, 50.0, key="bppcm")
    brt = "circle"  # Not used in new system

    if not st.button("🚀 Process All", type="primary", use_container_width=True, key="bgo"): return

    user_id = st.session_state.user["id"]
    sid = create_session(user_id, session_name, session_desc)
    prog = st.progress(0, "Starting...")
    ok = 0; fail = 0; total = len(images_to_process)

    for i, img_info in enumerate(images_to_process):
        fn = img_info["filename"]
        prog.progress((i+1)/total, f"Processing {i+1}/{total}: {fn}")
        try:
            res = auto_process(img_info["image"])
            meas = res["measurements"]
            ppc = batch_ppcm  # Use manual ppcm if provided
            if brc and not ppc:
                ref = detect_reference_object(res["seg_result"]["morphed"], ref_type=brt)
                ppc = calibrate_pixel_ratio(ref["size_px"], brc) if ref else None
            if ppc:
                meas = find_and_measure_contours(res["seg_result"]["morphed"], img_info["image"], pixel_per_cm=ppc, min_area_ratio=0.003)
            shapes = []
            for m in meas:
                sh, ci = classify_contour_shape(m["contour"])
                shapes.append({"shape":sh,"circularity":round(ci,3)})
            gray = cv2.cvtColor(img_info["image"], cv2.COLOR_BGR2GRAY)
            quality = bd(gray)
            ms = [{k:v for k,v in m.items() if k != "contour"} for m in meas]
            save_session_result(sid, {"filename":fn,"relative_path":img_info.get("relative_path",""),"image_width":img_info["width"],"image_height":img_info["height"],"filesize":img_info.get("filesize",0),"success":True,"strategy_used":res["strategy_name"],"objects_detected":len(meas),"measurements":ms,"shape_classes":shapes,"quality":quality,"processing_time":0})
            ok += 1
        except Exception as e:
            save_session_result(sid, {"filename":fn,"relative_path":img_info.get("relative_path",""),"image_width":img_info.get("width",0),"image_height":img_info.get("height",0),"success":False,"error_message":str(e)[:200],"objects_detected":0})
            fail += 1

    update_session_stats(sid, total, ok, fail, len(skipped_files))
    prog.progress(1.0, "Done!")
    st.markdown("---")
    st.markdown("### ✅ Batch Complete")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total",total); c2.metric("Success",ok); c3.metric("Failed",fail); c4.metric("Skipped",len(skipped_files))
    st.info(f"Saved to session: **{session_name}**. View in Sessions tab.")


def _show_tree(tree, indent):
    for dn, sub in tree.get("_dirs",{}).items():
        cnt = _tree_count(sub)
        st.caption(f"{indent}📁 **{dn}/** ({cnt} files)")
        _show_tree(sub, indent + "&nbsp;&nbsp;&nbsp;&nbsp;")
    for f in tree.get("_files",[])[:15]:
        st.caption(f"{indent}📄 {f}")
    rem = len(tree.get("_files",[])) - 15
    if rem > 0: st.caption(f"{indent}... +{rem} more")

def _tree_count(t):
    c = len(t.get("_files",[]));
    for s in t.get("_dirs",{}).values(): c += _tree_count(s)
    return c


def _render_sessions():
    from utils.session_manager import get_user_sessions, get_session_detail, get_session_statistics, delete_session

    st.markdown("### 📁 Sessions")
    uid = st.session_state.user["id"]
    sessions = get_user_sessions(uid)

    if not sessions:
        st.info("No sessions yet. Use Batch mode to create one.")
        return

    # Session selector
    names = [f"{s['name']} ({s['total_images']} imgs, {s['created_at'][:10]})" for s in sessions]
    sel = st.selectbox("Select session", names, key="sess_sel")
    sel_idx = names.index(sel)
    sid = sessions[sel_idx]["id"]

    session_info, results = get_session_detail(sid)
    if not session_info: st.error("Session not found."); return

    # Overview
    st.markdown(f"## 📊 {session_info['name']}")
    if session_info.get("description"): st.caption(session_info["description"])

    o1,o2,o3,o4,o5 = st.columns(5)
    o1.metric("Total", session_info["total_images"])
    o2.metric("Success", session_info["successful"])
    o3.metric("Failed", session_info["failed"])
    o4.metric("Skipped", session_info["skipped"])
    total_obj = sum(r.get("objects_detected",0) for r in results)
    o5.metric("Objects", total_obj)

    stats = get_session_statistics(sid)
    if not stats: return

    stab, ftab, ctab = st.tabs(["📊 Statistics", "📄 Files", "📈 Charts"])

    with stab:
        ds1,ds2,ds3 = st.columns(3)
        ds1.metric("Avg Obj/Image", f"{stats['avg_objects_per_image']:.1f}")
        ds2.metric("Avg Time", f"{stats['avg_processing_time']:.2f}s")
        ds3.metric("Total Time", f"{stats['total_processing_time']:.1f}s")
        if stats["strategies_used"]:
            st.markdown("**Strategies:**")
            st.dataframe(pd.DataFrame([{"Strategy":k,"Count":v} for k,v in stats["strategies_used"].items()]), use_container_width=True, hide_index=True)
        if stats["shapes_found"]:
            st.markdown("**Shapes:**")
            st.dataframe(pd.DataFrame([{"Shape":k,"Count":v} for k,v in sorted(stats["shapes_found"].items(),key=lambda x:-x[1])]), use_container_width=True, hide_index=True)
        if "area_stats" in stats:
            st.markdown("**Measurements:**")
            a = stats["area_stats"]
            a1,a2,a3,a4 = st.columns(4)
            a1.metric("Mean Area", f"{a['mean']:.2f}"); a2.metric("Median", f"{a['median']:.2f}")
            a3.metric("Std", f"{a['std']:.2f}"); a4.metric("Range", f"{a['min']:.1f}—{a['max']:.1f}")

    with ftab:
        fd = [{"Status":"✅" if r.get("success") else "❌","File":r.get("filename",""),"Path":r.get("relative_path",""),
               "Size":f"{r.get('image_width',0)}×{r.get('image_height',0)}","Objects":r.get("objects_detected",0),
               "Strategy":r.get("strategy_used",""),"Error":r.get("error_message","")} for r in results]
        st.dataframe(pd.DataFrame(fd), use_container_width=True, hide_index=True)
        csv = pd.DataFrame(fd).to_csv(index=False)
        st.download_button("📥 Export CSV", csv, f"session_{session_info['name']}.csv", "text/csv")

    with ctab:
        objs = [r.get("objects_detected",0) for r in results if r.get("success")]
        if objs:
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            fig.patch.set_facecolor('#0f172a')
            axes[0].hist(objs, bins=range(0,max(objs)+2), color="#6366f1", alpha=0.8, edgecolor="white")
            axes[0].set_title("Objects/Image", color="white"); axes[0].set_facecolor('#1e293b'); axes[0].tick_params(colors='#94a3b8')
            if "all_areas" in stats and stats["all_areas"]:
                axes[1].hist(stats["all_areas"], bins=20, color="#10b981", alpha=0.8, edgecolor="white")
                axes[1].set_title("Area Distribution", color="white")
            axes[1].set_facecolor('#1e293b'); axes[1].tick_params(colors='#94a3b8')
            if stats["strategies_used"]:
                labels = [l[:18] for l in stats["strategies_used"].keys()]
                sizes = list(stats["strategies_used"].values())
                colors = ["#6366f1","#10b981","#f59e0b","#ef4444","#8b5cf6","#06b6d4"]
                axes[2].pie(sizes, labels=labels, autopct="%1.0f%%", colors=colors[:len(sizes)], textprops={"color":"white","fontsize":8})
                axes[2].set_title("Strategies", color="white")
            plt.tight_layout(); st.pyplot(fig); plt.close()
        if "all_widths" in stats and stats["all_widths"]:
            fig2, ax = plt.subplots(figsize=(8,4)); fig2.patch.set_facecolor('#0f172a')
            ax.scatter(stats["all_widths"], stats["all_heights"], c="#6366f1", alpha=0.6, s=40, edgecolors="white", linewidth=0.5)
            ax.set_xlabel("Width",color="#94a3b8"); ax.set_ylabel("Height",color="#94a3b8")
            ax.set_title("Width vs Height", color="white"); ax.set_facecolor('#1e293b'); ax.tick_params(colors='#94a3b8')
            plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.markdown("---")
    if st.button("🗑️ Delete Session", key=f"del_{sid}"):
        delete_session(sid); st.session_state.pop("view_session_id", None); st.success("Deleted."); st.rerun()


def _run_full_analysis(original, filename, is_guest):
    """Full analysis UI for a single image — redesigned workflow."""
    from utils.texture_analysis import lbp_fast, gabor_filter_bank, glcm_features, morphological_gradient, color_space_decomposition
    from utils.calibration import QUICK_PRESETS, calibrate_from_object_selection, auto_pick_reference, UNIT_LABELS
    from utils.smart_features import (
        grade_objects_by_size, check_compliance, draw_compliance_overlay,
        compare_objects, isolate_object_at_point, analyze_size_distribution,
        crop_individual_objects, analyze_spacing, convex_hull_analysis,
        generate_report_data,
    )

    st.markdown("---")

    # ── STEP 1: MODE SELECTION ──
    st.markdown("### Step 1 — Choose Processing Mode")
    mode = st.radio("", ["🤖 Auto (recommended)","🔧 Manual","🩹 Wound/Anomaly"],
                    horizontal=True, label_visibility="collapsed", key=f"m_{filename}",
                    help="Auto tries multiple strategies and picks the best. Manual lets you tune every parameter.")
    is_auto = "Auto" in mode
    is_wound = "Wound" in mode

    # ── STEP 2: CALIBRATION ──
    st.markdown("### Step 2 — Calibrate Scale")
    st.caption("Tell VisionMeasure the real-world size of something in the image so it can measure in cm/mm/inches instead of pixels. Skip this step to get pixel-only measurements.")

    cal_mode = st.radio("Calibration method", [
        "⏭️ Skip (pixels only)",
        "📏 I know a dimension — I'll pick the object after detection",
        "🔢 I know the pixels-per-cm ratio already",
    ], key=f"cal_{filename}", label_visibility="collapsed")

    ref_size_cm = None
    manual_ppcm = None
    auto_detect_ref = False
    ref_type = "circle"  # Not used in new system but kept for compat
    cal_dimension_type = "width"
    cal_unit = "cm"

    if "I know a dimension" in cal_mode:
        c1, c2, c3 = st.columns(3)
        with c1:
            preset = st.selectbox("Quick preset", list(QUICK_PRESETS.keys()), key=f"preset_{filename}")
            preset_val = QUICK_PRESETS[preset]
        with c2:
            cal_unit = st.selectbox("Unit", ["cm", "mm", "inches"], key=f"unit_{filename}")
            if preset_val:
                default_size = preset_val["size_cm"]
                if cal_unit == "mm": default_size *= 10
                elif cal_unit == "inches": default_size /= 2.54
            else:
                default_size = 5.0
            ref_size_input = st.number_input(f"Real size ({cal_unit})", 0.01, 10000.0, default_size, key=f"refsize_{filename}")
        with c3:
            cal_dimension_type = st.selectbox("Which dimension?", ["width", "height", "diameter", "diagonal"], key=f"dim_{filename}",
                                               help="Which dimension of the reference object does your size value correspond to?")

        # Convert to cm internally
        if cal_unit == "mm":
            ref_size_cm = ref_size_input / 10.0
        elif cal_unit == "inches":
            ref_size_cm = ref_size_input * 2.54
        else:
            ref_size_cm = ref_size_input

        auto_detect_ref = True  # Will let user pick after detection
        st.info("After detection, you'll be able to pick which object is your reference.")

    elif "pixels-per-cm" in cal_mode:
        manual_ppcm = st.number_input("Pixels per cm", 1.0, 10000.0, 50.0, key=f"ppcm_{filename}",
                                       help="If you've previously calibrated or calculated this value.")
        ref_size_cm = None  # Not needed

    # ── STEP 3: ADDITIONAL SETTINGS ──
    # Wound mode settings
    if is_wound:
        with st.expander("🩹 Wound/Anomaly Settings", expanded=True):
            wound_type = st.selectbox("Detection target", ["Wound/Lesion", "Red anomaly", "Brown/Rust", "Dark spot", "Yellow discoloration"], key=f"wt_{filename}")
            wound_sensitivity = st.select_slider("Sensitivity", ["low", "medium", "high"], value="medium", key=f"ws_{filename}")

    use_roi = st.checkbox("✂️ Crop ROI", False, key=f"roi_{filename}")
    if use_roi:
        h,w = original.shape[:2]
        rc = st.columns(4)
        rx=rc[0].slider("Left%",0,90,0,key=f"rx_{filename}"); ry=rc[1].slider("Top%",0,90,0,key=f"ry_{filename}")
        rw=rc[2].slider("W%",10,100,100,key=f"rw_{filename}"); rh=rc[3].slider("H%",10,100,100,key=f"rh_{filename}")
        original = original[int(h*ry/100):min(h,int(h*(ry+rh)/100)), int(w*rx/100):min(w,int(w*(rx+rw)/100))]

    if not is_auto:
        with st.expander("🔧 Manual Parameters", expanded=True):
            m1,m2 = st.columns(2)
            with m1:
                filter_method=st.selectbox("Filter",["gaussian","median","bilateral"],key=f"mf_{filename}")
                kernel_size=st.slider("Kernel",3,15,5,step=2,key=f"mk_{filename}")
                apply_clahe=st.checkbox("CLAHE",True,key=f"mc_{filename}")
                threshold_method=st.selectbox("Threshold",["otsu","adaptive_gaussian","adaptive_mean","manual"],key=f"mt_{filename}")
                manual_thresh=st.slider("Val",0,255,127,key=f"mv_{filename}") if threshold_method=="manual" else 127
                adaptive_block=st.slider("Block",3,51,11,step=2,key=f"mb_{filename}") if threshold_method.startswith("adaptive") else 11
                adaptive_c=st.slider("C",0,20,2,key=f"mac_{filename}") if threshold_method.startswith("adaptive") else 2
                invert_binary=st.checkbox("Invert",False,key=f"mi_{filename}")
            with m2:
                morph_op=st.selectbox("Morph",["close","open","open_close","dilate","erode","none"],key=f"mm_{filename}")
                morph_kernel=st.slider("M.Kern",3,15,5,step=2,key=f"mmk_{filename}")
                morph_iter=st.slider("Iter",1,5,2,key=f"mmi_{filename}")
                edge_method=st.selectbox("Edge",["canny","sobel","laplacian"],key=f"me_{filename}")
                canny_low=st.slider("Low",0,255,50,key=f"mcl_{filename}") if edge_method=="canny" else 50
                canny_high=st.slider("High",0,255,150,key=f"mch_{filename}") if edge_method=="canny" else 150
                min_area=st.slider("Min%",0.01,5.0,0.1,step=0.01,key=f"ma_{filename}")
                max_objects=st.slider("MaxObj",1,30,10,key=f"mo_{filename}")

    if not st.button("🔍 Analyze", type="primary", use_container_width=True, key=f"go_{filename}"): return

    prog = st.progress(0, "Processing...")
    h, w = original.shape[:2]
    wound_result = None

    if is_wound:
        from utils.wound_segmentation import segment_wound, segment_color_anomaly, measure_wound_from_contours
        prog.progress(20, "Detecting wound region...")
        chars = analyze_image_characteristics(original)
        prep_result = preprocess_image(original)
        edge_result = detect_edges(prep_result["enhanced"])
        pixel_per_cm = None
        ref_info = None

        if "Wound" in wound_type:
            wound_result = segment_wound(original, sensitivity=wound_sensitivity)
        else:
            color_map = {"Red anomaly":"red", "Brown/Rust":"brown", "Dark spot":"dark", "Yellow discoloration":"yellow"}
            anom = segment_color_anomaly(original, target_color=color_map.get(wound_type,"red"), sensitivity=wound_sensitivity)
            wound_result = {"wound_mask":anom["mask"], "overlay":anom["overlay"], "wound_contours":anom["contours"],
                            "wound_area_px":anom["area_px"], "skin_area_px":h*w, "wound_ratio":anom["area_px"]/(h*w),
                            "skin_mask":np.ones((h,w),dtype=np.uint8)*255, "channel_masks":{}}

        seg_result = {"thresholded": wound_result["wound_mask"], "morphed": wound_result["wound_mask"]}
        strategy_name = f"Wound ({wound_sensitivity})"

        # Measure wound contours
        prog.progress(60, "Measuring wound...")
        pixel_per_cm = None
        if ref_size_cm and ref_size_cm > 0 and auto_detect_ref:
            ref_info = detect_reference_object(cv2.bitwise_not(wound_result["skin_mask"]), ref_type=ref_type)
            if ref_info:
                pixel_per_cm = calibrate_pixel_ratio(ref_info["size_px"], ref_size_cm)

        wound_meas = measure_wound_from_contours(wound_result["wound_contours"], pixel_per_cm=pixel_per_cm)

        # Convert wound measurement to same format as regular measurements
        measurements = []
        if wound_meas:
            m = {"width_px": wound_meas["width_px"], "height_px": wound_meas["height_px"],
                 "area_px": wound_meas["area_px"], "perimeter_px": wound_meas["perimeter_px"],
                 "centroid": wound_meas["centroid"], "min_rect": wound_meas["bounding_rect"],
                 "circularity": 0, "aspect_ratio": wound_meas["width_px"]/(wound_meas["height_px"]+1e-8),
                 "bbox": (0,0,w,h),
                 "width_cm": wound_meas.get("width_cm"), "height_cm": wound_meas.get("height_cm"),
                 "area_cm2": wound_meas.get("area_cm2"), "perimeter_cm": wound_meas.get("perimeter_cm")}
            # Need a contour for shape classification
            if wound_result["wound_contours"]:
                m["contour"] = np.vstack(wound_result["wound_contours"])
            else:
                m["contour"] = np.array([[[0,0]]])
            measurements = [m]

        annotated = wound_result["overlay"]
        ref_info = None

    elif is_auto:
        prog.progress(20, "Testing strategies...")
        auto_result = auto_process(original)
        prep_result=auto_result["prep_result"]; seg_result=auto_result["seg_result"]
        strategy_name=auto_result["strategy_name"]; chars=auto_result["characteristics"]
        edge_result = detect_edges(prep_result["enhanced"])

        prog.progress(60, "Calibrating...")
        pixel_per_cm = manual_ppcm  # Use manual ppcm if provided
        ref_info = None

        # First pass: measure in pixels to get contours
        prog.progress(70, "Detecting objects...")
        measurements = find_and_measure_contours(seg_result["morphed"], original, pixel_per_cm=None, min_area_ratio=0.003, max_objects=20)

        # If user wants to pick a reference object, show picker AFTER detection
        if ref_size_cm and ref_size_cm > 0 and auto_detect_ref and measurements and pixel_per_cm is None:
            ref_candidates = auto_pick_reference([m["contour"] for m in measurements],
                                                  original.shape[0] * original.shape[1])
            if ref_candidates:
                # Show object picker
                obj_labels = [f"Object {c['index']+1} — {c['shape']} ({c['width_px']:.0f}×{c['height_px']:.0f} px, conf: {c['confidence']:.0%})"
                              for c in ref_candidates[:8]]
                obj_labels.insert(0, "Auto-pick best candidate")
                sel = st.selectbox("🎯 Which object is your reference?", obj_labels, key=f"refpick_{filename}",
                                    help="Select the object whose real-world size you entered above.")
                if sel == "Auto-pick best candidate":
                    ref_idx = ref_candidates[0]["index"]
                else:
                    sel_i = obj_labels.index(sel) - 1
                    ref_idx = ref_candidates[sel_i]["index"]

                ref_contour = measurements[ref_idx]["contour"]
                pixel_per_cm = calibrate_from_object_selection(ref_contour, ref_size_cm, cal_dimension_type)
                ref_info = {"contour": ref_contour, "center": measurements[ref_idx]["centroid"],
                            "size_px": measurements[ref_idx]["width_px"]}

        # Re-measure with calibration
        if pixel_per_cm:
            measurements = find_and_measure_contours(seg_result["morphed"], original, pixel_per_cm=pixel_per_cm, min_area_ratio=0.003, max_objects=20)
            # Remove the reference object from results
            if ref_info and measurements:
                rcc = ref_info["center"]
                measurements = [m for m in measurements if np.linalg.norm(np.array(m["centroid"])-np.array(rcc)) > 30]

        annotated = draw_measurements(original, measurements)
        if ref_info: annotated = draw_reference_highlight(annotated, ref_info, ref_size_cm or 0)

    else:
        prog.progress(15, "Preprocessing...")
        prep_result = preprocess_image(original, method=filter_method, kernel_size=kernel_size, apply_clahe=apply_clahe)
        prog.progress(35, "Segmenting...")
        seg_result = segment_image(prep_result["enhanced"], threshold_method=threshold_method, adaptive_block_size=adaptive_block, adaptive_c=adaptive_c, manual_threshold=manual_thresh, morph_operation=morph_op, morph_kernel_size=morph_kernel, morph_iterations=morph_iter, invert=invert_binary)
        edge_result = detect_edges(prep_result["enhanced"], method=edge_method, canny_low=canny_low, canny_high=canny_high)
        strategy_name="Manual"; chars=analyze_image_characteristics(original)

        prog.progress(60, "Calibrating...")
        pixel_per_cm = manual_ppcm
        ref_info = None

        prog.progress(70, "Detecting objects...")
        m_min = min_area/100
        measurements = find_and_measure_contours(seg_result["morphed"], original, pixel_per_cm=None, min_area_ratio=m_min, max_objects=max_objects)

        if ref_size_cm and ref_size_cm > 0 and auto_detect_ref and measurements and pixel_per_cm is None:
            ref_candidates = auto_pick_reference([m["contour"] for m in measurements],
                                                  original.shape[0] * original.shape[1])
            if ref_candidates:
                obj_labels = [f"Object {c['index']+1} — {c['shape']} ({c['width_px']:.0f}×{c['height_px']:.0f} px)"
                              for c in ref_candidates[:8]]
                obj_labels.insert(0, "Auto-pick best candidate")
                sel = st.selectbox("🎯 Which object is your reference?", obj_labels, key=f"refpick_m_{filename}")
                if sel == "Auto-pick best candidate":
                    ref_idx = ref_candidates[0]["index"]
                else:
                    sel_i = obj_labels.index(sel) - 1
                    ref_idx = ref_candidates[sel_i]["index"]
                ref_contour = measurements[ref_idx]["contour"]
                pixel_per_cm = calibrate_from_object_selection(ref_contour, ref_size_cm, cal_dimension_type)
                ref_info = {"contour": ref_contour, "center": measurements[ref_idx]["centroid"],
                            "size_px": measurements[ref_idx]["width_px"]}

        if pixel_per_cm:
            measurements = find_and_measure_contours(seg_result["morphed"], original, pixel_per_cm=pixel_per_cm, min_area_ratio=m_min, max_objects=max_objects)
            if ref_info and measurements:
                rcc = ref_info["center"]
                measurements = [m for m in measurements if np.linalg.norm(np.array(m["centroid"])-np.array(rcc)) > 30]

        annotated = draw_measurements(original, measurements)
        if ref_info: annotated = draw_reference_highlight(annotated, ref_info, ref_size_cm or 0)

    if is_guest: record_guest_usage(st.session_state.guest_id)

    prog.progress(100, "Done!"); time.sleep(0.3); prog.empty()

    # ── RESULTS ──
    st.markdown("---")
    st.markdown("## 📊 Results")

    with st.expander("📋 Image Info"):
        ic = st.columns(5)
        ic[0].metric("Size", f"{w}×{h}"); ic[1].metric("Bright", f"{chars['brightness']:.0f}")
        ic[2].metric("Contrast", f"{chars['contrast']:.0f}"); ic[3].metric("Strategy", strategy_name)
        ic[4].metric("Colorful", "Yes" if chars.get("is_colorful") else "No")
        if pixel_per_cm: st.success(f"✅ Calibrated: {pixel_per_cm:.2f} px/cm")
        elif ref_size_cm is None: st.info("ℹ️ No reference selected — measurements in pixels.")
        else: st.warning("⚠️ Reference not detected — pixel units only.")
        if is_auto and not is_wound:
            try:
                strats = auto_result.get("all_strategies",[])
                if strats: st.dataframe(pd.DataFrame(strats), use_container_width=True, hide_index=True)
            except Exception:
                pass

    # ── Wound-specific results ──
    if is_wound and wound_result:
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Wound Detection Overlay", use_container_width=True)

        wc = st.columns(4)
        wc[0].metric("Wound Area", f"{wound_result['wound_area_px']:,} px²")
        wc[1].metric("Skin Area", f"{wound_result['skin_area_px']:,} px²")
        wc[2].metric("Wound/Skin", f"{wound_result['wound_ratio']*100:.2f}%")
        wc[3].metric("Regions", len(wound_result.get("wound_contours", [])))

        if measurements and measurements[0].get("width_cm"):
            st.success(f"Wound: {measurements[0]['width_cm']:.2f} × {measurements[0]['height_cm']:.2f} cm — Area: {measurements[0]['area_cm2']:.2f} cm²")
        elif measurements:
            st.info(f"Wound: {measurements[0]['width_px']:.0f} × {measurements[0]['height_px']:.0f} px — Area: {measurements[0]['area_px']:.0f} px²")

        # Channel masks visualization
        ch_masks = wound_result.get("channel_masks", {})
        if ch_masks:
            with st.expander("🔬 Detection Channel Breakdown"):
                st.caption("Each channel contributes to wound detection. Combined weighted result gives the final mask.")
                cols = st.columns(len(ch_masks))
                for i, (name, mask) in enumerate(ch_masks.items()):
                    cols[i].image(mask, caption=name, use_container_width=True, clamp=True)

        # Skin mask
        with st.expander("🧬 Skin Detection"):
            sk1, sk2 = st.columns(2)
            sk1.image(wound_result["skin_mask"], caption="Skin Mask", use_container_width=True, clamp=True)
            sk2.image(wound_result["wound_mask"], caption="Wound Mask", use_container_width=True, clamp=True)

        st.markdown("---")
        st.caption("⚕️ **Disclaimer:** VisionMeasure is not a medical device. Wound measurements are approximate and should not replace professional clinical assessment. Always consult a healthcare provider.")

    else:
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=f"{len(measurements)} objects", use_container_width=True)
    if not measurements:
        st.warning("No objects detected. Try Manual mode, ROI crop, or invert threshold.")
        return

    # Metrics
    mc = st.columns(4); mc[0].metric("Objects", len(measurements))
    if measurements[0].get("width_cm"):
        mc[1].metric("Avg W", f"{np.mean([m['width_cm'] for m in measurements]):.2f}cm")
        mc[2].metric("Avg A", f"{np.mean([m['area_cm2'] for m in measurements]):.2f}cm²")
        mc[3].metric("Total A", f"{np.sum([m['area_cm2'] for m in measurements]):.2f}cm²")

    # Table
    td = []
    for i, m in enumerate(measurements):
        row = {"#": i+1}
        if m.get("width_cm") is not None:
            row.update({"W(cm)":round(m["width_cm"],3),"H(cm)":round(m["height_cm"],3),"Area(cm²)":round(m["area_cm2"],3),"Perim(cm)":round(m["perimeter_cm"],3)})
        else:
            row.update({"W(px)":round(m["width_px"],1),"H(px)":round(m["height_px"],1),"Area(px²)":round(m["area_px"],1),"Perim(px)":round(m["perimeter_px"],1)})
        row.update({"Circ":round(m["circularity"],3),"Aspect":round(m["aspect_ratio"],3)})
        td.append(row)
    df = pd.DataFrame(td)
    st.dataframe(df, use_container_width=True, hide_index=True)

    e1,e2 = st.columns(2)
    with e1: st.download_button("📄 CSV", df.to_csv(index=False), f"results_{filename}.csv", "text/csv", key=f"csv_{filename}")
    with e2:
        buf = io.BytesIO(); Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)).save(buf, format="PNG")
        st.download_button("🖼️ Image", buf.getvalue(), f"annotated_{filename}.png", "image/png", key=f"img_{filename}")

    # ── UNIQUE FEATURES ──
    st.markdown("---")
    st.markdown("## 🔥 Smart Features")

    feat_tabs = st.tabs(["📊 Size Grading", "✅ Compliance Check", "🔍 Compare Objects",
                          "📐 Size Distribution", "🗺️ Spacing Analysis", "✂️ Object Crops"])

    with feat_tabs[0]:
        if len(measurements) >= 2:
            gc1, gc2 = st.columns(2)
            with gc1:
                n_grades = st.slider("Number of grades", 2, 5, 3, key=f"ng_{filename}")
                grade_metric = st.selectbox("Grade by", ["area", "width", "height"], key=f"gm_{filename}")
            with gc2:
                use_custom = st.checkbox("Custom thresholds", False, key=f"gc_{filename}")
                custom_thresh = None
                if use_custom:
                    thresh_str = st.text_input("Thresholds (comma separated)", key=f"gt_{filename}",
                                                placeholder="e.g. 100, 500, 2000")
                    if thresh_str:
                        try:
                            custom_thresh = [float(x.strip()) for x in thresh_str.split(",")]
                        except ValueError:
                            st.warning("Enter numbers separated by commas.")

            grading = grade_objects_by_size(measurements, n_grades=n_grades,
                                            custom_thresholds=custom_thresh, grade_by=grade_metric)
            # Color-coded graded image
            grade_img = original.copy()
            grade_colors = [(0,200,0), (255,200,0), (0,140,255), (0,0,255), (200,0,200)]
            for g in grading["graded_objects"]:
                idx = g["index"]
                if idx < len(measurements):
                    grade_idx = list(grading["grade_distribution"].keys()).index(g["grade"]) if g["grade"] in grading["grade_distribution"] else 0
                    color = grade_colors[grade_idx % len(grade_colors)]
                    cv2.drawContours(grade_img, [measurements[idx]["contour"]], -1, color, 3)
                    cv2.putText(grade_img, g["grade"], measurements[idx]["centroid"],
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            st.image(cv2.cvtColor(grade_img, cv2.COLOR_BGR2RGB), caption="Size Grading", use_container_width=True)

            gd_data = [{"#": g["index"]+1, "Grade": g["grade"], "Value": g["value"]} for g in grading["graded_objects"]]
            st.dataframe(pd.DataFrame(gd_data), use_container_width=True, hide_index=True)
            if grading["grade_summary"]:
                st.markdown("**Grade Summary:**")
                gs_rows = [{"Grade": k, "Count": v["count"], "Min": v["min"], "Max": v["max"], "Mean": v["mean"]}
                           for k, v in grading["grade_summary"].items()]
                st.dataframe(pd.DataFrame(gs_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Need at least 2 objects for size grading.")

    with feat_tabs[1]:
        if measurements:
            st.markdown("Set min/max dimensional tolerances to check which objects pass inspection.")
            is_cal = measurements[0].get("width_cm") is not None
            unit_l = "cm" if is_cal else "px"
            sc1, sc2 = st.columns(2)
            with sc1:
                spec_min_w = st.number_input(f"Min width ({unit_l})", 0.0, 99999.0, 0.0, key=f"sw1_{filename}")
                spec_max_w = st.number_input(f"Max width ({unit_l})", 0.0, 99999.0, 99999.0, key=f"sw2_{filename}")
                spec_min_circ = st.number_input("Min circularity", 0.0, 1.0, 0.0, key=f"sc1_{filename}")
            with sc2:
                spec_min_a = st.number_input(f"Min area ({unit_l}²)", 0.0, 99999.0, 0.0, key=f"sa1_{filename}")
                spec_max_a = st.number_input(f"Max area ({unit_l}²)", 0.0, 99999.0, 99999.0, key=f"sa2_{filename}")
                spec_max_ar = st.number_input("Max aspect ratio", 1.0, 50.0, 50.0, key=f"sar_{filename}")

            specs = {}
            if spec_min_w > 0: specs["min_width"] = spec_min_w
            if spec_max_w < 99999: specs["max_width"] = spec_max_w
            if spec_min_a > 0: specs["min_area"] = spec_min_a
            if spec_max_a < 99999: specs["max_area"] = spec_max_a
            if spec_min_circ > 0: specs["min_circularity"] = spec_min_circ
            if spec_max_ar < 50: specs["max_aspect_ratio"] = spec_max_ar

            if specs:
                comp = check_compliance(measurements, specs)
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Pass Rate", f"{comp['pass_rate']:.0f}%")
                mc2.metric("Passed", comp["passed"])
                mc3.metric("Failed", comp["failed"])

                comp_img = draw_compliance_overlay(original, measurements, comp)
                st.image(cv2.cvtColor(comp_img, cv2.COLOR_BGR2RGB), caption="Compliance Results", use_container_width=True)

                if comp["failed"] > 0:
                    st.markdown("**Failures:**")
                    for cr in comp["results"]:
                        if not cr["passed"]:
                            st.caption(f"Object #{cr['index']+1}: {'; '.join(cr['violations'])}")
            else:
                st.info("Set at least one threshold above to run compliance checking.")
        else:
            st.info("No objects detected.")

    with feat_tabs[2]:
        if len(measurements) >= 2:
            obj_options = [f"Object {i+1}" for i in range(len(measurements))]
            cc1, cc2 = st.columns(2)
            with cc1: sel_a = st.selectbox("First object", obj_options, index=0, key=f"cpa_{filename}")
            with cc2: sel_b = st.selectbox("Second object", obj_options, index=min(1, len(obj_options)-1), key=f"cpb_{filename}")
            idx_a = obj_options.index(sel_a)
            idx_b = obj_options.index(sel_b)
            if idx_a != idx_b:
                cmp = compare_objects(measurements[idx_a], measurements[idx_b])
                st.dataframe(pd.DataFrame(cmp["comparison"]), use_container_width=True, hide_index=True)
                # Side-by-side crops
                crops = crop_individual_objects(original, measurements)
                if idx_a < len(crops) and idx_b < len(crops):
                    cr1, cr2 = st.columns(2)
                    cr1.image(cv2.cvtColor(crops[idx_a]["cropped"], cv2.COLOR_BGR2RGB), caption=f"Object {idx_a+1}", use_container_width=True)
                    cr2.image(cv2.cvtColor(crops[idx_b]["cropped"], cv2.COLOR_BGR2RGB), caption=f"Object {idx_b+1}", use_container_width=True)
            else:
                st.warning("Select two different objects to compare.")
        else:
            st.info("Need at least 2 objects to compare.")

    with feat_tabs[3]:
        if len(measurements) >= 3:
            dist_metric = st.selectbox("Analyze", ["area", "width", "height"], key=f"dm_{filename}")
            dist = analyze_size_distribution(measurements, metric=dist_metric)
            if dist.get("status") == "ok":
                d1, d2, d3, d4 = st.columns(4)
                d1.metric("Mean", f"{dist['mean']:.3f}")
                d2.metric("Std Dev", f"{dist['std']:.3f}")
                d3.metric("CV", f"{dist['cv']:.1f}%")
                d4.metric("Uniformity", dist["uniformity"])
                d5, d6, d7, d8 = st.columns(4)
                d5.metric("Min", f"{dist['min']:.3f}")
                d6.metric("Max", f"{dist['max']:.3f}")
                d7.metric("Skewness", f"{dist['skewness']:.3f}")
                d8.metric("Kurtosis", f"{dist['kurtosis']:.3f}")
                if dist.get("normality_test"):
                    nt = dist["normality_test"]
                    st.caption(f"Shapiro-Wilk normality test: p={nt['p_value']:.4f} — {'Normal distribution' if nt['is_normal'] else 'Non-normal distribution'}")
                # Histogram
                fig, ax = plt.subplots(figsize=(8, 3)); fig.patch.set_facecolor('#0f172a'); ax.set_facecolor('#1e293b')
                ax.hist(dist["values"], bins=min(15, len(dist["values"])), color="#6366f1", alpha=0.8, edgecolor="white")
                ax.axvline(dist["mean"], color="#f59e0b", linestyle="--", label=f"Mean: {dist['mean']:.2f}")
                ax.legend(facecolor='#1e293b', edgecolor='#475569', labelcolor='white')
                ax.set_title(f"{dist_metric.title()} Distribution", color="white"); ax.tick_params(colors='#94a3b8')
                plt.tight_layout(); st.pyplot(fig); plt.close()
        else:
            st.info("Need at least 3 objects for distribution analysis.")

    with feat_tabs[4]:
        if len(measurements) >= 2:
            spacing = analyze_spacing(measurements)
            if spacing.get("status") == "ok":
                sp1, sp2, sp3 = st.columns(3)
                sp1.metric("Mean NN Distance", f"{spacing['mean_nn_distance']:.1f} px")
                sp2.metric("Clark-Evans R", f"{spacing['clark_evans_r']:.3f}")
                sp3.metric("Pattern", spacing["pattern"])
                st.caption("Clark-Evans R: <0.5 = clustered, 0.5–1.2 = random, >1.2 = regular/dispersed")
        else:
            st.info("Need at least 2 objects for spacing analysis.")

    with feat_tabs[5]:
        if measurements:
            crops = crop_individual_objects(original, measurements)
            cols_per_row = 4
            for row_start in range(0, len(crops), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    idx = row_start + j
                    if idx < len(crops):
                        col.image(cv2.cvtColor(crops[idx]["cropped"], cv2.COLOR_BGR2RGB),
                                  caption=f"Obj {idx+1} ({crops[idx]['size'][0]}×{crops[idx]['size'][1]})",
                                  use_container_width=True)
        else:
            st.info("No objects detected.")

    # Pipeline
    with st.expander("🔬 Pipeline Stages"):
        stages = [("Original",original),("Grayscale",prep_result["gray"]),("Denoised",prep_result["denoised"]),("Enhanced",prep_result["enhanced"]),("Threshold",seg_result["thresholded"]),("Morphology",seg_result["morphed"]),("Edges",edge_result),("Result",annotated)]
        for i in range(0,len(stages),4):
            cols = st.columns(4)
            for j,col in enumerate(cols):
                idx=i+j
                if idx < len(stages):
                    n,im = stages[idx]
                    if len(im.shape)==2: col.image(im, caption=n, use_container_width=True, clamp=True)
                    else: col.image(cv2.cvtColor(im,cv2.COLOR_BGR2RGB), caption=n, use_container_width=True)

    # ── ADVANCED DIP ──
    st.markdown("---")
    st.markdown("## 🧠 Advanced DIP Analysis")
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    at = st.tabs(["📊 Histogram","🔄 FFT","⭕ Hough","🔑 ORB","📐 Shapes","🔍 Quality","🧩 LBP","🌀 Gabor","📈 GLCM","🎨 Color Spaces"])

    with at[0]:
        hd = color_histogram_analysis(original)
        fig,axes=plt.subplots(1,2,figsize=(12,4)); fig.patch.set_facecolor('#0f172a')
        for n,c in [("Red","red"),("Green","green"),("Blue","blue")]: axes[0].plot(hd["bgr_histograms"][n],color=c,alpha=0.7)
        axes[0].set_title("RGB",color="white"); axes[0].set_facecolor('#1e293b'); axes[0].tick_params(colors='white')
        axes[1].bar(range(180),hd["h_hist"],color='#6366f1',alpha=0.7)
        axes[1].set_title("Hue",color="white"); axes[1].set_facecolor('#1e293b'); axes[1].tick_params(colors='white')
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.metric("Dominant Color", hd["dominant_color"])

    with at[1]:
        fd = fft_analysis(gray)
        c1,c2=st.columns(2)
        c1.image(fd["magnitude"],caption="Magnitude",use_container_width=True,clamp=True)
        c2.image(fd["phase"],caption="Phase",use_container_width=True,clamp=True)
        st.metric("Low Freq Energy", f"{fd['low_freq_ratio']:.1%}")
        st.caption("FFT: spatial → frequency domain. Bright center = low frequencies (smooth regions).")

    with at[2]:
        c1,c2=st.columns(2)
        with c1:
            circles = hough_circle_detection(gray)
            if circles: st.image(cv2.cvtColor(draw_hough_circles(original,circles),cv2.COLOR_BGR2RGB),caption=f"{len(circles)} circles",use_container_width=True)
            else: st.info("No circles.")
        with c2:
            lines = hough_line_detection(gray)
            if lines: st.image(cv2.cvtColor(draw_hough_lines(original,lines[:50]),cv2.COLOR_BGR2RGB),caption=f"{len(lines)} lines",use_container_width=True)
            else: st.info("No lines.")

    with at[3]:
        od = orb_feature_detection(gray)
        st.image(cv2.cvtColor(draw_orb_keypoints(original,od["keypoints"]),cv2.COLOR_BGR2RGB),caption=f"ORB: {od['count']} keypoints",use_container_width=True)
        st.caption("ORB: rotation/scale-invariant features for matching and recognition.")

    with at[4]:
        si = original.copy(); sd = []
        for i,m in enumerate(measurements):
            sh,ci = classify_contour_shape(m["contour"]); sd.append({"#":i+1,"Shape":sh,"Circ":round(ci,3)})
            cv2.putText(si,sh,m["centroid"],cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2)
            cv2.drawContours(si,[m["contour"]],-1,(0,255,0),2)
        st.image(cv2.cvtColor(si,cv2.COLOR_BGR2RGB),use_container_width=True)
        st.dataframe(pd.DataFrame(sd),use_container_width=True,hide_index=True)

    with at[5]:
        from utils.advanced_analysis import blur_detection as _bd, noise_estimation as _ne
        bl=_bd(gray); ns=_ne(gray)
        q1,q2,q3,q4=st.columns(4)
        q1.metric("Sharp",bl["quality"]); q2.metric("Score",f"{bl['score']}/5")
        q3.metric("Noise",ns["level"]); q4.metric("σ",f"{ns['sigma']:.2f}")

    with at[6]:
        with st.spinner("Computing LBP..."):
            li = cv2.resize(gray,None,fx=min(1,500/max(gray.shape)),fy=min(1,500/max(gray.shape))) if max(gray.shape)>500 else gray
            ld = lbp_fast(li)
        c1,c2=st.columns(2)
        c1.image(ld["lbp_image"],caption="LBP Map",use_container_width=True,clamp=True)
        with c2:
            fig,ax=plt.subplots(figsize=(6,3)); fig.patch.set_facecolor('#0f172a'); ax.set_facecolor('#1e293b')
            ax.bar(range(256),ld["histogram"],color="#6366f1",alpha=0.8); ax.set_title("LBP Histogram",color="white"); ax.tick_params(colors='white')
            plt.tight_layout(); st.pyplot(fig); plt.close()
        st.caption("LBP: encodes local texture by comparing center pixel with neighbors. Used for texture classification.")

    with at[7]:
        with st.spinner("Gabor filter bank..."):
            gi = cv2.resize(gray,None,fx=min(1,500/max(gray.shape)),fy=min(1,500/max(gray.shape))) if max(gray.shape)>500 else gray
            gd = gabor_filter_bank(gi)
        st.image(gd["energy_map"],caption="Gabor Energy Map",use_container_width=True,clamp=True)
        st.caption(f"{gd['n_filters']} Gabor filters applied (frequencies × orientations). Bio-inspired texture detection.")
        with st.expander("Filter responses"):
            for i in range(0,len(gd["responses"]),4):
                cols=st.columns(4)
                for j,col in enumerate(cols):
                    idx=i+j
                    if idx<len(gd["responses"]):
                        f,t,r=gd["responses"][idx]; rv=((r-r.min())/(r.max()-r.min()+1e-8)*255).astype(np.uint8)
                        col.image(rv,caption=f"f={f:.2f} θ={np.degrees(t):.0f}°",use_container_width=True,clamp=True)

    with at[8]:
        with st.spinner("GLCM features..."):
            gi = cv2.resize(gray,None,fx=min(1,400/max(gray.shape)),fy=min(1,400/max(gray.shape))) if max(gray.shape)>400 else gray
            gd = glcm_features(gi)
        a = gd["averaged"]
        c1,c2,c3=st.columns(3); c1.metric("Contrast",f"{a['contrast']:.3f}"); c2.metric("Homogeneity",f"{a['homogeneity']:.3f}"); c3.metric("Energy",f"{a['energy']:.4f}")
        c4,c5,c6=st.columns(3); c4.metric("Correlation",f"{a['correlation']:.3f}"); c5.metric("Entropy",f"{a['entropy']:.3f}"); c6.metric("Dissimilarity",f"{a['dissimilarity']:.3f}")
        st.caption("GLCM: second-order texture statistics from spatial pixel relationships.")
        with st.expander("Per-direction"): st.dataframe(pd.DataFrame(gd["per_direction"]),use_container_width=True,hide_index=True)

    with at[9]:
        cs = color_space_decomposition(original)
        chs = list(cs.items())
        for i in range(0,len(chs),4):
            cols=st.columns(4)
            for j,col in enumerate(cols):
                idx=i+j
                if idx<len(chs): col.image(chs[idx][1],caption=chs[idx][0],use_container_width=True,clamp=True)
        st.caption("RGB, HSV, LAB, YCrCb decomposition. LAB separates lightness; HSV separates color from intensity.")

    # Morphological gradient
    with st.expander("🔲 Morphological Gradient"):
        mg = morphological_gradient(gray)
        st.image(mg, caption="Dilation − Erosion", use_container_width=True, clamp=True)

    # ── ML/AI ANALYSIS ──
    st.markdown("---")
    st.markdown("## 🤖 ML / AI Analysis")
    st.caption("Machine learning–powered features: SVM shape classification, BRISQUE quality assessment, anomaly detection, superpixel segmentation, and more.")

    from utils.ml_analysis import (
        brisque_quality_score, train_shape_classifier, classify_shape_ml,
        slic_superpixels, pca_texture_analysis, detect_measurement_anomalies,
        classify_scene, hog_descriptor, gmm_color_segment,
        dbscan_cluster_objects, measurement_confidence_scorer,
        compute_similarity_matrix,
    )

    ml_tabs = st.tabs([
        "🏷️ SVM Shapes", "📊 BRISQUE IQA", "🧩 Superpixels",
        "🧠 PCA Texture", "🚨 Anomalies", "🎬 Scene",
        "🔲 HOG", "🎨 GMM Color", "📐 Confidence",
    ])

    with ml_tabs[0]:
        with st.spinner("Training SVM shape classifier on synthetic data..."):
            try:
                classifier = train_shape_classifier()
                if classifier and measurements:
                    st.success(f"SVM trained on {classifier['n_training']} synthetic samples across {len(classifier['labels'])} shape classes")
                    ml_shapes = []
                    si_ml = original.copy()
                    for i, m in enumerate(measurements):
                        pred, conf, all_probs = classify_shape_ml(m["contour"], classifier)
                        ml_shapes.append({"#": i+1, "ML Shape": pred, "Confidence": f"{conf:.1%}",
                                          "Top-2": ", ".join(f"{k}:{v:.0%}" for k, v in sorted(all_probs.items(), key=lambda x: -x[1])[:2])})
                        color = (0, 255, 0) if conf > 0.7 else (0, 255, 255) if conf > 0.4 else (0, 0, 255)
                        cv2.putText(si_ml, f"{pred} ({conf:.0%})", m["centroid"], cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                        cv2.drawContours(si_ml, [m["contour"]], -1, color, 2)
                    st.image(cv2.cvtColor(si_ml, cv2.COLOR_BGR2RGB), caption="SVM Shape Classification", use_container_width=True)
                    st.dataframe(pd.DataFrame(ml_shapes), use_container_width=True, hide_index=True)
                    st.caption("SVM with RBF kernel trained on Hu moments + 7 geometric features (circularity, solidity, convexity, aspect ratio, extent, eccentricity, vertex count).")
                elif not measurements:
                    st.info("No objects detected to classify.")
                else:
                    st.warning("Could not train shape classifier.")
            except Exception as e:
                st.error(f"SVM training error: {str(e)[:100]}")

    with ml_tabs[1]:
        with st.spinner("Computing BRISQUE quality score..."):
            bq = brisque_quality_score(gray)
        q1, q2 = st.columns(2)
        q1.metric("BRISQUE Score", f"{bq['score']:.0f}/100")
        q2.metric("Grade", bq["grade"])
        ds = bq.get("detail_scores", {})
        if ds:
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Variance", f"{ds.get('variance',0):.1f}/30")
            d2.metric("Skewness", f"{ds.get('skewness',0):.1f}/20")
            d3.metric("Kurtosis", f"{ds.get('kurtosis',0):.1f}/20")
            d4.metric("Symmetry", f"{ds.get('symmetry',0):.1f}/30")
        st.caption("BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator): extracts natural scene statistics from MSCN coefficients and fits asymmetric generalized Gaussian distributions to pairwise products. Higher score = better quality.")

    with ml_tabs[2]:
        with st.spinner("Computing SLIC superpixels..."):
            n_seg = st.slider("Superpixel count", 50, 500, 200, key="slic_n")
            sp = slic_superpixels(original, n_segments=n_seg)
        st.image(cv2.cvtColor(sp["boundary_overlay"], cv2.COLOR_BGR2RGB), caption=f"{sp['n_segments']} superpixels (SLIC)", use_container_width=True)
        s1, s2, s3 = st.columns(3)
        s1.metric("Segments", sp["n_segments"])
        s2.metric("Avg Size", f"{sp['avg_size']} px")
        s3.metric("Size Std", f"{sp['size_std']} px")
        st.caption("SLIC (Simple Linear Iterative Clustering): clusters pixels in 5D space (CIELAB + XY) to form perceptually coherent regions. Used as preprocessing for object proposals and semantic segmentation.")

    with ml_tabs[3]:
        with st.spinner("PCA texture analysis..."):
            gi = cv2.resize(gray, None, fx=min(1, 400/max(gray.shape)), fy=min(1, 400/max(gray.shape))) if max(gray.shape) > 400 else gray
            pca_result = pca_texture_analysis(gi)
        if "error" not in pca_result:
            p1, p2, p3 = st.columns(3)
            p1.metric("Texture Complexity", f"{pca_result['texture_complexity']}/{pca_result['n_components']}")
            p2.metric("Recon Error", f"{pca_result['reconstruction_error']:.4f}")
            p3.metric("Patches Analyzed", pca_result["n_patches"])
            st.image(pca_result["texture_map"], caption="PCA Texture Energy Map", use_container_width=True, clamp=True)
            if pca_result["components_vis"]:
                st.markdown("**Learned texture basis (principal components):**")
                cols = st.columns(min(4, len(pca_result["components_vis"])))
                for j, col in enumerate(cols):
                    if j < len(pca_result["components_vis"]):
                        col.image(pca_result["components_vis"][j], caption=f"PC{j+1} ({pca_result['variance_explained'][j]:.1%})", use_container_width=True, clamp=True)
            st.caption("PCA learns a compact texture representation from the image's own patch statistics. Complexity = number of components needed to explain 90% of variance.")
        else:
            st.info(pca_result["error"])

    with ml_tabs[4]:
        if len(measurements) >= 5:
            anom = detect_measurement_anomalies(measurements)
            if anom["n_anomalies"] > 0:
                st.warning(f"🚨 {anom['n_anomalies']} anomalous objects detected by Isolation Forest")
                for a in anom["anomalies"]:
                    st.caption(f"Object #{a['index']+1}: score={a['anomaly_score']:.3f} — {a['reason']}")
            else:
                st.success("No anomalous measurements detected.")
            # Show scores
            score_data = [{"#": i+1, "Anomaly Score": s, "Status": "⚠️ Anomaly" if s < 0 else "✅ Normal"}
                          for i, s in enumerate(anom["scores"])]
            st.dataframe(pd.DataFrame(score_data), use_container_width=True, hide_index=True)
            st.caption("Isolation Forest: ensemble method that isolates anomalies by randomly partitioning feature space. Objects that are isolated in fewer splits are more anomalous.")
        else:
            st.info("Need at least 5 objects for anomaly detection. Upload an image with more items.")

    with ml_tabs[5]:
        scene = classify_scene(original)
        sc1, sc2 = st.columns(2)
        sc1.metric("Scene Type", scene["scene_type"].replace("_", " ").title())
        sc2.metric("Confidence", f"{scene['confidence']:.0%}")
        feat = scene.get("features", {})
        if feat:
            f1, f2, f3 = st.columns(3)
            f1.metric("Edge Density", f"{feat.get('edge_density',0):.4f}")
            f2.metric("Uniformity", f"{feat.get('spatial_uniformity',0):.3f}")
            f3.metric("Low Freq", f"{feat.get('low_freq_ratio',0):.3f}")
        st.caption("Scene classification using spatial statistics, frequency analysis, and color distribution features. Helps auto-select optimal processing strategy.")

    with ml_tabs[6]:
        with st.spinner("Computing HOG descriptors..."):
            gi = cv2.resize(gray, None, fx=min(1, 400/max(gray.shape)), fy=min(1, 400/max(gray.shape))) if max(gray.shape) > 400 else gray
            hog_res = hog_descriptor(gi)
        if "error" not in hog_res:
            st.image(hog_res["visualization"], caption=f"HOG visualization ({hog_res['feature_length']} features)", use_container_width=True, clamp=True)
            h1, h2 = st.columns(2)
            h1.metric("Feature Length", hog_res["feature_length"])
            h2.metric("Cell Size", f"{hog_res['cell_size']}px")
            st.caption("Histogram of Oriented Gradients (Dalal & Triggs 2005): captures local edge direction distributions. Core feature in SVM-based object detection (e.g., pedestrian detection).")
        else:
            st.info(hog_res["error"])

    with ml_tabs[7]:
        with st.spinner("Fitting GMM color model..."):
            n_comp = st.slider("GMM components", 2, 6, 3, key="gmm_k")
            gmm_res = gmm_color_segment(original, n_components=n_comp)
        c1, c2 = st.columns(2)
        c1.image(cv2.cvtColor(gmm_res["colored"], cv2.COLOR_BGR2RGB), caption=f"GMM Segmentation ({n_comp} components)", use_container_width=True)
        c2.image(gmm_res["entropy_map"], caption="Uncertainty Map (entropy)", use_container_width=True, clamp=True)
        st.metric("BIC", f"{gmm_res['bic']:.0f}")
        st.caption("Gaussian Mixture Model: probabilistic color clustering that models each region as a multivariate Gaussian in CIELAB space. Entropy map shows classification uncertainty — bright = ambiguous boundaries. BIC (Bayesian Information Criterion) helps select optimal component count.")

    with ml_tabs[8]:
        if measurements:
            bq_for_conf = brisque_quality_score(gray)
            _is_calibrated = 'pixel_per_cm' in dir() and pixel_per_cm is not None
            conf_scores = measurement_confidence_scorer(measurements, bq_for_conf, calibrated=_is_calibrated)
            conf_data = [{"#": i+1, "Confidence": f"{c['confidence']:.0%}", "Grade": c["grade"]} for i, c in enumerate(conf_scores)]
            st.dataframe(pd.DataFrame(conf_data), use_container_width=True, hide_index=True)

            # Similarity matrix
            if len(measurements) >= 2:
                sim = compute_similarity_matrix(measurements)
                if sim.get("status") == "ok" and sim.get("most_similar"):
                    st.markdown("**Object Similarity (cosine):**")
                    for i, j, s in sim["most_similar"]:
                        st.caption(f"Objects #{i+1} & #{j+1}: similarity = {s:.3f}")

            # Spatial clustering
            if len(measurements) >= 3:
                clusters = dbscan_cluster_objects(measurements)
                if clusters["n_clusters"] > 0:
                    st.markdown(f"**DBSCAN Spatial Clusters:** {clusters['n_clusters']} groups found")
                    for cid, members in clusters.get("clusters", {}).items():
                        st.caption(f"Cluster {cid+1}: Objects {[m+1 for m in members]}")

            st.caption("Random Forest confidence scorer combines contour smoothness, image quality (BRISQUE), calibration status, and geometric features. Cosine similarity on standardized shape features. DBSCAN groups spatially proximate objects.")
        else:
            st.info("No measurements to score.")


# ──────────────────────────────────────────────
# DASHBOARD
# ──────────────────────────────────────────────
def render_dashboard():
    user = st.session_state.user
    if not user: st.session_state.page = "auth"; st.rerun(); return
    st.markdown(f"## 👤 {user['full_name'] or user['username']}")
    from utils.session_manager import get_user_sessions
    sessions = get_user_sessions(user["id"])
    s1,s2,s3 = st.columns(3)
    s1.metric("Analyses", get_user_analysis_count(user["id"]))
    s2.metric("Sessions", len(sessions))
    s3.metric("Since", user.get("created_at","")[:10])
    if sessions:
        st.markdown("### Recent Sessions")
        for s in sessions[:10]: st.caption(f"📁 **{s['name']}** — {s['total_images']} imgs — {s['created_at'][:16]}")
    st.markdown("---")
    if st.button("🚪 Sign Out"): st.session_state.user=None; st.session_state.page="home"; st.rerun()


# ──────────────────────────────────────────────
# ROUTER
# ──────────────────────────────────────────────
render_navbar()
page = st.session_state.page
if page == "home": render_home()
elif page == "auth": render_auth()
elif page == "analyze": render_analyze()
elif page == "dashboard": render_dashboard()
else: render_home()
