# 📐 VisionMeasure

**Automatic Object Dimension and Area Estimation from Images**

VisionMeasure is a digital image processing tool that estimates real-world dimensions (length, width, area, perimeter) of objects from a single photograph using a reference object of known size.

<!-- Add your live demo URL below -->
<!-- 🔗 **[Live Demo](https://visionmeasure.streamlit.app)** -->

---

## Features

- **Real-world measurements** — Calibrates using a reference object (coin, card, ruler) to convert pixels → centimeters
- **Multi-object detection** — Measures all objects in the frame simultaneously
- **Full DIP pipeline visualization** — See every processing stage: grayscale → denoising → thresholding → morphology → edges → contours → measurements
- **Interactive parameter tuning** — Adjust every parameter in real-time via sidebar controls
- **Auto reference detection** — Automatically detects circular (coins) or rectangular (cards) reference objects
- **Batch export** — Download results as CSV or annotated images

## Applications

| Domain | Use Case |
|---|---|
| Agriculture | Leaf area measurement, fruit sizing |
| Healthcare | Wound area estimation |
| Manufacturing | Component dimension verification |
| E-commerce | Product size documentation |
| Laboratory | Sample measurement |
| Education | Biology specimen analysis |

## DIP Techniques Used

```
Image → Noise Removal → Thresholding → Morphology → Edge Detection → Contour Extraction → Measurement
```

| Stage | Techniques |
|---|---|
| Preprocessing | Gaussian filter, Median filter, Bilateral filter, CLAHE |
| Segmentation | Otsu thresholding, Adaptive thresholding (mean/Gaussian), Manual threshold |
| Morphology | Erosion, Dilation, Opening, Closing |
| Edge Detection | Canny, Sobel, Laplacian |
| Analysis | Contour detection, Min-area bounding rectangle, Connected-component analysis, Pixel-to-cm calibration |

## Quick Start

### Prerequisites
- Python 3.10+

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/VisionMeasure.git
cd VisionMeasure
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## How to Use

1. Place your object(s) next to a reference item (e.g., a coin) on a plain background
2. Take a photo from directly above
3. Upload the image to VisionMeasure
4. Select your reference object type and known size from the sidebar
5. Adjust processing parameters if needed
6. View measurements and export results

## Accuracy Validation

| Object | Measured (cm) | Actual (cm) | Error (%) |
|---|---|---|---|
| | | | |

*Add your own validation results here*

## Project Structure

```
VisionMeasure/
├── app.py                    # Streamlit application
├── requirements.txt          # Dependencies
├── utils/
│   ├── preprocessing.py      # Noise removal, filtering
│   ├── segmentation.py       # Thresholding, morphology
│   ├── edge_detection.py     # Canny, Sobel, Laplacian
│   ├── contour_analysis.py   # Contour extraction & measurement
│   ├── calibration.py        # Pixel-to-cm conversion
│   └── visualization.py      # Result drawing & pipeline viz
└── test_images/              # Sample images
```

## Tech Stack

- **Python 3.10+**
- **OpenCV** — Image processing pipeline
- **NumPy / SciPy** — Numerical computation
- **scikit-image** — Additional image processing utilities
- **Streamlit** — Interactive web interface
- **Pandas** — Results tabulation and CSV export

## License

MIT License

## Author

**Your Name**
- Google Scholar: [Profile](https://scholar.google.com/citations?user=LlKQVegAAAAJ&hl=en)
- GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
