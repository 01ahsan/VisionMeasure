# VisionMeasure — Complete Setup Guide (Windows + VSCode)

Follow every step in order. Nothing is skipped.

---

## PHASE 1: ACCOUNTS YOU NEED (one-time setup)

### 1. GitHub Account
- Go to https://github.com
- Click "Sign up" → enter email, password, username
- Verify email
- This is where your project lives publicly

### 2. Streamlit Community Cloud Account (free hosting)
- Go to https://share.streamlit.io
- Click "Sign up" → sign in with your GitHub account
- This gives you free deployment directly from your GitHub repo
- No credit card needed, no domain purchase needed
- You get a free URL like: https://visionmeasure.streamlit.app

### 3. (Optional) HuggingFace Account
- Go to https://huggingface.co/join
- Alternative free deployment option
- Good for ML community visibility

---

## PHASE 2: LOCAL ENVIRONMENT SETUP (Windows + VSCode)

### Step 1: Install Python (if not already)
- Go to https://www.python.org/downloads/
- Download Python 3.10 or 3.11 (NOT 3.13 — some libraries lag behind)
- During install: CHECK "Add Python to PATH" ✅
- Verify: open Command Prompt → type `python --version`

### Step 2: Install VSCode (if not already)
- Go to https://code.visualstudio.com
- Download and install
- Install the "Python" extension by Microsoft (from Extensions tab)

### Step 3: Create project folder
Open Command Prompt (or VSCode terminal) and run:
```
mkdir VisionMeasure
cd VisionMeasure
```

### Step 4: Create virtual environment
```
python -m venv venv
venv\Scripts\activate
```
You should see `(venv)` at the start of your terminal line.

### Step 5: Open in VSCode
```
code .
```
Then in VSCode: Ctrl+Shift+P → "Python: Select Interpreter" → choose the one from your venv.

### Step 6: Install dependencies
With venv activated:
```
pip install -r requirements.txt
```
(The requirements.txt file is included in this project)

---

## PHASE 3: PROJECT STRUCTURE

```
VisionMeasure/
├── app.py                    # Main Streamlit app
├── requirements.txt          # Python dependencies
├── README.md                 # GitHub README
├── .gitignore                # Git ignore rules
├── Dockerfile                # Docker deployment (optional)
├── .streamlit/
│   └── config.toml           # Streamlit theme config
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py      # Noise removal, filtering
│   ├── segmentation.py       # Thresholding, morphology
│   ├── edge_detection.py     # Edge detection methods
│   ├── contour_analysis.py   # Contour extraction & measurement
│   ├── calibration.py        # Pixel-to-cm calibration
│   └── visualization.py      # Drawing results & pipeline viz
├── test_images/              # Sample test images
│   └── (add your test images here)
└── assets/
    └── banner.png            # README banner (optional)
```

---

## PHASE 4: RUN THE PROJECT LOCALLY

### Start the app
With venv activated, in the project root:
```
streamlit run app.py
```
This opens http://localhost:8501 in your browser.

### Test it
1. Place an object next to a coin (or any reference of known size) on a plain background
2. Take a photo with your phone
3. Upload it to the app
4. Enter the known size of your reference object
5. Click through the pipeline steps to see measurements

---

## PHASE 5: PUSH TO GITHUB

### Step 1: Initialize git
```
git init
git add .
git commit -m "Initial commit: VisionMeasure project"
```

### Step 2: Create GitHub repo
- Go to https://github.com/new
- Repository name: `VisionMeasure`
- Description: "Automatic Object Dimension and Area Estimation from Images"
- Make it Public
- Do NOT add README (you already have one)
- Click "Create repository"

### Step 3: Push code
GitHub will show you commands. Run:
```
git remote add origin https://github.com/YOUR_USERNAME/VisionMeasure.git
git branch -M main
git push -u origin main
```

---

## PHASE 6: DEPLOY TO STREAMLIT CLOUD (FREE)

### Step 1: Go to https://share.streamlit.io
### Step 2: Click "New app"
### Step 3: Select:
- Repository: `YOUR_USERNAME/VisionMeasure`
- Branch: `main`
- Main file path: `app.py`
### Step 4: Click "Deploy"

Wait 2-3 minutes. You get a live URL like:
```
https://visionmeasure.streamlit.app
```

Put this URL in your GitHub repo description and README.

---

## PHASE 7: GOOGLE COLAB ALTERNATIVE

If you want to run/develop in Colab:

```python
# Cell 1: Install dependencies
!pip install opencv-python-headless numpy scipy scikit-image streamlit

# Cell 2: Clone your repo
!git clone https://github.com/YOUR_USERNAME/VisionMeasure.git
%cd VisionMeasure

# Cell 3: Run with tunnel (for Streamlit in Colab)
!pip install localtunnel
!streamlit run app.py &>/content/logs.txt &
!npx localtunnel --port 8501
```

For just testing the processing functions without the UI, you can import
and run them directly in Colab cells — see the utils/ modules.

---

## TROUBLESHOOTING

| Problem | Fix |
|---|---|
| `cv2` import error | `pip install opencv-python-headless` |
| Streamlit not found | `pip install streamlit` then restart terminal |
| venv won't activate (PowerShell) | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` first |
| Camera permission denied | Use file upload instead of camera input |
| Black image after threshold | Try adjusting threshold values in the sidebar |
| Deployment fails | Check `requirements.txt` has exact versions, no GPU libraries |
