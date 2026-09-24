# Software Module for Diagnosing Heart Diseases Based on ECG Signal Analysis

A desktop application for automated ECG analysis and heart disease diagnosis using signal processing and machine learning.

---

## Features

- Load ECG files in **.csv**, **.txt**, or **.mat** formats
- Enter patient symptoms for enhanced AI diagnosis
- Visualize ECG signal with annotated R-peaks, ST segments, and arrhythmia regions
- Full medical-style ECG report including:
  - Heart Rate with classification (Normal / Tachycardia / Bradycardia)
  - Electrical Axis of the Heart
  - Rhythm analysis (Regular / Irregular + cause)
  - Ischemic signs (ST elevation/depression, T-wave inversion)
  - Conduction disorders (AV block, Bundle Branch Block, Long QT)
  - Final diagnosis + Confidence score

---

## Project Structure

```
ecg_diagnosis_app/
├── main.py               — Application entry point
├── gui.py                — PyQt5 main window and UI
├── ecg_loader.py         — ECG file loading (.csv, .txt, .mat)
├── preprocessing.py      — Signal filtering and baseline removal
├── r_peak_detection.py   — Pan-Tompkins R-peak detection algorithm
├── feature_extraction.py — Clinical ECG feature extraction
├── model.py              — Neural network model definition (PyTorch + sklearn)
├── train_model.py        — Model training on synthetic data
├── inference.py          — Full analysis pipeline
├── visualization.py      — Matplotlib ECG plotting utilities
├── requirements.txt      — Python dependencies
└── sample_data/
    ├── generate_samples.py   — Synthetic ECG data generator
    ├── ecg_normal.csv        — Normal sinus rhythm sample
    ├── ecg_tachycardia.csv   — Tachycardia sample
    ├── ecg_bradycardia.csv   — Bradycardia sample
    ├── ecg_arrhythmia.csv    — Arrhythmia / extrasystoles sample
    └── ecg_mi.csv            — Myocardial infarction signs sample
```

---

## Installation

### 1. Install Python 3.9+

Download from https://python.org

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** PyTorch installation may vary by platform. If the above fails for torch, install it manually from https://pytorch.org/get-started/locally/
> The app works without PyTorch (falls back to scikit-learn automatically).

---

## Running the Application

```bash
python main.py
```

The application will:
1. Launch the GUI automatically
2. Auto-train the ML model on first run (takes ~10 seconds)

---

## Training the Model Manually

```bash
python train_model.py
```

This generates `ecg_model.npy` in the project directory.

---

## Regenerating Sample Data

```bash
cd sample_data
python generate_samples.py
```

---

## Usage

1. Click **Upload ECG File** or select a sample from the **Quick Load Sample** panel
2. Optionally enter patient symptoms in the text box
3. Click **ANALYZE ECG**
4. View the annotated ECG graph and the full medical report

---

## Supported ECG File Formats

| Format | Notes |
|--------|-------|
| `.csv` | Two-column (time, amplitude) or single-column (amplitude) |
| `.txt` | Same as CSV, whitespace or comma delimited |
| `.mat` | MATLAB format, keys: `val`, `ecg`, `signal`, or `data` |

---

## Technical Details

### Signal Processing Pipeline
1. DC offset removal
2. Bandpass filter 0.5–40 Hz (4th-order Butterworth)
3. 50 Hz / 60 Hz notch filter (powerline interference)
4. Baseline wander removal (high-pass 0.5 Hz)
5. RMS normalization

### R-Peak Detection
Pan-Tompkins algorithm:
- Bandpass filter 5–15 Hz
- Differentiation → Squaring → Moving window integration
- Adaptive threshold with minimum distance constraint
- Sub-sample peak refinement

### Feature Extraction
- Heart rate and HRV (SDNN, RMSSD)
- RR interval variability coefficient
- Ectopic beat detection
- PQRST wave segmentation
- ST level measurement (J-point + 80 ms)
- QRS duration, PR interval, QT interval

### AI Classification
- Gradient Boosting Classifier (scikit-learn) trained on synthetic data
- 5 classes: Normal, Arrhythmia, Tachycardia, Bradycardia, Possible MI
- Symptom-based posterior probability adjustment
- Rule-based expert system fallback

---

## Disclaimer

This software is a research and educational prototype developed as a diploma project.
**It is not a certified medical device.** All results must be reviewed and confirmed by a
qualified cardiologist before any clinical decision-making.

---

*Diploma Project — 2026*
