"""
ecg_digitizer.py
================
Extracts a 1-D ECG signal from a photograph or scan of an ECG paper strip.

Pipeline
--------
1. Load image (PIL / OpenCV)
2. Convert to grayscale and auto-orient
3. Detect and remove background grid (paper lines)
4. Isolate the ECG trace (dark line on light background, or inverse)
5. For every x-column → find the y-position of the trace peak
6. Convert pixel coordinates → normalised amplitude
7. Estimate sampling rate from image width and standard ECG paper speed
8. Return (signal_array, fs)

Supported image formats
-----------------------
.png  .jpg  .jpeg  .bmp  .tiff  .tif  .gif
"""

import numpy as np

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def digitize_ecg_image(image_path, paper_speed_mm_s=25.0, dpi_hint=None):
    """
    Convert an ECG image file into a 1-D signal array.

    Parameters
    ----------
    image_path   : str   — path to image file
    paper_speed  : float — ECG paper speed in mm/s (standard = 25 mm/s)
    dpi_hint     : int   — image DPI if known; None = auto-estimate

    Returns
    -------
    signal : np.ndarray  — normalised ECG signal (float64)
    fs     : int         — estimated sampling rate in Hz
    """
    img_gray = _load_and_preprocess(image_path)
    trace_px  = _extract_trace(img_gray)
    signal    = _pixels_to_signal(trace_px, img_gray.shape[0])
    fs        = _estimate_fs(img_gray.shape[1], paper_speed_mm_s, dpi_hint)
    signal    = _smooth(signal, window=max(3, fs // 60))
    return signal, fs


# ---------------------------------------------------------------------------
# Step 1 — Load & preprocess
# ---------------------------------------------------------------------------

def _load_and_preprocess(image_path):
    """
    Load image → grayscale → uint8 normalised so that
    ECG trace is DARK (low value) on LIGHT background (high value).
    """
    img = _imread_gray(image_path)

    # Auto-rotate landscape if needed
    h, w = img.shape
    if h > w:
        import cv2
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    # Ensure trace is dark on light background
    # If median is < 128 → image is inverted (dark background)
    if float(np.median(img)) < 128:
        img = 255 - img

    return img.astype(np.float32)


def _imread_gray(path):
    """Load image as grayscale using OpenCV (preferred) or PIL fallback."""
    try:
        import cv2
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"OpenCV could not read: {path}")
        return img
    except ImportError:
        pass

    # PIL fallback
    from PIL import Image
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Step 2 — Isolate the ECG trace
# ---------------------------------------------------------------------------

def _extract_trace(img_gray):
    """
    For each x-column, find the y-coordinate of the ECG trace.

    Strategy:
    - Suppress the light background grid by subtracting a column-wise
      blurred version (highlight local dark structures = the trace).
    - Find the y-position of the minimum (darkest) pixel per column.
    - Median-filter the column positions to remove spikes.
    """
    img = img_gray.copy()
    h, w = img.shape

    # --- Remove slow-varying background (baseline / paper colour) ----------
    from scipy.ndimage import uniform_filter1d
    col_mean = img.mean(axis=0)                   # shape (w,)
    row_mean = img.mean(axis=1, keepdims=True)    # shape (h,1)

    # Subtract row-wise baseline to flatten horizontal grid lines
    img_flat = img - row_mean
    img_flat = img_flat - img_flat.min()

    # --- Binary threshold: keep darkest 20% of pixels ----------------------
    threshold = np.percentile(img_flat, 20)
    binary = (img_flat <= threshold).astype(np.float32)

    # --- Per-column: weighted centroid of dark pixels ----------------------
    y_coords = np.arange(h, dtype=np.float32)
    trace_y = np.zeros(w, dtype=np.float32)

    for x in range(w):
        col = binary[:, x]
        total = col.sum()
        if total > 0:
            trace_y[x] = float(np.dot(col, y_coords) / total)
        else:
            trace_y[x] = h / 2.0   # fallback: centre

    # --- Median filter to remove outlier columns ---------------------------
    from scipy.signal import medfilt
    trace_y = medfilt(trace_y, kernel_size=min(15, w // 50 * 2 + 1))

    return trace_y


# ---------------------------------------------------------------------------
# Step 3 — Convert pixels → normalised amplitude
# ---------------------------------------------------------------------------

def _pixels_to_signal(trace_y, img_height):
    """
    Invert y-axis (image y grows downward) and normalise to [-1, 1].
    """
    # Invert: large y = bottom = negative in ECG convention
    signal = float(img_height) - trace_y

    # Remove DC offset
    signal = signal - np.mean(signal)

    # Normalise
    peak = np.max(np.abs(signal))
    if peak > 1e-6:
        signal = signal / peak

    return signal.astype(np.float64)


# ---------------------------------------------------------------------------
# Step 4 — Estimate sampling rate
# ---------------------------------------------------------------------------

def _estimate_fs(img_width_px, paper_speed_mm_s=25.0, dpi=None):
    """
    Estimate fs from image width and ECG paper speed.

    Standard ECG paper: 25 mm/s.
    At 96 DPI: 1 mm = 96/25.4 ≈ 3.78 px → 1 s = 25 mm = ~94 px

    fs = (img_width_px / px_per_mm) / (total_duration_s)
       = img_width_px * (25.4 / dpi) * paper_speed_mm_s

    If DPI unknown, assume a 10-second strip at standard paper speed.
    10 seconds is the most common ECG recording length.
    """
    if dpi is not None and dpi > 0:
        px_per_mm = dpi / 25.4
        duration_s = img_width_px / (px_per_mm * paper_speed_mm_s)
        fs = int(round(img_width_px / duration_s))
    else:
        # Assume 10-second strip → fs = width / 10
        assumed_duration_s = 10.0
        fs = int(round(img_width_px / assumed_duration_s))

    # Clamp to realistic ECG sampling rates
    fs = max(100, min(fs, 1000))
    return fs


# ---------------------------------------------------------------------------
# Step 5 — Light smoothing
# ---------------------------------------------------------------------------

def _smooth(signal, window=5):
    """Apply a simple moving-average to reduce digitisation noise."""
    if window < 2:
        return signal
    kernel = np.ones(window) / window
    return np.convolve(signal, kernel, mode="same")


# ---------------------------------------------------------------------------
# Diagnostic helper (optional — saves debug image)
# ---------------------------------------------------------------------------

def save_debug_image(image_path, trace_y, output_path="debug_trace.png"):
    """Draw the extracted trace on the original image and save."""
    try:
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            return
        h, w = img.shape[:2]
        for x in range(min(len(trace_y) - 1, w - 1)):
            y1 = int(np.clip(trace_y[x],     0, h - 1))
            y2 = int(np.clip(trace_y[x + 1], 0, h - 1))
            cv2.line(img, (x, y1), (x + 1, y2), (0, 0, 255), 2)
        cv2.imwrite(output_path, img)
        print(f"Debug image saved: {output_path}")
    except Exception as e:
        print(f"Could not save debug image: {e}")
