"""
dataset_loader.py
=================
Loads the MIT-BIH Arrhythmia Database from PhysioNet using the wfdb library,
extracts per-beat feature vectors, and maps annotations to 5 diagnostic classes:

    0 – Normal Sinus Rhythm
    1 – Arrhythmia
    2 – Tachycardia
    3 – Bradycardia
    4 – Possible Myocardial Infarction

Records are cached in the  data/  folder to avoid repeated downloads.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import preprocess_ecg
from r_peak_detection import detect_r_peaks
from preprocessing import compute_pqrst_segments

# ---------------------------------------------------------------------------
# MIT-BIH record list (all 48 records)
# ---------------------------------------------------------------------------
MIT_BIH_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]

# AAMI beat annotation mapping
# Normal beats → candidate for Normal/Tachy/Brady (decided by local HR)
NORMAL_SYMBOLS = {"N", "L", "R", "B", "e", "j", "n", "."}
# Ectopic / arrhythmic beats → class 1
ARRHYTHMIA_SYMBOLS = {"A", "a", "J", "S", "V", "E", "F"}
# Skip these (pacemaker, noise, non-beat markers)
SKIP_SYMBOLS = {"/", "Q", "~", "|", "+", "x", "[", "]", "!", '"', "s",
                "T", "P", "U", "M", "u", "f", "p", "t", "u"}

# Diagnostic class names (same order as CLASS_NAMES in model.py)
CLASS_NAMES = [
    "Нормальний синусовий ритм",
    "Аритмія",
    "Тахікардія",
    "Брадикардія",
    "Можливий інфаркт міокарда",
]

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_mitbih_features(records=None, verbose=True):
    """
    Download / load MIT-BIH records, extract per-beat features and labels.

    Returns
    -------
    X : np.ndarray, shape (n_beats, 20)
    y : np.ndarray, shape (n_beats,)   — integer class labels 0-4
    """
    try:
        import wfdb
    except ImportError:
        raise ImportError(
            "wfdb is required. Install it with:  pip install wfdb"
        )

    os.makedirs(DATA_DIR, exist_ok=True)

    if records is None:
        records = MIT_BIH_RECORDS

    all_X, all_y = [], []
    failed = []

    for rec_name in records:
        try:
            X_rec, y_rec = _process_record(rec_name, wfdb, verbose)
            if len(X_rec) > 0:
                all_X.append(X_rec)
                all_y.append(y_rec)
                if verbose:
                    print(f"  [{rec_name}]  {len(X_rec)} beats extracted")
        except Exception as exc:
            failed.append(rec_name)
            if verbose:
                print(f"  [{rec_name}]  FAILED: {exc}")

    if failed and verbose:
        print(f"\nSkipped records: {failed}")

    if not all_X:
        raise RuntimeError("No records were loaded successfully.")

    X = np.vstack(all_X).astype(np.float32)
    y = np.concatenate(all_y).astype(np.int64)

    if verbose:
        _print_class_distribution(y)

    return X, y


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _process_record(rec_name, wfdb, verbose):
    """Load one MIT-BIH record, preprocess, segment, extract features."""
    # --- 1. Load (cache locally) -------------------------------------------
    cache_path = os.path.join(DATA_DIR, rec_name)
    if os.path.exists(cache_path + ".hea"):
        record = wfdb.rdrecord(cache_path)
        annotation = wfdb.rdann(cache_path, "atr")
    else:
        if verbose:
            print(f"  [{rec_name}]  downloading from PhysioNet...")
        record = wfdb.rdrecord(rec_name, pn_dir="mitdb",
                               sampto=None, smooth_frames=True)
        annotation = wfdb.rdann(rec_name, "atr", pn_dir="mitdb")
        # Save locally for future runs
        wfdb.wrsamp(
            rec_name,
            fs=record.fs,
            units=record.units,
            sig_name=record.sig_name,
            p_signal=record.p_signal,
            write_dir=DATA_DIR,
        )
        wfdb.wrann(rec_name, "atr",
                   annotation.sample, annotation.symbol,
                   write_dir=DATA_DIR)

    fs = record.fs  # 360 Hz for MIT-BIH

    # --- 2. Extract MLII lead (channel 0) ----------------------------------
    raw = record.p_signal[:, 0].astype(float)
    nan_mask = np.isnan(raw)
    if nan_mask.any():
        raw[nan_mask] = np.interp(
            np.where(nan_mask)[0],
            np.where(~nan_mask)[0],
            raw[~nan_mask],
        )

    # --- 3. Preprocess -------------------------------------------------------
    ecg = preprocess_ecg(raw, fs)

    # --- 4. Use annotated R-peak positions as ground truth ------------------
    ann_samples = annotation.sample
    ann_symbols = annotation.symbol

    # Filter to beats only (skip non-beat markers)
    beat_mask = np.array([
        s in NORMAL_SYMBOLS or s in ARRHYTHMIA_SYMBOLS
        for s in ann_symbols
    ])
    beat_samples = ann_samples[beat_mask]
    beat_symbols = np.array(ann_symbols)[beat_mask]

    if len(beat_samples) < 5:
        return np.empty((0, 20)), np.empty(0)

    # --- 5. Compute RR intervals -------------------------------------------
    rr_sec = np.diff(beat_samples) / fs
    # Pad so rr_sec has same length as beat_samples
    rr_sec = np.concatenate([[rr_sec[0]], rr_sec])

    # --- 6. Compute PQRST segments -----------------------------------------
    segments = compute_pqrst_segments(ecg, beat_samples, fs)

    # --- 7. Build per-beat feature vectors ----------------------------------
    X_list, y_list = [], []

    median_rr = float(np.median(rr_sec))

    for i, (seg, sym) in enumerate(zip(segments, beat_symbols)):
        feat = _beat_feature_vector(seg, i, rr_sec, beat_samples, fs, ecg, median_rr)
        label = _assign_label(sym, feat, median_rr)
        X_list.append(feat)
        y_list.append(label)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)


def _beat_feature_vector(seg, idx, rr_sec, beat_samples, fs, ecg, median_rr):
    """
    Build the 20-dimensional feature vector for a single beat.
    Matches the layout expected by build_feature_vector() / model.py.
    """
    rr_i = rr_sec[idx]

    # Local RR window (up to 4 neighbours)
    lo = max(0, idx - 2)
    hi = min(len(rr_sec), idx + 3)
    local_rr = rr_sec[lo:hi]
    mean_rr = float(np.mean(local_rr))
    std_rr = float(np.std(local_rr))

    hr = 60.0 / mean_rr if mean_rr > 0 else 0.0
    hrv_sdnn = std_rr * 1000                              # ms
    hrv_rmssd = float(np.sqrt(np.mean(np.diff(local_rr) ** 2))) * 1000 if len(local_rr) > 1 else 0.0
    rr_cv = std_rr / mean_rr if mean_rr > 0 else 0.0
    ectopic = float(abs(rr_i - median_rr) > 0.2 * median_rr)

    st = seg.get("st_level", 0.0) * 100   # scale to match train_model units
    r_amp = seg.get("r_amp", 0.0)

    # QRS duration
    q_idx = seg.get("q_idx", seg["r_idx"])
    s_idx = seg.get("s_idx", seg["r_idx"])
    qrs_ms = max(0.0, (s_idx - q_idx) / fs * 1000)

    # PR interval
    p_idx = seg.get("p_idx", seg["r_idx"] - int(0.15 * fs))
    pr_ms = (seg["r_idx"] - p_idx) / fs * 1000
    pr_ms = float(np.clip(pr_ms, 50, 400))

    # QT interval
    t_idx = seg.get("t_idx", seg["r_idx"] + int(0.2 * fs))
    qt_ms = (t_idx - q_idx) / fs * 1000
    qt_ms = float(np.clip(qt_ms, 150, 700))

    t_amp = seg.get("t_amp", 0.0)
    t_inv = float(t_amp < -0.05)
    p_amp = seg.get("p_amp", 0.0)
    p_absent = float(abs(p_amp) < 0.03)

    # Local signal statistics
    win_start = max(0, seg["r_idx"] - int(0.3 * fs))
    win_end = min(len(ecg), seg["r_idx"] + int(0.5 * fs))
    win = ecg[win_start:win_end]
    skew = _skewness(win)
    kurt = _kurtosis(win)

    vec = np.array([
        hr,
        hrv_sdnn,
        hrv_rmssd,
        rr_cv,
        ectopic,
        st,           # st_deviation_mean * 100
        st,           # st_elevation_max * 100
        st,           # st_depression_max * 100
        0.0,          # st_std (single beat → 0)
        t_amp * 10,
        t_inv,
        r_amp,
        0.0,          # r_amplitude_std (single beat → 0)
        qrs_ms,
        pr_ms,
        qt_ms,
        p_absent,
        skew,
        kurt,
        float(len(beat_samples)),
    ], dtype=np.float32)

    return vec


def _assign_label(symbol, feat_vec, median_rr):
    """
    Map an annotation symbol + local features to one of 5 classes.

    Priority:
      Arrhythmia annotation  → 1
      Tachycardia (HR > 100) → 2
      Bradycardia (HR < 60)  → 3
      ST elevation           → 4
      Otherwise              → 0 (Normal)
    """
    if symbol in ARRHYTHMIA_SYMBOLS:
        return 1

    hr = feat_vec[0]
    st_elev = feat_vec[6]   # st_elevation_max * 100
    t_inv = feat_vec[10]

    if hr > 100:
        return 2
    if hr < 60:
        return 3
    if st_elev > 5.0 or t_inv > 0:   # > 0.05 mV after ×100 scaling
        return 4

    return 0


def _print_class_distribution(y):
    print("\n=== Розподіл класів у датасеті ===")
    for cls_idx, name in enumerate(CLASS_NAMES):
        count = int(np.sum(y == cls_idx))
        pct = 100.0 * count / len(y) if len(y) > 0 else 0
        print(f"  {cls_idx} – {name:<38}: {count:>6}  ({pct:.1f}%)")
    print(f"  Всього: {len(y)} зразків\n")


def _skewness(x):
    if len(x) < 3:
        return 0.0
    m = np.mean(x)
    s = np.std(x)
    return float(np.mean(((x - m) / s) ** 3)) if s > 1e-10 else 0.0


def _kurtosis(x):
    if len(x) < 4:
        return 3.0
    m = np.mean(x)
    s = np.std(x)
    return float(np.mean(((x - m) / s) ** 4)) if s > 1e-10 else 3.0
