import numpy as np
from scipy import signal as sp_signal


def detect_r_peaks(ecg, fs=360):
    r_peaks = _pan_tompkins(ecg, fs)
    r_peaks = _refine_peaks(ecg, r_peaks, fs)
    return r_peaks


def _pan_tompkins(ecg, fs):
    # 1. Bandpass filter 5-15 Hz
    nyq = 0.5 * fs
    low = max(5.0 / nyq, 1e-4)
    high = min(15.0 / nyq, 0.99)
    if low < high:
        try:
            b, a = sp_signal.butter(4, [low, high], btype="band")
            filtered = sp_signal.filtfilt(b, a, ecg)
        except Exception:
            filtered = ecg.copy()
    else:
        filtered = ecg.copy()

    # 2. Differentiation
    diff = np.diff(filtered, prepend=filtered[0])

    # 3. Squaring
    squared = diff ** 2

    # 4. Moving window integration
    win_size = max(1, int(0.15 * fs))
    kernel = np.ones(win_size) / win_size
    integrated = np.convolve(squared, kernel, mode="same")

    # 5. Adaptive threshold and peak detection
    threshold = 0.5 * np.mean(integrated)
    min_distance = int(0.2 * fs)

    peaks, properties = sp_signal.find_peaks(
        integrated,
        height=threshold,
        distance=min_distance
    )

    if len(peaks) < 3:
        # Fallback: lower threshold
        threshold = 0.1 * np.max(integrated)
        peaks, _ = sp_signal.find_peaks(
            integrated,
            height=threshold,
            distance=min_distance
        )

    return peaks


def _refine_peaks(ecg, peaks, fs):
    search_radius = int(0.05 * fs)
    refined = []
    for p in peaks:
        start = max(0, p - search_radius)
        end = min(len(ecg), p + search_radius)
        local_max = start + np.argmax(ecg[start:end])
        refined.append(local_max)
    refined = sorted(set(refined))
    return np.array(refined, dtype=int)


def compute_rr_intervals(r_peaks, fs):
    if len(r_peaks) < 2:
        return np.array([])
    rr = np.diff(r_peaks) / fs
    return rr


def filter_ectopic_beats(rr_intervals, threshold=0.2):
    if len(rr_intervals) < 3:
        return rr_intervals, np.ones(len(rr_intervals), dtype=bool)

    median_rr = np.median(rr_intervals)
    normal_mask = np.abs(rr_intervals - median_rr) < threshold * median_rr
    return rr_intervals[normal_mask], normal_mask
