import numpy as np
from scipy import signal as sp_signal


def preprocess_ecg(ecg_signal, fs=360):
    ecg = ecg_signal.astype(float)
    ecg = ecg - np.mean(ecg)
    ecg = _bandpass_filter(ecg, 0.5, 40.0, fs)
    powerline_freq = 50.0
    if fs > 2 * powerline_freq:
        ecg = _notch_filter(ecg, powerline_freq, fs)
    if fs > 120:
        ecg = _notch_filter(ecg, 60.0, fs)
    ecg = _remove_baseline_wander(ecg, fs)
    ecg = _normalize(ecg)
    return ecg


def _bandpass_filter(ecg, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = min(highcut / nyq, 0.99)
    if low <= 0 or high <= low:
        return ecg
    try:
        b, a = sp_signal.butter(order, [low, high], btype="band")
        return sp_signal.filtfilt(b, a, ecg)
    except Exception:
        return ecg


def _notch_filter(ecg, freq, fs, quality=30):
    nyq = 0.5 * fs
    freq_norm = freq / nyq
    if freq_norm >= 1.0 or freq_norm <= 0:
        return ecg
    try:
        b, a = sp_signal.iirnotch(freq_norm, quality)
        return sp_signal.filtfilt(b, a, ecg)
    except Exception:
        return ecg


def _remove_baseline_wander(ecg, fs):
    nyq = 0.5 * fs
    cutoff = 0.5 / nyq
    if cutoff >= 1.0 or cutoff <= 0:
        return ecg
    try:
        b, a = sp_signal.butter(2, cutoff, btype="high")
        return sp_signal.filtfilt(b, a, ecg)
    except Exception:
        return ecg


def _normalize(ecg):
    rms = np.sqrt(np.mean(ecg ** 2))
    if rms > 1e-10:
        return ecg / (rms * 3.0)
    return ecg


def compute_pqrst_segments(ecg, r_peaks, fs):
    segments = []
    for rp in r_peaks:
        seg = {}
        pre = int(0.2 * fs)
        post = int(0.4 * fs)
        start = max(0, rp - pre)
        end = min(len(ecg), rp + post)
        seg["r_idx"] = rp
        seg["start"] = start
        seg["end"] = end
        seg["r_amp"] = float(ecg[rp])

        q_search_start = max(0, rp - int(0.05 * fs))
        q_search_end = rp
        if q_search_end > q_search_start:
            q_local = np.argmin(ecg[q_search_start:q_search_end])
            seg["q_idx"] = q_search_start + q_local
            seg["q_amp"] = float(ecg[seg["q_idx"]])
        else:
            seg["q_idx"] = rp
            seg["q_amp"] = 0.0

        s_search_start = rp
        s_search_end = min(len(ecg), rp + int(0.05 * fs))
        if s_search_end > s_search_start:
            s_local = np.argmin(ecg[s_search_start:s_search_end])
            seg["s_idx"] = s_search_start + s_local
            seg["s_amp"] = float(ecg[seg["s_idx"]])
        else:
            seg["s_idx"] = rp
            seg["s_amp"] = 0.0

        j_point = min(len(ecg) - 1, seg["s_idx"] + int(0.04 * fs))
        st_end = min(len(ecg) - 1, seg["s_idx"] + int(0.12 * fs))
        if st_end > j_point:
            seg["st_level"] = float(np.mean(ecg[j_point:st_end]))
        else:
            seg["st_level"] = 0.0

        t_start = min(len(ecg) - 1, seg["s_idx"] + int(0.08 * fs))
        t_end = min(len(ecg), rp + int(0.35 * fs))
        if t_end > t_start:
            t_local = np.argmax(np.abs(ecg[t_start:t_end]))
            seg["t_idx"] = t_start + t_local
            seg["t_amp"] = float(ecg[seg["t_idx"]])
        else:
            seg["t_idx"] = rp + int(0.2 * fs)
            seg["t_amp"] = 0.0

        p_start = max(0, rp - int(0.25 * fs))
        p_end = max(0, rp - int(0.05 * fs))
        if p_end > p_start:
            p_local = np.argmax(ecg[p_start:p_end])
            seg["p_idx"] = p_start + p_local
            seg["p_amp"] = float(ecg[seg["p_idx"]])
        else:
            seg["p_idx"] = max(0, rp - int(0.15 * fs))
            seg["p_amp"] = 0.0

        segments.append(seg)
    return segments
