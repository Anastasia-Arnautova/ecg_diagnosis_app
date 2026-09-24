import numpy as np
from r_peak_detection import compute_rr_intervals, filter_ectopic_beats
from preprocessing import compute_pqrst_segments


def extract_features(ecg, r_peaks, fs):
    features = {}

    rr = compute_rr_intervals(r_peaks, fs)
    features["rr_intervals"] = rr

    if len(rr) > 0:
        normal_rr, normal_mask = filter_ectopic_beats(rr)
        mean_rr = np.mean(normal_rr) if len(normal_rr) > 0 else np.mean(rr)
        features["mean_rr"] = float(mean_rr)
        features["heart_rate"] = float(60.0 / mean_rr) if mean_rr > 0 else 0.0
        features["hrv_sdnn"] = float(np.std(rr)) if len(rr) > 1 else 0.0
        features["hrv_rmssd"] = float(np.sqrt(np.mean(np.diff(rr) ** 2))) if len(rr) > 1 else 0.0
        features["rr_cv"] = float(np.std(rr) / np.mean(rr)) if np.mean(rr) > 0 else 0.0
        features["irregular_beats"] = int(np.sum(~normal_mask))
        features["ectopic_ratio"] = float(np.sum(~normal_mask) / len(rr)) if len(rr) > 0 else 0.0
    else:
        features["mean_rr"] = 0.0
        features["heart_rate"] = 0.0
        features["hrv_sdnn"] = 0.0
        features["hrv_rmssd"] = 0.0
        features["rr_cv"] = 0.0
        features["irregular_beats"] = 0
        features["ectopic_ratio"] = 0.0

    segments = compute_pqrst_segments(ecg, r_peaks, fs)
    features["segments"] = segments

    if segments:
        st_levels = [s["st_level"] for s in segments]
        features["st_deviation_mean"] = float(np.mean(st_levels))
        features["st_elevation_max"] = float(np.max(st_levels))
        features["st_depression_max"] = float(np.min(st_levels))
        features["st_std"] = float(np.std(st_levels))

        t_amps = [s["t_amp"] for s in segments]
        features["t_wave_mean"] = float(np.mean(t_amps))
        features["t_wave_inverted_count"] = int(sum(1 for t in t_amps if t < -0.05))

        r_amps = [s["r_amp"] for s in segments]
        features["r_amplitude_mean"] = float(np.mean(r_amps))
        features["r_amplitude_std"] = float(np.std(r_amps))

        qrs_durations = []
        for s in segments:
            qrs_dur = (s["s_idx"] - s["q_idx"]) / fs * 1000
            qrs_durations.append(max(0.0, qrs_dur))
        features["qrs_duration_ms"] = float(np.mean(qrs_durations)) if qrs_durations else 0.0

        pr_intervals = []
        for s in segments:
            pr = (s["r_idx"] - s["p_idx"]) / fs * 1000
            if 50 < pr < 400:
                pr_intervals.append(pr)
        features["pr_interval_ms"] = float(np.mean(pr_intervals)) if pr_intervals else 160.0

        qt_intervals = []
        for s in segments:
            qt = (s["t_idx"] - s["q_idx"]) / fs * 1000
            if 200 < qt < 700:
                qt_intervals.append(qt)
        features["qt_interval_ms"] = float(np.mean(qt_intervals)) if qt_intervals else 400.0

        p_amps = [s["p_amp"] for s in segments]
        features["p_wave_mean"] = float(np.mean(p_amps))
        features["p_wave_absent"] = bool(np.mean(np.abs(p_amps)) < 0.03)

    else:
        features["st_deviation_mean"] = 0.0
        features["st_elevation_max"] = 0.0
        features["st_depression_max"] = 0.0
        features["st_std"] = 0.0
        features["t_wave_mean"] = 0.0
        features["t_wave_inverted_count"] = 0
        features["r_amplitude_mean"] = 0.0
        features["r_amplitude_std"] = 0.0
        features["qrs_duration_ms"] = 80.0
        features["pr_interval_ms"] = 160.0
        features["qt_interval_ms"] = 400.0
        features["p_wave_mean"] = 0.0
        features["p_wave_absent"] = False

    features["signal_power"] = float(np.mean(ecg ** 2))
    features["signal_skewness"] = float(_skewness(ecg))
    features["signal_kurtosis"] = float(_kurtosis(ecg))
    features["num_r_peaks"] = int(len(r_peaks))

    signal_length_s = len(ecg) / fs
    features["estimated_hr_from_peaks"] = float(60.0 * len(r_peaks) / signal_length_s) if signal_length_s > 0 else 0.0

    return features


def build_feature_vector(features):
    vec = np.array([
        features.get("heart_rate", 0),
        features.get("hrv_sdnn", 0) * 1000,
        features.get("hrv_rmssd", 0) * 1000,
        features.get("rr_cv", 0),
        features.get("ectopic_ratio", 0),
        features.get("st_deviation_mean", 0) * 100,
        features.get("st_elevation_max", 0) * 100,
        features.get("st_depression_max", 0) * 100,
        features.get("st_std", 0) * 100,
        features.get("t_wave_mean", 0) * 10,
        features.get("t_wave_inverted_count", 0),
        features.get("r_amplitude_mean", 0),
        features.get("r_amplitude_std", 0),
        features.get("qrs_duration_ms", 80),
        features.get("pr_interval_ms", 160),
        features.get("qt_interval_ms", 400),
        float(features.get("p_wave_absent", False)),
        features.get("signal_skewness", 0),
        features.get("signal_kurtosis", 3),
        features.get("num_r_peaks", 0),
    ], dtype=np.float32)
    return vec


def _skewness(x):
    n = len(x)
    if n < 3:
        return 0.0
    mean = np.mean(x)
    std = np.std(x)
    if std < 1e-10:
        return 0.0
    return float(np.mean(((x - mean) / std) ** 3))


def _kurtosis(x):
    n = len(x)
    if n < 4:
        return 3.0
    mean = np.mean(x)
    std = np.std(x)
    if std < 1e-10:
        return 3.0
    return float(np.mean(((x - mean) / std) ** 4))
