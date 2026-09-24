"""
Generate synthetic ECG sample files for the ECG Diagnosis System.
Run:  python generate_samples.py
"""

import os
import numpy as np

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(42)

FS = 360


def gaussian_pulse(t, center, amplitude, width):
    return amplitude * np.exp(-((t - center) ** 2) / (2 * width ** 2))


def ecg_beat(t_beat, r_amplitude=1.0, st_offset=0.0, t_amplitude=0.3, qrs_width=0.04):
    """Generate one PQRST beat starting at t=0 with beat duration ~0.8 s"""
    ecg = np.zeros_like(t_beat)

    ecg += gaussian_pulse(t_beat, 0.1, 0.12, 0.025)

    ecg += gaussian_pulse(t_beat, 0.175, -0.08, 0.012)
    ecg += gaussian_pulse(t_beat, 0.20, r_amplitude, qrs_width / 2)
    ecg += gaussian_pulse(t_beat, 0.225, -0.15, 0.012)

    st_start = 0.24
    st_end = 0.36
    mask = (t_beat >= st_start) & (t_beat <= st_end)
    ecg[mask] += st_offset * np.linspace(1, 1, np.sum(mask))

    t_center = 0.38
    ecg += gaussian_pulse(t_beat, t_center, t_amplitude, 0.04)

    return ecg


def generate_signal(duration_s, fs, beat_fn, rr_mean, rr_std=0.0, noise_level=0.02):
    n_samples = int(duration_s * fs)
    signal = np.zeros(n_samples)
    t_all = np.arange(n_samples) / fs

    current_sample = int(0.3 * fs)
    while current_sample < n_samples - int(1.2 * fs):
        rr = rng.normal(rr_mean, rr_std) if rr_std > 0 else rr_mean
        rr = max(0.3, min(2.0, rr))
        beat_samples = int(rr * fs)
        end_sample = min(current_sample + beat_samples, n_samples)
        t_beat = np.arange(end_sample - current_sample) / fs
        beat = beat_fn(t_beat)
        signal[current_sample:end_sample] += beat[:end_sample - current_sample]
        current_sample += beat_samples

    noise = rng.normal(0, noise_level, n_samples)
    signal += noise

    return signal, fs


def save_csv(signal, fs, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    n = len(signal)
    time = np.arange(n) / fs
    data = np.column_stack([time, signal])
    header = f"# Synthetic ECG — {filename}\n# time_s,amplitude_mV"
    np.savetxt(path, data, delimiter=",", header=header, comments="", fmt="%.6f")
    print(f"  Saved: {path}  ({n} samples @ {fs} Hz, {n/fs:.1f} s)")


def generate_normal():
    def beat(t):
        return ecg_beat(t, r_amplitude=1.0, st_offset=0.0, t_amplitude=0.30, qrs_width=0.04)
    return generate_signal(30, FS, beat, rr_mean=0.833, rr_std=0.025)


def generate_tachycardia():
    def beat(t):
        return ecg_beat(t, r_amplitude=1.05, st_offset=0.0, t_amplitude=0.28, qrs_width=0.038)
    return generate_signal(20, FS, beat, rr_mean=0.50, rr_std=0.015)


def generate_bradycardia():
    def beat(t):
        return ecg_beat(t, r_amplitude=1.0, st_offset=0.0, t_amplitude=0.35, qrs_width=0.045)
    return generate_signal(40, FS, beat, rr_mean=1.30, rr_std=0.025)


def generate_arrhythmia():
    beat_count = [0]

    def beat(t):
        beat_count[0] += 1
        if beat_count[0] % 5 == 0:
            return ecg_beat(t, r_amplitude=0.6, st_offset=0.0, t_amplitude=0.15, qrs_width=0.06)
        return ecg_beat(t, r_amplitude=1.0, st_offset=0.0, t_amplitude=0.28, qrs_width=0.04)

    return generate_signal(30, FS, beat, rr_mean=0.78, rr_std=0.12)


def generate_myocardial_infarction():
    def beat(t):
        b = ecg_beat(t, r_amplitude=0.65, st_offset=0.12, t_amplitude=-0.18, qrs_width=0.055)
        return b

    return generate_signal(25, FS, beat, rr_mean=0.70, rr_std=0.04)


if __name__ == "__main__":
    print("Generating synthetic ECG sample files...")

    samples = [
        ("ecg_normal.csv",      generate_normal),
        ("ecg_tachycardia.csv", generate_tachycardia),
        ("ecg_bradycardia.csv", generate_bradycardia),
        ("ecg_arrhythmia.csv",  generate_arrhythmia),
        ("ecg_mi.csv",          generate_myocardial_infarction),
    ]

    for fname, gen_fn in samples:
        signal, fs = gen_fn()
        save_csv(signal, fs, fname)

    print("\nAll sample files generated successfully.")
    print("You can now load them from the application's 'Quick Load Sample' panel.")
