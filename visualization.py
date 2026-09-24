import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure


DARK_BG = "#0d1b2a"
PANEL_BG = "#0a1520"
ECG_COLOR = "#00e5ff"
R_PEAK_COLOR = "#ff4081"
ST_COLOR = "#ff6b00"
ARRHYTHMIA_COLOR = "#cc00ff"
GRID_COLOR = "#1e3a5f"
ACCENT = "#00c8ff"
TEXT_COLOR = "#8ab4cc"


def create_ecg_figure(ecg_signal, fs, r_peaks=None, st_regions=None,
                      arrhythmia_regions=None, title="ECG Signal"):
    fig = Figure(figsize=(14, 5), facecolor=DARK_BG, tight_layout=True)
    ax = fig.add_subplot(111)
    ax.set_facecolor(PANEL_BG)

    max_samples = int(fs * 15)
    ecg_d = ecg_signal[:max_samples] if len(ecg_signal) > max_samples else ecg_signal
    time = np.arange(len(ecg_d)) / fs

    r_d = [r for r in (r_peaks or []) if r < len(ecg_d)]

    ax.plot(time, ecg_d, color=ECG_COLOR, linewidth=0.85, alpha=0.95, label="ECG Signal")

    if r_d:
        ax.scatter(
            np.array(r_d) / fs,
            ecg_d[r_d],
            color=R_PEAK_COLOR,
            s=40,
            zorder=5,
            label="R-peaks",
            marker="^",
        )

    if st_regions:
        first = True
        for seg in st_regions:
            t0, t1 = seg[0] / fs, min(seg[1] / fs, time[-1])
            lbl = "ST change" if first else "_"
            ax.axvspan(t0, t1, alpha=0.28, color=ST_COLOR, label=lbl)
            first = False

    if arrhythmia_regions:
        first = True
        for seg in arrhythmia_regions:
            t0, t1 = seg[0] / fs, min(seg[1] / fs, time[-1])
            lbl = "Arrhythmia" if first else "_"
            ax.axvspan(t0, t1, alpha=0.18, color=ARRHYTHMIA_COLOR, label=lbl)
            first = False

    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.set_xlabel("Time (s)", color=TEXT_COLOR, fontsize=10)
    ax.set_ylabel("Amplitude", color=TEXT_COLOR, fontsize=10)
    ax.set_title(title, color=ACCENT, fontsize=12, fontweight="bold")
    ax.grid(True, color=GRID_COLOR, linewidth=0.4, alpha=0.7)

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if unique:
        ax.legend(
            unique.values(), unique.keys(),
            facecolor="#162840",
            edgecolor=GRID_COLOR,
            labelcolor=TEXT_COLOR,
            fontsize=9,
            loc="upper right",
        )

    return fig


def save_ecg_plot(ecg_signal, fs, output_path, r_peaks=None, st_regions=None,
                  arrhythmia_regions=None, title="ECG Signal"):
    fig = create_ecg_figure(ecg_signal, fs, r_peaks, st_regions, arrhythmia_regions, title)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close("all")
