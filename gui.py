import os
import traceback

import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon, QPixmap
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QPushButton, QLabel, QTextEdit, QFileDialog, QFrame,
    QProgressBar, QStatusBar, QGroupBox, QScrollArea, QMessageBox,
    QSizePolicy, QTabWidget,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar2QT
import matplotlib.pyplot as plt
from ua_patches import patch_dialog

from ecg_loader import load_ecg, IMAGE_EXTENSIONS
from inference import analyze_ecg


class UkrainianNavigationToolbar(NavigationToolbar2QT):
    """NavigationToolbar with Ukrainian dialog translations."""

    def configure_subplots(self):
        super().configure_subplots()
        # The subplot tool is stored as self.subplot_tool after the call
        tool = getattr(self, "subplot_tool", None)
        if tool is not None:
            patch_dialog(tool)

    def edit_parameters(self):
        super().edit_parameters()
        # Figure options dialog is a child of the canvas manager's window
        from PyQt5.QtWidgets import QApplication
        for widget in QApplication.topLevelWidgets():
            title = getattr(widget, "windowTitle", lambda: "")()
            if title in ("Figure options", "Параметри графіку"):
                patch_dialog(widget)
                break


DARK_BG = "#0d1b2a"
PANEL_BG = "#112233"
CARD_BG = "#162840"
ACCENT = "#00c8ff"
ACCENT2 = "#00ff9f"
TEXT_PRIMARY = "#e8f4fd"
TEXT_SECONDARY = "#8ab4cc"
BORDER = "#1e3a5f"
WARNING = "#ff6b6b"
SUCCESS = "#00ff9f"
MUTED = "#3a5a78"


STYLE = f"""
QMainWindow {{
    background-color: {DARK_BG};
}}
QWidget {{
    background-color: {DARK_BG};
    color: {TEXT_PRIMARY};
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
}}
QGroupBox {{
    background-color: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 8px;
    margin-top: 12px;
    padding: 8px;
    font-size: 12px;
    font-weight: bold;
    color: {ACCENT};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 2px 10px;
    color: {ACCENT};
    font-weight: bold;
    font-size: 12px;
    letter-spacing: 1px;
    text-transform: uppercase;
}}
QPushButton {{
    background-color: {ACCENT};
    color: white;
    border: none;
    border-radius: 6px;
    padding: 10px 18px;
    font-weight: bold;
    font-size: 13px;
    letter-spacing: 0.5px;
    min-height: 38px;
}}
QPushButton:hover {{
    background-color: #33d6ff;
}}
QPushButton:pressed {{
    background-color: #0099cc;
}}
QPushButton:disabled {{
    background-color: {MUTED};
    color: {TEXT_SECONDARY};
}}
QPushButton#analyze_btn {{
    background-color: {ACCENT2};
    color: white;
    font-size: 14px;
    min-height: 44px;
    border-radius: 8px;
    font-weight: bold;
    letter-spacing: 1px;
}}
QPushButton#analyze_btn:hover {{
    background-color: #33ffb8;
}}
QPushButton#analyze_btn:disabled {{
    background-color: {MUTED};
    color: {TEXT_SECONDARY};
}}
QTextEdit {{
    background-color: {PANEL_BG};
    border: 1px solid {BORDER};
    border-radius: 6px;
    color: {TEXT_PRIMARY};
    padding: 8px;
    font-size: 13px;
    selection-background-color: {ACCENT};
    selection-color: #001a2e;
}}
QLabel {{
    background-color: transparent;
    color: {TEXT_SECONDARY};
    font-size: 12px;
}}
QLabel#title_label {{
    color: {ACCENT};
    font-size: 18px;
    font-weight: bold;
    letter-spacing: 1px;
}}
QLabel#subtitle_label {{
    color: {TEXT_SECONDARY};
    font-size: 11px;
    letter-spacing: 0.5px;
}}
QLabel#file_label {{
    color: {TEXT_SECONDARY};
    font-size: 11px;
    padding: 4px;
    border: 1px dashed {MUTED};
    border-radius: 4px;
    background-color: {PANEL_BG};
}}
QLabel#file_label_loaded {{
    color: {ACCENT2};
    font-size: 11px;
    padding: 4px;
    border: 1px solid {ACCENT2};
    border-radius: 4px;
    background-color: {PANEL_BG};
}}
QProgressBar {{
    background-color: {PANEL_BG};
    border: 1px solid {BORDER};
    border-radius: 4px;
    text-align: center;
    color: {TEXT_PRIMARY};
    height: 16px;
    font-size: 11px;
}}
QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 3px;
}}
QScrollBar:vertical {{
    background: {PANEL_BG};
    width: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {MUTED};
    border-radius: 4px;
    min-height: 30px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QStatusBar {{
    background-color: {PANEL_BG};
    color: {TEXT_SECONDARY};
    border-top: 1px solid {BORDER};
    font-size: 11px;
}}
QSplitter::handle {{
    background-color: {BORDER};
}}
"""


class AnalysisWorker(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, ecg_signal, fs, symptoms):
        super().__init__()
        self.ecg_signal = ecg_signal
        self.fs = fs
        self.symptoms = symptoms

    def run(self):
        try:
            result = analyze_ecg(self.ecg_signal, self.fs, self.symptoms)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(traceback.format_exc())


class ECGCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig, self.ax = plt.subplots(figsize=(12, 4))
        self.fig.patch.set_facecolor(DARK_BG)
        self.ax.set_facecolor("#0a1520")
        self.ax.tick_params(colors=TEXT_SECONDARY)
        for spine in self.ax.spines.values():
            spine.set_edgecolor(BORDER)
        self.ax.set_xlabel("Time (s)", color=TEXT_SECONDARY, fontsize=10)
        self.ax.set_ylabel("Amplitude (mV)", color=TEXT_SECONDARY, fontsize=10)
        self.ax.set_title("ECG Signal — No file loaded", color=ACCENT, fontsize=12, fontweight="bold")
        self.fig.tight_layout(pad=1.5)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def plot_ecg(self, ecg_signal, fs, r_peaks=None, st_regions=None, arrhythmia_regions=None,
                 title="ECG Signal"):
        self.ax.cla()
        self.ax.set_facecolor("#0a1520")

        time = np.arange(len(ecg_signal)) / fs

        max_display_samples = int(fs * 15)
        if len(ecg_signal) > max_display_samples:
            ecg_display = ecg_signal[:max_display_samples]
            time_display = time[:max_display_samples]
            r_peaks_display = [r for r in (r_peaks or []) if r < max_display_samples]
        else:
            ecg_display = ecg_signal
            time_display = time
            r_peaks_display = r_peaks or []

        self.ax.plot(time_display, ecg_display, color="#00e5ff", linewidth=0.9, alpha=0.95, label="ECG")

        if r_peaks_display:
            self.ax.scatter(
                np.array(r_peaks_display) / fs,
                ecg_display[r_peaks_display],
                color="#ff4081", s=35, zorder=5, label="R-peaks", marker="^"
            )

        if st_regions:
            for seg in st_regions:
                start_t = seg[0] / fs
                end_t = seg[1] / fs
                if start_t < time_display[-1]:
                    end_t = min(end_t, time_display[-1])
                    self.ax.axvspan(start_t, end_t, alpha=0.25, color="#ff6b00", label="ST change")

        if arrhythmia_regions:
            for seg in arrhythmia_regions:
                start_t = seg[0] / fs
                end_t = seg[1] / fs
                if start_t < time_display[-1]:
                    end_t = min(end_t, time_display[-1])
                    self.ax.axvspan(start_t, end_t, alpha=0.18, color="#cc00ff", label="Arrhythmia")

        for spine in self.ax.spines.values():
            spine.set_edgecolor(BORDER)
        self.ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
        self.ax.set_xlabel("Time (s)", color=TEXT_SECONDARY, fontsize=10)
        self.ax.set_ylabel("Amplitude (mV)", color=TEXT_SECONDARY, fontsize=10)
        self.ax.set_title(title, color=ACCENT, fontsize=12, fontweight="bold")
        self.ax.grid(True, color=BORDER, linewidth=0.4, alpha=0.7)

        handles, labels = self.ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        if unique:
            self.ax.legend(unique.values(), unique.keys(),
                           facecolor=CARD_BG, edgecolor=BORDER,
                           labelcolor=TEXT_PRIMARY, fontsize=9, loc="upper right")

        self.fig.tight_layout(pad=1.5)
        self.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ECG Diagnosis System — Heart Disease Analysis Module")
        self.setMinimumSize(1280, 800)
        self.resize(1400, 880)

        self.ecg_signal = None
        self.ecg_fs = 360
        self.analysis_result = None
        self.worker = None

        self.setStyleSheet(STYLE)
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        root_layout.addWidget(self._build_header())

        splitter_h = QSplitter(Qt.Horizontal)
        splitter_h.setHandleWidth(2)

        left_panel = self._build_left_panel()
        left_panel.setMinimumWidth(280)
        left_panel.setMaximumWidth(340)
        splitter_h.addWidget(left_panel)

        right_area = QWidget()
        right_layout = QVBoxLayout(right_area)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        splitter_v = QSplitter(Qt.Vertical)
        splitter_v.setHandleWidth(2)
        splitter_v.addWidget(self._build_graph_panel())
        splitter_v.addWidget(self._build_report_panel())
        splitter_v.setSizes([420, 300])

        right_layout.addWidget(splitter_v)
        splitter_h.addWidget(right_area)
        splitter_h.setSizes([310, 1060])

        root_layout.addWidget(splitter_h, 1)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Готово — завантажте ЕКГ файл для початку аналізу.")

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(200)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

    def _build_header(self):
        header = QWidget()
        header.setFixedHeight(70)
        header.setStyleSheet(f"background-color: {PANEL_BG}; border-bottom: 1px solid {BORDER};")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 8, 20, 8)

        icon_lbl = QLabel("♥")
        icon_lbl.setStyleSheet(f"color: {WARNING}; font-size: 28px; background: transparent;")
        layout.addWidget(icon_lbl)

        text_col = QVBoxLayout()
        title_lbl = QLabel("ECG DIAGNOSIS SYSTEM")
        title_lbl.setObjectName("title_label")
        subtitle_lbl = QLabel("Software Module for Diagnosing Heart Diseases Based on ECG Signal Analysis")
        subtitle_lbl.setObjectName("subtitle_label")
        text_col.addWidget(title_lbl)
        text_col.addWidget(subtitle_lbl)
        layout.addLayout(text_col)
        layout.addStretch()

        version_lbl = QLabel("Arnautova Anastasia  |  Diploma Project  |  2026")
        version_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px; background: transparent;")
        layout.addWidget(version_lbl)

        return header

    def _build_left_panel(self):
        panel = QWidget()
        panel.setStyleSheet(f"background-color: {PANEL_BG}; border-right: 1px solid {BORDER};")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 14, 12, 14)
        layout.setSpacing(14)

        load_group = QGroupBox("ЕКГ ФАЙЛ")
        load_layout = QVBoxLayout(load_group)

        self.file_label = QLabel("Файл не обрано")
        self.file_label.setObjectName("file_label")
        self.file_label.setWordWrap(True)
        self.file_label.setAlignment(Qt.AlignCenter)
        self.file_label.setMinimumHeight(40)
        load_layout.addWidget(self.file_label)

        load_btn = QPushButton("  Завантажити ЕКГ файл")
        load_btn.setToolTip("Підтримувані формати: .csv, .txt, .dat, .mat")
        load_btn.clicked.connect(self._load_ecg_file)
        load_layout.addWidget(load_btn)

        format_lbl = QLabel("Формати: .csv  .txt  .dat  .mat\n.png  .jpg  .jpeg  .bmp  .tiff")
        format_lbl.setAlignment(Qt.AlignCenter)
        format_lbl.setStyleSheet(f"color: {MUTED}; font-size: 10px; background: transparent;")
        load_layout.addWidget(format_lbl)

        layout.addWidget(load_group)

        sym_group = QGroupBox("СИМПТОМИ ПАЦІЄНТА")
        sym_layout = QVBoxLayout(sym_group)
        sym_hint = QLabel("Введіть симптоми (кожен з нового рядка або через кому):")
        sym_hint.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px; background: transparent;")
        sym_hint.setWordWrap(True)
        sym_layout.addWidget(sym_hint)

        self.symptoms_edit = QTextEdit()
        self.symptoms_edit.setPlaceholderText(
            "Наприклад:\nбіль у грудях\nзапаморочення\nзадишка\nслабкість\nсерцебиття\nаритмія"
        )
        self.symptoms_edit.setFixedHeight(130)
        sym_layout.addWidget(self.symptoms_edit)
        layout.addWidget(sym_group)

        self.analyze_btn = QPushButton("  АНАЛІЗУВАТИ ЕКГ")
        self.analyze_btn.setObjectName("analyze_btn")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.clicked.connect(self._run_analysis)
        layout.addWidget(self.analyze_btn)

        sample_group = QGroupBox("ШВИДКЕ ЗАВАНТАЖЕННЯ")
        sample_layout = QVBoxLayout(sample_group)

        for label, fname in [("Норма",                  "ecg_normal.csv"),
                              ("Аритмія",                "ecg_arrhythmia.csv"),
                              ("Тахікардія",             "ecg_tachycardia.csv"),
                              ("Брадикардія",            "ecg_bradycardia.csv"),
                              ("Інфаркт міокарда",       "ecg_mi.csv")]:
            btn = QPushButton(label)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {CARD_BG};
                    color: {TEXT_PRIMARY};
                    border: 1px solid {BORDER};
                    border-radius: 5px;
                    padding: 6px 10px;
                    font-size: 12px;
                    min-height: 28px;
                }}
                QPushButton:hover {{
                    background-color: {BORDER};
                    color: {ACCENT};
                }}
            """)
            btn.setProperty("sample_file", fname)
            btn.clicked.connect(self._load_sample)
            sample_layout.addWidget(btn)

        layout.addWidget(sample_group)
        layout.addStretch()

        info_lbl = QLabel("ЕКГ-аналіз на основі НМ\nПравила + Нейронна мережа")
        info_lbl.setAlignment(Qt.AlignCenter)
        info_lbl.setStyleSheet(f"color: {MUTED}; font-size: 10px; background: transparent;")
        layout.addWidget(info_lbl)

        return panel

    def _build_graph_panel(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 6)
        layout.setSpacing(4)

        graph_group = QGroupBox("ВІЗУАЛІЗАЦІЯ ЕКГ СИГНАЛУ")
        graph_layout = QVBoxLayout(graph_group)
        graph_layout.setContentsMargins(4, 8, 4, 4)

        # Tab widget: [Оцифрований сигнал] + [Оригінальне фото] (hidden by default)
        self.graph_tabs = QTabWidget()
        self.graph_tabs.setStyleSheet(f"""
            QTabWidget::pane {{ border: none; background: {CARD_BG}; }}
            QTabBar::tab {{
                background: {PANEL_BG}; color: {TEXT_SECONDARY};
                padding: 6px 16px; border-radius: 4px 4px 0 0;
                border: 1px solid {BORDER};
            }}
            QTabBar::tab:selected {{ background: {CARD_BG}; color: {ACCENT}; border-bottom: 2px solid {ACCENT}; }}
        """)

        # --- Tab 1: digitised signal (always visible) ----------------------
        signal_tab = QWidget()
        signal_layout = QVBoxLayout(signal_tab)
        signal_layout.setContentsMargins(0, 0, 0, 0)
        self.ecg_canvas = ECGCanvas()
        toolbar = UkrainianNavigationToolbar(self.ecg_canvas, container)
        toolbar.setStyleSheet(f"background-color: {CARD_BG}; color: {TEXT_SECONDARY};")
        signal_layout.addWidget(toolbar)
        signal_layout.addWidget(self.ecg_canvas)
        self.graph_tabs.addTab(signal_tab, "Оцифрований сигнал")

        # --- Tab 2: original image (shown only when image is loaded) -------
        self.image_tab = QWidget()
        image_layout = QVBoxLayout(self.image_tab)
        image_layout.setContentsMargins(4, 4, 4, 4)
        self.image_preview = QLabel("Оригінальне зображення ЕКГ")
        self.image_preview.setAlignment(Qt.AlignCenter)
        self.image_preview.setStyleSheet(f"background: {PANEL_BG}; color: {TEXT_SECONDARY};")
        self.image_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        image_layout.addWidget(self.image_preview)
        self.graph_tabs.addTab(self.image_tab, "Оригінальне фото")
        self.graph_tabs.setTabVisible(1, False)   # hidden until image is loaded

        graph_layout.addWidget(self.graph_tabs)
        layout.addWidget(graph_group)
        return container

    def _show_image_tab(self, path):
        """Display the original ECG image in tab 2."""
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(
                self.image_preview.width() or 800,
                self.image_preview.height() or 300,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.image_preview.setPixmap(scaled)
        self.graph_tabs.setTabVisible(1, True)

    def _hide_image_tab(self):
        self.graph_tabs.setTabVisible(1, False)
        self.image_preview.setPixmap(QPixmap())

    def _build_report_panel(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 4, 10, 10)
        layout.setSpacing(4)

        report_group = QGroupBox("ДІАГНОСТИЧНИЙ ЗВІТ ЕКГ")
        report_layout = QVBoxLayout(report_group)
        report_layout.setContentsMargins(8, 8, 8, 8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"background-color: {CARD_BG}; border: none;")

        self.report_label = QLabel(self._default_report_text())
        self.report_label.setTextFormat(Qt.RichText)
        self.report_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.report_label.setWordWrap(True)
        self.report_label.setStyleSheet(
            f"background-color: {CARD_BG}; color: {TEXT_PRIMARY}; "
            f"padding: 12px; font-family: 'Consolas', 'Courier New', monospace; font-size: 13px;"
        )
        scroll.setWidget(self.report_label)
        report_layout.addWidget(scroll)
        layout.addWidget(report_group)
        return container

    def _default_report_text(self):
        return (
            f"<span style='color:{MUTED};'>"
            "Аналіз ще не виконано.<br><br>"
            "Щоб почати:<br>"
            "1. Завантажте ЕКГ файл (або оберіть зразок)<br>"
            "2. За бажанням введіть симптоми пацієнта<br>"
            "3. Натисніть <b>АНАЛІЗУВАТИ ЕКГ</b>"
            "</span>"
        )

    def _load_ecg_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Відкрити ЕКГ файл", "",
            "Всі підтримувані формати (*.csv *.txt *.dat *.mat *.png *.jpg *.jpeg *.bmp *.tiff *.tif);;"
            "Зображення ЕКГ (*.png *.jpg *.jpeg *.bmp *.tiff *.tif);;"
            "Сигнальні файли (*.csv *.txt *.dat *.mat);;"
            "Всі файли (*)"
        )
        if path:
            self._load_file(path)

    def _load_sample(self):
        btn = self.sender()
        fname = btn.property("sample_file")
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data")
        path = os.path.join(base, fname)
        if not os.path.exists(path):
            QMessageBox.warning(self, "Зразок не знайдено",
                                f"Файл зразка не знайдено:\n{path}\n\n"
                                "Запустіть  python sample_data/generate_samples.py  для генерації зразків.")
            return
        self._load_file(path)

    def _load_file(self, path):
        try:
            ext = os.path.splitext(path)[1].lower()
            is_image = ext in IMAGE_EXTENSIONS

            signal, fs = load_ecg(path)
            self.ecg_signal = signal
            self.ecg_fs = fs
            self._current_image_path = path if is_image else None

            fname = os.path.basename(path)
            src_label = " [фото ЕКГ]" if is_image else ""
            self.file_label.setText(
                f"{fname}{src_label}\n{len(signal)} зразків @ {fs} Гц\n({len(signal)/fs:.1f} с)"
            )
            self.file_label.setObjectName("file_label_loaded")
            self.file_label.setStyleSheet(
                f"color: {ACCENT2}; font-size: 11px; padding: 4px; "
                f"border: 1px solid {ACCENT2}; border-radius: 4px; background-color: {PANEL_BG};"
            )
            self.analyze_btn.setEnabled(True)
            self.status_bar.showMessage(
                f"Завантажено: {fname}{src_label}  |  {len(signal)} зразків @ {fs} Гц"
            )
            self.ecg_canvas.plot_ecg(signal, fs, title=f"ЕКГ: {fname}")

            # Show original image tab if image was loaded
            if is_image:
                self._show_image_tab(path)
            else:
                self._hide_image_tab()

            self.report_label.setText(self._default_report_text())
        except Exception as e:
            QMessageBox.critical(self, "Помилка завантаження",
                                 f"Не вдалося завантажити ЕКГ файл:\n\n{traceback.format_exc()}")
            self.status_bar.showMessage("Помилка завантаження файлу.")

    def _run_analysis(self):
        if self.ecg_signal is None:
            return
        symptoms = self.symptoms_edit.toPlainText().strip()
        self.analyze_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.status_bar.showMessage("Аналіз ЕКГ сигналу — зачекайте...")
        self.report_label.setText(
            f"<span style='color:{ACCENT};'>Аналіз ЕКГ сигналу... зачекайте.</span>"
        )

        self.worker = AnalysisWorker(self.ecg_signal, self.ecg_fs, symptoms)
        self.worker.finished.connect(self._on_analysis_done)
        self.worker.error.connect(self._on_analysis_error)
        self.worker.start()

    def _on_analysis_done(self, result):
        self.analysis_result = result
        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self.status_bar.showMessage(
            f"Аналіз завершено — Діагноз: {result.get('diagnosis', 'N/A')}  |  "
            f"Впевненість: {result.get('confidence', 0)*100:.1f}%"
        )
        self.ecg_canvas.plot_ecg(
            self.ecg_signal, self.ecg_fs,
            r_peaks=result.get("r_peaks"),
            st_regions=result.get("st_regions"),
            arrhythmia_regions=result.get("arrhythmia_regions"),
            title=f"Аналіз ЕКГ — {result.get('diagnosis', '')} (Впевненість: {min(result.get('confidence', 0), 0.989)*100:.1f}%)"
        )
        self.report_label.setText(self._format_report(result))

    def _on_analysis_error(self, msg):
        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self.status_bar.showMessage("Аналіз завершився з помилкою.")
        QMessageBox.critical(self, "Помилка аналізу", f"Помилка під час аналізу:\n\n{msg}")

    def _format_report(self, r):
        hr = r.get("heart_rate", 0)
        hr_cls = r.get("hr_classification", "Невідомо")
        hr_color = ACCENT2 if hr_cls == "Норма" else WARNING

        rhythm = r.get("rhythm", "Невідомо")
        rhythm_color = ACCENT2 if rhythm == "Регулярний" else WARNING
        rhythm_cause = r.get("rhythm_cause", "")

        axis = r.get("electrical_axis", "Нормальна")
        axis_color = ACCENT2 if axis == "Нормальна" else WARNING

        ischemic = r.get("ischemic_signs", [])
        ischemic_color = WARNING if ischemic else ACCENT2
        ischemic_text = "; ".join(ischemic) if ischemic else "Не виявлено"

        conduction = r.get("conduction_disorders", [])
        conduction_color = WARNING if conduction else ACCENT2
        conduction_text = "; ".join(conduction) if conduction else "Не виявлено"

        diagnosis = r.get("diagnosis", "Невідомо")
        confidence = min(r.get("confidence", 0), 0.989) * 100
        diag_color = WARNING if diagnosis != "Нормальний синусовий ритм" else ACCENT2

        notes = r.get("clinical_notes", [])
        sep = f"<span style='color:{MUTED};'>" + "─" * 58 + "</span>"

        html = f"""
<pre style='font-family: Consolas, monospace; font-size:13px; line-height:1.6;'>
<span style='color:{ACCENT}; font-size:15px; font-weight:bold;'>╔══════════════════════════════════════════════════════╗
║           ДІАГНОСТИЧНИЙ ЗВІТ ЕКГ                     ║
╚══════════════════════════════════════════════════════╝</span>

{sep}
<span style='color:{TEXT_SECONDARY};'>  ЧАСТОТА СЕРЦЕВИХ СКОРОЧЕНЬ (ЧСС):</span>
    Виміряна:       <span style='color:{hr_color};'><b>{hr:.0f} уд/хв</b></span>
    Класифікація:   <span style='color:{hr_color};'><b>{hr_cls}</b></span>
    Норма:          60 – 100 уд/хв

{sep}
<span style='color:{TEXT_SECONDARY};'>  ЕЛЕКТРИЧНА ВІС СЕРЦЯ:</span>
    Результат:      <span style='color:{axis_color};'><b>{axis}</b></span>

{sep}
<span style='color:{TEXT_SECONDARY};'>  АНАЛІЗ РИТМУ:</span>
    Ритм:           <span style='color:{rhythm_color};'><b>{rhythm}</b></span>"""

        if rhythm_cause:
            html += f"\n    Причина:        <span style='color:{WARNING};'><b>{rhythm_cause}</b></span>"

        html += f"""

{sep}
<span style='color:{TEXT_SECONDARY};'>  ІШЕМІЧНІ ОЗНАКИ:</span>
    Висновок:       <span style='color:{ischemic_color};'><b>{ischemic_text}</b></span>

{sep}
<span style='color:{TEXT_SECONDARY};'>  ПОРУШЕННЯ ПРОВІДНОСТІ:</span>
    Висновок:       <span style='color:{conduction_color};'><b>{conduction_text}</b></span>

{sep}
<span style='color:{TEXT_SECONDARY};'>  СЕГМЕНТ ST:</span>
    Відхилення:     <span style='color:{ACCENT2};'><b>{r.get("st_deviation_mean", 0)*1000:.2f} мВ (середнє)</b></span>
    Макс. елевація: <span style='color:{ACCENT2};'><b>{r.get("st_elevation_max", 0)*1000:.2f} мВ</b></span>
    Макс. депресія: <span style='color:{ACCENT2};'><b>{r.get("st_depression_max", 0)*1000:.2f} мВ</b></span>

{sep}
<span style='color:{TEXT_SECONDARY};'>  ІНТЕРВАЛИ / КОМПЛЕКСИ:</span>
    Тривалість QRS: <span style='color:{ACCENT2};'><b>{r.get("qrs_duration_ms", 0):.0f} мс</b></span>
    Інтервал PR:    <span style='color:{ACCENT2};'><b>{r.get("pr_interval_ms", 0):.0f} мс</b></span>
    Інтервал QT:    <span style='color:{ACCENT2};'><b>{r.get("qt_interval_ms", 0):.0f} мс</b></span>
    ВСР (SDNN):     <span style='color:{ACCENT2};'><b>{r.get("hrv_sdnn", 0)*1000:.1f} мс</b></span>

{sep}
<span style='color:{diag_color}; font-size:15px;'>  КІНЦЕВИЙ ДІАГНОЗ:  {diagnosis}</span>
<span style='color:{ACCENT};'>  ВПЕВНЕНІСТЬ :   {confidence:.1f}%</span>

{sep}"""

        if notes:
            html += f"\n<span style='color:{TEXT_SECONDARY};'>  КЛІНІЧНІ ПРИМІТКИ:</span>"
            for note in notes:
                html += f"\n    • <span style='color:{WARNING};'>{note}</span>"

        html += f"""

{sep}
<span style='color:{MUTED}; font-size:10px;'>  Звіт згенеровано системою .
  Результати потребують підтвердження кваліфікованим лікарем-кардіологом.
  Не є самостійним медичним висновком.</span>
{sep}
</pre>"""
        return html
