"""
Monkey-patch matplotlib Qt dialogs to display Ukrainian text.
Applied via UkrainianNavigationToolbar in gui.py.
"""

from PyQt5.QtWidgets import (
    QLabel, QPushButton, QGroupBox, QTabWidget, QCheckBox,
    QDialog, QDialogButtonBox, QWidget,
)
from PyQt5.QtCore import Qt

_T = {
    # SubplotTool
    "Borders":                       "Відступи",
    "Spacings":                      "Інтервали",
    "top":                           "верх",
    "bottom":                        "низ",
    "left":                          "ліво",
    "right":                         "право",
    "hspace":                        "гор. відст.",
    "wspace":                        "вер. відст.",
    "Tight layout":                  "Щільне розміщення",
    "Reset":                         "Скинути",
    "Export values":                 "Експорт значень",
    "Close":                         "Закрити",
    # Figure options dialog
    "Figure options":                "Параметри графіку",
    "Axes":                          "Осі",
    "Curves":                        "Криві",
    "Title":                         "Назва",
    "X-Axis":                        "Вісь X",
    "Y-Axis":                        "Вісь Y",
    "Min":                           "Мін",
    "Max":                           "Макс",
    "Label":                         "Підпис",
    "Scale":                         "Масштаб",
    "(Re-)Generate automatic legend":"(Пере)генерувати легенду",
    "Inverted":                      "Інвертована",
    "linear":                        "лінійний",
    "log":                           "логарифмічний",
    "symlog":                        "симетр. логарифмічний",
    "logit":                         "logit",
    "OK":                            "ОК",
    "Cancel":                        "Скасувати",
    "Apply":                         "Застосувати",
    # General matplotlib toolbar tooltips (tooltips on buttons)
    "Home":                          "Початок",
    "Back":                          "Назад",
    "Forward":                       "Вперед",
    "Pan":                           "Переміщення",
    "Zoom":                          "Масштаб",
    "Subplots":                      "Підграфіки",
    "Customize":                     "Налаштування",
    "Save":                          "Зберегти",
}


def _tr(text):
    return _T.get(text, text)


def _patch_widget_tree(widget):
    """Recursively walk a QWidget tree and replace known English strings."""
    if widget is None:
        return

    if isinstance(widget, QLabel):
        widget.setText(_tr(widget.text()))

    elif isinstance(widget, QPushButton):
        widget.setText(_tr(widget.text()))

    elif isinstance(widget, QGroupBox):
        widget.setTitle(_tr(widget.title()))

    elif isinstance(widget, QTabWidget):
        for i in range(widget.count()):
            widget.setTabText(i, _tr(widget.tabText(i)))

    elif isinstance(widget, QCheckBox):
        widget.setText(_tr(widget.text()))

    elif isinstance(widget, QDialogButtonBox):
        for btn in widget.buttons():
            btn.setText(_tr(btn.text()))

    elif isinstance(widget, QDialog):
        widget.setWindowTitle(_tr(widget.windowTitle()))

    for child in widget.findChildren(QWidget):
        _patch_widget_tree_shallow(child)


def _patch_widget_tree_shallow(widget):
    """Patch a single widget (non-recursive, used after findChildren)."""
    if isinstance(widget, QLabel):
        t = widget.text()
        if t in _T:
            widget.setText(_T[t])
    elif isinstance(widget, QPushButton):
        t = widget.text()
        if t in _T:
            widget.setText(_T[t])
    elif isinstance(widget, QGroupBox):
        t = widget.title()
        if t in _T:
            widget.setTitle(_T[t])
    elif isinstance(widget, QTabWidget):
        for i in range(widget.count()):
            t = widget.tabText(i)
            if t in _T:
                widget.setTabText(i, _T[t])
    elif isinstance(widget, QCheckBox):
        t = widget.text()
        if t in _T:
            widget.setText(_T[t])
    elif isinstance(widget, QDialogButtonBox):
        for btn in widget.buttons():
            t = btn.text()
            clean = t.replace("&", "")
            if clean in _T:
                btn.setText(_T[clean])


def patch_dialog(dialog):
    """Patch all translatable strings in a top-level dialog."""
    if dialog is None:
        return
    if hasattr(dialog, "windowTitle"):
        dialog.setWindowTitle(_tr(dialog.windowTitle()))
    for child in dialog.findChildren(QWidget):
        _patch_widget_tree_shallow(child)
