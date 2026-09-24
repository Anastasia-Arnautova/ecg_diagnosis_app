"""
evaluation.py
=============
Evaluates the trained ECG classification model on the MIT-BIH test set
(or any labeled dataset).

Usage
-----
    python evaluation.py                   # evaluate saved_model/model.npy
    python evaluation.py --synthetic       # quick check with synthetic data
    python evaluation.py --records 200 201 # specific records
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from model import ECGModel, CLASS_NAMES

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "saved_model", "model.npy"
)

# Short class labels for confusion matrix display
SHORT_NAMES = ["Норма", "Аритмія", "Тахікардія", "Брадикардія", "ІМ ризик"]


def evaluate(X_test, y_test, model_path=MODEL_PATH):
    """
    Run evaluation of the saved model on the provided test data.

    Parameters
    ----------
    X_test : np.ndarray, shape (n, 20)
    y_test : np.ndarray, shape (n,)
    model_path : str
    """
    # --- Load model --------------------------------------------------------
    if not os.path.exists(model_path):
        print(f"[ПОМИЛКА] Модель не знайдена: {model_path}")
        print("Запустіть спочатку:  python train_model.py")
        sys.exit(1)

    m = ECGModel()
    ok = m.load(model_path)
    if not ok:
        print(f"[ПОМИЛКА] Не вдалося завантажити модель з {model_path}")
        sys.exit(1)

    print(f"Модель завантажена: {model_path}\n")

    # --- Predict -----------------------------------------------------------
    scaler_mean = m.scaler_mean
    scaler_std = m.scaler_std
    X_sc = (X_test - scaler_mean) / (scaler_std + 1e-8)

    y_pred = np.array([m.model.predict((x).reshape(1, -1))[0] for x in X_sc])

    # --- Accuracy ----------------------------------------------------------
    acc = accuracy_score(y_test, y_pred)
    print(f"{'='*58}")
    print(f"  ЗАГАЛЬНА ТОЧНІСТЬ: {acc*100:.2f}%")
    print(f"{'='*58}\n")

    # --- Classification report --------------------------------------------
    print("=== Детальний звіт по класах ===")
    print(classification_report(
        y_test, y_pred,
        target_names=CLASS_NAMES,
        zero_division=0,
        digits=3,
    ))

    # --- Confusion matrix --------------------------------------------------
    cm = confusion_matrix(y_test, y_pred)
    print("=== Матриця плутанини ===")
    _print_confusion_matrix(cm, SHORT_NAMES)

    # --- Per-class stats ---------------------------------------------------
    print("\n=== Статистика по класах ===")
    for i, name in enumerate(CLASS_NAMES):
        total = int(np.sum(y_test == i))
        correct = int(cm[i, i]) if i < cm.shape[0] else 0
        pct = 100.0 * correct / total if total > 0 else 0.0
        print(f"  {i} – {name:<38}: {correct:>5}/{total:<5}  ({pct:.1f}%)")

    return acc, cm


def _print_confusion_matrix(cm, labels):
    n = len(labels)
    col_w = 11
    # Header
    header = " " * 14 + "".join(f"{l:>{col_w}}" for l in labels)
    print(header)
    print("-" * (14 + col_w * n))
    for i, row_label in enumerate(labels):
        row = f"  {row_label:<12}" + "".join(
            f"{cm[i, j]:>{col_w}}" for j in range(n)
        )
        print(row)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Оцінка точності ЕКГ-класифікатора"
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Використати синтетичні дані для швидкої перевірки"
    )
    parser.add_argument(
        "--records", nargs="+", default=None,
        metavar="ID",
        help="Конкретні записи MIT-BIH (напр.: --records 200 201 208)"
    )
    parser.add_argument(
        "--model", default=MODEL_PATH,
        help=f"Шлях до файлу моделі (default: {MODEL_PATH})"
    )
    args = parser.parse_args()

    if args.synthetic:
        # Quick evaluation on synthetic data
        from sklearn.model_selection import train_test_split
        import train_model as tm
        X, y = tm._generate_synthetic_inline()
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        print(f"Синтетичний тест: {len(X_test)} зразків\n")
    else:
        from dataset_loader import load_mitbih_features
        from sklearn.model_selection import train_test_split
        print("Завантаження MIT-BIH для оцінки...\n")
        X, y = load_mitbih_features(records=args.records, verbose=True)
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.15, stratify=y, random_state=42
        )
        print(f"Тестова вибірка: {len(X_test)} зразків\n")

    evaluate(X_test, y_test, model_path=args.model)
