"""
╔══════════════════════════════════════════════════════════════════════╗
║          НЕЙРОННА МЕРЕЖА ДЛЯ АНАЛІЗУ ЕКГ                    ║
║          Дипломний проєкт — DeepECGNet                              ║
║                                                                      ║
║  Модель навчається з нуля (from scratch) на власних даних            ║
║  з папки  data/  (MIT-BIH Arrhythmia Database)                      ║
║                                                                      ║
║  Без готових вагів. Без pretrained models.                           ║
╚══════════════════════════════════════════════════════════════════════╝

АРХІТЕКТУРА DeepECGNet:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Вхід            : 20 числових ознак ЕКГ (RR, HRV, QRS, ST, PR, QT...)
  Вхідна проєкція : Linear(20→256) + BatchNorm + ReLU + Dropout(0.3)
  ResBlock 1–2    : Linear(256→256) × 2 + skip-з'єднання   [256 нейронів]
  Компресія       : Linear(256→128) + BatchNorm + ReLU + Dropout(0.25)
  ResBlock 3–4    : Linear(128→128) × 2 + skip-з'єднання   [128 нейронів]
  Класифікатор    : Linear(128→64) + ReLU + Dropout(0.2) + Linear(64→5)
  Temperature T   : logits / T + Softmax  [калібрування впевненості]
  Параметрів      : ~377 710
  Класів виходу   : 5  (Норма / Аритмія / Тахікардія / Брадикардія / ІМ)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ЗАПУСК:
    python train_my_model.py               # навчання на реальних даних з data/
    python train_my_model.py --epochs 150  # більше епох
    python train_my_model.py --fast        # швидкий тест на 3 записах

РЕЗУЛЬТАТ:
    saved_model/model.pth   ← PyTorch checkpoint + temperature T
"""

import os
import sys
import argparse
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Шлях до папки з даними
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
SAVE_DIR   = os.path.join(BASE_DIR, "saved_model")
MODEL_PATH = os.path.join(SAVE_DIR, "model.pth")

sys.path.insert(0, BASE_DIR)

# Параметри
FEATURE_DIM  = 20     # розмірність вектора ознак
NUM_CLASSES  = 5      # кількість класів
RANDOM_SEED  = 42
BATCH_SIZE   = 256    # менший батч → частіші оновлення → краща точність
LABEL_EPS    = 0.05   # label smoothing epsilon
MIXUP_ALPHA  = 0.3    # mixup alpha (менший → чіткіші межі класів)
EARLY_STOP   = 30     # більше терпіння перед зупинкою

CLASS_NAMES = [
    "Нормальний синусовий ритм",
    "Аритмія",
    "Тахікардія",
    "Брадикардія",
    "Можливий інфаркт міокарда",
]


# ════════════════════════════════════════════════════════════════════
#  КРОК 1: ЗАВАНТАЖЕННЯ ДАНИХ З ПАПКИ data/
# ════════════════════════════════════════════════════════════════════

def load_local_data(records=None, verbose=True):
    """
    Завантажує ЕКГ записи MIT-BIH із локальної папки data/.
    Кожен запис → набір кардіоударів → вектор 20 ознак.

    Формат файлів у data/:
        100.dat  — бінарний ЕКГ сигнал (int16)
        100.hea  — заголовок (частота дискретизації, розмір...)
        100.atr  — анотації (мітки кардіоударів)

    Повертає:
        X : numpy array (n_beats, 20)  — матриця ознак
        y : numpy array (n_beats,)     — мітки класів 0-4
    """
    try:
        import wfdb
    except ImportError:
        raise ImportError("Встановіть wfdb: pip install wfdb")

    from preprocessing import preprocess_ecg
    from preprocessing import compute_pqrst_segments
    from dataset_loader import (
        NORMAL_SYMBOLS, ARRHYTHMIA_SYMBOLS,
        _beat_feature_vector, _assign_label, _print_class_distribution,
        MIT_BIH_RECORDS,
    )

    # Список записів
    if records is None:
        records = MIT_BIH_RECORDS

    if verbose:
        print(f"Дані: {DATA_DIR}")
        print(f"Записів для обробки: {len(records)}\n")

    all_X, all_y = [], []
    skipped = []

    for rec_name in records:
        rec_path = os.path.join(DATA_DIR, rec_name)

        # Перевірка що файли є локально
        if not os.path.exists(rec_path + ".hea"):
            skipped.append(f"{rec_name} (файл не знайдено)")
            continue

        try:
            # Читаємо локальний запис (без інтернету)
            record     = wfdb.rdrecord(rec_path)
            annotation = wfdb.rdann(rec_path, "atr")
            fs         = record.fs   # 360 Гц для MIT-BIH

            # Виділяємо відведення MLII (канал 0)
            raw = record.p_signal[:, 0].astype(float)

            # Заповнюємо NaN через лінійну інтерполяцію
            nan_mask = np.isnan(raw)
            if nan_mask.any():
                raw[nan_mask] = np.interp(
                    np.where(nan_mask)[0],
                    np.where(~nan_mask)[0],
                    raw[~nan_mask],
                )

            # Крок 2: Попередня обробка сигналу
            ecg = preprocess_ecg(raw, fs)

            # Крок 3: Вибірка тільки размічених кардіоударів
            ann_samples = annotation.sample
            ann_symbols = np.array(annotation.symbol)
            beat_mask   = np.array([
                s in NORMAL_SYMBOLS or s in ARRHYTHMIA_SYMBOLS
                for s in ann_symbols
            ])
            beat_samples = ann_samples[beat_mask]
            beat_symbols = ann_symbols[beat_mask]

            if len(beat_samples) < 5:
                skipped.append(f"{rec_name} (замало ударів)")
                continue

            # RR-інтервали
            rr_sec = np.diff(beat_samples) / fs
            rr_sec = np.concatenate([[rr_sec[0]], rr_sec])
            median_rr = float(np.median(rr_sec))

            # Крок 4: Сегментація PQRST
            segments = compute_pqrst_segments(ecg, beat_samples, fs)

            # Крок 5: Вектори ознак для кожного удару
            X_rec, y_rec = [], []
            for i, (seg, sym) in enumerate(zip(segments, beat_symbols)):
                feat  = _beat_feature_vector(seg, i, rr_sec, beat_samples, fs, ecg, median_rr)
                label = _assign_label(sym, feat, median_rr)
                X_rec.append(feat)
                y_rec.append(label)

            all_X.append(np.array(X_rec, dtype=np.float32))
            all_y.append(np.array(y_rec, dtype=np.int64))

            if verbose:
                print(f"  [{rec_name}]  {len(X_rec):>5} ударів  (fs={fs} Гц)")

        except Exception as exc:
            skipped.append(f"{rec_name} ({exc})")

    if skipped:
        print(f"\nПропущено: {skipped}")

    if not all_X:
        raise RuntimeError("Жодного запису не завантажено. Перевір папку data/")

    X = np.vstack(all_X)
    y = np.concatenate(all_y)

    if verbose:
        _print_class_distribution(y)

    return X, y


# ════════════════════════════════════════════════════════════════════
#  ВИРІВНЮВАННЯ КЛАСІВ (Oversampling + SMOTE-подібне)
# ════════════════════════════════════════════════════════════════════

def balance_classes(X, y, strategy="oversample", target_ratio=0.85):
    """
    Вирівнює дисбаланс класів у датасеті.

    Проблема: MIT-BIH має ~29% Норма, але лише ~10% Аритмія.
    Мережа навчається ігнорувати рідкісні класи → низька точність.

    Стратегії:
        "oversample" — дублюємо рідкісні зразки + додаємо невеликий шум
        "undersample" — зменшуємо домінуючі класи

    target_ratio: до якої частки найбільшого класу підтягувати малі
    """
    rng = np.random.default_rng(RANDOM_SEED)
    counts = np.bincount(y, minlength=NUM_CLASSES)
    max_count = int(counts.max() * target_ratio)

    X_balanced, y_balanced = [X.copy()], [y.copy()]

    print("Вирівнювання класів (oversampling):")
    for cls in range(NUM_CLASSES):
        cls_count = counts[cls]
        if cls_count >= max_count:
            print(f"  [{cls}] {CLASS_NAMES[cls]:<38}: {cls_count:>6} (без змін)")
            continue

        # Скільки додаткових зразків потрібно
        n_add = max_count - cls_count
        cls_idx = np.where(y == cls)[0]

        # Вибираємо випадкові зразки з повтореннями
        chosen = rng.choice(cls_idx, size=n_add, replace=True)
        X_add  = X[chosen].copy()

        # Додаємо невеликий Gaussian шум (SMOTE-like jitter)
        # Шум 1% від std ознак → різноманітність без зміни класу
        noise_std = X[cls_idx].std(axis=0) * 0.01
        X_add    += rng.normal(0, noise_std, X_add.shape).astype(np.float32)

        X_balanced.append(X_add)
        y_balanced.append(np.full(n_add, cls, dtype=np.int64))
        print(f"  [{cls}] {CLASS_NAMES[cls]:<38}: {cls_count:>6} → {cls_count+n_add:>6} (+{n_add})")

    X_out = np.vstack(X_balanced)
    y_out = np.concatenate(y_balanced)

    # Перемішуємо
    perm  = rng.permutation(len(y_out))
    print(f"\n  Всього після вирівнювання: {len(y_out):,} зразків\n")
    return X_out[perm], y_out[perm]


# ════════════════════════════════════════════════════════════════════
#  КРОК 2: АРХІТЕКТУРА НЕЙРОМЕРЕЖІ (DeepECGNet)
# ════════════════════════════════════════════════════════════════════

def build_deepecgnet(input_dim=FEATURE_DIM, num_classes=NUM_CLASSES):
    """
    Будує DeepECGNet з нуля — без завантаження будь-яких готових вагів.

    Архітектура:
    ┌─────────────────────────────────────────────────────────────┐
    │ Вхідна проєкція: Linear(20→256) + BatchNorm + ReLU + Drop  │
    ├─────────────────────────────────────────────────────────────┤
    │ ResBlock 1: Linear(256→256) + BN + ReLU + Drop             │
    │             └── skip: вихід = F(x) + x                     │
    │ ResBlock 2: Linear(256→256) + BN + ReLU + Drop             │
    │             └── skip: вихід = F(x) + x                     │
    ├─────────────────────────────────────────────────────────────┤
    │ Компресія:  Linear(256→128) + BatchNorm + ReLU + Drop      │
    ├─────────────────────────────────────────────────────────────┤
    │ ResBlock 3: Linear(128→128) + BN + ReLU + Drop             │
    │ ResBlock 4: Linear(128→128) + BN + ReLU + Drop             │
    ├─────────────────────────────────────────────────────────────┤
    │ Голова:     Linear(128→64) + ReLU + Drop + Linear(64→5)    │
    └─────────────────────────────────────────────────────────────┘

    Параметрів: ~377 710
    Всі ваги ініціалізуються випадково (Xavier uniform).
    """
    import torch
    import torch.nn as nn

    class ResBlock(nn.Module):
        """
        Residual Block — основна будівельна одиниця DeepECGNet.

        Ідея: вихід = ReLU(F(x) + x)
        де F(x) — два лінійних шари з нормалізацією і dropout.
        Skip-з'єднання (+x) дозволяє градієнту текти напряму назад →
        мережа навчається стабільно навіть при великій глибині.
        """
        def __init__(self, dim, dropout=0.2):
            super().__init__()
            self.transform = nn.Sequential(
                nn.Linear(dim, dim),       # Лінійне перетворення
                nn.BatchNorm1d(dim),       # Нормалізація батчу
                nn.ReLU(),                 # Нелінійна активація
                nn.Dropout(dropout),       # Регуляризація (вимкнення нейронів)
                nn.Linear(dim, dim),       # Друге лінійне перетворення
                nn.BatchNorm1d(dim),       # Нормалізація батчу
            )
            self.activation = nn.ReLU()

        def forward(self, x):
            # Skip connection: додаємо вхід до виходу трансформації
            return self.activation(self.transform(x) + x)

    class DeepECGNet(nn.Module):
        """
        Повна архітектура класифікатора ЕКГ.
        Навчається з нуля на власних даних.
        """
        def __init__(self):
            super().__init__()

            # ── Шар 1: вхідна проєкція 20 → 256 ────────────────────────
            self.input_projection = nn.Sequential(
                nn.Linear(input_dim, 256),   # вхід: 20 ознак, вихід: 256
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),             # 30% нейронів вимикається
            )

            # ── Шари 2–3: два residual-блоки 256×256 ────────────────────
            self.resblock_1 = ResBlock(256, dropout=0.2)
            self.resblock_2 = ResBlock(256, dropout=0.2)

            # ── Шар 4: компресія 256 → 128 ──────────────────────────────
            self.compression = nn.Sequential(
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.25),
            )

            # ── Шари 5–6: два residual-блоки 128×128 ────────────────────
            self.resblock_3 = ResBlock(128, dropout=0.15)
            self.resblock_4 = ResBlock(128, dropout=0.15)

            # ── Шари 7–8: класифікуючий блок ────────────────────────────
            self.classifier = nn.Sequential(
                nn.Linear(128, 64),          # 128 → 64
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, num_classes),  # 64 → 5 класів
            )

            # Ініціалізація вагів Xavier (краща стартова точка)
            self._init_weights()

        def _init_weights(self):
            """Ініціалізація всіх лінійних шарів методом Xavier uniform."""
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

        def forward(self, x):
            """Пряме поширення (forward pass) через усі шари."""
            x = self.input_projection(x)   # 20 → 256
            x = self.resblock_1(x)         # 256 → 256  (з skip)
            x = self.resblock_2(x)         # 256 → 256  (з skip)
            x = self.compression(x)        # 256 → 128
            x = self.resblock_3(x)         # 128 → 128  (з skip)
            x = self.resblock_4(x)         # 128 → 128  (з skip)
            return self.classifier(x)      # 128 → 5 (logits)

    model = DeepECGNet()

    # Підрахунок параметрів (тільки при прямому запуску навчання)
    if __name__ == "__main__" or os.environ.get("ECG_VERBOSE"):
        total  = sum(p.numel() for p in model.parameters())
        train_ = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Архітектура DeepECGNet:")
        print(f"  Всього параметрів     : {total:,}")
        print(f"  Параметрів для навчання: {train_:,}")
        print(f"  (Всі ваги — випадкові, навчаються з нуля)\n")

    return model


# ════════════════════════════════════════════════════════════════════
#  КРОК 3: ДОПОМІЖНІ ФУНКЦІЇ НАВЧАННЯ
# ════════════════════════════════════════════════════════════════════

def label_smoothing_loss(logits, targets, num_classes, eps=0.05, weight=None):
    """
    Label Smoothing Cross-Entropy (ε = 0.05).

    Замість жорсткої мітки [0, 0, 1, 0, 0] використовуємо м'яку:
        [ε/K, ε/K, 1-ε+ε/K, ε/K, ε/K]   де K = кількість класів

    Переваги:
    - Запобігає перевпевненості мережі
    - Збільшує margin (відстань) між класами
    - Краща генералізація на нових даних
    """
    import torch
    import torch.nn.functional as F

    log_probs = F.log_softmax(logits, dim=1)
    smooth    = eps / num_classes
    one_hot   = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1.0)
    soft_tgt  = one_hot * (1.0 - eps) + smooth

    loss = -(soft_tgt * log_probs).sum(dim=1)

    if weight is not None:
        loss = loss * weight[targets] * num_classes

    return loss.mean()


def mixup_augmentation(x, y, alpha=0.4):
    """
    Mixup Augmentation.

    Мікшує два випадкових зразки у батчі:
        x_new = λ · x_i + (1-λ) · x_j
        де λ ~ Beta(alpha, alpha)

    Переваги:
    - Регуляризує границі між класами
    - Покращує стійкість до шуму
    - Підвищує впевненість на реальних даних

    Повертає: (x_mixed, y_a, y_b, lambda)
    Лос = λ · Loss(y_a) + (1-λ) · Loss(y_b)
    """
    import torch

    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1.0 - lam) * x[idx]
    return x_mix, y, y[idx], lam


def find_optimal_temperature(model, X_val, y_val, device):
    """
    Temperature Scaling — калібрування впевненості.

    Шукає T ∈ [0.3, 2.0] що мінімізує NLL на валідаційній вибірці.
    T < 1.0 → загострює розподіл → вища впевненість (підходить для 95%+)
    T > 1.0 → пом'якшує розподіл → чесніша впевненість

    Математично:
        probs = softmax(logits / T)
    """
    import torch
    import torch.nn.functional as F

    model.eval()
    with torch.no_grad():
        logits_list = []
        for i in range(0, len(X_val), 1024):
            xb = X_val[i:i+1024].to(device)
            logits_list.append(model(xb).cpu())
        logits_all = torch.cat(logits_list)

    best_T   = 1.0
    best_nll = float("inf")
    y_t      = torch.tensor(y_val, dtype=torch.long)

    for T in np.arange(0.30, 2.01, 0.025):
        nll = F.cross_entropy(logits_all / T, y_t).item()
        if nll < best_nll:
            best_nll = nll
            best_T   = float(T)

    return best_T


# ════════════════════════════════════════════════════════════════════
#  КРОК 4: ОСНОВНИЙ НАВЧАЛЬНИЙ ЦИКЛ
# ════════════════════════════════════════════════════════════════════

def train(epochs=100, fast_mode=False):
    """
    Повний пайплайн навчання:
    1. Завантаження даних з data/
    2. Нормалізація (Z-score)
    3. Розбивка train/val/test
    4. Навчання DeepECGNet (100 епох за замовчуванням)
    5. Temperature Scaling
    6. Збереження model.pth
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    os.makedirs(SAVE_DIR, exist_ok=True)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    print("=" * 65)
    print("  DeepECGNet — Навчання власної нейромережі з нуля")
    print("=" * 65)
    print()

    # ------------------------------------------------------------------
    # 1. Завантаження даних з локальної папки data/
    # ------------------------------------------------------------------
    print("[ Крок 1 ] Завантаження ЕКГ даних із data/ ...\n")

    if fast_mode:
        # Швидкий тест — лише 3 записи
        records = ["100", "101", "200"]
        print("(Швидкий режим: 3 записи для тесту)\n")
    else:
        records = None   # всі доступні

    t0 = time.time()
    X, y = load_local_data(records=records, verbose=True)
    print(f"Час завантаження: {time.time()-t0:.1f} с\n")

    # ------------------------------------------------------------------
    # 2. Розбивка → нормалізація → oversampling тільки train
    # ------------------------------------------------------------------
    print("[ Крок 2 ] Розбивка, нормалізація та вирівнювання класів ...\n")

    # Стратифікований split: Train 70% / Val 15% / Test 15%
    # ВАЖЛИВО: split до oversampling щоб у val/test не було дублікатів
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=RANDOM_SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.15/0.85, stratify=y_tmp, random_state=RANDOM_SEED
    )

    # Z-score нормалізація (fit тільки на train)
    mean = X_train.mean(axis=0).astype(np.float32)
    std  = (X_train.std(axis=0) + 1e-8).astype(np.float32)

    X_train = ((X_train - mean) / std).astype(np.float32)
    X_val   = ((X_val   - mean) / std).astype(np.float32)
    X_test  = ((X_test  - mean) / std).astype(np.float32)

    print(f"До вирівнювання:")
    print(f"  Train : {len(X_train):>7,} зразків")
    print(f"  Val   : {len(X_val):>7,} зразків")
    print(f"  Test  : {len(X_test):>7,} зразків\n")

    # Oversampling тільки на train (val і test залишаються реальними)
    X_train, y_train = balance_classes(X_train, y_train)

    # Рівні ваги після oversampling (класи вирівняні)
    counts  = np.bincount(y_train, minlength=NUM_CLASSES).astype(float)
    weights = np.ones(NUM_CLASSES, dtype=np.float32) / NUM_CLASSES
    print("Ваги класів: рівні (після oversampling всі класи збалансовані)")
    print()

    # ------------------------------------------------------------------
    # 3. Побудова нейромережі з нуля
    # ------------------------------------------------------------------
    print("[ Крок 3 ] Побудова DeepECGNet ...\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Пристрій: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("(CPU — навчання займе ~5-20 хв залежно від кількості даних)")
    print()

    model = build_deepecgnet(FEATURE_DIM, NUM_CLASSES).to(device)

    # ------------------------------------------------------------------
    # 4. Навчання
    # ------------------------------------------------------------------
    print("[ Крок 4 ] Навчання нейромережі ...\n")

    # Тензори
    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.long)
    X_v_t  = torch.tensor(X_val,   dtype=torch.float32)
    y_v_t  = torch.tensor(y_val,   dtype=torch.long)
    X_te_t = torch.tensor(X_test,  dtype=torch.float32)
    y_te_t = torch.tensor(y_test,  dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        TensorDataset(X_v_t, y_v_t),
        batch_size=1024, shuffle=False, num_workers=0
    )

    w_t = torch.tensor(weights, dtype=torch.float32).to(device)

    # AdamW optimizer з трохи меншим lr для стабільнішого навчання
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=5e-4, weight_decay=1e-4
    )

    # OneCycleLR — найкращий scheduler для швидкої сходимості
    # Спочатку піднімає lr до max, потім плавно знижує (cosine)
    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-3,               # пікове значення lr
        total_steps=total_steps,
        pct_start=0.15,            # перші 15% кроків — warmup
        anneal_strategy="cos",
        div_factor=10.0,           # стартовий lr = max_lr / 10
        final_div_factor=1000.0,   # фінальний lr = стартовий / 1000
    )

    best_val_acc  = 0.0
    patience_cnt  = 0
    history = {"train_loss": [], "train_acc": [], "val_acc": []}

    header = (f"{'Епоха':>6}  {'Train Loss':>11}  {'Train Acc':>10}"
              f"  {'Val Acc':>9}  {'LR':>9}  {'Статус':}")
    print(header)
    print("─" * 70)

    for epoch in range(1, epochs + 1):
        # ── Train ───────────────────────────────────────────────────
        model.train()
        total_loss = correct = total = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            # Mixup Augmentation — мікшування пар зразків
            x_mix, y_a, y_b, lam = mixup_augmentation(xb, yb, MIXUP_ALPHA)

            optimizer.zero_grad()
            logits = model(x_mix)

            # Mixup loss = λ·Loss(y_a) + (1-λ)·Loss(y_b)
            loss = (lam * label_smoothing_loss(logits, y_a, NUM_CLASSES, LABEL_EPS, w_t) +
                    (1 - lam) * label_smoothing_loss(logits, y_b, NUM_CLASSES, LABEL_EPS, w_t))
            loss.backward()

            # Gradient Clipping — обмеження норми градієнта (запобігає "вибуху")
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * len(yb)
            # Точність без mixup (по оригінальному yb)
            with torch.no_grad():
                correct += (model(xb).argmax(1) == yb).sum().item()
            total += len(yb)
            scheduler.step()   # OneCycleLR: крок після кожного батча
        train_loss = total_loss / total
        train_acc  = correct    / total

        # ── Validation ──────────────────────────────────────────────
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_correct += (model(xb).argmax(1) == yb).sum().item()
                val_total   += len(yb)
        val_acc = val_correct / val_total
        lr_now  = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # ── Збереження найкращої моделі ─────────────────────────────
        status = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_cnt = 0
            status = "* зберігаємо"
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_acc":     val_acc,
                "scaler_mean": mean,
                "scaler_std":  std,
                "input_dim":   FEATURE_DIM,
                "num_classes": NUM_CLASSES,
                "temperature": 1.0,   # буде оновлено нижче
            }, MODEL_PATH)
        else:
            patience_cnt += 1
            if patience_cnt >= EARLY_STOP:
                print(f"\n[Early Stopping] Немає покращення {EARLY_STOP} епох → зупиняємось.")
                break

        # Лог кожні 5 епох (або перша)
        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:>6}  {train_loss:>11.4f}  {train_acc*100:>9.2f}%"
                  f"  {val_acc*100:>9.2f}%  {lr_now:>9.2e}  {status}")

    print(f"\nНайкраща Val Accuracy: {best_val_acc*100:.2f}%\n")

    # ------------------------------------------------------------------
    # 5. Temperature Scaling — калібрування після навчання
    # ------------------------------------------------------------------
    print("[ Крок 5 ] Temperature Scaling — калібрування впевненості ...\n")

    # Завантажуємо найкращу модель
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    T_opt = find_optimal_temperature(model, X_v_t, y_val, device)
    print(f"  Оптимальна температура T = {T_opt:.3f}")
    if T_opt < 1.0:
        print(f"  T < 1 → загострює розподіл → передбачення ≥ 95% впевненості")
    else:
        print(f"  T ≥ 1 → пом'якшує розподіл → чесніший розподіл впевненості")

    # Зберігаємо T разом з моделлю
    ckpt["temperature"] = T_opt
    torch.save(ckpt, MODEL_PATH)

    # ------------------------------------------------------------------
    # 6. Фінальна оцінка на тестовій вибірці
    # ------------------------------------------------------------------
    print("\n[ Крок 6 ] Оцінка на тестовій вибірці ...\n")

    model.eval()
    preds, confs = [], []
    with torch.no_grad():
        for i in range(0, len(X_te_t), 1024):
            xb     = X_te_t[i:i+1024].to(device)
            logits = model(xb) / T_opt
            probs  = F.softmax(logits, dim=1)
            preds.append(probs.argmax(1).cpu().numpy())
            confs.append(probs.max(1).values.cpu().numpy())

    y_pred = np.concatenate(preds)
    confs  = np.concatenate(confs)
    y_true = y_te_t.numpy()

    test_acc = accuracy_score(y_true, y_pred)

    print("=" * 65)
    print(f"  ТОЧНІСТЬ НА ТЕСТОВІЙ ВИБІРЦІ : {test_acc*100:.2f}%")
    print(f"  Середня впевненість (правильні): "
          f"{confs[y_pred == y_true].mean()*100:.1f}%")
    print(f"  % передбачень з впевненістю ≥ 95%: "
          f"{(confs >= 0.95).mean()*100:.1f}%")
    print("=" * 65)
    print()
    print(classification_report(y_true, y_pred,
                                target_names=CLASS_NAMES, zero_division=0))

    print("Матриця плутанини (рядки=істинні, стовпці=передбачені):")
    cm = confusion_matrix(y_true, y_pred)
    header_row = "         " + "  ".join(f"{i:>6}" for i in range(NUM_CLASSES))
    print(header_row)
    for i, row in enumerate(cm):
        bar = "".join("█" if j == np.argmax(row) else "░"
                      for j in range(NUM_CLASSES))
        print(f"  [{i}] {bar}  " + "  ".join(f"{v:>6}" for v in row))

    # ------------------------------------------------------------------
    # 7. Збереження фінальних параметрів нормалізації
    # ------------------------------------------------------------------
    print(f"\nМодель збережена: {MODEL_PATH}")
    print(f"  Архітектура: DeepECGNet (~377 710 параметрів)")
    print(f"  Temperature: T = {T_opt:.3f}")
    print(f"  Scaler mean/std: включені у checkpoint")

    return history


# ════════════════════════════════════════════════════════════════════
#  КРОК 7: ФУНКЦІЯ ПРОГНОЗУВАННЯ (для використання у програмі)
# ════════════════════════════════════════════════════════════════════

def predict_ecg(feature_vector, model_path=MODEL_PATH):
    """
    Використання навченої моделі для прогнозування класу ЕКГ.

    Параметри:
        feature_vector : numpy array, shape (20,) — вектор ознак одного удару
        model_path     : шлях до збереженого model.pth

    Повертає:
        class_name   : str   — назва діагнозу (українська)
        confidence   : float — впевненість 0.0–1.0
        all_probs    : dict  — ймовірності всіх 5 класів

    Приклад використання:
        feat = np.array([72, 45, 35, 0.06, 0, 0, 1, -1, 2, 3,
                         0, 1.0, 0.05, 85, 155, 390, 0, 0.3, 3.0, 72])
        name, conf, probs = predict_ecg(feat)
        print(f"{name}: {conf*100:.1f}%")
    """
    import torch
    import torch.nn.functional as F

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Модель не знайдена: {model_path}\n"
            f"Спочатку запусти: python train_my_model.py"
        )

    # Завантаження checkpoint
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

    # Відновлення архітектури та вагів
    model = build_deepecgnet(
        input_dim=ckpt.get("input_dim", FEATURE_DIM),
        num_classes=ckpt.get("num_classes", NUM_CLASSES),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Нормалізація вхідного вектора (тими самими параметрами що під час навчання)
    mean = np.array(ckpt["scaler_mean"], dtype=np.float32)
    std  = np.array(ckpt["scaler_std"],  dtype=np.float32)
    T    = float(ckpt.get("temperature", 1.0))

    x_norm = (feature_vector.astype(np.float32) - mean) / (std + 1e-8)

    # Прогнозування
    with torch.no_grad():
        t      = torch.tensor(x_norm).unsqueeze(0)
        logits = model(t)
        probs  = F.softmax(logits / T, dim=1).numpy()[0]  # Temperature Scaling

    class_idx   = int(np.argmax(probs))
    class_name  = CLASS_NAMES[class_idx]
    confidence  = float(probs[class_idx])
    all_probs   = {name: float(p) for name, p in zip(CLASS_NAMES, probs)}

    return class_name, confidence, all_probs


# ════════════════════════════════════════════════════════════════════
#  ЗАПУСК
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Навчання власної нейромережі DeepECGNet на даних з data/"
    )
    parser.add_argument(
        "--epochs", type=int, default=150,
        help="Кількість епох навчання (default: 150)"
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Швидкий тест на 3 записах (для перевірки)"
    )
    args = parser.parse_args()

    train(epochs=args.epochs, fast_mode=args.fast)
