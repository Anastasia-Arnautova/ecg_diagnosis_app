"""
train_model.py
==============
DeepECGNet — глибока нейромережа з 4 ResBlock (~377k параметрів).

Особливості:
  - LabelSmoothingLoss (ε=0.05)      — запобігає перевпевненості
  - Mixup Augmentation (α=0.4)        — кращий generalization
  - Temperature Scaling               — калібрування до 95%+ впевненості
  - CosineAnnealingWarmRestarts LR    — стабільніше навчання
  - Gradient Clipping (norm=1.0)

Навчання — ОДИН РАЗ:
    python train_model.py              # MIT-BIH (основний режим)
    python train_model.py --synthetic  # синтетика (без інтернету)
    python train_model.py --epochs 150

Output:
    saved_model/model.pth  — PyTorch checkpoint + temperature T
    saved_model/model.npy  — sklearn fallback (якщо немає PyTorch)
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from model import ECGModel, CLASS_NAMES, FEATURE_DIM, NUM_CLASSES

SAVE_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_model")
MODEL_PTH   = os.path.join(SAVE_DIR, "model.pth")
MODEL_NPY   = os.path.join(SAVE_DIR, "model.npy")

RANDOM_SEED = 42
TEST_SIZE   = 0.15
VAL_SIZE    = 0.15


# ============================================================
#  ENTRY POINT
# ============================================================

def train(use_synthetic=False, records=None, epochs=100):
    os.makedirs(SAVE_DIR, exist_ok=True)

    # --------------------------------------------------------
    # 1. Завантаження даних
    # --------------------------------------------------------
    if use_synthetic:
        print("Режим: синтетичні дані...")
        X, y = _generate_synthetic_inline()
    else:
        print("Завантаження MIT-BIH Arrhythmia Database...")
        print("(Перший запуск: ~2-5 хв, потім кеш у data/)\n")
        try:
            from dataset_loader import load_mitbih_features
            X, y = load_mitbih_features(records=records, verbose=True)
        except Exception as e:
            print(f"[УВАГА] MIT-BIH недоступна: {e}")
            print("Перемикаємося на синтетичні дані...\n")
            X, y = _generate_synthetic_inline()

    print(f"\nДатасет: {X.shape[0]:,} зразків | {X.shape[1]} ознак | "
          f"{len(np.unique(y))} класів\n")

    # --------------------------------------------------------
    # 2. Train / Validation / Test split (стратифікований)
    # --------------------------------------------------------
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )
    val_ratio = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=RANDOM_SEED
    )

    print("Розбивка датасету:")
    print(f"  Навчальна  : {len(X_train):>7,} зразків  ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Валідаційна: {len(X_val):>7,} зразків  ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Тестова    : {len(X_test):>7,} зразків  ({len(X_test)/len(X)*100:.1f}%)\n")

    # --------------------------------------------------------
    # 3. Нормалізація (fit тільки на train)
    # --------------------------------------------------------
    scaler_mean = X_train.mean(axis=0).astype(np.float32)
    scaler_std  = (X_train.std(axis=0) + 1e-8).astype(np.float32)

    X_train_sc = ((X_train - scaler_mean) / scaler_std).astype(np.float32)
    X_val_sc   = ((X_val   - scaler_mean) / scaler_std).astype(np.float32)
    X_test_sc  = ((X_test  - scaler_mean) / scaler_std).astype(np.float32)

    # --------------------------------------------------------
    # 4. Ваги класів (компенсація дисбалансу)
    # --------------------------------------------------------
    class_counts  = np.bincount(y_train, minlength=NUM_CLASSES).astype(float)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights /= class_weights.sum()
    print("Ваги класів (обернені до частоти):")
    for i, (name, w) in enumerate(zip(CLASS_NAMES, class_weights)):
        print(f"  {i} {name:<40}: {w:.4f}")
    print()

    # --------------------------------------------------------
    # 5. Основна нейронна мережа — PyTorch DeepECGNet
    # --------------------------------------------------------
    trained_torch = _train_pytorch(
        X_train_sc, y_train,
        X_val_sc,   y_val,
        X_test_sc,  y_test,
        scaler_mean, scaler_std,
        class_weights,
        epochs=epochs,
    )

    # --------------------------------------------------------
    # 6. Fallback — RandomForest (якщо PyTorch недоступний)
    # --------------------------------------------------------
    if not trained_torch:
        print("[FALLBACK] Навчання RandomForestClassifier...")
        _train_sklearn_fallback(
            X_train_sc, y_train,
            X_val_sc,   y_val,
            X_test_sc,  y_test,
            scaler_mean, scaler_std,
        )


# ============================================================
#  DeepECGNet — архітектура (4 ResBlock, ~377k параметрів)
# ============================================================

def _build_deep_model(input_dim, num_classes):
    """
    DeepECGNet:
      input_proj : Linear 20→256  + BN + ReLU + Dropout(0.3)
      res1, res2 : ResBlock 256→256  (×2)
      mid        : Linear 256→128 + BN + ReLU + Dropout(0.25)
      res3, res4 : ResBlock 128→128  (×2)
      head       : Linear 128→64 → ReLU → Dropout(0.2) → Linear 64→5
    Temperature Scaling застосовується при інференсі в model.py
    """
    try:
        import torch.nn as nn

        class ResBlock(nn.Module):
            def __init__(self, dim, dropout=0.2):
                super().__init__()
                self.block = nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.BatchNorm1d(dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim, dim),
                    nn.BatchNorm1d(dim),
                )
                self.relu = nn.ReLU()

            def forward(self, x):
                return self.relu(self.block(x) + x)

        class DeepECGNet(nn.Module):
            def __init__(self):
                super().__init__()
                # Вхідна проєкція: 20 → 256
                self.input_proj = nn.Sequential(
                    nn.Linear(input_dim, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                )
                # Блок 1-2: глибока нелінійна обробка
                self.res1 = ResBlock(256, dropout=0.2)
                self.res2 = ResBlock(256, dropout=0.2)

                # Компресія: 256 → 128
                self.mid = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                )
                # Блок 3-4: тонке розрізнення класів
                self.res3 = ResBlock(128, dropout=0.15)
                self.res4 = ResBlock(128, dropout=0.15)

                # Класифікатор
                self.head = nn.Sequential(
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, num_classes),
                )

            def forward(self, x):
                x = self.input_proj(x)
                x = self.res1(x)
                x = self.res2(x)
                x = self.mid(x)
                x = self.res3(x)
                x = self.res4(x)
                return self.head(x)

        return DeepECGNet()

    except Exception:
        from model import ECGClassifierNet
        return ECGClassifierNet(input_dim, num_classes)


# ============================================================
#  Label Smoothing Loss (ε=0.05)
# ============================================================

def _label_smoothing_loss(logits, targets, num_classes, eps=0.05, weight=None):
    """
    Замість жорстких міток [0,0,1,0,0] використовуємо
    [(eps/K), ..., (1-eps + eps/K), ..., (eps/K)].
    Запобігає перевпевненості та збільшує margin між класами.
    """
    import torch
    import torch.nn.functional as F

    log_probs = F.log_softmax(logits, dim=1)

    # Soft targets
    smooth = eps / num_classes
    one_hot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1.0)
    soft_targets = one_hot * (1.0 - eps) + smooth

    loss = -(soft_targets * log_probs).sum(dim=1)

    # Ваги класів
    if weight is not None:
        w = weight[targets]
        loss = loss * w * num_classes   # масштабуємо щоб зберегти рівень лосу

    return loss.mean()


# ============================================================
#  Mixup Augmentation
# ============================================================

def _mixup_batch(xb, yb, alpha=0.4):
    """
    Мікшує пари зразків: x_new = λ·x_i + (1-λ)·x_j
    Повертає (x_mixed, y_a, y_b, lam) для обчислення лосу.
    """
    import torch
    import numpy as np

    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    batch_size = xb.size(0)
    idx = torch.randperm(batch_size, device=xb.device)

    x_mix = lam * xb + (1.0 - lam) * xb[idx]
    y_a, y_b = yb, yb[idx]
    return x_mix, y_a, y_b, lam


def _mixup_loss(logits, y_a, y_b, lam, num_classes, weight=None, eps=0.05):
    loss_a = _label_smoothing_loss(logits, y_a, num_classes, eps=eps, weight=weight)
    loss_b = _label_smoothing_loss(logits, y_b, num_classes, eps=eps, weight=weight)
    return lam * loss_a + (1.0 - lam) * loss_b


# ============================================================
#  Temperature Scaling (пошук оптимального T на val set)
# ============================================================

def _find_temperature(model, X_val_t, y_val_t, device):
    """
    Шукає T ∈ [0.3, 2.0] що мінімізує NLL на валідаційній вибірці.
    T < 1 → загострює розподіл (вища впевненість).
    T > 1 → згладжує (менша впевненість, більш чесна).
    """
    import torch
    import torch.nn.functional as F

    model.eval()
    with torch.no_grad():
        logits_all = []
        for i in range(0, len(X_val_t), 1024):
            xb = X_val_t[i:i+1024].to(device)
            logits_all.append(model(xb).cpu())
        logits_all = torch.cat(logits_all, dim=0)

    best_T   = 1.0
    best_nll = float("inf")

    for T in np.arange(0.30, 2.01, 0.025):
        scaled = logits_all / T
        nll = F.cross_entropy(scaled, y_val_t).item()
        if nll < best_nll:
            best_nll = nll
            best_T   = float(T)

    return best_T


# ============================================================
#  PyTorch Training
# ============================================================

def _train_pytorch(X_train, y_train, X_val, y_val,
                   X_test, y_test,
                   scaler_mean, scaler_std,
                   class_weights, epochs=100):
    try:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
        from model import ECGClassifierNet
    except (ImportError, OSError) as e:
        print(f"[УВАГА] PyTorch недоступний ({e})")
        print("Рекомендація: встановіть Python 3.11/3.12 та  pip install torch\n")
        return False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch пристрій: {device}")
    if device.type == "cpu":
        print("  (GPU не знайдено — навчання на CPU, ~5-15 хв для 100k зразків)\n")
    else:
        print(f"  GPU: {torch.cuda.get_device_name(0)}\n")

    # --- Tensors ---
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    X_v  = torch.tensor(X_val,   dtype=torch.float32)
    y_v  = torch.tensor(y_val,   dtype=torch.long)
    X_te = torch.tensor(X_test,  dtype=torch.float32)
    y_te = torch.tensor(y_test,  dtype=torch.long)

    train_dl = DataLoader(TensorDataset(X_tr, y_tr),
                          batch_size=512, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(TensorDataset(X_v,  y_v),
                          batch_size=1024, shuffle=False, num_workers=0)

    # --- Модель DeepECGNet (~377k параметрів) ---
    model = _build_deep_model(FEATURE_DIM, NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"DeepECGNet: {total_params:,} параметрів\n")

    # --- Ваги класів ---
    w = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # --- Optimizer: AdamW + CosineAnnealingWarmRestarts ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=2e-4
    )
    # T_0=10: перший цикл 10 епох, T_mult=2: кожен наступний вдвічі довший
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-5
    )

    # --- Early stopping ---
    best_val_acc = 0.0
    patience_cnt = 0
    EARLY_STOP   = 25

    print(f"{'Епоха':>6} {'Train Loss':>12} {'Train Acc':>10} "
          f"{'Val Acc':>10} {'LR':>10}")
    print("-" * 57)

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        total_loss = correct = total = 0

        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)

            # Mixup augmentation
            x_mix, y_a, y_b, lam = _mixup_batch(xb, yb, alpha=0.4)

            optimizer.zero_grad()
            logits = model(x_mix)
            loss   = _mixup_loss(logits, y_a, y_b, lam,
                                 NUM_CLASSES, weight=w, eps=0.05)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * len(yb)
            # Точність рахуємо по оригінальному yb (не mixup)
            with torch.no_grad():
                orig_logits = model(xb)
                correct += (orig_logits.argmax(1) == yb).sum().item()
            total += len(yb)

        scheduler.step()
        train_loss = total_loss / total
        train_acc  = correct / total

        # --- Validate ---
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                preds       = model(xb).argmax(1)
                val_correct += (preds == yb).sum().item()
                val_total   += len(yb)
        val_acc = val_correct / val_total
        lr_now  = optimizer.param_groups[0]["lr"]

        if epoch % 5 == 0 or epoch == 1:
            marker = " *" if val_acc > best_val_acc else ""
            print(f"{epoch:>6} {train_loss:>12.4f} {train_acc*100:>9.2f}%"
                  f" {val_acc*100:>9.2f}%{marker:2} {lr_now:>10.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_cnt = 0
            # Зберігаємо тимчасово (без temperature — додамо після)
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "val_acc":      val_acc,
                "scaler_mean":  scaler_mean,
                "scaler_std":   scaler_std,
                "input_dim":    FEATURE_DIM,
                "num_classes":  NUM_CLASSES,
                "temperature":  1.0,   # оновимо нижче
            }, MODEL_PTH)
        else:
            patience_cnt += 1
            if patience_cnt >= EARLY_STOP:
                print(f"\n[Early stopping] Немає покращення {EARLY_STOP} епох.")
                break

    print(f"\nНайкраща точність на валідації: {best_val_acc*100:.2f}%")

    # --------------------------------------------------------
    # Temperature Scaling — пошук оптимального T на val set
    # --------------------------------------------------------
    print("\nКалібрування впевненості (Temperature Scaling)...")
    ckpt = torch.load(MODEL_PTH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    opt_T = _find_temperature(model, X_v, y_v, device)
    print(f"  Оптимальна температура T = {opt_T:.3f}")
    if opt_T < 1.0:
        print(f"  T < 1 → загострює розподіл → вища впевненість (95%+)")
    else:
        print(f"  T > 1 → пом'якшує розподіл → чесніша впевненість")

    # Зберігаємо остаточний checkpoint з temperature
    ckpt["temperature"] = opt_T
    torch.save(ckpt, MODEL_PTH)

    # --------------------------------------------------------
    # Фінальна оцінка на тестовій вибірці (з Temperature Scaling)
    # --------------------------------------------------------
    model.eval()
    test_preds  = []
    test_confs  = []
    with torch.no_grad():
        for i in range(0, len(X_te), 1024):
            xb     = X_te[i:i+1024].to(device)
            logits = model(xb) / opt_T
            probs  = torch.softmax(logits, dim=1)
            test_preds.append(probs.argmax(1).cpu().numpy())
            test_confs.append(probs.max(1).values.cpu().numpy())

    y_pred     = np.concatenate(test_preds)
    confidences = np.concatenate(test_confs)
    test_acc   = accuracy_score(y_te.numpy(), y_pred)

    print(f"\n{'='*60}")
    print(f"  ФІНАЛЬНА ТОЧНІСТЬ НА ТЕСТОВІЙ ВИБІРЦІ: {test_acc*100:.2f}%")
    print(f"  Середня впевненість (правильні): "
          f"{confidences[y_pred == y_te.numpy()].mean()*100:.1f}%")
    print(f"  % передбачень з впевненістю ≥ 95%: "
          f"{(confidences >= 0.95).mean()*100:.1f}%")
    print(f"{'='*60}\n")
    print(classification_report(
        y_te.numpy(), y_pred, target_names=CLASS_NAMES, zero_division=0
    ))
    print(f"PyTorch модель збережена: {MODEL_PTH}")
    return True


# ============================================================
#  SKLEARN FALLBACK
# ============================================================

def _train_sklearn_fallback(X_train, y_train, X_val, y_val,
                             X_test, y_test, scaler_mean, scaler_std):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV

    print("Навчання RandomForestClassifier (fallback)...")
    clf = RandomForestClassifier(
        n_estimators=300, max_depth=20,
        min_samples_leaf=3, n_jobs=-1,
        class_weight="balanced", random_state=RANDOM_SEED,
    )
    clf.fit(X_train, y_train)

    print("Калібрування ймовірностей...")
    cal = CalibratedClassifierCV(clf, method="isotonic", cv="prefit")
    cal.fit(X_val, y_val)

    y_pred   = cal.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"\nТочність (sklearn fallback): {test_acc*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))

    m = ECGModel()
    m.save(MODEL_NPY, cal, scaler_mean, scaler_std, backend="sklearn")
    print(f"Sklearn модель збережена: {MODEL_NPY}")


# ============================================================
#  SYNTHETIC DATA
# ============================================================

_rng = np.random.default_rng(RANDOM_SEED)

def _synth_sample(label):
    r = _rng
    if label == 0:   # Норма
        return np.array([r.normal(72,8), r.normal(45,10), r.normal(35,8),
            r.normal(0.06,0.01), r.uniform(0,0.03), r.normal(0,2),
            r.normal(1,1.5), r.normal(-1,1.5), r.normal(2,1), r.normal(3,0.5),
            0, r.normal(1.0,0.15), r.normal(0.05,0.02), r.normal(85,8),
            r.normal(155,15), r.normal(390,20), 0,
            r.normal(0.3,0.3), r.normal(3.0,0.5),
            float(r.normal(72,8)*r.uniform(0.9,1.1))], dtype=np.float32)
    elif label == 1:  # Аритмія
        return np.array([r.normal(75,15), r.normal(80,25), r.normal(65,20),
            r.normal(0.18,0.06), r.uniform(0.1,0.4), r.normal(0,3),
            r.normal(2,2), r.normal(-2,2), r.normal(4,2), r.normal(1.5,1.5),
            int(r.integers(0,4)), r.normal(0.9,0.3), r.normal(0.2,0.08),
            r.normal(90,15), r.normal(165,30), r.normal(410,30),
            int(r.random()>0.5), r.normal(0.5,0.5), r.normal(3.5,0.8),
            float(r.normal(75,15)*r.uniform(0.9,1.1))], dtype=np.float32)
    elif label == 2:  # Тахікардія
        return np.array([r.normal(118,15), r.normal(30,10), r.normal(20,8),
            r.normal(0.05,0.01), r.uniform(0,0.05), r.normal(0,2),
            r.normal(1,1.5), r.normal(-1,1.5), r.normal(2,1), r.normal(2.5,0.5),
            0, r.normal(1.1,0.15), r.normal(0.06,0.02), r.normal(82,8),
            r.normal(145,10), r.normal(360,20), 0,
            r.normal(0.4,0.3), r.normal(3.2,0.4),
            float(r.normal(118,15)*r.uniform(0.9,1.1))], dtype=np.float32)
    elif label == 3:  # Брадикардія
        return np.array([r.normal(48,6), r.normal(55,12), r.normal(45,10),
            r.normal(0.07,0.02), r.uniform(0,0.05), r.normal(0,2),
            r.normal(1,1.5), r.normal(-1,1.5), r.normal(2,1), r.normal(3.5,0.5),
            0, r.normal(1.05,0.15), r.normal(0.05,0.02), r.normal(88,10),
            r.normal(175,20), r.normal(440,25), 0,
            r.normal(0.3,0.2), r.normal(3.0,0.4),
            float(r.normal(48,6)*r.uniform(0.9,1.1))], dtype=np.float32)
    else:             # ІМ
        return np.array([r.normal(90,20), r.normal(35,12), r.normal(28,10),
            r.normal(0.08,0.02), r.uniform(0.05,0.2), r.normal(4,2),
            r.normal(8,3), r.normal(-3,2), r.normal(6,2), r.normal(-2,1.5),
            int(r.integers(2,6)), r.normal(0.7,0.2), r.normal(0.15,0.05),
            r.normal(110,20), r.normal(180,25), r.normal(460,30),
            int(r.random()>0.7), r.normal(0.8,0.4), r.normal(4.5,0.8),
            float(r.normal(90,20)*r.uniform(0.9,1.1))], dtype=np.float32)


def _generate_synthetic_inline(n_per_class=600):
    X, y = [], []
    for lbl in range(5):
        for _ in range(n_per_class):
            X.append(_synth_sample(lbl))
            y.append(lbl)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# ============================================================
#  CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Навчання DeepECGNet (MIT-BIH / PyTorch)"
    )
    parser.add_argument("--synthetic", action="store_true",
                        help="Синтетичні дані замість MIT-BIH")
    parser.add_argument("--records", nargs="+", default=None, metavar="ID",
                        help="Конкретні записи MIT-BIH (напр.: 100 101 200)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Кількість епох навчання (default: 100)")
    args = parser.parse_args()

    train(use_synthetic=args.synthetic,
          records=args.records,
          epochs=args.epochs)
