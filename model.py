"""
model.py
========
ECGModel — уніфікована обгортка PyTorch (.pth) + sklearn (.npy).

predict() застосовує Temperature Scaling:
    probs = softmax(logits / T)
де T зберігається у checkpoint після train_model.py.
"""

import os
import numpy as np

FEATURE_DIM = 20
NUM_CLASSES  = 5

CLASS_NAMES = [
    "Нормальний синусовий ритм",
    "Аритмія",
    "Тахікардія",
    "Брадикардія",
    "Можливий інфаркт міокарда",
]

_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except (ImportError, OSError):
    pass


# ============================================================
#  Базова архітектура (fallback для старих .pth без DeepECGNet)
# ============================================================

if _TORCH_AVAILABLE:

    class ECGClassifierNet(nn.Module):
        """Стандартний MLP — fallback для .pth без DeepECGNet."""
        def __init__(self, input_dim=FEATURE_DIM, num_classes=NUM_CLASSES):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, num_classes),
            )

        def forward(self, x):
            return self.net(x)


# ============================================================
#  ECGModel — уніфікована обгортка PyTorch + sklearn
# ============================================================

class ECGModel:
    def __init__(self):
        self.model        = None
        self.scaler_mean  = None
        self.scaler_std   = None
        self.temperature  = 1.0   # Temperature Scaling параметр
        self.is_loaded    = False
        self.backend      = "none"   # "torch" | "sklearn"

    # --------------------------------------------------------
    # Завантаження
    # --------------------------------------------------------

    def load(self, model_path):
        if not os.path.exists(model_path):
            return False
        ext = os.path.splitext(model_path)[1].lower()
        if ext == ".pth":
            return self._load_pth(model_path)
        elif ext == ".npy":
            return self._load_npy(model_path)
        return self._load_pth(model_path) or self._load_npy(model_path)

    def _load_pth(self, path):
        if not _TORCH_AVAILABLE:
            return False
        try:
            import torch
            ckpt = torch.load(path, map_location="cpu", weights_only=False)

            input_dim   = ckpt.get("input_dim",   FEATURE_DIM)
            num_classes = ckpt.get("num_classes",  NUM_CLASSES)
            state       = ckpt["model_state"]

            # Пробуємо завантажити архітектуру — підтримуємо обидва скрипти навчання
            net = None
            errors = []

            # Варіант 1: train_my_model.py (build_deepecgnet — назви: input_projection, resblock_*)
            try:
                from train_my_model import build_deepecgnet
                candidate = build_deepecgnet(input_dim, num_classes)
                candidate.load_state_dict(state, strict=True)
                net = candidate
            except Exception as e1:
                errors.append(f"train_my_model: {e1}")

            # Варіант 2: train_model.py (_build_deep_model — назви: input_proj, res1..res4)
            if net is None:
                try:
                    from train_model import _build_deep_model
                    candidate = _build_deep_model(input_dim, num_classes)
                    candidate.load_state_dict(state, strict=True)
                    net = candidate
                except Exception as e2:
                    errors.append(f"train_model: {e2}")

            # Варіант 3: non-strict (часткове завантаження)
            if net is None:
                try:
                    from train_my_model import build_deepecgnet
                    candidate = build_deepecgnet(input_dim, num_classes)
                    candidate.load_state_dict(state, strict=False)
                    net = candidate
                    print("[model.py] Завантажено з strict=False (часткове)")
                except Exception as e3:
                    errors.append(f"non-strict: {e3}")

            if net is None:
                raise RuntimeError(f"Жоден варіант архітектури не підійшов: {errors}")

            net.eval()
            self.model       = net
            self.scaler_mean = np.array(ckpt["scaler_mean"], dtype=np.float32)
            self.scaler_std  = np.array(ckpt["scaler_std"],  dtype=np.float32)
            self.temperature = float(ckpt.get("temperature", 1.0))
            self.backend     = "torch"
            self.is_loaded   = True

            T = self.temperature
            if T < 1.0:
                temp_str = f"T={T:.3f} → загострений розподіл (висока впевненість)"
            elif abs(T - 1.0) < 0.01:
                temp_str = f"T={T:.3f} (без калібрування)"
            else:
                temp_str = f"T={T:.3f} → пом'якшений розподіл"
            print(f"[model.py] DeepECGNet завантажено | {temp_str}")
            return True
        except Exception as e:
            print(f"[model.py] Не вдалося завантажити .pth: {e}")
            return False

    def _load_npy(self, path):
        try:
            data = np.load(path, allow_pickle=True).item()
            self.model       = data["model"]
            self.scaler_mean = np.array(data.get("scaler_mean", np.zeros(FEATURE_DIM)), dtype=np.float32)
            self.scaler_std  = np.array(data.get("scaler_std",  np.ones(FEATURE_DIM)),  dtype=np.float32)
            self.temperature = float(data.get("temperature", 1.0))
            self.backend     = "sklearn"
            self.is_loaded   = True
            print("[model.py] sklearn модель завантажено")
            return True
        except Exception as e:
            print(f"[model.py] Не вдалося завантажити .npy: {e}")
            return False

    # --------------------------------------------------------
    # Інференс з Temperature Scaling
    # --------------------------------------------------------

    def predict(self, feature_vector):
        """
        Повертає (class_idx, probs_array).
        PyTorch: probs = softmax(logits / T)  ← Temperature Scaling
        sklearn:  probs = model.predict_proba()  (без T)
        """
        if not self.is_loaded or self.model is None:
            return None, None

        x = (feature_vector.astype(np.float32) - self.scaler_mean) / (self.scaler_std + 1e-8)

        if self.backend == "torch" and _TORCH_AVAILABLE:
            import torch
            import torch.nn.functional as F
            self.model.eval()
            with torch.no_grad():
                t      = torch.tensor(x).unsqueeze(0)
                logits = self.model(t)
                # Temperature Scaling: ділимо logits на T перед softmax
                scaled = logits / self.temperature
                probs  = F.softmax(scaled, dim=1).numpy()[0]
            return int(np.argmax(probs)), probs

        else:  # sklearn fallback
            proba = self.model.predict_proba(x.reshape(1, -1))[0]
            return int(np.argmax(proba)), proba

    # --------------------------------------------------------
    # Збереження (sklearn fallback)
    # --------------------------------------------------------

    def save(self, model_path, model_obj, scaler_mean, scaler_std,
             backend="sklearn", temperature=1.0):
        data = {
            "backend":      backend,
            "scaler_mean":  scaler_mean,
            "scaler_std":   scaler_std,
            "model":        model_obj,
            "temperature":  temperature,
        }
        np.save(model_path, data)


# ============================================================
#  Singleton
# ============================================================

_shared_model = ECGModel()

def get_model():
    return _shared_model
