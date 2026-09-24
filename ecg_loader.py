import os
import io
import numpy as np
import scipy.io

_TEXT_ENCODINGS = ("utf-8", "utf-8-sig", "latin-1", "cp1251", "cp1252", "iso-8859-1")


def _open_text(filepath):
    """Return file content as a string, trying common encodings."""
    for enc in _TEXT_ENCODINGS:
        try:
            with open(filepath, "r", encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, LookupError):
            continue
    # Last resort: ignore undecodable bytes
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _loadtxt_safe(filepath, **kwargs):
    """np.loadtxt wrapper that handles encoding automatically."""
    text = _open_text(filepath)
    return np.loadtxt(io.StringIO(text), **kwargs)


def _genfromtxt_safe(filepath, **kwargs):
    """np.genfromtxt wrapper that handles encoding automatically."""
    text = _open_text(filepath)
    return np.genfromtxt(io.StringIO(text), **kwargs)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".gif"}


def load_ecg(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        return _load_csv(filepath)
    elif ext in (".txt", ".dat"):
        return _load_dat(filepath)
    elif ext == ".mat":
        return _load_mat(filepath)
    elif ext in IMAGE_EXTENSIONS:
        return _load_image(filepath)
    else:
        raise ValueError(
            f"Непідтримуваний формат: '{ext}'. "
            "Використовуйте: .csv .txt .dat .mat .png .jpg .jpeg .bmp .tiff"
        )


def _load_image(filepath):
    """Digitise an ECG image and return (signal, fs)."""
    from ecg_digitizer import digitize_ecg_image
    signal, fs = digitize_ecg_image(filepath)
    return signal, fs


def _load_csv(filepath):
    try:
        data = _loadtxt_safe(filepath, delimiter=",", comments="#")
    except ValueError:
        data = _genfromtxt_safe(filepath, delimiter=",", comments="#", filling_values=0)

    return _parse_2d(data)


def _load_dat(filepath):
    # --- Strategy 1: binary .dat (16-bit integers, e.g. MIT-BIH) ---
    try:
        raw = np.fromfile(filepath, dtype=np.int16)
        if len(raw) > 100:
            signal = raw.astype(float)
            signal = signal - np.mean(signal)
            return signal, 360
    except Exception:
        pass

    # --- Strategy 2: text (whitespace or comma separated) ---
    text = _open_text(filepath)
    lines = [l.strip() for l in text.splitlines() if l.strip() and not l.startswith("#")]
    if not lines:
        raise ValueError("File appears to be empty.")

    # Detect delimiter
    first_data = next((l for l in lines if l and l[0].lstrip("-").replace(".", "").replace("e", "").replace("+", "").isdigit()), lines[0])
    delimiter = "," if "," in first_data else None

    try:
        data = np.loadtxt(io.StringIO(text), delimiter=delimiter, comments="#")
        return _parse_2d(data)
    except ValueError:
        pass

    # --- Strategy 3: line-by-line, take first numeric token ---
    values = []
    for line in lines:
        token = line.split(",")[0].split()[0] if line else ""
        try:
            values.append(float(token))
        except (ValueError, IndexError):
            continue

    if not values:
        raise ValueError("Could not extract numeric values from the file.")

    return np.array(values, dtype=float), 360


def _parse_2d(data):
    if data.ndim == 1:
        return data.astype(float), 360

    if data.shape[1] >= 2:
        time_col = data[:, 0]
        diffs = np.diff(time_col)
        if len(diffs) > 0 and np.std(diffs) < 1e-3 and np.mean(diffs) > 1e-6:
            dt = np.mean(diffs)
            fs = int(round(1.0 / dt)) if dt > 1e-6 else 360
            return data[:, 1].astype(float), max(fs, 1)
        return data[:, 0].astype(float), 360

    return data[:, 0].astype(float), 360


def _load_mat(filepath):
    mat = scipy.io.loadmat(filepath)
    fs = 360

    for fs_key in ("fs", "Fs", "FS", "samplerate", "SampleRate"):
        if fs_key in mat:
            val = mat[fs_key]
            if hasattr(val, "flatten"):
                val = val.flatten()
            fs = int(val[0]) if len(val) > 0 else 360
            break

    for sig_key in ("val", "ecg", "signal", "data", "ECG", "ecgSignal"):
        if sig_key in mat:
            sig = mat[sig_key]
            if isinstance(sig, np.ndarray):
                return sig.flatten().astype(float), fs

    for key, val in mat.items():
        if key.startswith("_"):
            continue
        if isinstance(val, np.ndarray) and val.size > 100:
            return val.flatten().astype(float), fs

    raise ValueError("Cannot find ECG signal array in MAT file. "
                     "Expected key: 'val', 'ecg', 'signal', or 'data'.")
