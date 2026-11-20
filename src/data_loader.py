"""Load the AI4I 2020 dataset (download once, then fully offline).
Falls back to a CLEARLY LABELLED synthetic demo dataset if download is impossible."""
import io
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from src.config import DATASET_FILE, DATASET_URL, SYNTHETIC_FILE, FEATURES, TARGET


def download_dataset(timeout: int = 30) -> bool:
    """Download the UCI zip and extract ai4i2020.csv. Returns True on success."""
    try:
        with urllib.request.urlopen(DATASET_URL, timeout=timeout) as resp:
            data = resp.read()
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
            DATASET_FILE.write_bytes(zf.read(name))
        return True
    except Exception as exc:  # network down, URL changed, etc.
        print(f"[data_loader] Could not download AI4I dataset: {exc}")
        return False


def generate_synthetic(n: int = 10000, seed: int = 42) -> pd.DataFrame:
    """SYNTHETIC DEMO DATA ONLY - mimics AI4I value ranges, not real records."""
    rng = np.random.default_rng(seed)
    air = rng.normal(300, 2, n)
    proc = air + 10 + rng.normal(0, 1, n)
    rpm = rng.normal(1540, 180, n).clip(1150, 2900)
    torque = (40 + (1540 - rpm) * 0.04 + rng.normal(0, 8, n)).clip(3, 77)
    wear = rng.integers(0, 254, n)
    power = torque * rpm * 2 * np.pi / 60
    fail = ((wear > 200) & (torque > 50)) | (power < 3500) | (power > 9000) | (
        (proc - air < 8.6) & (rpm < 1380))
    fail = fail | (rng.random(n) < 0.001)
    return pd.DataFrame({
        "UDI": np.arange(1, n + 1), "Product ID": "SYN", "Type": rng.choice(["L", "M", "H"], n),
        FEATURES[0]: air.round(1), FEATURES[1]: proc.round(1), FEATURES[2]: rpm.round(0),
        FEATURES[3]: torque.round(1), FEATURES[4]: wear, TARGET: fail.astype(int),
    })


def load_dataset(allow_download: bool = True):
    """Return (dataframe, source_label). source_label is 'AI4I 2020 (UCI)' or 'SYNTHETIC DEMO DATA'."""
    if not DATASET_FILE.exists() and allow_download:
        download_dataset()
    if DATASET_FILE.exists():
        return pd.read_csv(DATASET_FILE), "AI4I 2020 (UCI)"
    if not SYNTHETIC_FILE.exists():
        generate_synthetic().to_csv(SYNTHETIC_FILE, index=False)
    return pd.read_csv(SYNTHETIC_FILE), "SYNTHETIC DEMO DATA"


def basic_eda(df: pd.DataFrame) -> dict:
    """Small EDA summary used by training logs and the dashboard."""
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_values": int(df[FEATURES + [TARGET]].isna().sum().sum()),
        "failure_rate": float(df[TARGET].mean()),
        "failures": int(df[TARGET].sum()),
        "describe": df[FEATURES].describe().round(2),
    }
