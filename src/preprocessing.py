"""Preprocessing: missing values, leakage removal, stratified split."""
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import FEATURES, TARGET, LEAKAGE_COLUMNS, PROCESSED_DIR


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in LEAKAGE_COLUMNS if c in df.columns])
    df = df.dropna(subset=[TARGET]).copy()
    for col in FEATURES:  # fill any missing numeric value with the median
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())
    return df


def split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    X, y = df[FEATURES], df[TARGET].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    pd.concat([X_tr, y_tr], axis=1).to_csv(PROCESSED_DIR / "train.csv", index=False)
    pd.concat([X_te, y_te], axis=1).to_csv(PROCESSED_DIR / "test.csv", index=False)
    return X_tr, X_te, y_tr, y_te


def to_frame(reading: dict) -> pd.DataFrame:
    """Convert a reading dict (short or full names) into a 1-row model input frame."""
    from src.config import SHORT_NAMES
    inv = {v: k for k, v in SHORT_NAMES.items()}
    row = {inv.get(k, k): float(v) for k, v in reading.items() if inv.get(k, k) in FEATURES}
    missing = set(FEATURES) - set(row)
    if missing:
        raise ValueError(f"Missing sensor values: {sorted(missing)}")
    return pd.DataFrame([row])[FEATURES]
