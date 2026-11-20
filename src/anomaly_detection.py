"""Isolation Forest anomaly detection on sensor readings."""
import joblib

from src.config import ANOMALY_MODEL
from src.preprocessing import to_frame

_iso = None


def load_detector(force: bool = False):
    global _iso
    if _iso is None or force:
        if not ANOMALY_MODEL.exists():
            raise FileNotFoundError("No anomaly model found. Run: python -m src.train")
        _iso = joblib.load(ANOMALY_MODEL)
    return _iso


def detect(reading: dict) -> dict:
    iso = load_detector()
    X = to_frame(reading)
    label = int(iso.predict(X)[0])          # 1 = normal, -1 = anomaly
    score = float(iso.decision_function(X)[0])  # < 0 means anomalous
    return {"status": "Anomaly" if label == -1 else "Normal", "anomaly_score": round(score, 4)}
