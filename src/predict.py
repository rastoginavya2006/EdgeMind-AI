"""Failure prediction using the locally saved, approved Random Forest model."""
import joblib

from src.config import APPROVED_MODEL, RISK_MEDIUM, RISK_HIGH, FEATURES
from src.preprocessing import to_frame

_model = None


def load_model(force: bool = False):
    global _model
    if _model is None or force:
        if not APPROVED_MODEL.exists():
            raise FileNotFoundError("No approved model found. Run: python -m src.train")
        _model = joblib.load(APPROVED_MODEL)
    return _model


def risk_category(score: float) -> str:
    if score >= RISK_HIGH:
        return "High"
    if score >= RISK_MEDIUM:
        return "Medium"
    return "Low"


def explain(reading: dict) -> list[str]:
    """Simple human-readable hints based on known AI4I failure physics."""
    from src.config import SHORT_NAMES as S
    r = {S.get(k, k): float(v) for k, v in reading.items()}
    notes = []
    power = r["torque"] * r["rpm"] * 2 * 3.14159 / 60
    if r["tool_wear"] > 200:
        notes.append(f"Tool wear is high ({r['tool_wear']:.0f} min) - tool wear failures rise above ~200 min.")
    if power < 3500 or power > 9000:
        notes.append(f"Mechanical power {power:.0f} W is outside the typical 3500-9000 W band (power failure risk).")
    if r["process_temp"] - r["air_temp"] < 8.6 and r["rpm"] < 1380:
        notes.append("Small temperature difference with low speed - heat dissipation failure pattern.")
    if r["torque"] * r["tool_wear"] > 11000:
        notes.append("Torque x tool wear is very high - overstrain failure pattern.")
    return notes or ["All readings are within typical operating ranges."]


def predict(reading: dict) -> dict:
    model = load_model()
    X = to_frame(reading)
    score = float(model.predict_proba(X)[0][1])  # share of trees voting 'failure' (risk score)
    return {
        "risk_score": round(score, 4),
        "risk_category": risk_category(score),
        "prediction": "FAILURE LIKELY" if model.predict(X)[0] == 1 else "NO FAILURE EXPECTED",
        "explanation": explain(reading),
        "feature_importance": dict(zip(FEATURES, map(float, model.feature_importances_))),
    }
