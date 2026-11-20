"""Glue: reading -> predict + detect -> SQLite."""
from src import database as db
from src.predict import predict
from src.anomaly_detection import detect


def process(reading: dict, source: str = "SIMULATED") -> dict:
    db.init_db()
    rid = db.insert_reading(reading, source)
    p = predict(reading)
    a = detect(reading)
    db.insert_prediction(rid, p)
    db.insert_anomaly(rid, a)
    if p["risk_category"] == "High":
        db.log(f"HIGH failure risk ({p['risk_score']:.2f}) on reading #{rid}", "ALERT")
    if a["status"] == "Anomaly":
        db.log(f"Anomaly detected on reading #{rid} (score {a['anomaly_score']:.3f})", "WARNING")
    return {"reading_id": rid, "reading": reading, "prediction": p, "anomaly": a}
