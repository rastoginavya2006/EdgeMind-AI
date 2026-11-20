"""Run: python -m pytest -q   (trains models first if missing)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pytest

from src.config import APPROVED_MODEL, ANOMALY_MODEL, LEAKAGE_COLUMNS, FEATURES
from src import database as db, mlops


@pytest.fixture(scope="session", autouse=True)
def models():
    if not (APPROVED_MODEL.exists() and ANOMALY_MODEL.exists()):
        from src.train import train
        train(n_estimators=50)


def test_dataset_loads():
    from src.data_loader import load_dataset
    df, src = load_dataset()
    assert len(df) > 1000 and all(f in df.columns for f in FEATURES)


def test_no_leakage_columns():
    from src.data_loader import load_dataset
    from src.preprocessing import clean
    df = clean(load_dataset()[0])
    assert not set(LEAKAGE_COLUMNS) & set(df.columns)


def test_prediction_range():
    from src.predict import predict
    from src.sensor_simulator import generate_reading
    p = predict(generate_reading("Normal"))
    assert 0 <= p["risk_score"] <= 1 and p["risk_category"] in ("Low", "Medium", "High")


def test_risk_thresholds():
    from src.predict import risk_category
    assert risk_category(0.1) == "Low" and risk_category(0.3) == "Medium" and risk_category(0.6) == "High"


def test_anomaly_extreme_reading():
    from src.anomaly_detection import detect
    a = detect({"air_temp": 309, "process_temp": 300, "rpm": 2900, "torque": 3, "tool_wear": 255})
    assert a["status"] == "Anomaly"


def test_database_roundtrip(tmp_path):
    p = tmp_path / "t.db"
    db.init_db(p)
    rid = db.insert_reading({"air_temp": 300, "process_temp": 310, "rpm": 1500, "torque": 40, "tool_wear": 10}, db_path=p)
    db.insert_prediction(rid, {"risk_score": .7, "risk_category": "High", "prediction": "FAILURE LIKELY"}, p)
    db.insert_anomaly(rid, {"status": "Normal", "anomaly_score": .1}, p)
    assert db.counts(p) == {"readings": 1, "anomalies": 0, "normal": 1, "high_risk": 1}


def test_promotion_rule():
    assert mlops.should_promote({"recall": .4, "f1": .9}, None)[0] is False
    assert mlops.should_promote({"recall": .7, "f1": .7}, None)[0] is True
    assert mlops.should_promote({"recall": .7, "f1": .6}, {"f1": .7})[0] is False
    assert mlops.should_promote({"recall": .7, "f1": .75}, {"f1": .7})[0] is True


def test_mlflow_runs_logged():
    assert not mlops.list_runs().empty
