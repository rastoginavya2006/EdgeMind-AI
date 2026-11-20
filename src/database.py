"""SQLite local persistence. Database file is created automatically."""
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import pandas as pd

from src.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS sensor_readings (
  id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, source TEXT NOT NULL,
  air_temp REAL, process_temp REAL, rpm REAL, torque REAL, tool_wear REAL);
CREATE TABLE IF NOT EXISTS prediction_history (
  id INTEGER PRIMARY KEY AUTOINCREMENT, reading_id INTEGER, timestamp TEXT NOT NULL,
  risk_score REAL, risk_category TEXT, prediction TEXT,
  FOREIGN KEY(reading_id) REFERENCES sensor_readings(id));
CREATE TABLE IF NOT EXISTS anomaly_history (
  id INTEGER PRIMARY KEY AUTOINCREMENT, reading_id INTEGER, timestamp TEXT NOT NULL,
  status TEXT, anomaly_score REAL, FOREIGN KEY(reading_id) REFERENCES sensor_readings(id));
CREATE TABLE IF NOT EXISTS system_logs (
  id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, level TEXT, message TEXT);
"""


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@contextmanager
def connect(db_path=None):
    conn = None
    try:
        conn = sqlite3.connect(db_path or DB_PATH, timeout=10)
        yield conn
        conn.commit()
    except sqlite3.Error as exc:
        if conn:
            conn.rollback()
        raise RuntimeError(f"Database error: {exc}") from exc
    finally:
        if conn:
            conn.close()


def init_db(db_path=None):
    with connect(db_path) as c:
        c.executescript(SCHEMA)


def insert_reading(r: dict, source: str = "SIMULATED", db_path=None) -> int:
    with connect(db_path) as c:
        cur = c.execute(
            "INSERT INTO sensor_readings(timestamp,source,air_temp,process_temp,rpm,torque,tool_wear)"
            " VALUES (?,?,?,?,?,?,?)",
            (now(), source, r["air_temp"], r["process_temp"], r["rpm"], r["torque"], r["tool_wear"]))
        return cur.lastrowid


def insert_prediction(reading_id: int, p: dict, db_path=None):
    with connect(db_path) as c:
        c.execute("INSERT INTO prediction_history(reading_id,timestamp,risk_score,risk_category,prediction)"
                  " VALUES (?,?,?,?,?)", (reading_id, now(), p["risk_score"], p["risk_category"], p["prediction"]))


def insert_anomaly(reading_id: int, a: dict, db_path=None):
    with connect(db_path) as c:
        c.execute("INSERT INTO anomaly_history(reading_id,timestamp,status,anomaly_score) VALUES (?,?,?,?)",
                  (reading_id, now(), a["status"], a["anomaly_score"]))


def log(message: str, level: str = "INFO", db_path=None):
    try:
        with connect(db_path) as c:
            c.execute("INSERT INTO system_logs(timestamp,level,message) VALUES (?,?,?)", (now(), level, message))
    except RuntimeError as exc:
        print(f"[log failed] {exc}")


def query(sql: str, params=(), db_path=None) -> pd.DataFrame:
    with connect(db_path) as c:
        return pd.read_sql_query(sql, c, params=params)


def history(limit: int = 500, db_path=None) -> pd.DataFrame:
    """Joined view of readings + prediction + anomaly results, newest first."""
    return query("""
      SELECT s.id, s.timestamp, s.source, s.air_temp, s.process_temp, s.rpm, s.torque, s.tool_wear,
             p.risk_score, p.risk_category, p.prediction, a.status AS anomaly_status, a.anomaly_score
      FROM sensor_readings s
      LEFT JOIN prediction_history p ON p.reading_id = s.id
      LEFT JOIN anomaly_history a ON a.reading_id = s.id
      ORDER BY s.id DESC LIMIT ?""", (limit,), db_path)


def counts(db_path=None) -> dict:
    q = lambda s: int(query(s, db_path=db_path).iloc[0, 0] or 0)
    return {
        "readings": q("SELECT COUNT(*) FROM sensor_readings"),
        "anomalies": q("SELECT COUNT(*) FROM anomaly_history WHERE status='Anomaly'"),
        "normal": q("SELECT COUNT(*) FROM anomaly_history WHERE status='Normal'"),
        "high_risk": q("SELECT COUNT(*) FROM prediction_history WHERE risk_category='High'"),
    }


def clear_all(db_path=None):
    with connect(db_path) as c:
        for t in ("prediction_history", "anomaly_history", "sensor_readings", "system_logs"):
            c.execute(f"DELETE FROM {t}")
