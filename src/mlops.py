"""Lightweight local MLOps: MLflow file tracking + a JSON model registry with promotion rule."""
import json
import shutil
from datetime import datetime

import joblib
import mlflow

from src.config import MLRUNS_DIR, REGISTRY_DIR, REGISTRY_INDEX, APPROVED_MODEL

EXPERIMENT = "EdgeMind-Failure-Prediction"
MIN_RECALL = 0.50   # quality gate: candidate must catch at least half of failures
MIN_F1 = 0.50


TRACKING_URI = f"sqlite:///{(MLRUNS_DIR / 'mlflow.db').resolve().as_posix()}"


def setup_mlflow():
    """Local MLflow tracking: SQLite metadata + local artifact folder. No server needed."""
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(MLRUNS_DIR / "artifacts").resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT)


def load_registry() -> dict:
    if REGISTRY_INDEX.exists():
        return json.loads(REGISTRY_INDEX.read_text())
    return {"approved_version": None, "versions": []}


def save_registry(reg: dict):
    REGISTRY_INDEX.write_text(json.dumps(reg, indent=2))


def should_promote(candidate: dict, approved: dict | None) -> tuple[bool, str]:
    """Promotion rule: pass quality gate AND F1 >= current approved F1."""
    if candidate["recall"] < MIN_RECALL or candidate["f1"] < MIN_F1:
        return False, f"Rejected: recall/F1 below quality gate ({MIN_RECALL}/{MIN_F1})."
    if approved is None:
        return True, "Promoted: no approved model existed."
    if candidate["f1"] >= approved["f1"]:
        return True, f"Promoted: F1 {candidate['f1']:.3f} >= approved {approved['f1']:.3f}."
    return False, f"Retained existing: candidate F1 {candidate['f1']:.3f} < approved {approved['f1']:.3f}."


def register(model, metrics: dict, params: dict, run_id: str, data_source: str) -> dict:
    reg = load_registry()
    version = len(reg["versions"]) + 1
    path = REGISTRY_DIR / f"model_v{version}.joblib"
    joblib.dump(model, path)
    approved = next((v for v in reg["versions"] if v["version"] == reg["approved_version"]), None)
    promote, reason = should_promote(metrics, approved and approved["metrics"])
    entry = {
        "version": version, "file": path.name, "trained_at": datetime.now().isoformat(timespec="seconds"),
        "algorithm": "RandomForestClassifier", "params": params, "metrics": metrics,
        "mlflow_run_id": run_id, "data_source": data_source, "status": "approved" if promote else "candidate",
        "decision": reason,
    }
    if promote:
        for v in reg["versions"]:
            if v["status"] == "approved":
                v["status"] = "archived"
        reg["approved_version"] = version
        shutil.copy(path, APPROVED_MODEL)
    reg["versions"].append(entry)
    save_registry(reg)
    return entry


def list_runs():
    """Return MLflow runs as a DataFrame (empty if none)."""
    import pandas as pd
    try:
        setup_mlflow()
        return mlflow.search_runs(experiment_names=[EXPERIMENT])
    except Exception:
        return pd.DataFrame()
