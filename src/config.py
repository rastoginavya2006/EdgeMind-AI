"""Central configuration: all paths are local to the project folder (offline-first)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
STREAMING_DIR = ROOT / "data" / "streaming"
MODELS_DIR = ROOT / "models"
REGISTRY_DIR = MODELS_DIR / "model_registry"
DB_PATH = ROOT / "database" / "edgemind.db"
MLRUNS_DIR = ROOT / "mlruns"

DATASET_FILE = RAW_DIR / "ai4i2020.csv"
SYNTHETIC_FILE = RAW_DIR / "synthetic_demo_data.csv"
DATASET_URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"

# Model input features (AI4I 2020). Failure-mode columns (TWF, HDF, PWF, OSF, RNF) are NEVER used.
FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
SHORT_NAMES = {
    "Air temperature [K]": "air_temp",
    "Process temperature [K]": "process_temp",
    "Rotational speed [rpm]": "rpm",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
}
TARGET = "Machine failure"
LEAKAGE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

APPROVED_MODEL = REGISTRY_DIR / "approved_model.joblib"
ANOMALY_MODEL = REGISTRY_DIR / "anomaly_model.joblib"
REGISTRY_INDEX = REGISTRY_DIR / "registry.json"

# Risk score thresholds (risk score = model vote share, not a calibrated probability)
RISK_MEDIUM = 0.30
RISK_HIGH = 0.60

for d in (RAW_DIR, PROCESSED_DIR, STREAMING_DIR, REGISTRY_DIR, DB_PATH.parent, MLRUNS_DIR):
    d.mkdir(parents=True, exist_ok=True)
