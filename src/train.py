"""Train Random Forest (failure prediction) + Isolation Forest (anomaly). Run: python -m src.train"""
import argparse

import joblib
import mlflow
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.config import ANOMALY_MODEL, TARGET
from src.data_loader import load_dataset, basic_eda
from src.preprocessing import clean, split
from src import mlops


def train(n_estimators: int = 200, max_depth: int | None = 12, seed: int = 42, allow_download: bool = True) -> dict:
    df, source = load_dataset(allow_download=allow_download)
    eda = basic_eda(df)
    print(f"[train] Data source: {source} | rows={eda['rows']} failures={eda['failures']} "
          f"({eda['failure_rate']:.2%}) missing={eda['missing_values']}")
    df = clean(df)
    X_tr, X_te, y_tr, y_te = split(df, seed=seed)

    params = {"n_estimators": n_estimators, "max_depth": max_depth,
              "class_weight": "balanced_subsample", "random_state": seed}
    clf = RandomForestClassifier(n_jobs=-1, **params).fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    metrics = {
        "accuracy": float(accuracy_score(y_te, pred)),
        "precision": float(precision_score(y_te, pred, zero_division=0)),
        "recall": float(recall_score(y_te, pred, zero_division=0)),
        "f1": float(f1_score(y_te, pred, zero_division=0)),
    }
    cm = confusion_matrix(y_te, pred).tolist()

    # Isolation Forest trained only on NORMAL (non-failure) training rows
    iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=seed)
    iso.fit(X_tr[y_tr == 0])
    joblib.dump(iso, ANOMALY_MODEL)

    mlops.setup_mlflow()
    with mlflow.start_run(run_name=f"rf_{n_estimators}_{max_depth}") as run:
        mlflow.log_params({"model_name": "EdgeMind-FailurePredictor", "algorithm": "RandomForestClassifier",
                           "data_source": source, **params})
        mlflow.log_metrics(metrics)
        mlflow.log_dict({"confusion_matrix": cm, "labels": ["no_failure", "failure"]}, "confusion_matrix.json")
        entry = mlops.register(clf, metrics, params, run.info.run_id, source)
        mlflow.set_tag("registry_version", entry["version"])
        mlflow.set_tag("promotion_decision", entry["decision"])
        mlflow.log_artifact(str(mlops.REGISTRY_DIR / entry["file"]))

    entry["confusion_matrix"] = cm
    print(f"[train] Metrics: {metrics}")
    print(f"[train] Confusion matrix [[TN, FP],[FN, TP]]: {cm}")
    print(f"[train] {entry['decision']}")
    return entry


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--max-depth", type=int, default=12)
    ap.add_argument("--offline", action="store_true", help="never try to download the dataset")
    a = ap.parse_args()
    train(a.n_estimators, a.max_depth, allow_download=not a.offline)
