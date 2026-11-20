# EdgeMind AI
### Offline Autonomous AI Engineer for Local Systems — *Intelligence Without Internet*

> Educational prototype for predictive maintenance. **Not** a certified industrial safety system.
> All live readings are **SIMULATED DATA** — the app is not connected to real hardware.

## 1. Introduction
EdgeMind AI is an offline-first Edge AI + MLOps platform. It monitors machine sensor readings, predicts machine failure risk, detects unusual behaviour, stores everything locally in SQLite and shows alerts on a Streamlit dashboard.

## 2. Problem Statement
Unplanned machine failures cause downtime and cost. Many factories have poor or no internet, so cloud AI is not always practical.

## 3. Proposed Solution
Run the full AI pipeline locally: local models (Joblib), local database (SQLite), local experiment tracking (MLflow), local dashboard (Streamlit on localhost).

## 4. Objectives
- Predict machine failure from 5 sensor readings (Random Forest).
- Detect anomalous readings (Isolation Forest).
- Simulate a sensor stream with normal and abnormal scenarios.
- Track experiments and promote models with a quality rule (MLOps).
- Work without internet after setup.

## 5. Technology Stack
Python 3.11 · Pandas · NumPy · Scikit-learn · Streamlit · Plotly · SQLite · MLflow · Joblib

## 6. System Architecture
```
Sensor Simulator / Manual input
          │  reading (5 sensors)
          ▼
   src/pipeline.py ──► predict.py (Random Forest)  ──► risk score + category
          │         └─► anomaly_detection.py (Isolation Forest) ──► Normal / Anomaly
          ▼
   SQLite (database/edgemind.db) ──► Streamlit dashboard (Plotly charts, alerts)

   train.py ──► MLflow (mlruns/mlflow.db) ──► mlops.py registry ──► approved_model.joblib
```

## 7. Machine Learning Algorithms
- **Random Forest Classifier** (`class_weight=balanced_subsample`) — failure prediction. Output is a **risk score** (share of trees voting "failure"), not a calibrated probability. Categories: Low < 0.30 ≤ Medium < 0.60 ≤ High.
- **Isolation Forest** — trained on non-failure rows only; flags readings that look unlike normal operation.
- Evaluation: accuracy, precision, recall, F1, confusion matrix (accuracy alone is misleading: only ~3.4% of rows are failures).
- Leakage prevention: failure-mode columns **TWF, HDF, PWF, OSF, RNF are dropped** and never used as features.

## 8. Dataset
[AI4I 2020 Predictive Maintenance Dataset (UCI)](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) — 10,000 rows.
Features: Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min]. Target: `Machine failure`.
Downloaded automatically on first training to `data/raw/ai4i2020.csv`. If download fails, a **clearly labelled SYNTHETIC DEMO DATA** file is generated instead and the dashboard shows this. The dataset has no vibration data, so none is used.

## 9. Project Structure
```
EdgeMind-AI/
├── dashboard/  app.py, components.py
├── src/        config.py, data_loader.py, preprocessing.py, train.py, predict.py,
│               anomaly_detection.py, sensor_simulator.py, pipeline.py, database.py, mlops.py
├── data/       raw/, processed/, streaming/
├── models/model_registry/   (versioned .joblib + registry.json)
├── database/   (edgemind.db created automatically)
├── mlruns/     (MLflow tracking DB + artifacts)
├── tests/test_edgemind.py
├── .streamlit/config.toml   (dark theme, localhost only)
├── requirements.txt  README.md  .gitignore
```

## 10. Installation (Windows PowerShell)
Install Python 3.11 from https://www.python.org/downloads/ (tick "Add Python to PATH").
```powershell
cd EdgeMind-AI
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1      # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 11. Execution
```powershell
python -m src.train                 # downloads dataset once, trains, logs to MLflow, promotes model
python -m pytest -q                 # run tests
streamlit run dashboard/app.py      # opens http://localhost:8501
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db   # optional: http://127.0.0.1:5000
```
Offline retrain (no download attempt): `python -m src.train --offline`

## 12. MLOps
- Every training run logs model name, algorithm, hyperparameters, data source, accuracy, precision, recall, F1, confusion matrix and model artifact to local MLflow.
- Each run creates a new version `models/model_registry/model_vN.joblib` with timestamp in `registry.json`.
- **Promotion rule:** candidate is promoted only if recall ≥ 0.50, F1 ≥ 0.50, **and** F1 ≥ current approved model. Otherwise the existing approved model is kept.
- Retrain from the dashboard **MLOps** page with different hyperparameters.
- No automated drift detection is implemented.

## 13. Offline Functionality
After installing dependencies and training once, the app needs no internet: models load from local files, data goes to SQLite, MLflow uses a local database, Streamlit runs on localhost, and no cloud/AI APIs are called. The app does not disable your internet; turn off Wi-Fi yourself to demonstrate.

## 14. Demo Workflow
1. `streamlit run dashboard/app.py` → **Dashboard Overview** → click **Load demo mode**.
2. **Live Monitoring** → choose scenario → **Start simulation** → watch readings, gauge, anomaly status → **Stop**.
3. Try scenarios *High Tool Wear* / *Power Overload* to trigger High risk; *Random Spike* for anomalies.
4. **AI Prediction** / **Anomaly Detection** → enter custom values.
5. **Analytics** → trends and distributions. **MLOps** → registry, runs, retrain. **System Information** → offline status, logs.
6. Show `database/edgemind.db` (e.g. DB Browser for SQLite) and the MLflow UI.

## 15. Screenshots
_Add screenshots here: Overview, Live Monitoring, Analytics, MLOps, MLflow UI._

## 16. Limitations
- Simulated data only; no real sensors/PLC integration.
- Risk score is not a calibrated probability.
- Recall ~0.6 on the real dataset: some failures are missed.
- Single machine, single user, no authentication.

## 17. Future Enhancements
- MQTT/OPC-UA connection to real sensors, probability calibration, data drift detection, per-failure-type models, SHAP explanations, multi-machine fleet view.
