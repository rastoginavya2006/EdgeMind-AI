"""EdgeMind AI - Streamlit dashboard. Run: streamlit run dashboard/app.py"""
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components import inject_css, metric_card, badge, risk_gauge, style, COLORS
from src import database as db
from src import mlops
from src.config import APPROVED_MODEL, ANOMALY_MODEL, DB_PATH, DATASET_FILE, SYNTHETIC_FILE, MLRUNS_DIR
from src.pipeline import process
from src.predict import predict, load_model
from src.anomaly_detection import detect, load_detector
from src.sensor_simulator import generate_reading, SCENARIOS, save_to_csv

st.set_page_config(page_title="EdgeMind AI", page_icon="⚙️", layout="wide")
inject_css()
db.init_db()

models_ready = APPROVED_MODEL.exists() and ANOMALY_MODEL.exists()
ss = st.session_state
ss.setdefault("running", False)
ss.setdefault("last", None)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("### ⚙️ EdgeMind AI")
    st.caption("Intelligence Without Internet")
    page = st.radio("Navigate", ["Dashboard Overview", "Live Monitoring", "AI Prediction",
                                 "Anomaly Detection", "Analytics", "MLOps", "System Information"])
    st.divider()
    st.markdown(badge("SIMULATED DATA", "sim"), unsafe_allow_html=True)
    st.caption("Readings come from a software simulator, not real hardware.")

st.markdown('<div class="em-title">EdgeMind AI</div>'
            '<div class="em-sub">Offline Autonomous AI Engineer for Local Systems</div>', unsafe_allow_html=True)
st.write("")

if not models_ready:
    st.error("No trained models found. Click below (or run `python -m src.train`).")
    if st.button("Train models now", type="primary"):
        from src.train import train
        with st.spinner("Training Random Forest + Isolation Forest..."):
            e = train()
        st.success(e["decision"])
        st.rerun()
    st.stop()

hist = db.history(1000)
c = db.counts()


def sensor_form(key: str) -> dict:
    cols = st.columns(5)
    base = ss.last["reading"] if ss.last else generate_reading("Normal")
    return {
        "air_temp": cols[0].number_input("Air temp [K]", 290.0, 310.0, float(base["air_temp"]), 0.1, key=key + "a"),
        "process_temp": cols[1].number_input("Process temp [K]", 300.0, 320.0, float(base["process_temp"]), 0.1, key=key + "p"),
        "rpm": cols[2].number_input("Rotational speed [rpm]", 1000.0, 3000.0, float(base["rpm"]), 10.0, key=key + "r"),
        "torque": cols[3].number_input("Torque [Nm]", 0.0, 80.0, float(base["torque"]), 0.5, key=key + "t"),
        "tool_wear": cols[4].number_input("Tool wear [min]", 0.0, 260.0, float(base["tool_wear"]), 1.0, key=key + "w"),
    }


def show_result(res):
    p, a = res["prediction"], res["anomaly"]
    k = {"Low": "ok", "Medium": "warn", "High": "bad"}[p["risk_category"]]
    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(risk_gauge(p["risk_score"]), width="stretch")
    with c2:
        st.markdown(f"**Risk category:** {badge(p['risk_category'], k)}", unsafe_allow_html=True)
        st.markdown(f"**Prediction:** {p['prediction']}")
        st.markdown(f"**Anomaly status:** {badge(a['status'], 'bad' if a['status']=='Anomaly' else 'ok')} "
                    f"(score {a['anomaly_score']:.3f})", unsafe_allow_html=True)
        st.markdown("**Why?**")
        for n in p["explanation"]:
            st.markdown(f"- {n}")
    st.caption("Risk score = share of Random Forest trees voting 'failure'. It is not a calibrated probability.")


# ---------------- Pages ----------------
if page == "Dashboard Overview":
    cols = st.columns(5)
    with cols[0]: metric_card("Total readings", c["readings"])
    with cols[1]: metric_card("Normal", c["normal"], COLORS["Normal"])
    with cols[2]: metric_card("Anomalies", c["anomalies"], COLORS["Anomaly"])
    with cols[3]: metric_card("High-risk", c["high_risk"], COLORS["High"])
    status = "CRITICAL" if (not hist.empty and hist.iloc[0]["risk_category"] == "High") else "OPERATIONAL"
    with cols[4]: metric_card("System status", status, COLORS["High"] if status == "CRITICAL" else COLORS["Low"])
    st.write("")
    l, r = st.columns([2, 1])
    with l:
        st.subheader("Recent readings")
        st.dataframe(hist.head(15), width="stretch", hide_index=True)
    with r:
        st.subheader("Recent alerts")
        logs = db.query("SELECT timestamp, level, message FROM system_logs WHERE level IN ('ALERT','WARNING') "
                        "ORDER BY id DESC LIMIT 12")
        if logs.empty:
            st.info("No alerts yet. Start the simulator in Live Monitoring.")
        for _, row in logs.iterrows():
            icon = "🔴" if row.level == "ALERT" else "🟠"
            st.markdown(f"{icon} `{row.timestamp[11:]}` {row.message}")
    st.write("")
    if st.button("▶ Load demo mode (20 sample readings)"):
        for i in range(20):
            sc = SCENARIOS[0] if i % 4 else SCENARIOS[1 + (i // 4) % 4]
            process(generate_reading(sc), "SIMULATED-DEMO")
        db.log("Demo mode: 20 simulated readings loaded")
        st.rerun()

elif page == "Live Monitoring":
    st.markdown(badge("SIMULATED DATA", "sim"), unsafe_allow_html=True)
    a, b, cc, d = st.columns([2, 1, 1, 1])
    scenario = a.selectbox("Scenario", SCENARIOS)
    interval = b.number_input("Interval (s)", 0.5, 10.0, 1.5, 0.5)
    if cc.button("▶ Start simulation", type="primary", width="stretch"):
        ss.running = True; db.log(f"Simulation started ({scenario})")
    if d.button("■ Stop simulation", width="stretch"):
        ss.running = False; db.log("Simulation stopped")
    st.markdown(f"Simulator: {badge('RUNNING','ok') if ss.running else badge('STOPPED','warn')}",
                unsafe_allow_html=True)

    if ss.running:
        r = generate_reading(scenario)
        save_to_csv(r, scenario)
        ss.last = process(r, "SIMULATED")
    if ss.last:
        r = ss.last["reading"]
        cols = st.columns(5)
        for col, (lbl, k, u) in zip(cols, [("Air temp", "air_temp", "K"), ("Process temp", "process_temp", "K"),
                                           ("Speed", "rpm", "rpm"), ("Torque", "torque", "Nm"), ("Tool wear", "tool_wear", "min")]):
            with col: metric_card(lbl, f"{r[k]:.1f} {u}")
        st.write("")
        show_result(ss.last)
        if not hist.empty:
            t = hist.head(60).iloc[::-1]
            st.plotly_chart(style(px.line(t, x="timestamp", y=["rpm"], title="Rotational speed (live)")),
                            width="stretch")

    st.divider()
    st.subheader("Manual sensor input")
    manual = sensor_form("m")
    x1, x2, x3 = st.columns(3)
    if x1.button("Predict failure", width="stretch"):
        p = predict(manual); st.markdown(f"**{p['prediction']}** — risk {p['risk_score']:.2f} ({p['risk_category']})")
    if x2.button("Detect anomaly", width="stretch"):
        an = detect(manual); st.markdown(f"**{an['status']}** (score {an['anomaly_score']:.3f})")
    if x3.button("Run both & save", width="stretch"):
        ss.last = process(manual, "MANUAL"); st.rerun()

    if ss.running:
        time.sleep(interval)
        st.rerun()

elif page == "AI Prediction":
    st.subheader("Machine failure prediction")
    reading = sensor_form("p")
    if st.button("Predict", type="primary"):
        res = {"prediction": predict(reading), "anomaly": detect(reading)}
        show_result(res)
        fi = pd.Series(res["prediction"]["feature_importance"]).sort_values()
        st.plotly_chart(style(px.bar(fi, orientation="h", title="Model feature importance")), width="stretch")
    reg = mlops.load_registry()
    cur = next((v for v in reg["versions"] if v["version"] == reg["approved_version"]), None)
    if cur:
        st.subheader("Model information")
        st.json({k: cur[k] for k in ("version", "algorithm", "trained_at", "data_source", "metrics", "params")})

elif page == "Anomaly Detection":
    st.subheader("Isolation Forest anomaly detection")
    reading = sensor_form("a")
    if st.button("Detect", type="primary"):
        an = detect(reading)
        st.markdown(f"Result: {badge(an['status'], 'bad' if an['status']=='Anomaly' else 'ok')} "
                    f"(decision score {an['anomaly_score']:.3f}; negative = anomalous)", unsafe_allow_html=True)
    st.subheader("Recent anomaly history")
    an_hist = hist[hist["anomaly_status"] == "Anomaly"]
    if an_hist.empty:
        st.info("No anomalies recorded yet.")
    else:
        st.dataframe(an_hist.head(30), width="stretch", hide_index=True)

elif page == "Analytics":
    if hist.empty:
        st.info("No data yet. Load demo mode from the Overview page.")
    else:
        t = hist.iloc[::-1]
        sensor = st.selectbox("Sensor", ["air_temp", "process_temp", "rpm", "torque", "tool_wear"])
        st.plotly_chart(style(px.line(t, x="timestamp", y=sensor, color_discrete_sequence=["#F5A524"],
                                      title=f"Sensor trend: {sensor}")), width="stretch")
        l, r = st.columns(2)
        l.plotly_chart(style(px.pie(t, names="risk_category", color="risk_category", color_discrete_map=COLORS,
                                    title="Failure risk distribution", hole=.5)), width="stretch")
        r.plotly_chart(style(px.pie(t, names="anomaly_status", color="anomaly_status", color_discrete_map=COLORS,
                                    title="Anomaly distribution", hole=.5)), width="stretch")
        st.plotly_chart(style(px.scatter(t, x="timestamp", y="risk_score", color="risk_category",
                                         color_discrete_map=COLORS, title="Prediction history")),
                        width="stretch")
        st.plotly_chart(style(px.scatter(t, x="rpm", y="torque", color="anomaly_status", color_discrete_map=COLORS,
                                         title="Speed vs torque (anomalies highlighted)")), width="stretch")

elif page == "MLOps":
    reg = mlops.load_registry()
    st.subheader("Local model registry")
    if reg["versions"]:
        rows = [{"version": v["version"], "status": v["status"], "trained_at": v["trained_at"],
                 **{k: round(x, 4) for k, x in v["metrics"].items()}, "decision": v["decision"]}
                for v in reg["versions"]]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.subheader("MLflow experiment runs")
    runs = mlops.list_runs()
    if not runs.empty:
        cols = [c for c in runs.columns if c.startswith(("metrics.", "params.n_", "params.max")) or c in ("run_id", "start_time")]
        st.dataframe(runs[cols], width="stretch", hide_index=True)
    st.caption(f"Tracking URI: {mlops.TRACKING_URI}  •  View full UI: `mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`")
    st.subheader("Retrain candidate model")
    a, b = st.columns(2)
    n = a.slider("n_estimators", 50, 500, 200, 50)
    depth = b.slider("max_depth", 3, 30, 12)
    if st.button("Train candidate & evaluate", type="primary"):
        from src.train import train
        with st.spinner("Training..."):
            e = train(n, depth, allow_download=False)
        load_model(force=True); load_detector(force=True)
        (st.success if e["status"] == "approved" else st.warning)(e["decision"])
        st.write("Confusion matrix [[TN, FP], [FN, TP]]:", e["confusion_matrix"])
    st.caption(f"Promotion rule: recall ≥ {mlops.MIN_RECALL}, F1 ≥ {mlops.MIN_F1}, and F1 ≥ current approved model.")

elif page == "System Information":
    reg = mlops.load_registry()
    info = {
        "Python version": platform.python_version(),
        "Platform": platform.platform(),
        "Model status": f"Approved v{reg['approved_version']}" if APPROVED_MODEL.exists() else "Missing",
        "Anomaly model": "Loaded" if ANOMALY_MODEL.exists() else "Missing",
        "Database": f"SQLite OK — {DB_PATH}",
        "MLflow tracking": f"Local file store — {MLRUNS_DIR}",
        "Dataset": "AI4I 2020 (UCI), local CSV" if DATASET_FILE.exists() else
                   ("SYNTHETIC DEMO DATA (AI4I unavailable)" if SYNTHETIC_FILE.exists() else "Not loaded"),
        "Offline operation": "All models, data, DB and tracking are read from local files. "
                             "No cloud or external AI APIs are called. (Internet is not disabled automatically.)",
    }
    for k, v in info.items():
        st.markdown(f"**{k}:** {v}")
    st.subheader("System logs")
    st.dataframe(db.query("SELECT * FROM system_logs ORDER BY id DESC LIMIT 50"), width="stretch", hide_index=True)
    if st.button("Clear all stored readings"):
        db.clear_all(); st.rerun()
