"""Reusable UI pieces for the EdgeMind AI dashboard."""
import plotly.graph_objects as go
import streamlit as st

CSS = """
<style>
.block-container {padding-top: 1.5rem;}
.em-card {background:#161B22;border:1px solid #2A313C;border-radius:12px;padding:16px 18px;}
.em-card .lbl {color:#8B949E;font-size:.78rem;text-transform:uppercase;letter-spacing:.08em;}
.em-card .val {font-size:1.9rem;font-weight:700;margin-top:4px;}
.em-badge {display:inline-block;padding:4px 12px;border-radius:999px;font-weight:700;font-size:.85rem;}
.ok {background:#12361F;color:#3FB950;} .warn {background:#3D2E00;color:#F5A524;} .bad {background:#3D1214;color:#F85149;}
.sim {background:#1F2A44;color:#79A6FF;}
.em-title {font-size:2rem;font-weight:800;letter-spacing:-.02em;}
.em-sub {color:#8B949E;margin-top:-6px;}
</style>
"""
COLORS = {"Low": "#3FB950", "Medium": "#F5A524", "High": "#F85149", "Normal": "#3FB950", "Anomaly": "#F85149"}


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def metric_card(label, value, color="#E6EDF3"):
    st.markdown(f'<div class="em-card"><div class="lbl">{label}</div>'
                f'<div class="val" style="color:{color}">{value}</div></div>', unsafe_allow_html=True)


def badge(text, kind="ok"):
    return f'<span class="em-badge {kind}">{text}</span>'


def risk_gauge(score: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score * 100, number={"suffix": "%"},
        title={"text": "Failure Risk Score"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#E6EDF3"},
               "steps": [{"range": [0, 30], "color": "#12361F"}, {"range": [30, 60], "color": "#3D2E00"},
                         {"range": [60, 100], "color": "#3D1214"}]}))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      font_color="#E6EDF3")
    return fig


def style(fig, height=320):
    fig.update_layout(height=height, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=40, b=10))
    return fig
