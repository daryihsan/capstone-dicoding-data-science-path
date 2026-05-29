# ============================================================
# RuangRasa — Dashboard 
# CC26-PSU309 | Dary Ihsan Amanullah
# ============================================================

import io
import os
import random
import sys
from collections import Counter
from datetime import date, timedelta

import numpy as np
import pandas as pd

# GroupBy compatibility patch for Pandas 2.2+ / Plotly Express compatibility
try:
    if not hasattr(pd.core.groupby.groupby.GroupBy, "_is_patched"):
        orig_get_group = pd.core.groupby.groupby.GroupBy.get_group
        def _patched_get_group(self, name, *args, **kwargs):
            try:
                return orig_get_group(self, name, *args, **kwargs)
            except KeyError:
                if not isinstance(name, tuple):
                    return orig_get_group(self, (name,), *args, **kwargs)
                raise
        pd.core.groupby.groupby.GroupBy.get_group = _patched_get_group
        pd.core.groupby.groupby.GroupBy._is_patched = True
except Exception:
    pass

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Pastikan utils bisa diimport dari folder manapun
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="RuangRasa — Mental Well-being Platform",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    background-color: #F8FAFC !important;
}

/* ── Sidebar (light, Notion-style) ── */
[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid #E2E8F0 !important;
}
[data-testid="stSidebar"] * {
    color: #374151 !important;
}
[data-testid="stSidebar"] .stButton > button {
    display: inline-flex !important;
    justify-content: flex-start !important;
    align-items: center !important;
    text-align: left !important;
    width: 100% !important;
    border-radius: 8px !important;
    border: none !important;
    background: transparent !important;
    color: #374151 !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    padding: 8px 14px !important;
    transition: background 0.15s ease !important;
}
[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] .stButton > button * {
    text-align: left !important;
    justify-content: flex-start !important;
    align-items: center !important;
}
[data-testid="stSidebar"] .stButton > button p {
    margin: 0 !important;
    text-align: left !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #F1F5F9 !important;
    color: #1E3A5F !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: #EFF6FF !important;
    color: #1D4ED8 !important;
    font-weight: 600 !important;
    border-left: 3px solid #2563EB !important;
    border-radius: 0 8px 8px 0 !important;
}

/* ── Main area ── */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
    background-color: #F8FAFC !important;
}

/* ── Typography ── */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Plus Jakarta Sans', 'Inter', sans-serif !important;
    color: #0F172A !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
}
h1 { font-size: 1.875rem !important; margin-bottom: 0.25rem !important; }
h2 { font-size: 1.5rem !important; margin-top: 1.5rem !important; margin-bottom: 0.75rem !important; }
h3 { font-size: 1.125rem !important; font-weight: 600 !important; margin-top: 1rem !important; margin-bottom: 0.5rem !important; }
p, span, a, li { color: #374151 !important; font-size: 0.9375rem; }
strong { color: #0F172A !important; font-weight: 600 !important; }

/* ── KPI Cards ── */
.kpi-card {
    background: #FFFFFF;
    border-radius: 10px;
    padding: 18px 20px;
    border: 1px solid #E2E8F0;
    border-top: 3px solid #2563EB;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    height: 100%;
}
.kpi-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #64748B !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 1.4rem;
    font-weight: 700 !important;
    color: #0F172A !important;
    line-height: 1.1;
    font-family: 'Plus Jakarta Sans', sans-serif;
}
.kpi-sub { font-size: 0.78rem; color: #94A3B8 !important; margin-top: 4px; }
.kpi-accent { border-top-color: #059669; }
.kpi-warn   { border-top-color: #D97706; }
.kpi-alert  { border-top-color: #DC2626; }

/* ── Insight box ── */
.insight-box {
    background: #F0F9FF;
    border-radius: 8px;
    padding: 16px 20px;
    border-left: 4px solid #2563EB;
    margin: 8px 0;
    color: #0F172A !important;
    font-size: 0.9rem;
    line-height: 1.7;
}
.insight-box strong { color: #1E3A5F !important; font-weight: 600 !important; }

/* ── Result card ── */
.result-card {
    background: #FFFFFF;
    border-radius: 10px;
    padding: 20px 24px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    color: #0F172A !important;
}
.result-card * { color: #0F172A !important; }

/* ── Risk badge ── */
.badge-low    { background: #D1FAE5; color: #065F46; padding: 3px 12px; border-radius: 20px; font-weight: 600; font-size: 0.82rem; }
.badge-medium { background: #FEF3C7; color: #92400E; padding: 3px 12px; border-radius: 20px; font-weight: 600; font-size: 0.82rem; }
.badge-high   { background: #FEE2E2; color: #991B1B; padding: 3px 12px; border-radius: 20px; font-weight: 600; font-size: 0.82rem; }

/* ── Section headers ── */
.section-title {
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: #0F172A !important;
    border-bottom: 1px solid #E2E8F0 !important;
    padding-bottom: 8px !important;
    margin-bottom: 14px !important;
    display: block !important;
    letter-spacing: 0.01em;
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* ── Disclaimer ── */
.disclaimer-box {
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-radius: 8px;
    padding: 10px 14px;
    color: #78350F !important;
    font-size: 0.84rem;
}

/* ── st.metric ── */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border-radius: 8px;
    padding: 14px 16px;
    border: 1px solid #E2E8F0;
    border-top: 2px solid #2563EB;
}
[data-testid="stMetric"] > div {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
}
[data-testid="stMetricLabel"] { 
    width: 100%; 
    color: #64748B !important; 
    font-size: 0.8rem !important; 
    font-weight: 500 !important; 
}
[data-testid="stMetricValue"] { 
    margin-right: 12px;
    font-size: 1.6rem !important; 
    font-weight: 700 !important; 
    color: #0F172A !important; 
}
[data-testid="stMetricDelta"] { 
    color: #64748B !important; 
}

/* ── Journal entry ── */
.journal-entry {
    background: #FFFFFF;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 6px;
    border: 1px solid #E2E8F0;
    border-left: 3px solid #2563EB;
    font-size: 0.875rem;
    color: #374151 !important;
}
.journal-entry * { color: #374151 !important; }

/* ── Tab styling ── */
[data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid #E2E8F0; }
[data-baseweb="tab"] {
    border-radius: 6px 6px 0 0 !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    color: #64748B !important;
}

/* ── Progress bar ── */
.progress-outer { background: #F1F5F9; border-radius: 20px; height: 6px; margin: 6px 0 16px; }
.progress-inner { background: linear-gradient(90deg, #2563EB, #059669); border-radius: 20px; height: 6px; transition: width 0.3s ease; }

/* ── Buttons ── */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
}

/* ── Sidebar labels ── */
[data-testid="stSidebar"] label {
    color: #374151 !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] small {
    color: #6B7280 !important;
}
[data-testid="stSidebar"] h3 {
    color: #0F172A !important;
    font-size: 0.95rem !important;
}

/* ── Form inputs ── */
[data-baseweb="select"] { color: #0F172A !important; }
.stDateInput label { color: #374151 !important; font-weight: 500 !important; }

/* Multiselect pills */
span[data-baseweb="tag"] {
    background-color: #4464AD !important;
    color: white !important;
}
span[data-baseweb="tag"] span {
    color: white !important;
}
span[data-baseweb="tag"] svg {
    fill: white !important;
}

/* Primary buttons in main area */
.block-container button[kind="primary"],
.block-container [data-testid="stButton"] button[kind="primary"],
.block-container .stButton > button[kind="primary"] {
    background-color: #4464AD !important;
    color: #FFFFFF !important;
    border-color: #4464AD !important;
}
.block-container button[kind="primary"] *,
.block-container [data-testid="stButton"] button[kind="primary"] *,
.block-container .stButton > button[kind="primary"] * {
    color: #FFFFFF !important;
}
.block-container button[kind="primary"]:hover,
.block-container [data-testid="stButton"] button[kind="primary"]:hover,
.block-container .stButton > button[kind="primary"]:hover {
    background-color: #33508a !important;
    border-color: #33508a !important;
}

/* Date input border */
div[data-testid="stDateInput"] div[data-baseweb="input"] {
    border: 1px solid #CBD5E1 !important;
    border-radius: 4px;
}

/* ── Universal text ── */
.block-container { color: #374151 !important; }
.block-container label { color: #374151 !important; font-weight: 500 !important; }

/* ── Page header strip ── */
.page-header {
    padding: 0 0 20px 0;
    border-bottom: 1px solid #E2E8F0;
    margin-bottom: 24px;
}
.page-header h2 {
    margin: 0 0 4px 0 !important;
    color: #0F172A !important;
}
.page-header p {
    margin: 0;
    color: #64748B !important;
    font-size: 0.9rem;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #F8FAFC !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    # Logo RuangRasa
    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.png")
    if os.path.exists(logo_path):
        import base64
        with open(logo_path, "rb") as f:
            logo_base64 = base64.b64encode(f.read()).decode()
        st.markdown(
            f"<div style='display: flex; justify-content: center; align-items: center; padding-top: 16px; margin-bottom: 8px;'><img src='data:image/png;base64,{logo_base64}' width='80'></div>",
            unsafe_allow_html=True
        )
        
    st.markdown("""
    <div style='padding: 8px 16px 16px; border-bottom: 1px solid #E2E8F0; margin-bottom: 8px;'>
        <div style='font-family:"Plus Jakarta Sans",sans-serif; font-size:1.1rem; font-weight:700; color:#0F172A; letter-spacing:-0.02em; text-align: center;'>RuangRasa</div>
        <div style='font-size:0.75rem; color:#64748B; margin-top:2px; text-align: center;'>Mental Well-being Analytics</div>
    </div>
    """, unsafe_allow_html=True)


# Define pages using st.navigation
pages = {
    "NAVIGASI": [
        st.Page("views/home.py", title="Beranda", icon="🏠"),
        st.Page("views/analytics.py", title="Analitik", icon="📊"),
        st.Page("views/ai_lab.py", title="AI Lab Demo AI", icon="🧪"),
    ]
}

pg = st.navigation(pages)

with st.sidebar:
    st.markdown("<div style='margin-top:32px;padding:0 16px 16px;font-size:0.72rem;color:#94A3B8;'>© 2026 CC26-PSU309 · RuangRasa<br>Dary Ihsan Amanullah</div>", unsafe_allow_html=True)

# Run the selected page
pg.run()
