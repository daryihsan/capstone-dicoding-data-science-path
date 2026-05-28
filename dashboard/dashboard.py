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
    orig_get_group = pd.core.groupby.groupby.GroupBy.get_group
    def _patched_get_group(self, name, *args, **kwargs):
        try:
            return orig_get_group(self, name, *args, **kwargs)
        except KeyError:
            if not isinstance(name, tuple):
                return orig_get_group(self, (name,), *args, **kwargs)
            raise
    pd.core.groupby.groupby.GroupBy.get_group = _patched_get_group
except Exception:
    pass

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Pastikan utils bisa diimport dari folder manapun
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.constants import (
    ACCENT_COLOR, DAY_ORDER, DAY_ORDER_ID, EMOTION_COLORS, EMOTION_COLORMAPS,
    EMOTION_EMOJI, EMOTION_LABELS_ID, EMOTION_ORDER, HOTLINE_INFO,
    NB_AKURASI_FULL, NB_AKURASI_SEL, NB_CLASS_NEG, NB_CLASS_POS,
    NB_FITUR_AWAL, NB_FITUR_FINAL, NB_METRICS, NB_PENGURANGAN, NB_RATIO,
    PRIMARY_COLOR, SECONDARY_COLOR, RISK_COLORS, RISK_LABELS, SENTIMENT_COLORS, STRESS_COLS,
)
from utils.data_loader import load_data, load_screening_data
from utils.ai_inference import (
    MODEL_STATUS, load_emotion_model, load_screening_model,
    predict_emotion, predict_risk
)

# Preload models
load_emotion_model()
load_screening_model()

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

/* ── Hotline ── */
.hotline-box {
    background: #FFF5F5;
    border: 1px solid #FECACA;
    border-radius: 8px;
    padding: 14px 18px;
    color: #991B1B !important;
    font-size: 0.875rem;
}
.hotline-box strong { color: #7F1D1D !important; font-weight: 600 !important; }

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
# SESSION STATE INIT
# ============================================================
if "page" not in st.session_state:
    st.session_state.page = "Beranda"
if "journal_history" not in st.session_state:
    st.session_state.journal_history = []  # [{text, emotion, sentiment, timestamp}]
if "last_screening" not in st.session_state:
    st.session_state.last_screening = None  # dict hasil screening
if "screening_answers" not in st.session_state:
    st.session_state.screening_answers = {}
if "ai_result" not in st.session_state:
    st.session_state.ai_result = None
if "user_name" not in st.session_state:
    st.session_state.user_name = "Pengguna"

# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data(show_spinner=False)
def _load_all():
    try:
        df_j = load_data()
    except Exception:
        df_j = pd.DataFrame()
    try:
        df_s = load_screening_data()
        scr_ok = True
    except Exception:
        df_s = pd.DataFrame()
        scr_ok = False
    return df_j, df_s, scr_ok

with st.spinner("Memuat data…"):
    df_all, df_screening_all, SCREENING_AVAILABLE = _load_all()

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

    st.markdown("<div style='padding:8px 16px 4px; font-size:0.68rem; font-weight:600; color:#94A3B8; letter-spacing:0.08em; text-transform:uppercase;'>NAVIGASI</div>", unsafe_allow_html=True)
    for p, label, icon in [("Beranda", "Beranda", "🏠"), ("Analitik", "Analitik", "📊"), ("AI Lab", "AI Lab Demo AI", "🧪")]:
        is_active = st.session_state.page == p
        if st.button(f"{icon}  {label}", use_container_width=True, type="primary" if is_active else "secondary", key=f"nav_{p}"):
            st.session_state.page = p
            st.session_state.ai_result = None
            st.rerun()

    # Filter Analitik (hanya muncul saat halaman Analitik aktif)
    if st.session_state.page == "Analitik":
        st.markdown("<div style='height:1px;background:#E2E8F0;margin:12px 0;'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("<p style='font-size:0.68rem;color:#94A3B8;letter-spacing:0.08em;font-weight:600;text-transform:uppercase;margin-bottom:8px;margin-top:0;'>Filter Analitik</p>", unsafe_allow_html=True)
            if not df_all.empty:
                from utils.constants import EMOTION_LABELS_ID as _EL
                _MIN_DATE = df_all["timestamp"].min().date()
                _MAX_DATE = df_all["timestamp"].max().date()
                try:
                    _date_range = st.date_input(
                        "Rentang Tanggal", [_MIN_DATE, _MAX_DATE],
                        min_value=_MIN_DATE, max_value=_MAX_DATE,
                        key="analytics_date_sb"
                    )
                    if isinstance(_date_range, (list, tuple)) and len(_date_range) == 2:
                        st.session_state["_sb_date_range"] = _date_range
                    else:
                        st.info("Pilih tanggal awal dan akhir secara lengkap.")
                        st.session_state["_sb_date_range"] = (_MIN_DATE, _MAX_DATE)
                except Exception:
                    st.info("Filter tanggal gagal. Menampilkan semua data.")
                    st.session_state["_sb_date_range"] = (_MIN_DATE, _MAX_DATE)
                _ALL_EM = sorted(df_all["label_emosi"].unique().tolist())
                _sel_em = st.multiselect(
                    "Filter Emosi",
                    options=_ALL_EM,
                    default=_ALL_EM,
                    format_func=lambda x: _EL.get(x, x),
                    key="analytics_em_sb"
                )
                st.session_state["_sb_sel_em"] = _sel_em if _sel_em else _ALL_EM
            else:
                st.session_state["_sb_date_range"] = None
                st.session_state["_sb_sel_em"] = []

    st.markdown("<div style='margin-top:32px;padding:0 16px 16px;font-size:0.72rem;color:#94A3B8;'>© 2026 CC26-PSU309 · RuangRasa<br>Dary Ihsan Amanullah</div>", unsafe_allow_html=True)

# ============================================================
# HELPERS — CHART FUNCTIONS
# ============================================================
def _plotly_clean_layout(fig, title=""):
    """Layout Plotly profesional — background putih, grid abu muda, font Inter."""
    fig.update_layout(
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#475569", family="Inter, Plus Jakarta Sans, sans-serif", size=12),
        title_font=dict(size=14, color="#0F172A", family="Plus Jakarta Sans, Inter, sans-serif"),
        margin=dict(l=20, r=20, t=45, b=20),
        xaxis=dict(gridcolor="#F1F5F9", linecolor="#E2E8F0", tickfont=dict(color="#64748B")),
        yaxis=dict(gridcolor="#F1F5F9", linecolor="#E2E8F0", tickfont=dict(color="#64748B")),
        legend=dict(font=dict(color="#374151")),
    )
    if title:
        fig.update_layout(title=dict(text=title, x=0.01, font=dict(color="#0F172A", size=14)))
    return fig

def chart_emotion_bar(df):
    counts = df["label_emosi"].value_counts()
    fig = go.Figure(go.Bar(
        x=[EMOTION_LABELS_ID.get(e, e) for e in counts.index],
        y=counts.values,
        marker_color=PRIMARY_COLOR,
        text=counts.values, textposition="outside",
    ))
    _plotly_clean_layout(fig, "Distribusi Emosi")
    fig.update_layout(showlegend=False, yaxis_title="Jumlah Jurnal")
    return fig

def chart_sentiment_pie(df):
    grp = df["emotion_group"].value_counts()
    colors = [PRIMARY_COLOR if x == grp.idxmax() else "#758EC9" for x in grp.index]
    fig = go.Figure(go.Pie(
        labels=grp.index, values=grp.values,
        marker_colors=colors,
        hole=0.45,
    ))
    _plotly_clean_layout(fig, "Distribusi Sentimen")
    return fig

def chart_daily_trend(df):
    daily = df.groupby("date").size().reset_index(name="count")
    fig = px.line(daily, x="date", y="count", color_discrete_sequence=[PRIMARY_COLOR])
    _plotly_clean_layout(fig, "Tren Jurnal Harian")
    fig.update_traces(line_width=2, fill="tozeroy", fillcolor="rgba(68,100,173,0.12)")
    return fig

def chart_boxplot_text_length(df):
    avg_order = df.groupby("label_emosi")["jumlah_kata"].mean().sort_values(ascending=False).index.tolist()
    fig = px.box(df, x="label_emosi", y="jumlah_kata",
                 color_discrete_sequence=[PRIMARY_COLOR],
                 category_orders={"label_emosi": avg_order},
                 labels={"label_emosi": "Emosi", "jumlah_kata": "Jumlah Kata"})
    _plotly_clean_layout(fig, "Distribusi Panjang Jurnal per Emosi")
    fig.update_layout(showlegend=False)
    return fig

def chart_avg_word_count(df):
    avg = df.groupby("label_emosi")["jumlah_kata"].mean().sort_values(ascending=False)
    fig = go.Figure(go.Bar(
        x=[EMOTION_LABELS_ID.get(e, e) for e in avg.index],
        y=avg.values.round(1),
        marker_color=PRIMARY_COLOR,
        text=avg.values.round(1), textposition="outside",
    ))
    _plotly_clean_layout(fig, "Rata-rata Kata per Emosi")
    fig.update_layout(showlegend=False)
    return fig

def chart_hourly_emotion(df):
    hh = df.groupby(["hour", "label_emosi"]).size().reset_index(name="count")
    fig = px.bar(hh, x="hour", y="count", color="label_emosi",
                 color_discrete_map=EMOTION_COLORS, barmode="stack",
                 labels={"hour": "Jam", "count": "Jumlah", "label_emosi": "Emosi"})
    _plotly_clean_layout(fig, "Pola Penulisan per Jam")
    return fig

def chart_day_of_week(df):
    d = df.groupby("day_of_week").size().reindex(DAY_ORDER, fill_value=0)
    fig = go.Figure(go.Bar(
        x=[DAY_ORDER_ID[dd] for dd in d.index], y=d.values,
        marker_color=PRIMARY_COLOR, text=d.values, textposition="outside",
    ))
    _plotly_clean_layout(fig, "Distribusi per Hari")
    return fig

def chart_heatmap_day_hour(df):
    pivot = df.groupby(["day_of_week", "hour"]).size().unstack(fill_value=0)
    pivot = pivot.reindex(DAY_ORDER, fill_value=0)
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=[DAY_ORDER_ID[d] for d in pivot.index],
        colorscale="Blues",
    ))
    _plotly_clean_layout(fig, "Heatmap Hari x Jam")
    return fig

def make_wordcloud_image(df, emotion):
    """Generate WordCloud as PIL Image, color-themed by emotion."""
    from wordcloud import WordCloud
    
    sub = df[df["label_emosi"] == emotion]
    if len(sub) < 5:
        return None
    text = " ".join(sub["text_clean"].astype(str).tolist())
    if not text.strip():
        return None
    
    cmap = EMOTION_COLORMAPS.get(emotion, "Blues")
    wc = WordCloud(
        width=600, height=300,
        background_color="white",
        colormap=cmap,
        max_words=80,
        random_state=42
    ).generate(text)
    return wc.to_image()

def chart_top_words(df, emotion, top_n=15):
    sub = df[df["label_emosi"] == emotion]
    all_words = " ".join(sub["text_clean"].astype(str)).split()
    stop = {"yang", "dan", "di", "ke", "dari", "ini", "itu", "tidak", "saya",
            "aku", "kamu", "untuk", "dengan", "ada", "atau", "sudah", "juga",
            "sama", "bisa", "mau", "tapi", "kita", "mereka", "saja", "lagi",
            "aja", "ya", "yg", "gue", "gw", "lo", "nya", "ku", "mu"}
    cnt = Counter(w for w in all_words if w not in stop and len(w) > 2)
    top = pd.DataFrame(cnt.most_common(top_n), columns=["Kata", "Frekuensi"])
    fig = px.bar(top.sort_values("Frekuensi"), x="Frekuensi", y="Kata",
                 orientation="h", color_discrete_sequence=[PRIMARY_COLOR])
    _plotly_clean_layout(fig, f"Top {top_n} Kata — {EMOTION_LABELS_ID.get(emotion, emotion)}")
    return fig

def chart_age_distribution(df_scr):
    ag = df_scr["age_group"].value_counts().sort_index()
    fig = px.bar(x=ag.index.astype(str), y=ag.values,
                 color_discrete_sequence=[PRIMARY_COLOR],
                 labels={"x": "Kelompok Usia", "y": "Jumlah"})
    _plotly_clean_layout(fig, "Distribusi Usia Responden")
    return fig

def chart_gender_pie(df_scr):
    g = df_scr["Gender"].value_counts()
    fig = go.Figure(go.Pie(labels=g.index, values=g.values,
                           hole=0.4, marker_colors=["#4464AD", "#758EC9", "#95A5A6"]))
    _plotly_clean_layout(fig, "Distribusi Gender")
    return fig

def chart_stress_radar(df_scr):
    means = df_scr[list(STRESS_COLS.keys())].mean()
    cats  = [STRESS_COLS[c] for c in means.index]
    vals  = means.values.tolist()
    vals += vals[:1]
    cats  += cats[:1]
    fig = go.Figure(go.Scatterpolar(r=vals, theta=cats, fill="toself",
                                    fillcolor="rgba(68,100,173,0.15)",
                                    line=dict(color=PRIMARY_COLOR)))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(range=[0, 10], tickfont=dict(color="#64748B")),
            angularaxis=dict(tickfont=dict(color="#374151")),
            bgcolor="#FAFAFA"
        ),
        paper_bgcolor="#FFFFFF",
        font=dict(color="#374151"),
        margin=dict(l=20, r=20, t=45, b=20),
    )
    _plotly_clean_layout(fig, "Profil Stres Rata-rata")
    return fig

def chart_lifestyle_boxplot(df_scr):
    cols = ["Sleep_Hours_Night", "Screen_Time_Hours_Day", "Social_Media_Hours_Day", "Work_Hours_Per_Week"]
    labels = {"Sleep_Hours_Night": "Tidur (j)", "Screen_Time_Hours_Day": "Screen Time (j)",
              "Social_Media_Hours_Day": "Sosmed (j)", "Work_Hours_Per_Week": "Kerja (j/mg)"}
    melted = df_scr[cols].rename(columns=labels).melt(var_name="Faktor", value_name="Nilai")
    fig = px.box(melted, x="Faktor", y="Nilai", color="Faktor",
                 color_discrete_sequence=["#1a5276", "#2980b9", "#7fb3d3", "#a8c8f0"])
    _plotly_clean_layout(fig, "Distribusi Gaya Hidup")
    fig.update_layout(showlegend=False)
    return fig

def chart_screening_trend(df_scr):
    t = df_scr.groupby(df_scr["timestamp"].dt.date).size().reset_index(name="count")
    t.columns = ["date", "count"]
    fig = px.line(t, x="date", y="count", color_discrete_sequence=[PRIMARY_COLOR])
    _plotly_clean_layout(fig, "Tren Screening Harian")
    fig.update_traces(fill="tozeroy", fillcolor="rgba(68,100,173,0.12)")
    return fig

def chart_sleep_category(df_scr):
    s = df_scr["sleep_category"].value_counts()
    fig = px.bar(x=s.index.astype(str), y=s.values,
                 color_discrete_sequence=["#4464AD", "#758EC9", "#95A5A6"],
                 labels={"x": "Kategori Tidur", "y": "Jumlah"})
    _plotly_clean_layout(fig, "Kualitas Tidur Responden")
    return fig

def chart_trauma_previously(df_scr):
    t = pd.DataFrame({
        "Kategori": ["Riwayat Trauma", "Pernah Didiagnosis", "Tidak Ada Keduanya"],
        "Jumlah": [
            df_scr["Trauma_History"].sum(),
            df_scr["Previously_Diagnosed"].sum(),
            ((df_scr["Trauma_History"] == 0) & (df_scr["Previously_Diagnosed"] == 0)).sum(),
        ]
    })
    fig = px.bar(t, x="Kategori", y="Jumlah", color="Kategori",
                 color_discrete_sequence=["#E74C3C", "#F4D03F", "#27AE60"])
    _plotly_clean_layout(fig, "Riwayat Kesehatan Responden")
    fig.update_layout(showlegend=False)
    return fig

def chart_correlation_heatmap(df_scr):
    num_cols = ["Work_Stress_Level","Financial_Stress","Mood_Swings","Loneliness",
                "Sleep_Hours_Night","Screen_Time_Hours_Day","Social_Media_Hours_Day",
                "Work_Hours_Per_Week","risk_score"]
    corr = df_scr[num_cols].corr()
    fig = px.imshow(corr.round(2), text_auto=True, color_continuous_scale="RdBu_r",
                    aspect="auto", zmin=-1, zmax=1)
    _plotly_clean_layout(fig, "Matriks Korelasi Variabel Wellbeing")
    return fig

def chart_class_imbalance():
    labels = ["Data Minoritas (Tidak Ada Masalah)", "Data Mayoritas (Ada Masalah)"]
    values = [NB_CLASS_NEG, NB_CLASS_POS]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.45,
        marker_colors=["#a8c8f0", "#2e6db4"],
    ))
    _plotly_clean_layout(fig, f"Ketidakseimbangan Kelas (Rasio {NB_RATIO}:1)")
    return fig

def chart_model_performance():
    metrics = list(NB_METRICS.keys())
    values  = list(NB_METRICS.values())
    colors = ['#003d5c', '#7a4f99', '#ef527a', '#ffa600']
    fig = go.Figure(go.Bar(
        x=metrics, y=values,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside"
    ))
    _plotly_clean_layout(fig, "Performa Model Setelah SMOTE (%)")
    fig.update_layout(yaxis=dict(range=[0, 105]), xaxis=dict(tickangle=-30))
    return fig

def chart_stress_histogram(df_scr, col):
    fig = px.histogram(df_scr, x=col, nbins=10,
                       color_discrete_sequence=[PRIMARY_COLOR],
                       labels={col: STRESS_COLS.get(col, col)})
    _plotly_clean_layout(fig, f"Distribusi {STRESS_COLS.get(col, col)}")
    return fig

def chart_faktor_risiko_utama():
    features_data = [
        ("Smoking", "Kebiasaan Merokok", "#464c89", "Gaya Hidup", 0.0694),
        ("Family_History_Mental_Illness", "Riwayat Keluarga dengan Gangguan Mental", "#ff6b59", "Riwayat Kesehatan", 0.0654),
        ("Marital_Status", "Status Pernikahan", "#ffa600", "Demografis", 0.0595),
        ("Remote_Work", "Mode Kerja (Remote/Hybrid)", "#ffa600", "Demografis", 0.0512),
        ("Trauma_History", "Riwayat Trauma", "#ff6b59", "Riwayat Kesehatan", 0.0406),
        ("Previously_Diagnosed", "Pernah Terdiagnosis Sebelumnya", "#ff6b59", "Riwayat Kesehatan", 0.0321),
        ("Discuss_Mental_Health", "Keterbukaan Bicara soal Mental", "#dd4d88", "Sosial & Dukungan", 0.0268),
        ("On_Therapy_Now", "Sedang Menjalani Terapi", "#ff6b59", "Riwayat Kesehatan", 0.0238),
        ("Work_Stress_Level", "Tingkat Stres Kerja", "#954e9b", "Pekerjaan & Ekonomi", 0.0216),
        ("Loneliness", "Tingkat Kesepian", "#dd4d88", "Sosial & Dukungan", 0.0207),
        ("Financial_Stress", "Tekanan Keuangan", "#954e9b", "Pekerjaan & Ekonomi", 0.0195),
        ("Ever_Sought_Treatment", "Pernah Mencari Pertolongan", "#ff6b59", "Riwayat Kesehatan", 0.0192),
        ("Mood_Swings", "Perubahan Suasana Hati Drastis", "#003d5c", "Gejala Psikologis", 0.0185),
        ("Social_Media_Hours_Day", "Penggunaan Media Sosial", "#464c89", "Gaya Hidup", 0.0183),
        ("Screen_Time_Hours_Day", "Waktu Layar per Hari", "#464c89", "Gaya Hidup", 0.0171),
        ("Sleep_Hours_Night", "Kualitas & Durasi Tidur", "#464c89", "Gaya Hidup", 0.0170),
    ]
    
    def truncate(text, max_words=2):
        words = text.split()
        if len(words) > max_words:
            return " ".join(words[:max_words]) + "..."
        return text

    df_feat = pd.DataFrame({
        "Fitur": [truncate(f[1]) for f in features_data],
        "Fitur_Full": [f[1] for f in features_data],
        "Skor": [f[4] for f in features_data],
        "Warna": [f[2] for f in features_data],
        "Kategori": [f[3] for f in features_data]
    }).sort_values("Skor", ascending=True)
    
    fig = go.Figure()
    categories = ["Gejala Psikologis", "Gaya Hidup", "Pekerjaan & Ekonomi", "Sosial & Dukungan", "Riwayat Kesehatan", "Demografis"]
    cat_colors = {
        "Gejala Psikologis": "#003d5c",
        "Gaya Hidup": "#464c89",
        "Pekerjaan & Ekonomi": "#954e9b",
        "Sosial & Dukungan": "#dd4d88",
        "Riwayat Kesehatan": "#ff6b59",
        "Demografis": "#ffa600"
    }
    
    for cat in categories:
        sub = df_feat[df_feat["Kategori"] == cat]
        if not sub.empty:
            fig.add_trace(go.Bar(
                y=sub["Fitur"],
                x=sub["Skor"],
                orientation="h",
                name=cat,
                marker_color=cat_colors[cat],
                text=sub["Skor"].apply(lambda x: f"{x:.4f}"),
                textposition="outside",
                hovertext=sub["Fitur_Full"],
                customdata=sub["Kategori"],
                hovertemplate="<b>%{hovertext}</b><br>Kategori: %{customdata}<br>Skor: %{x:.4f}<extra></extra>"
            ))
            
    _plotly_clean_layout(fig, "15 Faktor Pemicu Utama Masalah Kesehatan Mental")
    fig.update_layout(
        barmode="stack",
        height=620,
        legend=dict(
            title="", 
            orientation="h", 
            y=-0.12, 
            x=0, 
            xanchor="left", 
            yanchor="top",
            traceorder="normal"
        ),
        xaxis=dict(range=[0, 0.08]),
        yaxis=dict(categoryorder='total ascending'),
        margin=dict(l=20, r=40, t=45, b=80)
    )
    return fig

def chart_cross_emotion_stress(df_j, df_scr):
    if df_j.empty or df_scr.empty:
        return None
    merged = df_j.merge(df_scr, on="user_id", how="inner")
    if merged.empty:
        return None
    grp = merged.groupby("label_emosi")[["Work_Stress_Level", "Financial_Stress",
                                          "Mood_Swings", "Loneliness"]].mean()
    grp["avg_stress"] = grp.mean(axis=1)
    grp = grp.reset_index()
    fig = px.bar(grp, x="label_emosi", y="avg_stress",
                 color_discrete_sequence=[PRIMARY_COLOR],
                 labels={"label_emosi": "Emosi Dominan", "avg_stress": "Rata-rata Stres (0-10)"})
    _plotly_clean_layout(fig, "Korelasi: Emosi Jurnal x Skor Stres Screening")
    fig.update_layout(showlegend=False)
    return fig

def chart_cross_sleep_emotion(df_j, df_scr):
    if df_j.empty or df_scr.empty:
        return None
    merged = df_j.merge(df_scr, on="user_id", how="inner")
    if merged.empty:
        return None
    fig = px.box(merged, x="label_emosi", y="Sleep_Hours_Night",
                 color_discrete_sequence=[PRIMARY_COLOR],
                 labels={"label_emosi": "Emosi", "Sleep_Hours_Night": "Jam Tidur per Malam"})
    _plotly_clean_layout(fig, "Kualitas Tidur vs Emosi Jurnal")
    fig.update_layout(showlegend=False)
    return fig

def chart_before_after_q3():
    fig = go.Figure()
    categories = ["Accuracy", "Recall", "Precision", "F1-Score"]
    before = [91.8, 93.2, 90.7, 91.8]
    after  = [91.4, 92.8, 90.3, 91.4]
    fig.add_trace(go.Bar(name="50 Fitur (Sebelum)", x=categories, y=before,
                         marker_color="#95A5A6", text=[f"{v:.1f}%" for v in before], textposition="outside"))
    fig.add_trace(go.Bar(name="15 Fitur (Sesudah)", x=categories, y=after,
                         marker_color="#4464AD", text=[f"{v:.1f}%" for v in after], textposition="outside"))
    _plotly_clean_layout(fig, "Perbandingan Performa: 50 vs 15 Fitur")
    fig.update_layout(barmode="group", yaxis=dict(range=[85, 100]), bargap=0.15, bargroupgap=0.05)
    return fig

def chart_jurnal_word_count_hist(df, emotion):
    sub = df[df["label_emosi"] == emotion]
    fig = px.histogram(
        sub, x="jumlah_kata", nbins=20,
        color_discrete_sequence=[PRIMARY_COLOR],
        labels={"jumlah_kata": "Jumlah Kata", "count": "Frekuensi"}
    )
    _plotly_clean_layout(fig, f"Distribusi Panjang Kata — {EMOTION_LABELS_ID.get(emotion, emotion)}")
    fig.update_layout(
        xaxis=dict(range=[0, 60], title="Jumlah Kata"),
        yaxis=dict(title="Frekuensi"),
        showlegend=False,
        bargap=0.15
    )
    return fig

# ============================================================
# ██████████████ HALAMAN 1: BERANDA ████████████████████████
# ============================================================
if st.session_state.page == "Beranda":
    st.markdown(f"""
    <div style='margin-bottom:20px;'>
        <h1 style='color:#0F172A; font-size:1.9rem; font-weight:800; margin-bottom:4px;'>
            RuangRasa — Analytics Platform
        </h1>
        <p style='color:#64748b; font-size:0.95rem; margin:0;'>
            Dashboard riset & analitik kesehatan mental berbasis AI. Eksplorasi insight dari <strong>9.700+ data jurnal emosi</strong>
            dan <strong>10.000 data screening</strong>, lengkap dengan demo model AI secara interaktif.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Pastikan ada data ──
    has_journal  = not df_all.empty
    has_screen   = SCREENING_AVAILABLE and not df_screening_all.empty
    has_session  = len(st.session_state.journal_history) > 0

    # ── KPI Cards ──
    col1, col2, col3, col4, col5 = st.columns(5)

    total_jurnal = len(df_all) if has_journal else 0
    with col1:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Data Jurnal Emosi</div>
            <div class="kpi-value">{total_jurnal:,}</div>
            <div class="kpi-sub">Total data penelitian</div>
        </div>""", unsafe_allow_html=True)

    dominant_em = df_all["label_emosi"].value_counts().idxmax() if has_journal else "—"
    dom_id      = EMOTION_LABELS_ID.get(dominant_em, dominant_em)
    with col2:
        st.markdown(f"""<div class="kpi-card kpi-accent">
            <div class="kpi-label">Mood Dominan</div>
            <div class="kpi-value">{dom_id}</div>
            <div class="kpi-sub">Terbanyak di dataset</div>
        </div>""", unsafe_allow_html=True)

    if has_session:
        last_j = st.session_state.journal_history[-1]
        session_em = last_j.get("emotion", "—")
        session_id = EMOTION_LABELS_ID.get(session_em, session_em)
    else:
        session_id = "Belum ada"
    scr_total = len(df_screening_all) if has_screen else 0
    with col3:
        st.markdown(f"""<div class="kpi-card kpi-warn">
            <div class="kpi-label">Data Screening</div>
            <div class="kpi-value">{scr_total:,}</div>
            <div class="kpi-sub">Total data penelitian</div>
        </div>""", unsafe_allow_html=True)

    if st.session_state.last_screening:
        risk_score = st.session_state.last_screening.get("risk_score", 0)
        risk_level = st.session_state.last_screening.get("risk_level", "Low")
        risk_color = RISK_COLORS[risk_level]
        risk_id    = RISK_LABELS[risk_level]
    elif has_screen:
        risk_score = df_screening_all["risk_score"].mean()
        risk_level = "Medium" if risk_score >= 4 else "Low"
        risk_color = RISK_COLORS[risk_level]
        risk_id    = "Rata-rata dataset"
    else:
        risk_score, risk_id, risk_color = 0, "—", "#95A5A6"
    with col4:
        st.markdown(f"""<div class="kpi-card kpi-alert">
            <div class="kpi-label">Risk Score</div>
            <div class="kpi-value" style="color:{risk_color};">{risk_score:.1f}<span style="font-size:1rem;color:#64748b;">/10</span></div>
            <div class="kpi-sub">{risk_id}</div>
        </div>""", unsafe_allow_html=True)

    avg_sleep = df_screening_all["Sleep_Hours_Night"].mean() if has_screen else 0
    sleep_cat = "Cukup" if avg_sleep >= 6 else "Kurang"
    with col5:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Rata-rata Tidur</div>
            <div class="kpi-value">{avg_sleep:.1f}<span style="font-size:1rem;color:#64748b;">j</span></div>
            <div class="kpi-sub">{sleep_cat}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts utama ──
    col_a, col_b = st.columns(2)
    with col_a:
        if has_journal:
            st.plotly_chart(chart_sentiment_pie(df_all), use_container_width=True)
    with col_b:
        if has_screen:
            st.plotly_chart(chart_sleep_category(df_screening_all), use_container_width=True)

    # ── Insight AI ──
    if has_journal:
        neg_pct  = (df_all["emotion_group"] == "Negative").sum() / len(df_all) * 100
        pos_pct  = (df_all["emotion_group"] == "Positive").sum() / len(df_all) * 100
        peak_h   = df_all.groupby("hour").size().idxmax()
        avg_w    = df_all["jumlah_kata"].mean()
        st.markdown(f"""<div class="insight-box">
            <strong>Insight Otomatis</strong><br><br>
            Dari <strong>{len(df_all):,} jurnal</strong> yang tersimpan, <strong>{neg_pct:.1f}%</strong> mengandung emosi negatif
            dan <strong>{pos_pct:.1f}%</strong> positif. Emosi dominan adalah <strong>{dom_id}</strong>.
            Puncak aktivitas menulis terjadi pada pukul <strong>{peak_h:02d}:00</strong>,
            dengan rata-rata <strong>{avg_w:.1f} kata</strong> per entri jurnal.
        </div>""", unsafe_allow_html=True)

    # ── Quick Actions ──
    st.markdown("<br><p class='section-title'>Eksplorasi Dashboard</p>", unsafe_allow_html=True)
    q1, q2 = st.columns(2)
    with q1:
        if st.button("🧪 Coba Demo AI (Emosi & Screening)", use_container_width=True, type="primary"):
            st.session_state.page = "AI Lab"
            st.rerun()
    with q2:
        if st.button("📊 Lihat Analitik Riset", use_container_width=True):
            st.session_state.page = "Analitik"
            st.rerun()

    # ── Disclaimer ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div class="disclaimer-box">
        <strong>Catatan Riset:</strong> Data yang ditampilkan di halaman ini bersumber dari dataset penelitian
        (<strong>9.700+ jurnal emosi</strong> dan <strong>10.000 data screening</strong>).
        Demo model AI tersedia di halaman <strong>AI Lab</strong>.
    </div>""", unsafe_allow_html=True)

# ============================================================
# ████████████ HALAMAN 2: JURNAL EMOSI + AI ████████████████
# ============================================================
elif st.session_state.page == "AI Lab":
    st.markdown("""
    <h2 style='color:#0F172A; font-weight:800; margin-bottom:4px;'>AI Lab Demo Interaktif</h2>
    <p style='color:#64748b; margin-bottom:4px;'>Uji coba langsung kedua model AI yang dikembangkan dalam riset ini secara real-time.</p>
    <p style='color:#94a3b8; font-size:0.82rem; margin-bottom:20px;'>Pilih tab untuk mulai demo masing-masing model.</p>
    """, unsafe_allow_html=True)

    tab_emosi, tab_screening = st.tabs(["Demo: Analisis Emosi (BiLSTM)", "Demo: Screening Risiko (DNN)"])

    # ============================================================
    # TAB 1 DEMO ANALISIS EMOSI (BiLSTM)
    # ============================================================
    with tab_emosi:
        st.markdown("""
        <p style='color:#64748b; margin-bottom:4px;'>Tulis teks dalam Bahasa Indonesia model <strong>BiLSTM</strong> akan mengklasifikasikan emosi secara real-time.</p>
        <p style='color:#94a3b8; font-size:0.82rem; margin-bottom:20px;'>Model dilatih dari <strong>9.700+ entri jurnal emosi</strong> berbahasa Indonesia dengan 6 kelas emosi.</p>
        """, unsafe_allow_html=True)

        if not MODEL_STATUS["emotion"]:
            st.info("Model AI belum tersedia di folder `models/`. Menggunakan **Mock Mode** (keyword-based). Hasil bersifat simulasi.")
    
        col_input, col_history = st.columns([1.4, 1])
    
        with col_input:
            st.markdown('<p class="section-title">Tulis Teks untuk Dianalisis</p>', unsafe_allow_html=True)
            journal_text = st.text_area(
                "Teks jurnal",
                height=180,
                placeholder="Contoh: Hari ini meeting-ku gagal total. Aku merasa malu dan frustrasi banget sama diri sendiri...",
                label_visibility="collapsed",
                key="ailab_journal_text",
            )
            char_count = len(journal_text)
            st.caption(f"{char_count} karakter | min. 20 karakter untuk analisis")
    
            col_btn1, col_btn2 = st.columns(2)
            col_btn1, col_btn2 = st.columns(2)
            analyze_clicked = col_btn1.button("Analisis AI", use_container_width=True, type="primary",
                                                disabled=(char_count < 20), key="ailab_analyze_btn")
            clear_clicked   = col_btn2.button("Bersihkan", use_container_width=True, key="ailab_clear_btn")
    
            if clear_clicked:
                st.session_state.ai_result = None
                st.rerun()
    
            if analyze_clicked and journal_text.strip():
                with st.spinner("Menganalisis emosi…"):
                    result = predict_emotion(journal_text)
                st.session_state.ai_result = result
                # Simpan ke riwayat sesi
                st.session_state.journal_history.append({
                    "text":      journal_text[:120] + ("…" if len(journal_text) > 120 else ""),
                    "emotion":   result.get("emotion", "Neutral"),
                    "sentiment": result.get("sentiment", "Neutral"),
                    "timestamp": pd.Timestamp.now().strftime("%H:%M"),
                })
    
            # ── Tampilkan Hasil AI ──
            if st.session_state.ai_result:
                r = st.session_state.ai_result
                if "error" in r:
                    st.error(r["error"])
                else:
                    em     = r["emotion"]
                    em_id  = r["emotion_id"]
                    sent   = r["sentiment"]
                    conf   = r["confidence"]
                    color  = EMOTION_COLORS.get(em, PRIMARY_COLOR)
                    sent_c = SENTIMENT_COLORS.get(sent, "#aaa")
    
                    with st.container(border=True):
                        st.markdown(f"""
                            <div style="display:flex; align-items:center; gap:16px; margin-bottom:16px;">
                                <div style="width: 48px; height: 48px; border-radius: 50%; background-color: {color}; flex-shrink: 0;"></div>
                                <div>
                                    <div style="font-size:0.75rem; color:#64748b; text-transform:uppercase; letter-spacing:.06em;">Emosi Terdeteksi</div>
                                    <div style="font-size:1.7rem; font-weight:800; color:{color};">{em_id}</div>
                                    <div style="font-size:0.82rem; color:{sent_c};">{sent} · {conf:.1f}% confidence</div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
    
                        # Probability bar
                        probs = r.get("probabilities", {})
                        if probs:
                            prob_df = pd.DataFrame([
                                {"Emosi": EMOTION_LABELS_ID.get(e, e), "Prob (%)": v, "color": EMOTION_COLORS.get(e, "#aaa")}
                                for e, v in probs.items()
                            ]).sort_values("Prob (%)", ascending=True)
                            fig_prob = go.Figure(go.Bar(
                                x=prob_df["Prob (%)"], y=prob_df["Emosi"],
                                orientation="h",
                                marker_color=prob_df["color"].tolist(),
                                text=[f"{v:.1f}%" for v in prob_df["Prob (%)"]],
                                textposition="outside",
                            ))
                            _plotly_clean_layout(fig_prob, "Distribusi Probabilitas Emosi")
                            fig_prob.update_layout(height=220, margin=dict(l=0, r=40, t=35, b=0),
                                                   xaxis=dict(range=[0, 110]))
                            st.plotly_chart(fig_prob, use_container_width=True)
    
                        st.markdown(f"""
                            <div style="background:rgba(0,0,0,0.02); border-radius:8px; padding:14px 16px; margin-bottom:12px; border: 1px solid #e2e8f0;">
                                <div style="font-size:0.75rem; color:#64748b; margin-bottom:6px; text-transform:uppercase; font-weight:600;">Validasi</div>
                                <div style="color:#374151; font-style:italic; line-height:1.7;">"{r.get('validation', '')}"</div>
                            </div>
                            <div style="margin-bottom:8px;">
                                <div style="font-size:0.75rem; color:#64748b; margin-bottom:8px; text-transform:uppercase; font-weight:600;">Rekomendasi</div>
                            </div>
                        """, unsafe_allow_html=True)
    
                        for rec in r.get("recommendations", []):
                            st.markdown(f"- {rec}")
    
                        st.markdown(f"""
                            <div style="margin-top:12px; font-size:0.75rem; color:#475569; border-top: 1px solid #e2e8f0; padding-top: 8px;">
                                Model: {r.get('model_used','—')} · Waktu inferensi: {r.get('inference_ms','—')} ms
                                {" · <em>(Simulasi)</em>" if r.get('is_mock') else ""}
                            </div>
                        """, unsafe_allow_html=True)
    
                    # Tombol Simpan
                    if st.button("Simpan Jurnal", use_container_width=True):
                        st.success("Jurnal berhasil disimpan ke riwayat sesi!")
                        st.balloons()
    
        with col_history:
            st.markdown('<p class="section-title">Riwayat Sesi Ini</p>', unsafe_allow_html=True)
            if not st.session_state.journal_history:
                st.markdown("<p style='color:#475569; font-size:0.88rem;'>Belum ada analisis di sesi ini.</p>", unsafe_allow_html=True)
            else:
                for entry in reversed(st.session_state.journal_history):
                    em_c = EMOTION_COLORS.get(entry["emotion"], PRIMARY_COLOR)
                    em_i = EMOTION_LABELS_ID.get(entry["emotion"], entry["emotion"])
                    st.markdown(f"""<div class="journal-entry">
                        <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                            <span style="color:{em_c}; font-weight:700; font-size:0.82rem;">{em_i}</span>
                            <span style="color:#475569; font-size:0.78rem;">{entry['timestamp']}</span>
                        </div>
                        <div style="color:#475569; font-size:0.85rem;">{entry['text']}</div>
                    </div>""", unsafe_allow_html=True)
    
        st.markdown("""<div class="disclaimer-box" style="margin-top:12px;">
            Analisis AI ini bersifat indikatif dan digunakan untuk tujuan demonstrasi riset, bukan sebagai diagnosis klinis.
        </div>""", unsafe_allow_html=True)

# ============================================================
# TAB 2 DEMO SCREENING RISIKO (DNN)
# ============================================================
    with tab_screening:
        st.markdown("""
        <p style='color:#64748b; margin-bottom:4px;'>Jawab 15 pertanyaan singkat model <strong>Deep Neural Network</strong> kami akan memprediksi level risiko dan memberikan rekomendasi.</p>
        <p style='color:#94a3b8; font-size:0.82rem; margin-bottom:20px;'>Model dilatih dari <strong>1.000+ responden</strong>. Kuesioner dioptimasi dari 50 pertanyaan menjadi <strong>15 fitur esensial</strong> (akurasi turun hanya 0.4%).</p>
        """, unsafe_allow_html=True)

        if not MODEL_STATUS["screening"]:
            st.info("Model screening dalam **Mock Mode**. Hasil menggunakan rule-based scoring.")
    
        # ── Form Screening ──
        # ── Form Screening ──
        SCR_QUESTIONS  = [
            {"key": "Sleep_Hours_Night",         "label": "Apakah Anda mendapatkan tidur malam yang cukup lama dalam sehari? (1 = Sangat Kurang, 10 = Sangat Cukup)", "min": 1, "max": 10, "default": 5},
            {"key": "Screen_Time_Hours_Day",     "label": "Seberapa sering kamu menggunakan gadget (smartphone, tab, PC) dalam sehari? (1 = Sangat Jarang, 10 = Sangat Sering)", "min": 1, "max": 10, "default": 5},
            {"key": "Social_Media_Hours_Day",    "label": "Seberapa sering kamu bermain media sosial (Instagram, X, Tiktok) dalam sehari? (1 = Sangat Jarang, 10 = Sangat Sering)", "min": 1, "max": 10, "default": 5},
            {"key": "Trauma_History",            "label": "Seberapa sering pikiran atau ingatan tentang pengalaman buruk masa lalu, bahkan trauma muncul kembali dan mengganggu aktivitasmu sehari-hari? (1 = Tidak Pernah, 10 = Sangat Sering)", "min": 1, "max": 10, "default": 5},
            {"key": "Previously_Diagnosed",      "label": "Seberapa besar dampak kondisi kesehatan mental yang pernah kamu alami (atau sedang kamu alami) terhadap kualitas hidupmu saat ini? (1 = Tidak Berdampak, 10 = Sangat Berdampak)", "min": 1, "max": 10, "default": 5},
            {"key": "Work_Hours_Per_Week",       "label": "Seberapa banyak kamu menghabiskan waktu atau energi untuk bekerja dalam satu minggu? (1 = Sangat Sedikit, 10 = Sangat Banyak)", "min": 1, "max": 10, "default": 5},
            {"key": "Work_Stress_Level",         "label": "Seberapa besar tekanan dan tuntutan pekerjaan membuat kamu merasa kewalahan, burnout, atau tidak mampu mengatasinya? (1 = Sangat Rendah, 10 = Sangat Tinggi)", "min": 1, "max": 10, "default": 5},
            {"key": "Financial_Stress",          "label": "Seberapa sering masalah keuangan membuat kamu kehilangan tidur, sulit fokus, atau merasa putus asa tentang masa depan? (1 = Tidak Pernah, 10 = Sangat Sering)", "min": 1, "max": 10, "default": 5},
            {"key": "Mood_Swings",               "label": "Seberapa sering suasana hatimu berubah secara tiba-tiba dan drastis tanpa alasan yang jelas hingga mengganggu hubungan atau aktivitasmu? (1 = Tidak Pernah, 10 = Sangat Sering)", "min": 1, "max": 10, "default": 5},
            {"key": "Loneliness",                "label": "Seberapa sering kamu merasa tidak ada yang benar-benar memahami kamu, atau merasa sendirian meskipun berada di tengah orang banyak? (1 = Tidak Pernah, 10 = Sangat Sering)", "min": 1, "max": 10, "default": 5},
            {"key": "Feeling_Sad_Down",          "label": "Seberapa sering kamu merasa sedih, murung, atau putus asa dalam kehidupan sehari-hari? (1 = Tidak Pernah, 10 = Sangat Sering)", "min": 1, "max": 10, "default": 5},
            {"key": "Anxious_Nervous",           "label": "Seberapa sering kamu merasa cemas, tegang, atau gugup dalam menghadapi aktivitas? (1 = Tidak Pernah, 10 = Sangat Sering)", "min": 1, "max": 10, "default": 5},
            {"key": "Social_Support",            "label": "Seberapa besar dukungan sosial (dari keluarga, teman, atau lingkungan sekitar) yang kamu rasakan? (1 = Sangat Rendah, 10 = Sangat Tinggi)", "min": 1, "max": 10, "default": 5},
            {"key": "Loss_Of_Interest",          "label": "Seberapa sering kamu kehilangan minat atau kesenangan dalam melakukan aktivitas sehari-hari yang biasa kamu sukai? (1 = Tidak Pernah, 10 = Sangat Sering)", "min": 1, "max": 10, "default": 5},
            {"key": "Sleep_Trouble",             "label": "Seberapa sering kamu mengalami kesulitan tidur atau gangguan pola tidur (seperti sulit tidur atau sering terbangun)? (1 = Tidak Pernah, 10 = Sangat Sering)", "min": 1, "max": 10, "default": 5},
        ]
    
        # ── Informasi Demografis ──
        st.markdown('<p class="section-title">Informasi Demografis</p>', unsafe_allow_html=True)
        col_age, col_gen = st.columns(2)
        with col_age:
            age_val = st.number_input("**Usia kamu saat ini?**", min_value=10, max_value=100, 
                                      value=st.session_state.screening_answers.get("Age", 22), 
                                      key="scr_Age")
            st.session_state.screening_answers["Age"] = age_val
        with col_gen:
            gender_opts = ["Male", "Female", "Non-binary", "Prefer not to say"]
            gender_idx = gender_opts.index(st.session_state.screening_answers.get("Gender", "Male")) if st.session_state.screening_answers.get("Gender", "Male") in gender_opts else 0
            gender_val = st.radio("**Jenis Kelamin / Gender**", gender_opts, index=gender_idx, horizontal=True, key="ailab_scr_Gender")
            st.session_state.screening_answers["Gender"] = gender_val
    
        st.markdown("---")
        st.markdown('<p class="section-title">Kuesioner Kesehatan Mental (Pilih angka 1 sampai 10)</p>', unsafe_allow_html=True)
    
        n_total = len(SCR_QUESTIONS)
        answered = sum(1 for q in SCR_QUESTIONS if q["key"] in st.session_state.screening_answers)
        pct = int(answered / n_total * 100)
        st.markdown(f"""
        <div style="margin-bottom:4px; font-size:0.82rem; color:#94a3b8;">{answered}/{n_total} pertanyaan kuesioner dijawab</div>
        <div class="progress-outer"><div class="progress-inner" style="width:{pct}%;"></div></div>
        """, unsafe_allow_html=True)
    
        for i, q in enumerate(SCR_QUESTIONS):
            key   = q["key"]
            label = f"**{i+1}. {q['label']}**"
            val = st.slider(label, q["min"], q["max"],
                            st.session_state.screening_answers.get(key, q["default"]),
                            key=f"ailab_scr_{key}")
            st.session_state.screening_answers[key] = val
    
        st.markdown("<br>", unsafe_allow_html=True)
    
        col_sub, col_reset = st.columns([1.5, 1])
        submit = col_sub.button("Lihat Hasil Screening", use_container_width=True, type="primary")
        if col_reset.button("Reset Form", use_container_width=True):
            st.session_state.screening_answers = {}
            st.session_state.last_screening    = None
            st.rerun()
    
        if submit:
            missing = [q["key"] for q in SCR_QUESTIONS if q["key"] not in st.session_state.screening_answers]
            if missing or "Age" not in st.session_state.screening_answers or "Gender" not in st.session_state.screening_answers:
                st.warning("Harap lengkapi semua informasi demografis dan jawab semua pertanyaan kuesioner sebelum melihat hasil.")
            else:
                with st.spinner("AI sedang menganalisis data kamu…"):
                    result = predict_risk(st.session_state.screening_answers)
                st.session_state.last_screening = result
    
        # Tampilkan Hasil
        if st.session_state.last_screening:
            r     = st.session_state.last_screening
            level = r["risk_level"]
            level_id = r["risk_level_id"]
            score = r["risk_score"]
            rcolor = RISK_COLORS[level]
            badge_class = f"badge-{level.lower()}"
    
            st.markdown("<hr>", unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown(f"""
                    <div style="display:flex; align-items:center; gap:20px; margin-bottom:20px;">
                        <div style="width: 48px; height: 48px; border-radius: 50%; background-color: {rcolor}; flex-shrink: 0;"></div>
                        <div>
                            <div style="font-size:0.75rem; color:#64748b; text-transform:uppercase; letter-spacing:.06em;">Risk Level</div>
                            <div style="font-size:2.2rem; font-weight:800; color:{rcolor};">{level_id.upper()}</div>
                            <div style="font-size:1rem; color:#94a3b8;">Skor: {score:.1f}/10 · Confidence: {r.get('confidence',0):.1f}%</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
    
                # Breakdown radar
                breakdown = r.get("breakdown", {})
                if breakdown:
                    cats = list(breakdown.keys())
                    vals = [breakdown[c] for c in cats]
                    vals_loop = vals + vals[:1]
                    cats_loop = cats + cats[:1]
                    fig_radar = go.Figure(go.Scatterpolar(
                        r=vals_loop, theta=cats_loop, fill="toself",
                        fillcolor=f"rgba{tuple(list(bytes.fromhex(rcolor[1:])) + [77])}",
                        line=dict(color=rcolor, width=2),
                    ))
                    fig_radar.update_layout(
                        polar=dict(radialaxis=dict(range=[0, 10], gridcolor="#e2e8f0", tickfont=dict(size=9, color="#475569"))),
                        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                        font=dict(color="#475569"),
                        height=280, margin=dict(l=30, r=30, t=30, b=30),
                        title=dict(text="Breakdown per Kategori", font=dict(size=13, color="#0f172a")),
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
    
                # Rekomendasi
                st.markdown('<div style="font-size:0.82rem; color:#64748b; text-transform:uppercase; margin-bottom:10px;">Rekomendasi Personal</div>', unsafe_allow_html=True)
                for rec in r.get("recommendations", []):
                    st.markdown(f"- {rec}")
    
                st.markdown(f"""
                    <div style="margin-top:12px; font-size:0.75rem; color:#475569; border-top: 1px solid #e2e8f0; padding-top: 8px;">
                        Model: {r.get('model_used','—')} · Waktu inferensi: {r.get('inference_ms','—')} ms
                        {" · <em>(Simulasi)</em>" if r.get('is_mock') else ""}
                    </div>
                """, unsafe_allow_html=True)
    
            # Hotline jika High
            if level == "High":
                st.markdown(f"""<div class="hotline-box" style="margin-top:16px;">
                    <strong>Kamu tidak sendirian.</strong> Pertimbangkan untuk menghubungi bantuan profesional:<br><br>
                    {HOTLINE_INFO}
                </div>""", unsafe_allow_html=True)
    
            st.markdown("""<div class="disclaimer-box" style="margin-top:12px;">
                <strong>Disclaimer:</strong> Hasil screening ini <em>bukan diagnosis klinis</em>.
                Ini adalah alat skrining awal berbasis pola data. Selalu konsultasikan kondisimu dengan tenaga profesional kesehatan mental.
            </div>""", unsafe_allow_html=True)

# ============================================================
# ████████████ HALAMAN 3: ANALITIK ████████████████████████████
# ============================================================
elif st.session_state.page == "Analitik":
    st.markdown("""
    <h2 style='color:#0F172A; font-weight:800; margin-bottom:4px;'>Analitik & Riset</h2>
    <p style='color:#64748b; margin-bottom:20px;'>Eksplorasi mendalam data jurnal, screening, dan analisis lintas dataset.</p>
    """, unsafe_allow_html=True)

    # Filter diambil dari session_state yang di-set sidebar utama
    _dr = st.session_state.get("_sb_date_range", None)
    _se = st.session_state.get("_sb_sel_em", None)
    if not df_all.empty:
        MIN_DATE = df_all["timestamp"].min().date()
        MAX_DATE = df_all["timestamp"].max().date()
        start_d = _dr[0] if (_dr and len(_dr) == 2) else MIN_DATE
        end_d   = _dr[1] if (_dr and len(_dr) == 2) else MAX_DATE
        ALL_EMOTIONS = sorted(df_all["label_emosi"].unique().tolist())
        sel_em = _se if _se else ALL_EMOTIONS

        df = df_all[
            (df_all["timestamp"].dt.date >= start_d) &
            (df_all["timestamp"].dt.date <= end_d) &
            (df_all["label_emosi"].isin(sel_em))
        ].copy()
    else:
        df = df_all.copy()

    if SCREENING_AVAILABLE:
        gender_opts = sorted(df_screening_all["Gender"].dropna().unique().tolist())
        sel_g = st.multiselect("Gender (Screening)", gender_opts, default=gender_opts, key="analytics_g")
        df_scr = df_screening_all[df_screening_all["Gender"].isin(sel_g)].copy() if sel_g else df_screening_all.copy()
    else:
        df_scr = pd.DataFrame()

    # Sub-tabs
    tab_j, tab_s = st.tabs([
        "Analisis Jurnal Emosi",
        "Analisis Screening Kesehatan"
    ])

    # ── TAB 1: ANALISIS JURNAL EMOSI ──
    with tab_j:
        if df.empty:
            st.warning("Tidak ada data jurnal untuk filter yang dipilih.")
        else:
            st.metric("Total Jurnal (filter)", f"{len(df):,}")
            
            # --- Pertanyaan 1 ---
            st.markdown("### Q1 · Proporsi & Distribusi Emosi")
            c1, c2 = st.columns(2)
            with c1: 
                st.plotly_chart(chart_sentiment_pie(df), use_container_width=True)
            with c2: 
                st.plotly_chart(chart_emotion_bar(df), use_container_width=True)
            
            st.markdown("""
            <div class="insight-box">
                <strong>Analisis & Insight Bisnis - Q1 (Proporsi & Distribusi Emosi):</strong><br>
                <ul>
                    <li><strong>Dominasi Sentimen Negatif (~53.6%):</strong> Gabungan emosi <em>Sedih</em>, <em>Marah</em>, dan <em>Takut</em> mendominasi dataset riil. Ini mengonfirmasi peran platform sebagai <strong>ruang katarsis (safe space)</strong> bagi pengguna untuk mencurahkan beban emosional mereka tanpa hambatan.</li>
                    <li><strong>Sebaran Kelas Siap Latih:</strong> Emosi <em>Senang (Joy)</em> memiliki frekuensi tertinggi, diikuti sangat dekat oleh <em>Marah (Anger)</em> dan <em>Sedih (Sadness)</em>. Kategori <em>Netral (Neutral)</em> menjadi minoritas.</li>
                    <li><strong>Strategi Model (Global Class Weights):</strong> Dibandingkan membuang data (downsampling) atau mensintesis teks buatan (oversampling), ketidakseimbangan kelas ini ditangani menggunakan bobot kelas seimbang (Class Weights). Hal ini melatih model BiLSTM secara objektif tanpa bias.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # --- Pertanyaan 2 ---
            st.markdown("### Q2 · Karakteristik Panjang Tulisan (Word Count)")
            c3, c4 = st.columns(2)
            with c3: 
                st.plotly_chart(chart_boxplot_text_length(df), use_container_width=True)
            with c4: 
                st.plotly_chart(chart_avg_word_count(df), use_container_width=True)
            
            st.markdown("#### Sebaran Kata per Emosi")
            sel_wc_em = st.selectbox("Pilih Emosi untuk Detail Histogram", sel_em, 
                                     format_func=lambda x: f"{EMOTION_LABELS_ID.get(x,x)} ({x})", key="wc_em_selector")
            st.plotly_chart(chart_jurnal_word_count_hist(df, sel_wc_em), use_container_width=True)
            
            st.markdown("""
            <div class="insight-box">
                <strong>Analisis & Insight Bisnis - Q2 (Panjang Teks):</strong><br>
                <ul>
                    <li><strong>Rata-rata 24 Kata per Jurnal:</strong> Menunjukkan pengguna cenderung mengetik cerita yang cukup padat dan kaya konteks. Panjang ini sangat ideal untuk sequence modeling.</li>
                    <li><strong>Cinta (Love) & Marah (Anger) Terpanjang:</strong> Rata-rata masing-masing mencapai <strong>25.6 kata</strong> dan <strong>25.0 kata</strong>. Secara psikologis, emosi intens mendorong penjelasan yang lebih panjang lebar dan deskriptif.</li>
                    <li><strong>Netral (Neutral) Paling Singkat:</strong> Rata-rata hanya <strong>22.2 kata</strong> dengan median terendah (19 kata), menunjukkan pengguna langsung to-the-point saat dalam kondisi santai.</li>
                    <li><strong>Implikasi Pemodelan AI:</strong> Sebaran data yang terpusat di kisaran 15-35 kata mendukung penetapan parameter <code>max_length = 45-50</code> pada tokenizer BiLSTM untuk mencegah truncation konteks.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # --- Pertanyaan 3 ---
            st.markdown("### Q3 · Analisis Kata Kunci (Leksikon) & Tema")
            sel_wc_em3 = st.selectbox("Pilih Emosi untuk Kata Populer & Top Kata", sel_em,
                                      format_func=lambda x: f"{EMOTION_LABELS_ID.get(x,x)} ({x})", key="wc_em_selector_q3")
            c7, c8 = st.columns(2)
            wc_img = make_wordcloud_image(df, sel_wc_em3)
            with c7:
                if wc_img:
                    st.image(wc_img, use_container_width=True)
                else:
                    st.warning("Data tidak cukup untuk visualisasi kata.")
            with c8:
                st.plotly_chart(chart_top_words(df, sel_wc_em3), use_container_width=True)
                
            st.markdown("""
            <div class="insight-box">
                <strong>Analisis & Insight Bisnis - Q3 (Sidik Jari Kata):</strong><br>
                <ul>
                    <li><strong>Stopwords sebagai Jangkar Konteks:</strong> Kata-kata ganti ("saya", "aku") dan konjungsi ("yang", "dan", "tapi") sengaja dipertahankan agar model BiLSTM memahami relasi semantik secara sekuensial.</li>
                    <li><strong>Fingerprint Leksikal Unik:</strong>
                        <ul>
                            <li><strong>Marah (Anger):</strong> Bertema konfrontasi eksternal, didominasi kata <em>kamu</em>, <em>orang</em>, <em>salah</em>, dan <em>benci</em>.</li>
                            <li><strong>Takut (Fear):</strong> Bertema kecemasan dan respons fisik/psikis, didominasi kata <em>takut</em>, <em>gugup</em>, <em>kaget</em>, dan <em>nanti</em>.</li>
                            <li><strong>Cinta (Love):</strong> Hubungan afeksi personal dekat, ditandai kata <em>suka</em>, <em>sayang</em>, <em>cinta</em>, <em>hati</em>, dan <em>kamu</em>.</li>
                            <li><strong>Senang (Joy):</strong> Didominasi rasa syukur, optimisme, dan spiritualitas seperti <em>semangat</em>, <em>semoga</em>, <em>alhamdulillah</em>, dan <em>bahagia</em>.</li>
                            <li><strong>Sedih (Sadness):</strong> Kecewa dan kesepian internal, ditandai kata <em>kecewa</em>, <em>sakit</em>, <em>sendiri</em>, <em>menangis</em>, dan <em>tapi</em>.</li>
                            <li><strong>Netral (Neutral):</strong> Rutinitas datar deskriptif, ditandai kata <em>saja</em>, <em>biasa</em>, <em>normal</em>, dan <em>kabar</em>.</li>
                        </ul>
                    </li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    # ── TAB 2: ANALISIS SCREENING KESEHATAN ──
    with tab_s:
        if not SCREENING_AVAILABLE or df_scr.empty:
            st.warning("Data screening tidak tersedia atau filter menghasilkan data kosong.")
        else:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Responden", f"{len(df_scr):,}")
            k2.metric("Rata-rata Usia", f"{df_scr['Age'].mean():.1f} th")
            k3.metric("Riwayat Trauma", f"{df_scr['Trauma_History'].mean()*100:.1f}%")
            risk_avg = df_scr[list(STRESS_COLS.keys())].mean(axis=1).mean()
            k4.metric("Risk Score Rata-rata", f"{risk_avg:.2f}/10")
            
            st.markdown("---")
            
            # --- Pertanyaan 1 ---
            st.markdown("### Q1 · Keandalan Sistem & Kesiapan Data")
            c1, c2 = st.columns(2)
            with c1: 
                st.plotly_chart(chart_class_imbalance(), use_container_width=True)
            with c2: 
                st.plotly_chart(chart_model_performance(), use_container_width=True)
                
            st.markdown("""
            <div class="insight-box">
                <strong>Analisis & Insight Bisnis - Q1 (Keandalan & Representasi Data):</strong><br>
                <ul>
                    <li><strong>Tantangan Class Imbalance:</strong> Data awal didominasi oleh 92.2% responden bermasalah kesehatan mental dan hanya 7.8% sehat (rasio 11.76:1). Tanpa penanganan, model AI akan bias memprediksi mayoritas responden memiliki masalah.</li>
                    <li><strong>Penyeimbangan SMOTE:</strong> Penerapan teknik penyeimbangan data (SMOTE) pada data training berhasil melatih model secara adil tanpa kehilangan data asli di test set.</li>
                    <li><strong>Performa Tinggi (Recall >90%):</strong> Model mencapai <strong>Recall 93.2%</strong> dan <strong>F1-Score 91.8%</strong>, menjamin keandalan sistem dalam mendeteksi risiko tanpa bias sistematis.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # --- Pertanyaan 2 ---
            st.markdown("### Q2 · 15 Faktor Risiko Utama")
            c3, c4 = st.columns(2)
            with c3: 
                st.plotly_chart(chart_faktor_risiko_utama(), use_container_width=True)
            with c4: 
                st.plotly_chart(chart_stress_radar(df_scr), use_container_width=True)
                
            st.markdown("""
            <div class="insight-box">
                <strong>Analisis & Insight Bisnis - Q2 (Faktor Risiko Dominan):</strong><br>
                <ul>
                    <li><strong>15 Fitur Konsensus:</strong> Terpilih melalui konsensus 3 metode (Mutual Information, Random Forest, RFE) yang mencakup 5 dimensi utama:
                        <ul>
                            <li><strong>Gejala Klinis:</strong> Perasaan sedih berkepanjangan, kecemasan, kehilangan minat, dan mood swings drastis.</li>
                            <li><strong>Gaya Hidup:</strong> Durasi tidur harian, screen time, dan frekuensi olahraga.</li>
                            <li><strong>Pekerjaan & Ekonomi:</strong> Beban jam kerja mingguan, tingkat stres kerja, dan tekanan finansial.</li>
                            <li><strong>Sosial:</strong> Tingkat kesepian dan kekuatan jaringan dukungan sosial.</li>
                            <li><strong>Riwayat:</strong> Riwayat trauma dan riwayat diagnosis medis sebelumnya.</li>
                        </ul>
                    </li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # --- Pertanyaan 3 ---
            st.markdown("### Q3 · Efisiensi UX Kuesioner")
            st.plotly_chart(chart_before_after_q3(), use_container_width=True)
            
            c1, c2 = st.columns(2)
            with c1:
                delta = NB_AKURASI_FULL - NB_AKURASI_SEL
                st.metric("Akurasi 50 Pertanyaan (Sebelum)", f"{NB_AKURASI_FULL:.1f}%")
                st.metric("Akurasi 15 Pertanyaan (Sesudah)", f"{NB_AKURASI_SEL:.1f}%", delta=f"-{delta:.1f}%")
            with c2:
                st.metric("Pengurangan Pertanyaan", f"{NB_PENGURANGAN}%")
                st.metric("Estimasi Waktu Pengisian", "6-8 menit (Sebelumnya ~20-25 menit)")
                
            st.markdown("""
            <div class="insight-box">
                <strong>Analisis & Insight Bisnis - Q3 (Efisiensi Kuesioner):</strong><br>
                <ul>
                    <li><strong>Pemangkasan Pertanyaan 70%:</strong> Dari 50 pertanyaan awal menjadi 15 pertanyaan esensial saja.</li>
                    <li><strong>Peningkatan Completion Rate:</strong> Memotong waktu pengisian dari ~20-25 menit menjadi hanya <strong>6-8 menit</strong>, mengurangi drop-off rate pengguna secara signifikan.</li>
                    <li><strong>Akurasi Klinis Terjaga:</strong> Penurunan akurasi model sangat kecil (hanya berkurang <strong>0.4%</strong>), membuktikan efisiensi optimal kuesioner tanpa mengorbankan kualitas deteksi awal.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
