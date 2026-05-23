# ============================================================
# RuangRasa — Dashboard Terpadu (Restrukturisasi User Journey)
# CC26-PSU309 | Dary Ihsan Amanullah
# ============================================================

import io
import os
import random
import sys
from collections import Counter
from datetime import date, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from wordcloud import WordCloud

# Pastikan utils bisa diimport dari folder manapun
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.constants import (
    ACCENT_COLOR, DAY_ORDER, DAY_ORDER_ID, EMOTION_COLORS, EMOTION_COLORMAPS,
    EMOTION_EMOJI, EMOTION_LABELS_ID, EMOTION_ORDER, HOTLINE_INFO,
    NB_AKURASI_FULL, NB_AKURASI_SEL, NB_CLASS_NEG, NB_CLASS_POS,
    NB_FITUR_AWAL, NB_FITUR_FINAL, NB_METRICS, NB_PENGURANGAN, NB_RATIO,
    PRIMARY_COLOR, RISK_COLORS, RISK_LABELS, SENTIMENT_COLORS, STRESS_COLS,
)
from utils.data_loader import load_data, load_screening_data

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="RuangRasa — Mental Well-being Platform",
    page_icon="💭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS — App-like feel
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    background-color: #ffffff !important;
    color: #000000 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #1a2535 100%) !important;
}
[data-testid="stSidebar"] * { 
    color: #e2e8f0 !important; 
}
[data-testid="stSidebar"] .stButton > button {
    border-radius: 10px !important;
    border: 1px solid rgba(68,100,173,0.3) !important;
    background: rgba(68,100,173,0.1) !important;
    color: #e2e8f0 !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}
[data-testid="stSidebar"] .stButton > button:hover,
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: rgba(68,100,173,0.55) !important;
    border-color: #4464AD !important;
    color: #fff !important;
    transform: translateX(3px) !important;
}

/* Main area */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1400px;
    background-color: #ffffff !important;
}

/* Headings dan Text - Dark for light theme */
h1, h2, h3, h4, h5, h6 {
    color: #1a1a1a !important;
    font-weight: 900 !important;
    letter-spacing: 0.02em;
}

h1 {
    font-size: 2.2rem !important;
    color: #000000 !important;
    margin-bottom: 0.5rem !important;
}

h2 {
    font-size: 1.8rem !important;
    color: #1a1a1a !important;
    margin-top: 1.5rem !important;
    margin-bottom: 0.75rem !important;
}

h3 {
    font-size: 1.4rem !important;
    color: #1a1a1a !important;
    font-weight: 800 !important;
    margin-top: 1rem !important;
    margin-bottom: 0.5rem !important;
}

p, span, a, li {
    color: #333333 !important;
}

p strong, span strong, div strong, a strong {
    color: #000000 !important;
    font-weight: 800 !important;
}

/* KPI Cards */
.kpi-card {
    background: linear-gradient(135deg, #f8f9fa 0%, #f0f1f3 100%);
    border-radius: 16px;
    padding: 20px 22px;
    border-left: 4px solid #4464AD;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    transition: transform 0.2s ease;
    height: 100%;
}
.kpi-card:hover { transform: translateY(-2px); }
.kpi-label { 
    font-size: 0.78rem; 
    font-weight: 700; 
    color: #666666 !important; 
    letter-spacing: 0.08em; 
    text-transform: uppercase; 
    margin-bottom: 8px; 
}
.kpi-value { 
    font-size: 2.2rem; 
    font-weight: 900 !important; 
    color: #000000 !important; 
    line-height: 1; 
}
.kpi-sub   { 
    font-size: 0.8rem; 
    color: #888888 !important; 
    margin-top: 6px; 
}
.kpi-accent { border-left-color: #27AE60; }
.kpi-warn   { border-left-color: #F4D03F; }
.kpi-alert  { border-left-color: #E74C3C; }

/* Insight box */
.insight-box {
    background: linear-gradient(135deg, #f0f4ff 0%, #f5f7ff 100%);
    border-radius: 14px;
    padding: 18px 22px;
    border-left: 4px solid #4464AD;
    margin: 8px 0;
    color: #1a1a1a !important;
    font-size: 0.93rem;
    line-height: 1.7;
}
.insight-box strong {
    color: #000000 !important;
    font-weight: 900 !important;
}

/* Result card AI */
.result-card {
    background: linear-gradient(135deg, #f8f9fa 0%, #f0f1f3 100%);
    border-radius: 16px;
    padding: 24px;
    border: 1px solid rgba(68,100,173,0.2);
    box-shadow: 0 8px 30px rgba(0,0,0,0.08);
    color: #1a1a1a !important;
}
.result-card * {
    color: #1a1a1a !important;
}
.result-card strong {
    color: #000000 !important;
    font-weight: 900 !important;
}

/* Risk badge */
.badge-low    { background: #27AE60; color: white; padding: 4px 14px; border-radius: 20px; font-weight: 700; }
.badge-medium { background: #F4D03F; color: #1a1a1a; padding: 4px 14px; border-radius: 20px; font-weight: 700; }
.badge-high   { background: #E74C3C; color: white; padding: 4px 14px; border-radius: 20px; font-weight: 700; }

/* Section headers */
.section-title {
    font-size: 1.4rem !important;
    font-weight: 900 !important;
    color: #000000 !important;
    border-bottom: 3px solid #4464AD !important;
    padding-bottom: 12px !important;
    margin-bottom: 16px !important;
    display: block !important;
    letter-spacing: 0.03em;
}

/* Disclaimer */
.disclaimer-box {
    background: rgba(244,208,63,0.1);
    border: 1px solid rgba(244,208,63,0.5);
    border-radius: 10px;
    padding: 12px 16px;
    color: #cc9900 !important;
    font-size: 0.84rem;
}
.disclaimer-box strong {
    color: #b38600 !important;
    font-weight: 700 !important;
}

/* Metric override */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #f8f9fa 0%, #f0f1f3 100%);
    border-radius: 12px;
    padding: 14px 18px;
    border-left: 4px solid #4464AD;
}
[data-testid="stMetricValue"] { 
    font-size: 1.8rem !important; 
    font-weight: 900 !important; 
    color: #000000 !important; 
}
[data-testid="stMetricLabel"] { 
    color: #333333 !important; 
    font-size: 0.85rem !important; 
    font-weight: 700 !important;
}
[data-testid="stMetricDelta"] {
    color: #666666 !important;
}

/* Scrollable journal list */
.journal-entry {
    background: rgba(68,100,173,0.06);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
    border-left: 3px solid #4464AD;
    font-size: 0.88rem;
    color: #1a1a1a !important;
}
.journal-entry * {
    color: #1a1a1a !important;
}

/* Tab styling */
[data-baseweb="tab-list"] { gap: 8px; }
[data-baseweb="tab"] {
    border-radius: 8px 8px 0 0 !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
}

/* Progress bar */
.progress-outer { background: rgba(255,255,255,0.08); border-radius: 20px; height: 8px; margin: 8px 0 20px; }
.progress-inner { background: linear-gradient(90deg, #4464AD, #27AE60); border-radius: 20px; height: 8px; transition: width 0.3s ease; }

/* Hotline */
.hotline-box {
    background: rgba(231,76,60,0.1);
    border: 1px solid rgba(231,76,60,0.6);
    border-radius: 12px;
    padding: 16px 20px;
    color: #d32f2f !important;
    font-size: 0.88rem;
}
.hotline-box strong {
    color: #b71c1c !important;
    font-weight: 700 !important;
}

/* Quick action buttons */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}

/* Universal text fix - override Streamlit defaults */
.block-container {
    color: #1a1a1a !important;
}

.element-container {
    color: #1a1a1a !important;
}

.markdown-text-container {
    color: #1a1a1a !important;
}

/* Specific text elements only */
.block-container p,
.block-container span,
.block-container a,
.markdown-text-container p,
.markdown-text-container span {
    color: #1a1a1a !important;
}

.block-container strong,
strong {
    color: #000000 !important;
    font-weight: 800 !important;
}

/* Form labels */
.block-container label,
label {
    color: #333333 !important;
    font-weight: 700 !important;
}

/* Caption/small text */
.block-container small,
small,
.caption,
caption {
    color: #666666 !important;
}

/* ========== SIDEBAR FORM ELEMENTS ========== */
/* Sidebar labels - light for dark background */
[data-testid="stSidebar"] label {
    color: #e2e8f0 !important;
    font-weight: 800 !important;
}

/* Sidebar input labels specifically */
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stDateInput label,
[data-testid="stSidebar"] .stNumberInput label {
    color: #e2e8f0 !important;
    font-weight: 800 !important;
}

/* Sidebar text/paragraph */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span:not(.stCheckbox) {
    color: #cbd5e1 !important;
}

/* ========== TAB AND DROPDOWN STYLING ========== */
/* Tab text */
[data-baseweb="tab"] {
    color: #1a1a1a !important;
}

/* Selectbox dropdown items */
[data-baseweb="select"] {
    color: #1a1a1a !important;
}

/* Date input label text */
.stDateInput label {
    color: #1a1a1a !important;
    font-weight: 800 !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE INIT
# ============================================================
if "page" not in st.session_state:
    st.session_state.page = "🏠 Beranda"
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
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 10px;'>
        <span style='font-size:2.8rem'>💭</span><br>
        <span style='font-size:1.4rem; font-weight:800; color:#4464AD; letter-spacing:-0.5px;'>RuangRasa</span><br>
        <span style='font-size:0.78rem; color:#64748b;'>AI Mental Well-being Platform</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    pages = [
        "🏠 Beranda",
        "📝 Jurnal Emosi",
        "🩺 Screening",
        "🤖 AI Lab",
        "📈 Analitik",
    ]
    st.markdown("<p style='font-size:0.78rem;color:#64748b;letter-spacing:0.08em;font-weight:600;margin-bottom:8px;'>NAVIGASI</p>", unsafe_allow_html=True)
    for p in pages:
        is_active = st.session_state.page == p
        if st.button(p, width='stretch', type="primary" if is_active else "secondary", key=f"nav_{p}"):
            st.session_state.page = p
            st.session_state.ai_result = None
            st.rerun()

    st.markdown("---")

    # Info model status
    from utils.ai_inference import MODEL_STATUS, load_emotion_model, load_screening_model
    load_emotion_model(); load_screening_model()
    st.markdown("<p style='font-size:0.75rem;color:#64748b;letter-spacing:0.08em;font-weight:600;'>STATUS MODEL AI</p>", unsafe_allow_html=True)
    em_status = "🟢 Aktif" if MODEL_STATUS["emotion"] else "🟡 Mock Mode"
    sc_status  = "🟢 Aktif" if MODEL_STATUS["screening"] else "🟡 Mock Mode"
    st.markdown(f"<small>Emosi: {em_status}<br>Screening: {sc_status}</small>", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("© 2025 CC26-PSU309 · RuangRasa\nDary Ihsan Amanullah")

# ============================================================
# HELPERS — CHART FUNCTIONS (dipindah dari dashboard lama)
# ============================================================

def _plotly_dark_layout(fig, title=""):
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1", family="Plus Jakarta Sans"),
        title_font=dict(size=14, color="#f1f5f9"),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    if title:
        fig.update_layout(title=dict(text=title, x=0.01))
    return fig


def chart_emotion_bar(df):
    counts = df["label_emosi"].value_counts().reindex(EMOTION_ORDER, fill_value=0)
    fig = go.Figure(go.Bar(
        x=[EMOTION_LABELS_ID.get(e, e) for e in counts.index],
        y=counts.values,
        marker_color=[EMOTION_COLORS[e] for e in counts.index],
        text=counts.values, textposition="outside",
    ))
    _plotly_dark_layout(fig, "Distribusi Emosi")
    fig.update_layout(showlegend=False, yaxis_title="Jumlah Jurnal")
    return fig


def chart_sentiment_pie(df):
    grp = df["emotion_group"].value_counts()
    fig = go.Figure(go.Pie(
        labels=grp.index, values=grp.values,
        marker_colors=[SENTIMENT_COLORS.get(g, "#aaa") for g in grp.index],
        hole=0.45,
    ))
    _plotly_dark_layout(fig, "Distribusi Sentimen")
    return fig


def chart_daily_trend(df):
    daily = df.groupby("date").size().reset_index(name="count")
    fig = px.line(daily, x="date", y="count", color_discrete_sequence=[PRIMARY_COLOR])
    _plotly_dark_layout(fig, "Tren Jurnal Harian")
    fig.update_traces(line_width=2, fill="tozeroy", fillcolor="rgba(68,100,173,0.12)")
    return fig


def chart_daily_emotion_trend(df):
    daily_em = df.groupby(["date", "label_emosi"]).size().reset_index(name="count")
    # Filter color map to only include emotions that exist in the data
    color_map = {e: EMOTION_COLORS[e] for e in daily_em["label_emosi"].unique() if e in EMOTION_COLORS}
    fig = px.line(daily_em, x="date", y="count", color="label_emosi",
                  color_discrete_map=color_map,
                  labels={"label_emosi": "Emosi", "count": "Jumlah"})
    _plotly_dark_layout(fig, "Tren Emosi Harian")
    return fig


def chart_boxplot_text_length(df):
    fig = px.box(df, x="label_emosi", y="jumlah_kata",
                 color="label_emosi", color_discrete_map=EMOTION_COLORS,
                 labels={"label_emosi": "Emosi", "jumlah_kata": "Jumlah Kata"})
    _plotly_dark_layout(fig, "Distribusi Panjang Jurnal per Emosi")
    fig.update_layout(showlegend=False)
    return fig


def chart_avg_word_count(df):
    avg = df.groupby("label_emosi")["jumlah_kata"].mean().reindex(EMOTION_ORDER)
    fig = go.Figure(go.Bar(
        x=[EMOTION_LABELS_ID.get(e, e) for e in avg.index],
        y=avg.values.round(1),
        marker_color=[EMOTION_COLORS[e] for e in avg.index],
        text=avg.values.round(1), textposition="outside",
    ))
    _plotly_dark_layout(fig, "Rata-rata Kata per Emosi")
    fig.update_layout(showlegend=False)
    return fig


def chart_hourly_emotion(df):
    hh = df.groupby(["hour", "label_emosi"]).size().reset_index(name="count")
    fig = px.bar(hh, x="hour", y="count", color="label_emosi",
                 color_discrete_map=EMOTION_COLORS, barmode="stack",
                 labels={"hour": "Jam", "count": "Jumlah", "label_emosi": "Emosi"})
    _plotly_dark_layout(fig, "Pola Penulisan per Jam")
    return fig


def chart_day_of_week(df):
    d = df.groupby("day_of_week").size().reindex(DAY_ORDER, fill_value=0)
    fig = go.Figure(go.Bar(
        x=[DAY_ORDER_ID[dd] for dd in d.index], y=d.values,
        marker_color=PRIMARY_COLOR, text=d.values, textposition="outside",
    ))
    _plotly_dark_layout(fig, "Distribusi per Hari")
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
    _plotly_dark_layout(fig, "Heatmap Hari × Jam")
    return fig


def make_wordcloud_image(df, emotion):
    sub = df[df["label_emosi"] == emotion]
    if len(sub) < 5:
        return None
    text = " ".join(sub["text_clean"].astype(str).tolist())
    if not text.strip():
        return None
    cmap = EMOTION_COLORMAPS.get(emotion, "Blues")
    wc = WordCloud(width=600, height=300, background_color=None,
                   mode="RGBA", colormap=cmap, max_words=80).generate(text)
    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor="none")
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig


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
                 orientation="h", color_discrete_sequence=[EMOTION_COLORS.get(emotion, PRIMARY_COLOR)])
    _plotly_dark_layout(fig, f"Top {top_n} Kata — {EMOTION_LABELS_ID.get(emotion, emotion)}")
    return fig


def chart_age_distribution(df_scr):
    ag = df_scr["age_group"].value_counts().sort_index()
    fig = px.bar(x=ag.index.astype(str), y=ag.values,
                 color_discrete_sequence=[PRIMARY_COLOR],
                 labels={"x": "Kelompok Usia", "y": "Jumlah"})
    _plotly_dark_layout(fig, "Distribusi Usia Responden")
    return fig


def chart_gender_pie(df_scr):
    g = df_scr["Gender"].value_counts()
    fig = go.Figure(go.Pie(labels=g.index, values=g.values,
                           hole=0.4, marker_colors=["#4464AD", "#E91E63", "#95A5A6"]))
    _plotly_dark_layout(fig, "Distribusi Gender")
    return fig


def chart_stress_radar(df_scr):
    means = df_scr[list(STRESS_COLS.keys())].mean()
    cats  = [STRESS_COLS[c] for c in means.index]
    vals  = means.values.tolist()
    vals += vals[:1]
    cats  += cats[:1]
    fig = go.Figure(go.Scatterpolar(r=vals, theta=cats, fill="toself",
                                    fillcolor="rgba(68,100,173,0.3)",
                                    line=dict(color=PRIMARY_COLOR)))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 10])),
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#cbd5e1"))
    _plotly_dark_layout(fig, "Profil Stres Rata-rata")
    return fig


def chart_lifestyle_boxplot(df_scr):
    cols = ["Sleep_Hours_Night", "Screen_Time_Hours_Day", "Social_Media_Hours_Day", "Work_Hours_Per_Week"]
    labels = {"Sleep_Hours_Night": "Tidur (j)", "Screen_Time_Hours_Day": "Screen Time (j)",
              "Social_Media_Hours_Day": "Sosmed (j)", "Work_Hours_Per_Week": "Kerja (j/mg)"}
    melted = df_scr[cols].rename(columns=labels).melt(var_name="Faktor", value_name="Nilai")
    fig = px.box(melted, x="Faktor", y="Nilai", color="Faktor",
                 color_discrete_sequence=["#4464AD","#27AE60","#E91E63","#F4D03F"])
    _plotly_dark_layout(fig, "Distribusi Gaya Hidup")
    fig.update_layout(showlegend=False)
    return fig


def chart_screening_trend(df_scr):
    t = df_scr.groupby(df_scr["timestamp"].dt.date).size().reset_index(name="count")
    t.columns = ["date", "count"]
    fig = px.line(t, x="date", y="count", color_discrete_sequence=[PRIMARY_COLOR])
    _plotly_dark_layout(fig, "Tren Screening Harian")
    fig.update_traces(fill="tozeroy", fillcolor="rgba(68,100,173,0.12)")
    return fig


def chart_sleep_category(df_scr):
    s = df_scr["sleep_category"].value_counts()
    fig = px.bar(x=s.index.astype(str), y=s.values,
                 color_discrete_sequence=["#27AE60","#F4D03F","#E74C3C"],
                 labels={"x": "Kategori Tidur", "y": "Jumlah"})
    _plotly_dark_layout(fig, "Kualitas Tidur Responden")
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
    _plotly_dark_layout(fig, "Riwayat Kesehatan Responden")
    fig.update_layout(showlegend=False)
    return fig


def chart_correlation_heatmap(df_scr):
    num_cols = ["Work_Stress_Level","Financial_Stress","Mood_Swings","Loneliness",
                "Sleep_Hours_Night","Screen_Time_Hours_Day","Social_Media_Hours_Day",
                "Work_Hours_Per_Week","risk_score"]
    corr = df_scr[num_cols].corr()
    fig = px.imshow(corr.round(2), text_auto=True, color_continuous_scale="RdBu_r",
                    aspect="auto", zmin=-1, zmax=1)
    _plotly_dark_layout(fig, "Matriks Korelasi Variabel Wellbeing")
    return fig


def chart_class_imbalance():
    labels = ["Ada Masalah (Positive)", "Tidak Ada (Negative)"]
    values = [NB_CLASS_POS, NB_CLASS_NEG]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.45,
        marker_colors=["#E74C3C", "#27AE60"],
    ))
    _plotly_dark_layout(fig, f"Ketidakseimbangan Kelas (Rasio {NB_RATIO}:1)")
    return fig


def chart_model_performance():
    metrics = list(NB_METRICS.keys())
    values  = list(NB_METRICS.values())
    fig = go.Figure(go.Bar(x=metrics, y=values, marker_color=PRIMARY_COLOR,
                           text=[f"{v:.1f}%" for v in values], textposition="outside"))
    _plotly_dark_layout(fig, "Performa Model Setelah SMOTE (%)")
    fig.update_layout(yaxis=dict(range=[0, 105]))
    return fig


def chart_stress_histogram(df_scr, col):
    fig = px.histogram(df_scr, x=col, nbins=10,
                       color_discrete_sequence=[PRIMARY_COLOR],
                       labels={col: STRESS_COLS.get(col, col)})
    _plotly_dark_layout(fig, f"Distribusi {STRESS_COLS.get(col, col)}")
    return fig


def chart_faktor_risiko_utama():
    fitur_emas = ["Mood_Swings", "Feeling_Sad_Down", "Loneliness", "Anxious_Nervous",
                  "Work_Stress_Level", "Financial_Stress", "Sleep_Hours_Night",
                  "Social_Support", "Trauma_History"]
    fitur_pendukung = ["Screen_Time_Hours_Day", "Social_Media_Hours_Day",
                       "Work_Hours_Per_Week", "Previously_Diagnosed", "Age", "Gender"]
    labels = fitur_emas + fitur_pendukung
    colors = ["#F4D03F"] * len(fitur_emas) + [PRIMARY_COLOR] * len(fitur_pendukung)
    scores = [9.2,8.8,8.5,8.3,7.9,7.6,7.4,7.1,6.8,
              6.2,5.9,5.7,5.5,5.2,5.0]

    fig = go.Figure(go.Bar(
        x=scores, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{s:.1f}" for s in scores], textposition="outside",
    ))
    _plotly_dark_layout(fig, "Pentingnya Fitur: 15 Fitur Final (Konsensus Multi-Metode)")
    from plotly.graph_objects import layout as go_layout
    fig.update_layout(
        height=500,
        annotations=[
            dict(x=0, y=1.05, xref="paper", yref="paper", showarrow=False,
                 text="🟡 Fitur Emas (konsensus 3 metode)  🔵 Fitur Pendukung (konsensus 2 metode)",
                 font=dict(size=11, color="#94a3b8"))
        ]
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
                 color="label_emosi", color_discrete_map=EMOTION_COLORS,
                 labels={"label_emosi": "Emosi Dominan", "avg_stress": "Rata-rata Stres (0–10)"})
    _plotly_dark_layout(fig, "Korelasi: Emosi Jurnal × Skor Stres Screening")
    fig.update_layout(showlegend=False)
    return fig


def chart_cross_sleep_emotion(df_j, df_scr):
    if df_j.empty or df_scr.empty:
        return None
    merged = df_j.merge(df_scr, on="user_id", how="inner")
    if merged.empty:
        return None
    fig = px.box(merged, x="label_emosi", y="Sleep_Hours_Night",
                 color="label_emosi", color_discrete_map=EMOTION_COLORS,
                 labels={"label_emosi": "Emosi", "Sleep_Hours_Night": "Jam Tidur per Malam"})
    _plotly_dark_layout(fig, "Kualitas Tidur vs Emosi Jurnal")
    fig.update_layout(showlegend=False)
    return fig


def chart_before_after_q3():
    fig = go.Figure()
    categories = ["Accuracy", "Recall", "Precision", "F1-Score"]
    before = [91.8, 93.2, 90.7, 91.8]
    after  = [91.4, 92.8, 90.3, 91.4]
    fig.add_trace(go.Bar(name="50 Fitur (Sebelum)", x=categories, y=before,
                         marker_color="#E74C3C", text=[f"{v:.1f}%" for v in before], textposition="outside"))
    fig.add_trace(go.Bar(name="15 Fitur (Sesudah)", x=categories, y=after,
                         marker_color=PRIMARY_COLOR, text=[f"{v:.1f}%" for v in after], textposition="outside"))
    _plotly_dark_layout(fig, "Perbandingan Performa: 50 vs 15 Fitur")
    fig.update_layout(barmode="group", yaxis=dict(range=[85, 100]))
    return fig


def _fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120, facecolor="none")
    buf.seek(0)
    return buf.read()

# ============================================================
# ██████████████ HALAMAN 1: BERANDA ████████████████████████
# ============================================================
if st.session_state.page == "🏠 Beranda":
    st.markdown(f"""
    <div style='margin-bottom:20px;'>
        <h1 style='color:#f1f5f9; font-size:1.9rem; font-weight:800; margin-bottom:4px;'>
            Halo, {st.session_state.user_name}! 👋
        </h1>
        <p style='color:#64748b; font-size:0.95rem; margin:0;'>
            Selamat datang di RuangRasa — ruang amanmu untuk memahami kondisi emosional.
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
            <div class="kpi-label">📔 Total Jurnal</div>
            <div class="kpi-value">{total_jurnal:,}</div>
            <div class="kpi-sub">Semua pengguna</div>
        </div>""", unsafe_allow_html=True)

    dominant_em = df_all["label_emosi"].value_counts().idxmax() if has_journal else "—"
    dom_emoji   = EMOTION_EMOJI.get(dominant_em, "🎭")
    dom_id      = EMOTION_LABELS_ID.get(dominant_em, dominant_em)
    with col2:
        st.markdown(f"""<div class="kpi-card kpi-accent">
            <div class="kpi-label">{dom_emoji} Mood Dominan</div>
            <div class="kpi-value" style="font-size:1.4rem;">{dom_id}</div>
            <div class="kpi-sub">Terbanyak di dataset</div>
        </div>""", unsafe_allow_html=True)

    if has_session:
        last_j = st.session_state.journal_history[-1]
        session_em = last_j.get("emotion", "—")
        session_id = EMOTION_LABELS_ID.get(session_em, session_em)
        session_emoji = EMOTION_EMOJI.get(session_em, "🎭")
    else:
        session_id = "Belum ada"
        session_emoji = "📝"
    with col3:
        st.markdown(f"""<div class="kpi-card kpi-warn">
            <div class="kpi-label">{session_emoji} Jurnal Terakhir</div>
            <div class="kpi-value" style="font-size:1.3rem;">{session_id}</div>
            <div class="kpi-sub">Dari sesi ini</div>
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
            <div class="kpi-label">⚠️ Risk Score</div>
            <div class="kpi-value" style="color:{risk_color};">{risk_score:.1f}<span style="font-size:1rem;color:#64748b;">/10</span></div>
            <div class="kpi-sub">{risk_id}</div>
        </div>""", unsafe_allow_html=True)

    avg_sleep = df_screening_all["Sleep_Hours_Night"].mean() if has_screen else 0
    sleep_cat = "Cukup 😴" if avg_sleep >= 6 else "Kurang 😵"
    with col5:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">😴 Rata-rata Tidur</div>
            <div class="kpi-value">{avg_sleep:.1f}<span style="font-size:1rem;color:#64748b;">j</span></div>
            <div class="kpi-sub">{sleep_cat}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts utama ──
    col_a, col_b = st.columns([1.6, 1])
    with col_a:
        if has_journal:
            st.plotly_chart(chart_daily_trend(df_all), width='stretch')
    with col_b:
        if has_journal:
            st.plotly_chart(chart_sentiment_pie(df_all), width='stretch')

    # ── Insight AI ──
    if has_journal:
        neg_pct  = (df_all["emotion_group"] == "Negative").sum() / len(df_all) * 100
        pos_pct  = (df_all["emotion_group"] == "Positive").sum() / len(df_all) * 100
        peak_h   = df_all.groupby("hour").size().idxmax()
        avg_w    = df_all["jumlah_kata"].mean()
        st.markdown(f"""<div class="insight-box">
            💡 <strong>Insight Otomatis</strong><br><br>
            Dari <strong>{len(df_all):,} jurnal</strong> yang tersimpan, <strong>{neg_pct:.1f}%</strong> mengandung emosi negatif
            dan <strong>{pos_pct:.1f}%</strong> positif. Emosi dominan adalah <strong>{dom_id}</strong>.
            Puncak aktivitas menulis terjadi pada pukul <strong>{peak_h:02d}:00</strong>,
            dengan rata-rata <strong>{avg_w:.1f} kata</strong> per entri jurnal.
            {f"<br><br>⚠️ Skor risiko rata-rata pengguna: <strong>{risk_score:.1f}/10</strong> — " + ("Perlu perhatian lebih." if risk_score >= 4 else "Dalam batas sehat.") if has_screen else ""}
        </div>""", unsafe_allow_html=True)

    # ── Quick Actions ──
    st.markdown("<br><p class='section-title'>⚡ Mulai Sekarang</p>", unsafe_allow_html=True)
    q1, q2, q3 = st.columns(3)
    with q1:
        if st.button("📝 Tulis Jurnal Hari Ini", width='stretch', type="primary"):
            st.session_state.page = "📝 Jurnal Emosi"
            st.rerun()
    with q2:
        if st.button("🩺 Mulai Screening Baru", width='stretch'):
            st.session_state.page = "🩺 Screening"
            st.rerun()
    with q3:
        if st.button("🤖 Coba AI Lab", width='stretch'):
            st.session_state.page = "🤖 AI Lab"
            st.rerun()

    # ── Disclaimer ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div class="disclaimer-box">
        ⚠️ <strong>Disclaimer:</strong> RuangRasa adalah alat <em>dukungan awal</em> dan <em>screening</em>,
        bukan pengganti diagnosis atau konsultasi profesional kesehatan mental.
        Data bersifat anonim dan hanya digunakan untuk analisis personal.
    </div>""", unsafe_allow_html=True)


# ============================================================
# ████████████ HALAMAN 2: JURNAL EMOSI + AI ████████████████
# ============================================================
elif st.session_state.page == "📝 Jurnal Emosi":
    st.markdown("""
    <h2 style='color:#f1f5f9; font-weight:800; margin-bottom:4px;'>📝 Jurnal Emosi</h2>
    <p style='color:#64748b; margin-bottom:20px;'>Tulis apa yang kamu rasakan — AI akan menganalisis emosimu secara real-time.</p>
    """, unsafe_allow_html=True)

    if not MODEL_STATUS["emotion"]:
        st.info("⚠️ Model AI belum tersedia di folder `models/`. Menggunakan **Mock Mode** (keyword-based). Hasil bersifat simulasi.", icon="🔬")

    col_input, col_history = st.columns([1.4, 1])

    with col_input:
        st.markdown('<p class="section-title">✍️ Tulis Jurnalmu</p>', unsafe_allow_html=True)
        journal_text = st.text_area(
            "Apa yang kamu rasakan hari ini?",
            height=180,
            placeholder="Contoh: Hari ini meeting-ku gagal total. Aku merasa malu dan frustrasi banget sama diri sendiri...",
            label_visibility="collapsed",
        )
        char_count = len(journal_text)
        st.caption(f"{char_count} karakter | min. 20 karakter untuk analisis")

        col_btn1, col_btn2 = st.columns(2)
        analyze_clicked = col_btn1.button("🔍 Analisis AI", width='stretch', type="primary",
                                           disabled=(char_count < 20))
        clear_clicked   = col_btn2.button("🗑️ Bersihkan", width='stretch')

        if clear_clicked:
            st.session_state.ai_result = None
            st.rerun()

        if analyze_clicked and journal_text.strip():
            from utils.ai_inference import predict_emotion
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
                emoji  = EMOTION_EMOJI.get(em, "🎭")
                color  = EMOTION_COLORS.get(em, PRIMARY_COLOR)
                sent_c = SENTIMENT_COLORS.get(sent, "#aaa")

                st.markdown(f"""<div class="result-card" style="margin-top:16px;">
                    <div style="display:flex; align-items:center; gap:16px; margin-bottom:16px;">
                        <span style="font-size:3rem;">{emoji}</span>
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
                    _plotly_dark_layout(fig_prob, "Distribusi Probabilitas Emosi")
                    fig_prob.update_layout(height=220, margin=dict(l=0, r=40, t=35, b=0),
                                           xaxis=dict(range=[0, 110]))
                    st.plotly_chart(fig_prob, width='stretch')

                st.markdown(f"""
                    <div style="background:rgba(255,255,255,0.04); border-radius:10px; padding:14px 16px; margin-bottom:12px;">
                        <div style="font-size:0.75rem; color:#64748b; margin-bottom:6px; text-transform:uppercase;">💬 Validasi</div>
                        <div style="color:#e2e8f0; font-style:italic; line-height:1.7;">"{r.get('validation', '')}"</div>
                    </div>
                    <div style="margin-bottom:8px;">
                        <div style="font-size:0.75rem; color:#64748b; margin-bottom:8px; text-transform:uppercase;">🎯 Rekomendasi</div>
                """, unsafe_allow_html=True)

                for rec in r.get("recommendations", []):
                    st.markdown(f"- {rec}")

                st.markdown(f"""
                    <div style="margin-top:12px; font-size:0.75rem; color:#475569;">
                        Model: {r.get('model_used','—')} · Waktu inferensi: {r.get('inference_ms','—')} ms
                        {" · <em>(Simulasi)</em>" if r.get('is_mock') else ""}
                    </div></div>
                """, unsafe_allow_html=True)

                # Tombol Simpan
                if st.button("✅ Simpan Jurnal", width='stretch'):
                    st.success("Jurnal berhasil disimpan ke riwayat sesi!")
                    st.balloons()

    with col_history:
        st.markdown('<p class="section-title">📖 Riwayat Sesi Ini</p>', unsafe_allow_html=True)
        if not st.session_state.journal_history:
            st.markdown("<p style='color:#475569; font-size:0.88rem;'>Belum ada jurnal di sesi ini.</p>", unsafe_allow_html=True)
        else:
            for entry in reversed(st.session_state.journal_history):
                em_c = EMOTION_COLORS.get(entry["emotion"], PRIMARY_COLOR)
                em_i = EMOTION_LABELS_ID.get(entry["emotion"], entry["emotion"])
                em_e = EMOTION_EMOJI.get(entry["emotion"], "🎭")
                st.markdown(f"""<div class="journal-entry">
                    <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                        <span style="color:{em_c}; font-weight:700; font-size:0.82rem;">{em_e} {em_i}</span>
                        <span style="color:#475569; font-size:0.78rem;">{entry['timestamp']}</span>
                    </div>
                    <div style="color:#94a3b8; font-size:0.85rem;">{entry['text']}</div>
                </div>""", unsafe_allow_html=True)

    # Dataset distribution
    if not df_all.empty:
        st.markdown("<br><p class='section-title'>📊 Distribusi Emosi Dataset</p>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(chart_emotion_bar(df_all), width='stretch')
        with c2:
            st.plotly_chart(chart_daily_emotion_trend(df_all), width='stretch')

    st.markdown("""<div class="disclaimer-box" style="margin-top:12px;">
        ⚠️ Analisis AI ini bersifat indikatif, bukan diagnosis klinis.
        Jika kamu merasa tidak baik-baik saja secara berkepanjangan, pertimbangkan untuk berbicara dengan profesional.
    </div>""", unsafe_allow_html=True)


# ============================================================
# ████████████ HALAMAN 3: SCREENING + AI ████████████████████
# ============================================================
elif st.session_state.page == "🩺 Screening":
    st.markdown("""
    <h2 style='color:#f1f5f9; font-weight:800; margin-bottom:4px;'>🩺 Screening Kesehatan Mental</h2>
    <p style='color:#64748b; margin-bottom:20px;'>Jawab 15 pertanyaan singkat — AI akan memprediksi level risiko dan memberikan rekomendasi personal.</p>
    """, unsafe_allow_html=True)

    if not MODEL_STATUS["screening"]:
        st.info("⚠️ Model screening dalam **Mock Mode**. Hasil menggunakan rule-based scoring.", icon="🔬")

    # ── Form Screening ──
    QUESTIONS = [
        {"key": "Age",                       "label": "Usia kamu saat ini?",                          "type": "number",  "min": 10, "max": 80, "default": 22},
        {"key": "Gender",                    "label": "Gender",                                        "type": "radio",   "options": ["Male", "Female", "Other"]},
        {"key": "Work_Stress_Level",         "label": "Seberapa stres kamu terkait pekerjaan/studi? (1=sangat rendah, 10=sangat tinggi)", "type": "slider", "min": 1, "max": 10, "default": 5},
        {"key": "Financial_Stress",          "label": "Seberapa besar tekanan finansial yang kamu rasakan? (1-10)", "type": "slider", "min": 1, "max": 10, "default": 5},
        {"key": "Mood_Swings",               "label": "Seberapa sering suasana hatimu berubah-ubah secara tiba-tiba? (1-10)", "type": "slider", "min": 1, "max": 10, "default": 5},
        {"key": "Loneliness",                "label": "Seberapa sering kamu merasa kesepian? (1=tidak pernah, 10=sangat sering)", "type": "slider", "min": 1, "max": 10, "default": 5},
        {"key": "Sleep_Hours_Night",         "label": "Rata-rata berapa jam kamu tidur per malam?",    "type": "number_float", "min": 2.0, "max": 14.0, "default": 7.0, "step": 0.5},
        {"key": "Screen_Time_Hours_Day",     "label": "Rata-rata berapa jam screen time per hari?",   "type": "number_float", "min": 0.0, "max": 20.0, "default": 5.0, "step": 0.5},
        {"key": "Social_Media_Hours_Day",    "label": "Berapa jam penggunaan media sosial per hari?", "type": "number_float", "min": 0.0, "max": 12.0, "default": 2.0, "step": 0.5},
        {"key": "Work_Hours_Per_Week",       "label": "Rata-rata berapa jam kerja/kuliah per minggu?","type": "number",  "min": 0,  "max": 100, "default": 40},
        {"key": "Previously_Diagnosed",      "label": "Apakah kamu pernah didiagnosis masalah kesehatan mental sebelumnya?", "type": "radio_yn"},
        {"key": "Trauma_History",            "label": "Apakah kamu pernah mengalami trauma atau kejadian sangat menekan?",   "type": "radio_yn"},
        {"key": "Feeling_Sad_Down",          "label": "Seberapa sering kamu merasa sedih/murung tanpa alasan jelas? (1-10)", "type": "slider", "min": 1, "max": 10, "default": 5},
        {"key": "Anxious_Nervous",           "label": "Seberapa sering kamu merasa cemas atau gugup? (1-10)",               "type": "slider", "min": 1, "max": 10, "default": 5},
        {"key": "Social_Support",            "label": "Seberapa besar dukungan sosial yang kamu rasakan dari orang sekitar? (1=sangat rendah, 10=sangat tinggi)", "type": "slider", "min": 1, "max": 10, "default": 6},
    ]

    n_total = len(QUESTIONS)
    # Progress
    answered = sum(1 for q in QUESTIONS if q["key"] in st.session_state.screening_answers)
    pct = int(answered / n_total * 100)
    st.markdown(f"""
    <div style="margin-bottom:4px; font-size:0.82rem; color:#94a3b8;">{answered}/{n_total} pertanyaan dijawab</div>
    <div class="progress-outer"><div class="progress-inner" style="width:{pct}%;"></div></div>
    """, unsafe_allow_html=True)

    # ── Pertanyaan ──
    for i, q in enumerate(QUESTIONS):
        key   = q["key"]
        label = f"**{i+1}. {q['label']}**"

        if q["type"] == "slider":
            val = st.slider(label, q["min"], q["max"],
                            st.session_state.screening_answers.get(key, q["default"]),
                            key=f"scr_{key}")
        elif q["type"] == "number":
            val = st.number_input(label, q["min"], q["max"],
                                  st.session_state.screening_answers.get(key, q["default"]),
                                  key=f"scr_{key}")
        elif q["type"] == "number_float":
            val = st.number_input(label, float(q["min"]), float(q["max"]),
                                  float(st.session_state.screening_answers.get(key, q["default"])),
                                  step=q.get("step", 0.5), key=f"scr_{key}")
        elif q["type"] == "radio":
            opts = q["options"]
            cur  = st.session_state.screening_answers.get(key, opts[0])
            idx  = opts.index(cur) if cur in opts else 0
            val  = st.radio(label, opts, index=idx, horizontal=True, key=f"scr_{key}")
        elif q["type"] == "radio_yn":
            opts = ["Ya", "Tidak"]
            cur_raw = st.session_state.screening_answers.get(key, 0)
            cur_str = "Ya" if cur_raw == 1 else "Tidak"
            sel  = st.radio(label, opts, index=opts.index(cur_str), horizontal=True, key=f"scr_{key}")
            val  = 1 if sel == "Ya" else 0
        else:
            val = None

        if val is not None:
            st.session_state.screening_answers[key] = val

    st.markdown("<br>", unsafe_allow_html=True)

    col_sub, col_reset = st.columns([1.5, 1])
    submit = col_sub.button("🔍 Lihat Hasil Screening", width='stretch', type="primary")
    if col_reset.button("🔄 Reset Form", width='stretch'):
        st.session_state.screening_answers = {}
        st.session_state.last_screening    = None
        st.rerun()

    if submit:
        from utils.ai_inference import predict_risk
        if len(st.session_state.screening_answers) < n_total:
            st.warning("Harap jawab semua pertanyaan sebelum melihat hasil.")
        else:
            with st.spinner("AI sedang menganalisis data kamu…"):
                result = predict_risk(st.session_state.screening_answers)
            st.session_state.last_screening = result

    # ── Tampilkan Hasil ──
    if st.session_state.last_screening:
        r     = st.session_state.last_screening
        level = r["risk_level"]
        level_id = r["risk_level_id"]
        score = r["risk_score"]
        rcolor = RISK_COLORS[level]
        badge_class = f"badge-{level.lower()}"

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(f"""<div class="result-card">
            <div style="display:flex; align-items:center; gap:20px; margin-bottom:20px;">
                <div style="font-size:3.5rem;">{'🟢' if level=='Low' else '🟡' if level=='Medium' else '🔴'}</div>
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
                polar=dict(radialaxis=dict(range=[0, 10], tickfont=dict(size=9))),
                paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#cbd5e1"),
                height=280, margin=dict(l=30, r=30, t=30, b=30),
                title=dict(text="Breakdown per Kategori", font=dict(size=13, color="#f1f5f9")),
            )
            st.plotly_chart(fig_radar, width='stretch')

        # Rekomendasi
        st.markdown('<div style="font-size:0.82rem; color:#64748b; text-transform:uppercase; margin-bottom:10px;">🎯 Rekomendasi Personal</div>', unsafe_allow_html=True)
        for rec in r.get("recommendations", []):
            st.markdown(f"- {rec}")

        st.markdown(f"""
            <div style="margin-top:12px; font-size:0.75rem; color:#475569;">
                Model: {r.get('model_used','—')} · Waktu inferensi: {r.get('inference_ms','—')} ms
                {" · <em>(Simulasi)</em>" if r.get('is_mock') else ""}
            </div>
        </div>""", unsafe_allow_html=True)

        # Hotline jika High
        if level == "High":
            st.markdown(f"""<div class="hotline-box" style="margin-top:16px;">
                🚨 <strong>Kamu tidak sendirian.</strong> Pertimbangkan untuk menghubungi bantuan profesional:<br><br>
                {HOTLINE_INFO}
            </div>""", unsafe_allow_html=True)

        st.markdown("""<div class="disclaimer-box" style="margin-top:12px;">
            ⚠️ <strong>Disclaimer:</strong> Hasil screening ini <em>bukan diagnosis klinis</em>.
            Ini adalah alat skrining awal berbasis pola data. Selalu konsultasikan kondisimu dengan tenaga profesional kesehatan mental.
        </div>""", unsafe_allow_html=True)


# ============================================================
# ████████████ HALAMAN 4: AI LAB ████████████████████████████
# ============================================================
elif st.session_state.page == "🤖 AI Lab":
    st.markdown("""
    <h2 style='color:#f1f5f9; font-weight:800; margin-bottom:4px;'>🤖 AI Lab</h2>
    <p style='color:#64748b; margin-bottom:20px;'>Demo model AI secara teknis. Eksplorasi output model emosi dan screening secara langsung.</p>
    """, unsafe_allow_html=True)

    model_choice = st.selectbox("Pilih Model AI", ["🎭 Model Emosi (BiLSTM)", "🩺 Model Screening (DNN)"], label_visibility="visible")

    col_meta1, col_meta2, col_meta3 = st.columns(3)
    if "Emosi" in model_choice:
        col_meta1.metric("Arsitektur",   "BiLSTM + Attention")
        col_meta2.metric("Output",       "6 kelas emosi")
        col_meta3.metric("Status",       "✅ Aktif" if MODEL_STATUS["emotion"] else "🟡 Mock")
        meta_info = {"Model": "emotion_bilstm.keras", "Input Shape": "(None, 100)", "Bahasa": "Indonesia",
                     "Preprocessing": "Sastrawi + regex + padding", "Kelas": "Anger, Sadness, Joy, Love, Fear, Neutral"}
    else:
        col_meta1.metric("Arsitektur",   "Deep Neural Network")
        col_meta2.metric("Output",       "Low / Medium / High")
        col_meta3.metric("Status",       "✅ Aktif" if MODEL_STATUS["screening"] else "🟡 Mock")
        meta_info = {"Model": "screening_risk.keras", "Fitur Input": "15 fitur", "Output": "Risk level",
                     "Scaler": "StandardScaler", "Trained on": "1,000 responden"}

    with st.expander("📋 Metadata Teknis Model"):
        for k, v in meta_info.items():
            st.markdown(f"- **{k}**: {v}")

    st.markdown("---")

    if "Emosi" in model_choice:
        st.markdown("### ✍️ Input Teks")
        col_a, col_b = st.columns(2)
        texts = [
            col_a.text_area("Input A", height=130, placeholder="Contoh: Hari ini aku sangat senang karena berhasil…", key="lab_a"),
            col_b.text_area("Input B (opsional, compare mode)", height=130, placeholder="Contoh: Aku takut dan cemas menghadapi besok…", key="lab_b"),
        ]

        if st.button("🔬 Jalankan Inferensi", type="primary", width='stretch'):
            from utils.ai_inference import predict_emotion
            results = []
            for t in texts:
                if t and len(t.strip()) >= 5:
                    with st.spinner("Inferensi…"):
                        results.append(predict_emotion(t))
                else:
                    results.append(None)

            cols_res = st.columns(2)
            for i, (res, col) in enumerate(zip(results, cols_res)):
                if res and "error" not in res:
                    em    = res["emotion"]
                    color = EMOTION_COLORS.get(em, PRIMARY_COLOR)
                    emoji = EMOTION_EMOJI.get(em, "🎭")
                    probs = res.get("probabilities", {})

                    with col:
                        st.markdown(f"""<div class="result-card">
                            <div style="font-size:1.1rem; font-weight:700; color:{color}; margin-bottom:10px;">
                                {emoji} {res['emotion_id']} ({res['confidence']:.1f}%)
                            </div>
                            <div style="font-size:0.8rem; color:#64748b; margin-bottom:10px;">
                                {res['sentiment']} · {res['model_used']} · {res['inference_ms']} ms
                            </div>
                        """, unsafe_allow_html=True)
                        if probs:
                            prob_df = pd.DataFrame(
                                [{"Emosi": EMOTION_LABELS_ID.get(e, e), "Prob (%)": v} for e, v in probs.items()]
                            ).sort_values("Prob (%)", ascending=True)
                            fig_p = go.Figure(go.Bar(
                                x=prob_df["Prob (%)"], y=prob_df["Emosi"],
                                orientation="h",
                                marker_color=[EMOTION_COLORS.get(e, PRIMARY_COLOR) for e in probs.keys()],
                                text=[f"{v:.1f}%" for v in prob_df["Prob (%)"]],
                                textposition="outside",
                            ))
                            _plotly_dark_layout(fig_p)
                            fig_p.update_layout(height=180, margin=dict(l=0,r=40,t=20,b=0),
                                                xaxis=dict(range=[0,110]))
                            st.plotly_chart(fig_p, width='stretch')
                        st.markdown("</div>", unsafe_allow_html=True)

    else:  # Screening model
        st.markdown("### 📋 Input Cepat (5 fitur utama)")
        c1, c2, c3 = st.columns(3)
        work_stress  = c1.slider("Stres Kerja", 1, 10, 5, key="lab_ws")
        sad          = c2.slider("Rasa Sedih", 1, 10, 5, key="lab_sad")
        lonely       = c3.slider("Kesepian", 1, 10, 5, key="lab_lon")
        c4, c5, _ = st.columns(3)
        sleep        = c4.number_input("Jam Tidur", 2.0, 14.0, 7.0, step=0.5, key="lab_slp")
        support      = c5.slider("Dukungan Sosial", 1, 10, 6, key="lab_sup")

        if st.button("🔬 Prediksi Risiko", type="primary", width='stretch'):
            from utils.ai_inference import predict_risk
            quick_answers = {
                "Age": 25, "Gender": "Male",
                "Work_Stress_Level": work_stress, "Financial_Stress": 5,
                "Mood_Swings": 5, "Loneliness": lonely,
                "Sleep_Hours_Night": sleep, "Screen_Time_Hours_Day": 5,
                "Social_Media_Hours_Day": 2, "Work_Hours_Per_Week": 40,
                "Previously_Diagnosed": 0, "Trauma_History": 0,
                "Feeling_Sad_Down": sad, "Anxious_Nervous": 5,
                "Social_Support": support,
            }
            with st.spinner("Memprediksi…"):
                r = predict_risk(quick_answers)
            level  = r["risk_level"]
            rcolor = RISK_COLORS[level]
            st.markdown(f"""<div class="result-card" style="margin-top:16px;">
                <div style="font-size:1.5rem; font-weight:800; color:{rcolor}; margin-bottom:10px;">
                    {'🟢' if level=='Low' else '🟡' if level=='Medium' else '🔴'}
                    Risk Level: {r['risk_level_id'].upper()} ({r['risk_score']:.1f}/10)
                </div>
                <div style="font-size:0.8rem; color:#64748b;">
                    Confidence: {r.get('confidence',0):.1f}% · {r.get('model_used','—')} · {r.get('inference_ms','—')} ms
                </div>
            </div>""", unsafe_allow_html=True)


# ============================================================
# ████████████ HALAMAN 5: ANALITIK ██████████████████████████
# ============================================================
elif st.session_state.page == "📈 Analitik":
    st.markdown("""
    <h2 style='color:#f1f5f9; font-weight:800; margin-bottom:4px;'>📈 Analitik & Riset</h2>
    <p style='color:#64748b; margin-bottom:20px;'>Eksplorasi mendalam data jurnal, screening, dan analisis lintas dataset.</p>
    """, unsafe_allow_html=True)

    # Filter sidebar untuk analitik
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🔎 Filter Analitik")
        if not df_all.empty:
            MIN_DATE = df_all["timestamp"].min().date()
            MAX_DATE = df_all["timestamp"].max().date()
            date_range = st.date_input("Rentang Tanggal", [MIN_DATE, MAX_DATE],
                                       min_value=MIN_DATE, max_value=MAX_DATE, key="analytics_date")
            if len(date_range) == 2:
                start_d, end_d = date_range
            else:
                start_d, end_d = MIN_DATE, MAX_DATE

            ALL_EMOTIONS = sorted(df_all["label_emosi"].unique().tolist())
            sel_em = st.multiselect("Filter Emosi", ALL_EMOTIONS, default=ALL_EMOTIONS, key="analytics_em",
                                    format_func=lambda x: f"{EMOTION_LABELS_ID.get(x,x)} ({x})")
            if not sel_em:
                sel_em = ALL_EMOTIONS

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
    tab_j, tab_s, tab_cross, tab_q1, tab_q2, tab_q3 = st.tabs([
        "📔 Jurnal", "🩺 Screening", "🔗 Cross-Analysis",
        "Q1 · Keandalan", "Q2 · Faktor Risiko", "Q3 · Efisiensi"
    ])

    # ── TAB JURNAL ──
    with tab_j:
        if df.empty:
            st.warning("Tidak ada data jurnal untuk filter yang dipilih.")
        else:
            st.metric("Total Jurnal (filter)", f"{len(df):,}")
            c1, c2 = st.columns(2)
            with c1: st.plotly_chart(chart_emotion_bar(df), width='stretch')
            with c2: st.plotly_chart(chart_sentiment_pie(df), width='stretch')
            st.plotly_chart(chart_daily_emotion_trend(df), width='stretch')
            st.markdown("---")
            c3, c4 = st.columns(2)
            with c3: st.plotly_chart(chart_boxplot_text_length(df), use_container_width=True)
            with c4: st.plotly_chart(chart_avg_word_count(df), use_container_width=True)
            st.markdown("---")
            st.subheader("⏰ Pola Waktu")
            st.plotly_chart(chart_hourly_emotion(df), use_container_width=True)
            c5, c6 = st.columns(2)
            with c5: st.plotly_chart(chart_day_of_week(df), use_container_width=True)
            with c6: st.plotly_chart(chart_heatmap_day_hour(df), use_container_width=True)
            st.markdown("---")
            st.subheader("☁️ Word Cloud per Emosi")
            emotions_to_show = [e for e in EMOTION_ORDER if e in sel_em]
            wc_tabs = st.tabs([EMOTION_LABELS_ID.get(e, e) for e in emotions_to_show])
            for i, em in enumerate(emotions_to_show):
                with wc_tabs[i]:
                    c7, c8 = st.columns(2)
                    wc_fig = make_wordcloud_image(df, em)
                    with c7:
                        if wc_fig:
                            buf = io.BytesIO()
                            wc_fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
                            buf.seek(0)
                            st.image(buf, use_container_width=True)
                            plt.close(wc_fig)
                        else:
                            st.warning("Data tidak cukup.")
                    with c8:
                        st.plotly_chart(chart_top_words(df, em), use_container_width=True)

    # ── TAB SCREENING ──
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
            c1, c2 = st.columns(2)
            with c1: st.plotly_chart(chart_age_distribution(df_scr), use_container_width=True)
            with c2: st.plotly_chart(chart_gender_pie(df_scr), use_container_width=True)
            c3, c4 = st.columns(2)
            with c3: st.plotly_chart(chart_stress_radar(df_scr), use_container_width=True)
            with c4: st.plotly_chart(chart_lifestyle_boxplot(df_scr), use_container_width=True)
            c5, c6 = st.columns(2)
            with c5: st.plotly_chart(chart_sleep_category(df_scr), use_container_width=True)
            with c6: st.plotly_chart(chart_trauma_previously(df_scr), use_container_width=True)
            st.plotly_chart(chart_screening_trend(df_scr), use_container_width=True)
            st.plotly_chart(chart_correlation_heatmap(df_scr), use_container_width=True)

    # ── TAB CROSS ANALYSIS ──
    with tab_cross:
        st.subheader("🔗 Korelasi Jurnal Emosi × Data Screening")
        f1 = chart_cross_emotion_stress(df_all, df_scr)
        if f1: st.plotly_chart(f1, use_container_width=True)
        else: st.warning("Tidak cukup user_id yang cocok di kedua dataset.")
        f2 = chart_cross_sleep_emotion(df_all, df_scr)
        if f2: st.plotly_chart(f2, use_container_width=True)

    # ── TAB Q1 ──
    with tab_q1:
        st.subheader("Q1: Apakah data cukup representatif?")
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(chart_class_imbalance(), use_container_width=True)
        with c2: st.plotly_chart(chart_model_performance(), use_container_width=True)
        st.info(f"💡 Class imbalance awal {NB_RATIO}:1. Setelah SMOTE: Recall {NB_METRICS['Keandalan Deteksi (Recall)']:.1f}%, F1 {NB_METRICS['F1-Score']:.1f}%")

    # ── TAB Q2 ──
    with tab_q2:
        st.subheader("Q2: Faktor risiko paling dominan?")
        st.plotly_chart(chart_faktor_risiko_utama(), use_container_width=True)
        st.info("💡 15 fitur final dari 50 awal via MI + RF + RFE. 9 fitur emas (konsensus 3 metode) + 6 pendukung.")
        if not df_scr.empty:
            st.plotly_chart(chart_stress_radar(df_scr), use_container_width=True)

    # ── TAB Q3 ──
    with tab_q3:
        st.subheader("Q3: Efisiensi kuesioner 15 pertanyaan")
        st.plotly_chart(chart_before_after_q3(), use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            delta = NB_AKURASI_FULL - NB_AKURASI_SEL
            st.metric("Akurasi 50 fitur", f"{NB_AKURASI_FULL:.1f}%")
            st.metric("Akurasi 15 fitur", f"{NB_AKURASI_SEL:.1f}%", delta=f"-{delta:.1f}%")
        with c2:
            st.metric("Pengurangan pertanyaan", f"{NB_PENGURANGAN}%")
            st.metric("Estimasi waktu pengisian", "6–8 menit")
        st.success(f"✅ Kuesioner berhasil dipangkas {NB_PENGURANGAN}% ({NB_FITUR_AWAL}→{NB_FITUR_FINAL} pertanyaan) dengan penurunan akurasi hanya {delta:.1f}%.")

