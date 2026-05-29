import streamlit as st
import pandas as pd
import os
import sys

# Ensure utils can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import EMOTION_LABELS_ID, RISK_COLORS, RISK_LABELS, PRIMARY_COLOR
from utils.charts import chart_sentiment_pie, chart_sleep_category
from utils.data_loader import load_data, load_screening_data

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

df_all, df_screening_all, SCREENING_AVAILABLE = _load_all()

st.markdown(f"""
<div style='margin-bottom:20px;'>
    <h1 style='color:#0F172A; font-size:1.9rem; font-weight:800; margin-bottom:4px;'>
        RuangRasa — Analytics Platform
    </h1>
    <p style='color:#64748b; font-size:0.95rem; margin:0;'>
        Dashboard riset & analitik kesehatan mental berbasis AI. Eksplorasi insight dari <strong>9.700 data jurnal emosi</strong>
        dan <strong>10.000 data screening</strong>, lengkap dengan demo model AI secara interaktif.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Pastikan ada data ──
has_journal  = not df_all.empty
has_screen   = SCREENING_AVAILABLE and not df_screening_all.empty
has_session  = len(st.session_state.get("journal_history", [])) > 0

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

if st.session_state.get("last_screening"):
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
        st.plotly_chart(chart_sentiment_pie(df_all), use_container_width=True, theme=None)
with col_b:
    if has_screen:
        st.plotly_chart(chart_sleep_category(df_screening_all), use_container_width=True, theme=None)

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
        st.switch_page("views/ai_lab.py")
with q2:
    if st.button("📊 Lihat Analitik Riset", use_container_width=True):
        st.switch_page("views/analytics.py")

# ── Disclaimer ──
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""<div class="disclaimer-box">
    <strong>Catatan Riset:</strong> Data yang ditampilkan di halaman ini bersumber dari dataset penelitian
    (<strong>9.700 jurnal emosi</strong> dan <strong>10.000 data screening</strong>).
    Demo model AI tersedia di halaman <strong>AI Lab</strong>.
</div>""", unsafe_allow_html=True)
