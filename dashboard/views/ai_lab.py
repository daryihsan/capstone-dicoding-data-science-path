import streamlit as st
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import EMOTION_COLORS, EMOTION_LABELS_ID, PRIMARY_COLOR, SENTIMENT_COLORS, RISK_COLORS, RISK_LABELS
from utils.charts import _plotly_clean_layout
import plotly.graph_objects as go
from utils.ai_inference import (
    MODEL_STATUS, load_emotion_model, load_screening_model,
    predict_emotion, predict_risk
)

# Load models ONLY when this page is accessed
with st.spinner("Menyiapkan model AI..."):
    load_emotion_model()
    load_screening_model()

if "journal_history" not in st.session_state:
    st.session_state.journal_history = []
if "last_screening" not in st.session_state:
    st.session_state.last_screening = None
if "screening_answers" not in st.session_state:
    st.session_state.screening_answers = {}
if "ai_result" not in st.session_state:
    st.session_state.ai_result = None

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
                        st.plotly_chart(fig_prob, use_container_width=True, theme=None, config={'displayModeBar': False})

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
                st.plotly_chart(fig_radar, use_container_width=True, theme=None, config={'displayModeBar': False})

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


        st.markdown("""<div class="disclaimer-box" style="margin-top:12px;">
            <strong>Disclaimer:</strong> Hasil screening ini <em>bukan diagnosis klinis</em>.
            Ini adalah alat skrining awal berbasis pola data. Selalu konsultasikan kondisimu dengan tenaga profesional kesehatan mental.
        </div>""", unsafe_allow_html=True)
