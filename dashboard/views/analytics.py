import streamlit as st
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import EMOTION_LABELS_ID, STRESS_COLS, NB_AKURASI_FULL, NB_AKURASI_SEL, NB_PENGURANGAN
from utils.charts import (
    chart_sentiment_pie, chart_emotion_bar, chart_boxplot_text_length, chart_avg_word_count,
    chart_jurnal_word_count_hist, make_wordcloud_image, chart_top_words,
    chart_class_imbalance, chart_model_performance, chart_faktor_risiko_utama,
    chart_stress_radar, chart_before_after_q3
)
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

st.markdown("""
<h2 style='color:#0F172A; font-weight:800; margin-bottom:4px;'>Analitik & Riset</h2>
<p style='color:#64748b; margin-bottom:20px;'>Eksplorasi mendalam data jurnal, screening, dan analisis lintas dataset.</p>
""", unsafe_allow_html=True)

if not df_all.empty:
    MIN_DATE = df_all["timestamp"].min().date()
    MAX_DATE = df_all["timestamp"].max().date()
    ALL_EMOTIONS = sorted(df_all["label_emosi"].unique().tolist())

    with st.sidebar:
        st.markdown("<hr style='margin: 16px 0 16px 0; border: none; border-top: 1px solid #E2E8F0;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-family:\"Plus Jakarta Sans\", sans-serif; font-size: 1rem; font-weight: 700; color: #0F172A; margin-bottom: 12px;'>Filter Analitik (Jurnal)</div>", unsafe_allow_html=True)
        _dr = st.date_input("Rentang Waktu Jurnal", [MIN_DATE, MAX_DATE], min_value=MIN_DATE, max_value=MAX_DATE)
        _se = st.multiselect("Emosi Spesifik", options=ALL_EMOTIONS, default=ALL_EMOTIONS)

    start_d = _dr[0] if (isinstance(_dr, (list, tuple)) and len(_dr) >= 1) else MIN_DATE
    end_d   = _dr[1] if (isinstance(_dr, (list, tuple)) and len(_dr) == 2) else MAX_DATE
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
        
        with st.expander("Q1 · Proporsi & Distribusi Emosi", expanded=True):
            c1, c2 = st.columns(2)
            with c1: 
                st.plotly_chart(chart_sentiment_pie(df), use_container_width=True, config={'displayModeBar': False})
            with c2: 
                st.plotly_chart(chart_emotion_bar(df), use_container_width=True, config={'displayModeBar': False})
            
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
        
        with st.expander("Q2 · Karakteristik Panjang Tulisan (Word Count)", expanded=False):
            c3, c4 = st.columns(2)
            with c3: 
                st.plotly_chart(chart_boxplot_text_length(df), use_container_width=True, config={'displayModeBar': False})
            with c4: 
                st.plotly_chart(chart_avg_word_count(df), use_container_width=True, config={'displayModeBar': False})
            
            st.markdown("#### Sebaran Kata per Emosi")
            sel_wc_em = st.selectbox("Pilih Emosi untuk Detail Histogram", sel_em, 
                                     format_func=lambda x: f"{EMOTION_LABELS_ID.get(x,x)} ({x})", key="wc_em_selector")
            st.plotly_chart(chart_jurnal_word_count_hist(df, sel_wc_em), use_container_width=True, config={'displayModeBar': False})
            
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
        
        with st.expander("Q3 · Analisis Kata Kunci (Leksikon) & Tema", expanded=False):
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
                st.plotly_chart(chart_top_words(df, sel_wc_em3), use_container_width=True, config={'displayModeBar': False})
                
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
        
        with st.expander("Q1 · Keandalan Sistem & Kesiapan Data", expanded=True):
            c1, c2 = st.columns(2)
            with c1: 
                st.plotly_chart(chart_class_imbalance(), use_container_width=True, config={'displayModeBar': False})
            with c2: 
                st.plotly_chart(chart_model_performance(), use_container_width=True, config={'displayModeBar': False})
                
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
        
        with st.expander("Q2 · 15 Faktor Risiko Utama", expanded=False):
            c3, c4 = st.columns(2)
            with c3: 
                st.plotly_chart(chart_faktor_risiko_utama(), use_container_width=True, config={'displayModeBar': False})
            with c4: 
                st.plotly_chart(chart_stress_radar(df_scr), use_container_width=True, config={'displayModeBar': False})
                
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
        
        with st.expander("Q3 · Efisiensi UX Kuesioner", expanded=False):
            st.plotly_chart(chart_before_after_q3(), use_container_width=True, config={'displayModeBar': False})
            
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
