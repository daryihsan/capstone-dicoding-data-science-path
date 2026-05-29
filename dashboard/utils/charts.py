import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from collections import Counter

from utils.constants import (
    PRIMARY_COLOR, EMOTION_LABELS_ID, EMOTION_COLORMAPS, STRESS_COLS,
    NB_CLASS_NEG, NB_CLASS_POS, NB_RATIO, NB_METRICS
)

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

@st.cache_data(show_spinner=False)
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

@st.cache_data(show_spinner=False)
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

@st.cache_data(show_spinner=False)
def chart_boxplot_text_length(df):
    avg_order = df.groupby("label_emosi")["jumlah_kata"].mean().sort_values(ascending=False).index.tolist()
    fig = px.box(df, x="label_emosi", y="jumlah_kata",
                 color_discrete_sequence=[PRIMARY_COLOR],
                 category_orders={"label_emosi": avg_order},
                 labels={"label_emosi": "Emosi", "jumlah_kata": "Jumlah Kata"})
    _plotly_clean_layout(fig, "Distribusi Panjang Jurnal per Emosi")
    fig.update_layout(showlegend=False)
    return fig

@st.cache_data(show_spinner=False)
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

@st.cache_data(show_spinner=False)
def make_wordcloud_image(df, emotion):
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

@st.cache_data(show_spinner=False)
def _get_top_words_data(df, emotion, top_n=15):
    sub = df[df["label_emosi"] == emotion]
    all_words = " ".join(sub["text_clean"].astype(str)).split()
    stop = {"yang", "dan", "di", "ke", "dari", "ini", "itu", "tidak", "saya",
            "aku", "kamu", "untuk", "dengan", "ada", "atau", "sudah", "juga",
            "sama", "bisa", "mau", "tapi", "kita", "mereka", "saja", "lagi",
            "aja", "ya", "yg", "gue", "gw", "lo", "nya", "ku", "mu"}
    cnt = Counter(w for w in all_words if w not in stop and len(w) > 2)
    return pd.DataFrame(cnt.most_common(top_n), columns=["Kata", "Frekuensi"])

def chart_top_words(df, emotion, top_n=15):
    top = _get_top_words_data(df, emotion, top_n)
    fig = px.bar(top.sort_values("Frekuensi"), x="Frekuensi", y="Kata",
                 orientation="h", color_discrete_sequence=[PRIMARY_COLOR])
    _plotly_clean_layout(fig, f"Top {top_n} Kata — {EMOTION_LABELS_ID.get(emotion, emotion)}")
    return fig

def chart_sleep_category(df_scr):
    s = df_scr["sleep_category"].value_counts()
    fig = px.bar(x=s.index.astype(str), y=s.values,
                 color_discrete_sequence=["#4464AD", "#758EC9", "#95A5A6"],
                 labels={"x": "Kategori Tidur", "y": "Jumlah"})
    _plotly_clean_layout(fig, "Kualitas Tidur")
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

def chart_before_after_q3():
    from utils.constants import NB_AKURASI_FULL, NB_AKURASI_SEL
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

@st.cache_data(show_spinner=False)
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
