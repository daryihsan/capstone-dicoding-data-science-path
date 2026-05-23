# ============================================================
# RuangRasa — Data Loader
# ============================================================

import os
import pandas as pd
import streamlit as st

from utils.constants import STRESS_COLS


def _data_dir() -> str:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Coba folder data/ dulu, lalu folder yang sama dengan dashboard.py
    for candidate in [os.path.join(base, "data"), base]:
        if os.path.isdir(candidate):
            return candidate
    return base


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Memuat dataset jurnal emosi."""
    path = os.path.join(_data_dir(), "dataset_ruangrasa_jurnal_ai.csv")
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["jumlah_kata"] = df["text_clean"].astype(str).apply(lambda x: len(x.split()))
    emotion_map = {
        "Sadness": "Negative", "Anger": "Negative", "Fear": "Negative",
        "Joy": "Positive", "Love": "Positive", "Neutral": "Neutral",
    }
    df["emotion_group"] = df["label_emosi"].map(emotion_map)
    df["hour"]       = df["timestamp"].dt.hour
    df["day_of_week"]= df["timestamp"].dt.day_name()
    df["date"]       = df["timestamp"].dt.date
    return df


@st.cache_data(show_spinner=False)
def load_screening_data() -> pd.DataFrame:
    """Memuat dataset screening wellbeing."""
    path = os.path.join(_data_dir(), "final_screening_dataset.csv")
    df = pd.read_csv(path, sep=";")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["risk_score"] = df[list(STRESS_COLS.keys())].mean(axis=1)
    df["age_group"] = pd.cut(
        df["Age"], bins=[17, 24, 34, 44, 54, 100],
        labels=["18–24", "25–34", "35–44", "45–54", "55+"],
    )
    df["sleep_category"] = pd.cut(
        df["Sleep_Hours_Night"], bins=[0, 5.9, 7.9, 99],
        labels=["Kurang (<6j)", "Cukup (6–8j)", "Lebih (>8j)"],
    )
    return df
