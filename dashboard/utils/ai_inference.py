# ============================================================
# RuangRasa — AI Inference
# Mendukung model .keras asli + fallback mock jika tidak tersedia
# ============================================================

import json
import os
import random
import re
import time
from typing import Optional

import numpy as np

from utils.constants import (
    EMOTION_COLORS, EMOTION_LABELS_ID, EMOTION_ORDER,
    EMPATHY_RESPONSES, RECOMMENDATIONS, RISK_COLORS, RISK_LABELS,
)

# ──────────────────────────────────────────────
# LOADER MODEL (dengan graceful fallback)
# ──────────────────────────────────────────────

_emotion_model   = None
_screening_model = None
_tokenizer_obj   = None
_scaler_obj      = None
MODEL_STATUS     = {"emotion": False, "screening": False}


def _get_model_dir() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def load_emotion_model():
    """Memuat model BiLSTM emosi. Return None jika file tidak ada."""
    global _emotion_model, _tokenizer_obj, MODEL_STATUS
    if _emotion_model is not None:
        return _emotion_model
    try:
        import tensorflow as tf
        model_path = os.path.join(_get_model_dir(), "emotion_bilstm.keras")
        tok_path   = os.path.join(_get_model_dir(), "tokenizer.json")
        if not os.path.exists(model_path):
            return None
        _emotion_model = tf.keras.models.load_model(model_path)
        if os.path.exists(tok_path):
            with open(tok_path, "r") as f:
                from tensorflow.keras.preprocessing.text import tokenizer_from_json
                _tokenizer_obj = tokenizer_from_json(f.read())
        MODEL_STATUS["emotion"] = True
        return _emotion_model
    except Exception:
        return None


def load_screening_model():
    """Memuat model DNN screening risiko. Return None jika file tidak ada."""
    global _screening_model, _scaler_obj, MODEL_STATUS
    if _screening_model is not None:
        return _screening_model
    try:
        import pickle
        import tensorflow as tf
        model_path  = os.path.join(_get_model_dir(), "screening_risk.keras")
        scaler_path = os.path.join(_get_model_dir(), "scaler.pkl")
        if not os.path.exists(model_path):
            return None
        _screening_model = tf.keras.models.load_model(model_path)
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                _scaler_obj = pickle.load(f)
        MODEL_STATUS["screening"] = True
        return _screening_model
    except Exception:
        return None


# ──────────────────────────────────────────────
# TEXT PREPROCESSING
# ──────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """Membersihkan teks jurnal: lowercase, hapus karakter non-alfabet, normalisasi spasi."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize_and_pad(text: str, maxlen: int = 100) -> np.ndarray:
    """Tokenisasi dan padding menggunakan tokenizer tersimpan atau fallback sederhana."""
    if _tokenizer_obj is not None:
        seq = _tokenizer_obj.texts_to_sequences([text])
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        return pad_sequences(seq, maxlen=maxlen, padding="post")
    # Fallback: bag-of-words-position sederhana (hanya jika tokenizer tidak ada)
    words  = text.split()[:maxlen]
    idx    = [hash(w) % 10000 + 1 for w in words]
    padded = idx + [0] * (maxlen - len(idx))
    return np.array([padded])


# ──────────────────────────────────────────────
# KEYWORD-BASED MOCK (fallback realistis)
# ──────────────────────────────────────────────

_KEYWORD_MAP = {
    "Anger":   ["marah", "kesal", "jengkel", "benci", "muak", "frustrasi", "dongkol", "emosi", "geram", "sebal"],
    "Sadness": ["sedih", "nangis", "menangis", "hancur", "kecewa", "patah", "putus asa", "murung", "galau", "sendu"],
    "Joy":     ["senang", "bahagia", "gembira", "seru", "asik", "happy", "lega", "puas", "alhamdulillah", "syukur"],
    "Love":    ["cinta", "sayang", "rindu", "kangen", "suka", "kasih", "menyayangi", "mencintai", "sahabat", "bersama"],
    "Fear":    ["takut", "khawatir", "cemas", "panik", "was-was", "gelisah", "anxiety", "tegang", "gugup", "waswas"],
    "Neutral": [],
}

_SENTIMENT_MAP = {"Anger": "Negative", "Sadness": "Negative", "Fear": "Negative",
                  "Joy": "Positive",  "Love": "Positive",  "Neutral": "Neutral"}


def _mock_predict_emotion(text: str) -> dict:
    """Prediksi emosi berbasis keyword matching (mock mode)."""
    text_lower = text.lower()
    scores = {}
    for emotion, keywords in _KEYWORD_MAP.items():
        hit = sum(1 for kw in keywords if kw in text_lower)
        scores[emotion] = hit

    total = sum(scores.values())
    if total == 0:
        # Default ke Neutral
        probs = {e: (0.85 if e == "Neutral" else random.uniform(0.01, 0.06)) for e in EMOTION_ORDER}
    else:
        # Distribusi berdasarkan hit, ditambah noise
        probs = {}
        for e in EMOTION_ORDER:
            base = scores[e] / total if total > 0 else 0
            probs[e] = max(0.01, base + random.uniform(-0.03, 0.05))
        # Normalisasi
        s = sum(probs.values())
        probs = {k: v / s for k, v in probs.items()}

    dominant = max(probs, key=probs.get)
    sentiment = _SENTIMENT_MAP[dominant]
    confidence = probs[dominant]

    return {
        "emotion":     dominant,
        "emotion_id":  EMOTION_LABELS_ID[dominant],
        "sentiment":   sentiment,
        "confidence":  round(confidence * 100, 1),
        "probabilities": {e: round(probs[e] * 100, 1) for e in EMOTION_ORDER},
        "validation":  random.choice(EMPATHY_RESPONSES[dominant]),
        "recommendations": RECOMMENDATIONS[dominant],
        "model_used":  "Mock (Keyword-based)",
        "inference_ms": random.randint(50, 200),
        "is_mock":     True,
    }


# ──────────────────────────────────────────────
# FUNGSI UTAMA: PREDIKSI EMOSI
# ──────────────────────────────────────────────

def predict_emotion(text: str) -> dict:
    """
    Prediksi emosi dari teks jurnal.

    Args:
        text: Teks jurnal bebas (Bahasa Indonesia)

    Returns:
        dict berisi emotion, sentiment, confidence, probabilities, validation, recommendations
    """
    if not text or len(text.strip()) < 5:
        return {"error": "Teks terlalu pendek untuk dianalisis."}

    cleaned = preprocess_text(text)
    model   = load_emotion_model()

    # ── Inferensi model nyata ──
    if model is not None:
        try:
            t0  = time.time()
            seq = _tokenize_and_pad(cleaned)
            raw = model.predict(seq, verbose=0)[0]
            ms  = int((time.time() - t0) * 1000)

            probs = {e: float(raw[i]) for i, e in enumerate(EMOTION_ORDER)}
            dominant   = max(probs, key=probs.get)
            confidence = probs[dominant]
            sentiment  = _SENTIMENT_MAP[dominant]

            return {
                "emotion":     dominant,
                "emotion_id":  EMOTION_LABELS_ID[dominant],
                "sentiment":   sentiment,
                "confidence":  round(confidence * 100, 1),
                "probabilities": {e: round(probs[e] * 100, 1) for e in EMOTION_ORDER},
                "validation":  random.choice(EMPATHY_RESPONSES[dominant]),
                "recommendations": RECOMMENDATIONS[dominant],
                "model_used":  "emotion_bilstm.keras",
                "inference_ms": ms,
                "is_mock":     False,
            }
        except Exception as e:
            pass  # Jatuh ke mock

    # ── Fallback mock ──
    return _mock_predict_emotion(cleaned)


# ──────────────────────────────────────────────
# RISK SCORING (mock berbasis skor rata-rata)
# ──────────────────────────────────────────────

def _compute_risk_score(answers: dict) -> float:
    """Hitung risk score sederhana dari jawaban kuesioner (0–10)."""
    stress_keys  = ["Work_Stress_Level", "Financial_Stress", "Mood_Swings", "Loneliness"]
    symptom_keys = ["Feeling_Sad_Down", "Anxious_Nervous"]
    lifestyle_keys = ["Sleep_Hours_Night"]  # rendah = lebih buruk

    s  = np.mean([answers.get(k, 5) for k in stress_keys])
    sy = np.mean([answers.get(k, 5) for k in symptom_keys])
    sleep = answers.get("Sleep_Hours_Night", 7)
    sleep_penalty = max(0, (7 - sleep)) * 0.5  # kurang tidur menambah risiko
    trauma = 1.5 if answers.get("Trauma_History", 0) == 1 else 0
    diagnosed = 1.0 if answers.get("Previously_Diagnosed", 0) == 1 else 0

    raw = (s * 0.4 + sy * 0.3 + sleep_penalty + trauma + diagnosed)
    return min(10.0, max(0.0, raw))


def _score_to_level(score: float) -> str:
    if score < 4.0:
        return "Low"
    elif score < 7.0:
        return "Medium"
    return "High"


def _mock_predict_risk(answers: dict) -> dict:
    """Prediksi risiko berbasis aturan (mock mode)."""
    t0    = time.time()
    score = _compute_risk_score(answers)
    level = _score_to_level(score)
    ms    = int((time.time() - t0) * 1000) + random.randint(100, 500)

    breakdown = {
        "Gejala Psikologis": min(10, (answers.get("Feeling_Sad_Down", 5) + answers.get("Anxious_Nervous", 5)) / 2),
        "Stres Kerja & Ekonomi": min(10, (answers.get("Work_Stress_Level", 5) + answers.get("Financial_Stress", 5)) / 2),
        "Kualitas Tidur": max(0, 10 - answers.get("Sleep_Hours_Night", 7)),
        "Kesepian & Dukungan": min(10, (answers.get("Loneliness", 5) + (10 - answers.get("Social_Support", 5))) / 2),
        "Riwayat Kesehatan": (answers.get("Trauma_History", 0) + answers.get("Previously_Diagnosed", 0)) * 5,
    }

    recs = {
        "Low":    ["Pertahankan rutinitas sehat yang sudah berjalan.", "Luangkan waktu untuk aktivitas yang menyenangkan.", "Tetap jaga koneksi sosial dengan orang terdekat."],
        "Medium": ["Pertimbangkan konsultasi dengan konselor atau psikolog.", "Prioritaskan kualitas tidur minimal 7 jam per malam.", "Batasi paparan media sosial dan berita negatif.", "Praktikkan mindfulness atau meditasi singkat harian."],
        "High":   ["Sangat disarankan untuk segera berkonsultasi dengan profesional kesehatan mental.", "Ceritakan kondisimu kepada orang yang kamu percaya.", "Hubungi hotline kesehatan mental jika membutuhkan bantuan segera.", "Kurangi paparan stresor eksternal sebisa mungkin."],
    }

    return {
        "risk_level":    level,
        "risk_level_id": RISK_LABELS[level],
        "risk_score":    round(score, 2),
        "confidence":    round(random.uniform(78, 96), 1),
        "breakdown":     {k: round(v, 2) for k, v in breakdown.items()},
        "recommendations": recs[level],
        "model_used":    "Mock (Rule-based)",
        "inference_ms":  ms,
        "is_mock":       True,
    }


def predict_risk(answers: dict) -> dict:
    """
    Prediksi level risiko kesehatan mental dari jawaban kuesioner.

    Args:
        answers: dict berisi 15 jawaban pertanyaan screening

    Returns:
        dict berisi risk_level, risk_score, breakdown, recommendations
    """
    model = load_screening_model()

    if model is not None:
        try:
            import pandas as pd
            t0 = time.time()
            features = [
                "Age", "Work_Stress_Level", "Financial_Stress", "Mood_Swings",
                "Loneliness", "Sleep_Hours_Night", "Screen_Time_Hours_Day",
                "Social_Media_Hours_Day", "Work_Hours_Per_Week",
                "Previously_Diagnosed", "Trauma_History",
                "Feeling_Sad_Down", "Anxious_Nervous", "Social_Support",
            ]
            gender_map = {"Male": 0, "Female": 1, "Other": 2}
            X_raw = [answers.get(f, 0) for f in features]
            # encode gender
            X_raw.insert(1, gender_map.get(answers.get("Gender", "Male"), 0))
            X = np.array([X_raw], dtype=float)

            if _scaler_obj is not None:
                X = _scaler_obj.transform(X)

            raw  = model.predict(X, verbose=0)[0]
            ms   = int((time.time() - t0) * 1000)
            level_idx = int(np.argmax(raw))
            level = ["Low", "Medium", "High"][level_idx % 3]
            score = _compute_risk_score(answers)

            breakdown = {
                "Gejala Psikologis": (answers.get("Feeling_Sad_Down", 5) + answers.get("Anxious_Nervous", 5)) / 2,
                "Stres Kerja & Ekonomi": (answers.get("Work_Stress_Level", 5) + answers.get("Financial_Stress", 5)) / 2,
                "Kualitas Tidur": max(0, 10 - answers.get("Sleep_Hours_Night", 7)),
                "Kesepian & Dukungan": (answers.get("Loneliness", 5) + (10 - answers.get("Social_Support", 5))) / 2,
                "Riwayat Kesehatan": (answers.get("Trauma_History", 0) + answers.get("Previously_Diagnosed", 0)) * 5,
            }

            recs = {"Low": ["Pertahankan gaya hidup sehat.", "Jaga koneksi sosial."],
                    "Medium": ["Pertimbangkan konsultasi psikolog.", "Prioritaskan tidur & istirahat."],
                    "High": ["Segera konsultasi profesional kesehatan mental.", "Hubungi hotline jika perlu."]}

            return {
                "risk_level":    level,
                "risk_level_id": RISK_LABELS[level],
                "risk_score":    round(score, 2),
                "confidence":    round(float(raw[level_idx % 3]) * 100, 1),
                "breakdown":     {k: round(v, 2) for k, v in breakdown.items()},
                "recommendations": recs[level],
                "model_used":    "screening_risk.keras",
                "inference_ms":  ms,
                "is_mock":       False,
            }
        except Exception:
            pass

    return _mock_predict_risk(answers)
