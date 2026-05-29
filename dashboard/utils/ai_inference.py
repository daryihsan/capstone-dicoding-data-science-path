# ============================================================
# RuangRasa — AI Inference
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

import streamlit as st


# ──────────────────────────────────────────────
# LOADER MODEL (dengan graceful fallback)
# ──────────────────────────────────────────────

try:
    import keras
    import tensorflow as tf
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False
    keras = None
    tf = None

# ──────────────────────────────────────────────
# CUSTOM LAYERS (hanya jika TensorFlow tersedia)
# ──────────────────────────────────────────────
if _TF_AVAILABLE:
    @keras.utils.register_keras_serializable(package="RuangRasa")
    class FeatureAttention(keras.layers.Layer):
        def __init__(self, units=16, **kwargs):
            super(FeatureAttention, self).__init__(**kwargs)
            self.units = units
            self.dense = keras.layers.Dense(self.units, activation="relu", name="dense")
            self.score = keras.layers.Dense(1, activation="sigmoid", name="score")

        def build(self, input_shape):
            self.dense.build((None, 1))
            self.score.build((None, self.units))
            super(FeatureAttention, self).build(input_shape)

        def call(self, inputs):
            expanded = tf.expand_dims(inputs, axis=-1)
            h = self.dense(expanded)
            a = self.score(h)
            a = tf.squeeze(a, axis=-1)
            return inputs * a

        def get_config(self):
            config = super(FeatureAttention, self).get_config()
            config.update({"units": self.units})
            return config

    @keras.utils.register_keras_serializable(package="RuangRasa")
    class TemporalContextAttention(keras.layers.Layer):
        """
        Custom gating attention layer dari notebook 02_journaling_model.
        Menggabungkan text repr dan context features dengan learned gate weight.
        """
        def __init__(self, units: int = 64, **kwargs):
            super().__init__(**kwargs)
            self.units  = units
            self.W_text = keras.layers.Dense(units, use_bias=True)
            self.W_ctx  = keras.layers.Dense(units, use_bias=True)
            self.V      = keras.layers.Dense(1, use_bias=True)

        def call(self, text_repr, context_repr):
            text_proj = self.W_text(text_repr)
            ctx_proj  = self.W_ctx(context_repr)
            combined  = tf.nn.tanh(text_proj + ctx_proj)
            score     = self.V(combined)
            gate      = tf.nn.sigmoid(score)
            text_out  = self.W_text(text_repr)
            ctx_out   = self.W_ctx(context_repr)
            return gate * text_out + (1 - gate) * ctx_out

        def get_config(self):
            config = super().get_config()
            config.update({"units": self.units})
            return config
else:
    # Placeholder classes agar referensi tidak error
    FeatureAttention = None
    TemporalContextAttention = None

_emotion_model   = None
_screening_model = None
_tokenizer_obj   = None
_scaler_obj      = None
MODEL_STATUS     = {"emotion": False, "screening": False}

def _get_model_dir() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

@st.cache_resource(show_spinner=False)
def _cached_load_emotion_model():
    model, tokenizer = None, None
    status = False
    if not _TF_AVAILABLE:
        return model, tokenizer, status
    try:
        import json as _json
        model_dir = os.path.join(_get_model_dir(), "jurnaling_model")
        config_path = os.path.join(model_dir, "config.json")
        weights_path = os.path.join(model_dir, "model.weights.h5")

        if not os.path.exists(config_path) or not os.path.exists(weights_path):
            old_path = os.path.join(_get_model_dir(), "emotion_bilstm.keras")
            if os.path.exists(old_path):
                model = keras.models.load_model(old_path)
                status = True
                return model, tokenizer, status
            return model, tokenizer, status

        with open(config_path, "r", encoding="utf-8") as f:
            config = _json.load(f)

        custom_objects = {
            "FeatureAttention": FeatureAttention,
            "TemporalContextAttention": TemporalContextAttention,
        }
        model = keras.models.model_from_json(
            _json.dumps(config), custom_objects=custom_objects
        )
        model.load_weights(weights_path)
        status = True

        tok_path = os.path.join(model_dir, "journaling_tokenizer.pkl")
        if os.path.exists(tok_path):
            import pickle as _pickle
            with open(tok_path, "rb") as f:
                tokenizer = _pickle.load(f)
        else:
            meta_path = os.path.join(model_dir, "metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = _json.load(f)
                tok_path_meta = meta.get("tokenizer_path", "")
                if tok_path_meta and os.path.exists(tok_path_meta):
                    from tensorflow.keras.preprocessing.text import tokenizer_from_json
                    with open(tok_path_meta, "r", encoding="utf-8") as f:
                        tokenizer = tokenizer_from_json(f.read())
        return model, tokenizer, status
    except Exception:
        return model, tokenizer, status

def load_emotion_model():
    global _emotion_model, _tokenizer_obj, MODEL_STATUS
    if _emotion_model is not None:
        return _emotion_model
    model, tokenizer, status = _cached_load_emotion_model()
    _emotion_model = model
    _tokenizer_obj = tokenizer
    MODEL_STATUS["emotion"] = status
    return _emotion_model

@st.cache_resource(show_spinner=False)
def _cached_load_screening_model():
    model, scaler = None, None
    status = False
    if not _TF_AVAILABLE:
        return model, scaler, status
    try:
        import pickle
        model_path = os.path.join(_get_model_dir(), "screening_model")
        if not os.path.exists(model_path):
            model_path = os.path.join(_get_model_dir(), "screening_risk.keras")
            
        scaler_path = os.path.join(_get_model_dir(), "scaler.pkl")
        
        if not os.path.exists(model_path):
            return model, scaler, status
            
        model = keras.models.load_model(model_path, custom_objects={"FeatureAttention": FeatureAttention})
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
        status = True
        return model, scaler, status
    except Exception:
        return model, scaler, status

def load_screening_model():
    global _screening_model, _scaler_obj, MODEL_STATUS
    if _screening_model is not None:
        return _screening_model
    model, scaler, status = _cached_load_screening_model()
    _screening_model = model
    _scaler_obj = scaler
    MODEL_STATUS["screening"] = status
    return _screening_model

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
    if _TF_AVAILABLE and _tokenizer_obj is not None:
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
# INTENSITY KEYWORDS
# ──────────────────────────────────────────────
_INTENSIFIER_NEGATIVE = {
    "banget", "sangat", "sekali", "amat", "parah", "bgt",
    "habis", "hancur", "benar-benar", "betul-betul",
    "sungguh", "luar biasa", "teramat", "nangis",
    "menangis", "putus asa", "hopeless", "menyerah",
}
_INTENSIFIER_POSITIVE = {
    "banget", "sangat", "sekali", "amat", "bahagia",
    "senang", "gembira", "bersyukur", "luar biasa",
    "lega", "semangat", "bersemangat", "sungguh",
}
_DIMINISHER = {
    "sedikit", "agak", "lumayan", "cukup", "kayaknya",
    "kayak", "sepertinya", "mungkin", "rada", "biasa",
}
_NEGATION_DETECT = {"tidak", "bukan", "belum", "jangan", "ga", "gak", "nggak", "ngga", "enggak", "tak"}
_NEGATIVE_EMOTIONS = {"Sadness", "Fear", "Anger"}
_POSITIVE_EMOTIONS = {"Joy", "Love"}

def _compute_emotion_intensity(
    raw_text: str, emotion: str, model_confidence: float,
    sentiment_score: float, text_length: int, journal_hour: int
) -> dict:
    """Hitung intensity emosi dinamis."""
    words_raw = raw_text.lower().split()
    base = model_confidence

    if emotion in _NEGATIVE_EMOTIONS:
        sentiment_modifier = 1.0 - sentiment_score
    elif emotion in _POSITIVE_EMOTIONS:
        sentiment_modifier = sentiment_score
    else:
        sentiment_modifier = abs(sentiment_score - 0.5) * 2

    has_negation = any(w in _NEGATION_DETECT for w in words_raw)
    negation_penalty = 0.3 if has_negation else 0.0

    intensifier_set = _INTENSIFIER_POSITIVE if emotion in _POSITIVE_EMOTIONS else _INTENSIFIER_NEGATIVE
    n_intensifier = sum(1 for w in words_raw if w in intensifier_set)
    n_diminisher  = sum(1 for w in words_raw if w in _DIMINISHER)
    keyword_modifier = min(n_intensifier * 0.05, 0.25) - min(n_diminisher * 0.05, 0.25)

    if text_length >= 30:   length_modifier = 0.08
    elif text_length >= 15: length_modifier = 0.04
    elif text_length <= 5:  length_modifier = -0.05
    else:                   length_modifier = 0.0

    time_modifier = 0.05 if 0 <= journal_hour <= 4 else 0.0

    raw_intensity = (
        0.45 * base + 0.35 * sentiment_modifier
        + keyword_modifier + length_modifier + time_modifier - negation_penalty
    )
    intensity_score = max(0.0, min(1.0, raw_intensity))

    if intensity_score < 0.25:   label = "ringan"
    elif intensity_score < 0.50: label = "sedang"
    elif intensity_score < 0.75: label = "berat"
    else:                        label = "sangat berat"

    return {"intensity_score": round(intensity_score, 3), "intensity_label": label}

def predict_emotion(text: str) -> dict:
    """
    Prediksi emosi dari teks jurnal menggunakan model BiLSTM multitask.

    Args:
        text: Teks jurnal bebas (Bahasa Indonesia)

    Returns:
        dict berisi emotion, sentiment, confidence, probabilities,
        intensity, validation, recommendations
    """
    if not text or len(text.strip()) < 5:
        return {"error": "Teks terlalu pendek untuk dianalisis."}

    cleaned = preprocess_text(text)
    model   = load_emotion_model()

    # ── Inferensi model BiLSTM multitask ──
    if model is not None:
        try:
            import datetime
            t0  = time.time()
            seq = _tokenize_and_pad(cleaned, maxlen=80)  # MAX_LEN dari notebook = 80
            
            # Construct extra_input
            text_length = len(text.split())
            now_dt = datetime.datetime.now()
            extra_raw = np.array([[
                now_dt.hour,
                now_dt.weekday(),
                now_dt.month,
                1 if (now_dt.hour >= 21 or now_dt.hour <= 5) else 0,
                1 if now_dt.weekday() >= 5 else 0,
                text_length
            ]], dtype="float32")

            # Load scaler_extra
            model_dir = os.path.join(_get_model_dir(), "jurnaling_model")
            scaler_path = os.path.join(model_dir, "journaling_scaler_extra.pkl")
            if os.path.exists(scaler_path):
                import pickle as _pickle
                with open(scaler_path, "rb") as f:
                    scaler_extra = _pickle.load(f)
                X_extra = scaler_extra.transform(extra_raw)
            else:
                X_extra = extra_raw

            raw_output = model.predict(
                {"text_input": seq, "extra_input": X_extra},
                verbose=0
            )
            ms  = int((time.time() - t0) * 1000)

            # Multitask model output: [emotion_output, sentiment_output]
            if isinstance(raw_output, dict):
                emotion_raw    = raw_output["emotion_output"][0]
                sentiment_raw  = float(raw_output["sentiment_output"][0][0])
            elif isinstance(raw_output, (list, tuple)) and len(raw_output) >= 2:
                emotion_raw    = raw_output[0][0]   # shape (6,) — 6 emosi
                sentiment_raw  = float(raw_output[1][0][0]) if raw_output[1].ndim >= 2 else float(raw_output[1][0])
            else:
                # Single output (backward compat)
                emotion_raw   = raw_output[0] if raw_output.ndim == 2 else raw_output
                sentiment_raw = 0.5  # default neutral

            EMOSI_MAP_INV = {
                0: "Anger",
                1: "Fear",
                2: "Joy",
                3: "Love",
                4: "Neutral",
                5: "Sadness",
            }
            probs      = {EMOSI_MAP_INV[i]: float(emotion_raw[i]) for i in range(6)}
            dominant   = max(probs, key=probs.get)
            confidence = probs[dominant]
            sentiment  = _SENTIMENT_MAP[dominant]

            # Hitung intensit
            journal_hour = now_dt.hour
            intensity    = _compute_emotion_intensity(
                raw_text=text, emotion=dominant,
                model_confidence=confidence,
                sentiment_score=sentiment_raw,
                text_length=text_length,
                journal_hour=journal_hour,
            )

            return {
                "emotion":      dominant,
                "emotion_id":   EMOTION_LABELS_ID[dominant],
                "sentiment":    sentiment,
                "confidence":   round(confidence * 100, 1),
                "probabilities": {e: round(probs[e] * 100, 1) for e in EMOTION_ORDER},
                "intensity":    intensity,
                "validation":   random.choice(EMPATHY_RESPONSES[dominant]),
                "recommendations": RECOMMENDATIONS[dominant],
                "model_used":   "jurnaling_model (BiLSTM Multitask)",
                "inference_ms": ms,
                "is_mock":      False,
            }
        except Exception:
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
    trauma = 1.5 if answers.get("Trauma_History", 0) >= 7 else 0
    diagnosed = 1.0 if answers.get("Previously_Diagnosed", 0) >= 7 else 0

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
        "High":   ["Sangat disarankan untuk segera berkonsultasi dengan profesional kesehatan mental.", "Ceritakan kondisimu kepada orang yang kamu percaya.", "Kurangi paparan stresor eksternal sebisa mungkin."],
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
        answers: dict berisi jawaban kuesioner screening

    Returns:
        dict berisi risk_level, risk_score, breakdown, recommendations
    """
    model = load_screening_model()

    if model is not None:
        try:
            import pandas as pd
            t0 = time.time()
            
            # Cek apakah model yang dimuat adalah model multitask fungsional baru
            is_multitask = False
            if hasattr(model, "inputs") and model.inputs:
                input_names = [getattr(inp, "name", "") for inp in model.inputs]
                if any("sleep_risk" in name for name in input_names):
                    is_multitask = True

            if is_multitask:
                # ── Inferensi model multitask functional (.keras folder / SavedModel) ──
                # Ambil rating slider 1-10 dari answers
                def get_rating(key, default=5):
                    return float(answers.get(key, default))
                
                age_val = float(answers.get("Age", 22))
                gender_str = str(answers.get("Gender", "Male"))
                
                sleep_rating = get_rating("Sleep_Hours_Night", 7)
                screen_rating = get_rating("Screen_Time_Hours_Day", 5)
                social_rating = get_rating("Social_Media_Hours_Day", 3)
                work_rating = get_rating("Work_Hours_Per_Week", 5)
                
                # Petakan rating 1-10 kembali ke range data latihan (raw scale)
                sleep_val = 3.0 + (sleep_rating - 1.0) * 8.0 / 9.0
                screen_val = 1.0 + (screen_rating - 1.0) * 15.0 / 9.0
                socmed_val = 0.0 + (social_rating - 1.0) * 12.0 / 9.0
                work_val = 0.0 + (work_rating - 1.0) * 74.0 / 9.0
                
                # Threshold >= 7 menghindari slider 5 (default) langsung set binary=1
                # Secara klinis: 7-10 = ada dampak signifikan ("sering" / "berdampak besar")
                trauma_val = 1.0 if get_rating("Trauma_History", 0) >= 7.0 else 0.0
                diagnosed_val = 1.0 if get_rating("Previously_Diagnosed", 0) >= 7.0 else 0.0
                
                work_stress_val = get_rating("Work_Stress_Level", 5)
                financial_val = get_rating("Financial_Stress", 5)
                mood_swings_val = get_rating("Mood_Swings", 5)
                loneliness_val = get_rating("Loneliness", 5)
                
                # Hitung derived risk features — HARUS SESUAI notebook add_engineered_features_to_single_user
                # work_stress_risk = (Work_Stress_Level - 1) / 9
                # financial_risk = (Financial_Stress - 1) / 9
                # mood_swings_risk = Mood_Swings / 10
                # loneliness_risk = (Loneliness - 1) / 9
                # sleep_risk: piecewise — < 7 jam → (7 - x) / 4, > 9 jam → (x - 9) / 2, sinyal 0
                # screen_time_risk = max(0, (x - 6) / 10)
                # social_media_risk = max(0, (x - 3) / 9)
                # work_hours_risk = max(0, (x - 40) / 34)
                
                if sleep_val < 7.0:
                    sleep_r = min((7.0 - sleep_val) / 4.0, 1.0)
                elif sleep_val > 9.0:
                    sleep_r = min((sleep_val - 9.0) / 2.0, 1.0)
                else:
                    sleep_r = 0.0
                
                screen_r = min(max((screen_val - 6.0) / 10.0, 0.0), 1.0)
                social_r = min(max((socmed_val - 3.0) / 9.0, 0.0), 1.0)
                work_r   = min(max((work_val - 40.0) / 34.0, 0.0), 1.0)
                
                work_stress_r = (work_stress_val - 1.0) / 9.0
                financial_r   = (financial_val - 1.0) / 9.0
                mood_swings_r = mood_swings_val / 10.0
                loneliness_r  = (loneliness_val - 1.0) / 9.0

                # Derivasikan fitur berbasis waktu pengisian
                now = pd.Timestamp.now()
                scr_hour = float(now.hour)
                scr_dayofweek = float(now.dayofweek)
                scr_month = float(now.month)
                
                # Susun input dictionary dengan format tensor/numpy 
                inputs_dict = {
                    "Gender": tf.constant([[gender_str]], dtype=tf.string),
                    "Age": np.array([[age_val]], dtype=np.float32),
                    "Sleep_Hours_Night": np.array([[sleep_val]], dtype=np.float32),
                    "Screen_Time_Hours_Day": np.array([[screen_val]], dtype=np.float32),
                    "Social_Media_Hours_Day": np.array([[socmed_val]], dtype=np.float32),
                    "Trauma_History": np.array([[trauma_val]], dtype=np.float32),
                    "Previously_Diagnosed": np.array([[diagnosed_val]], dtype=np.float32),
                    "Work_Hours_Per_Week": np.array([[work_val]], dtype=np.float32),
                    "Work_Stress_Level": np.array([[work_stress_val]], dtype=np.float32),
                    "Financial_Stress": np.array([[financial_val]], dtype=np.float32),
                    "Mood_Swings": np.array([[mood_swings_val]], dtype=np.float32),
                    "Loneliness": np.array([[loneliness_val]], dtype=np.float32),
                    "screening_hour": np.array([[scr_hour]], dtype=np.float32),
                    "screening_dayofweek": np.array([[scr_dayofweek]], dtype=np.float32),
                    "screening_month": np.array([[scr_month]], dtype=np.float32),
                    "sleep_risk": np.array([[float(sleep_r)]], dtype=np.float32),
                    "screen_time_risk": np.array([[float(screen_r)]], dtype=np.float32),
                    "social_media_risk": np.array([[float(social_r)]], dtype=np.float32),
                    "work_hours_risk": np.array([[float(work_r)]], dtype=np.float32),
                    "work_stress_risk": np.array([[float(work_stress_r)]], dtype=np.float32),
                    "financial_risk": np.array([[float(financial_r)]], dtype=np.float32),
                    "mood_swings_risk": np.array([[float(mood_swings_r)]], dtype=np.float32),
                    "loneliness_risk": np.array([[float(loneliness_r)]], dtype=np.float32),
                }
                
                # Jalankan inferensi model
                outputs = model.predict(inputs_dict, verbose=0)
                ms = int((time.time() - t0) * 1000)
                
                risk_class_probs = outputs["risk_class"][0]
                raw_score = float(outputs["risk_score"][0][0])
                score = raw_score * 10.0
                
                # Gunakan risk_score (regresi) bukan argmax(risk_class_probs) yang selalu saturated
                # risk_class_probs selalu mendekati [1,0,0] atau [0,0,1], tidak pernah Medium
                if raw_score < 0.30:
                    level = "Low"
                    # Confidence: 99% saat score=0, turun ke ~45% saat mendekati batas 0.30
                    t = 1.0 - (raw_score / 0.30)
                    confidence = 45.0 + 54.0 * t
                elif raw_score < 0.65:
                    level = "Medium"
                    # Confidence: puncak ~80% di tengah (0.475), turun di pinggir
                    t = 1.0 - abs(raw_score - 0.475) / 0.175
                    confidence = 40.0 + 40.0 * t
                else:
                    level = "High"
                    # Confidence: ~45% saat baru lewat batas 0.65, naik ke 99% saat score=1.0
                    t = (raw_score - 0.65) / 0.35
                    confidence = 45.0 + 54.0 * t
                confidence = round(max(40.0, min(99.0, confidence)), 1)
                
                # Hitung breakdown pada skala 0-10 untuk radar chart
                sad_val = get_rating("Feeling_Sad_Down", 5)
                anxious_val = get_rating("Anxious_Nervous", 5)
                support_val = get_rating("Social_Support", 6)
                
                breakdown = {
                    "Gejala Psikologis": (sad_val + anxious_val) / 2.0,
                    "Stres Kerja & Ekonomi": (work_stress_val + financial_val) / 2.0,
                    "Kualitas Tidur": 10.0 - sleep_rating,
                    "Kesepian & Dukungan": (loneliness_val + (11.0 - support_val)) / 2.0,
                    "Riwayat Kesehatan": (get_rating("Trauma_History", 0) + get_rating("Previously_Diagnosed", 0)) / 2.0,
                }
                
                recs = {
                    "Low": ["Pertahankan gaya hidup sehat yang seimbang.", "Jaga koneksi sosial dengan keluarga dan teman."],
                    "Medium": ["Pertimbangkan untuk berkonsultasi dengan konselor.", "Prioritaskan tidur yang cukup dan batasi screen time.", "Praktikkan teknik relaksasi atau journaling."],
                    "High": ["Sangat disarankan untuk segera berkonsultasi dengan profesional kesehatan mental.", "Ceritakan kondisimu kepada orang terdekat."]
                }
                
                return {
                    "risk_level": level,
                    "risk_level_id": RISK_LABELS[level],
                    "risk_score": round(score, 2),
                    "confidence": round(confidence, 1),
                    "breakdown": {k: round(v, 2) for k, v in breakdown.items()},
                    "recommendations": recs[level],
                    "model_used": "screening_model (Functional DNN)",
                    "inference_ms": ms,
                    "is_mock": False,
                }
            
            else:
                # ── Fallback ke model sequential lama (.keras file tunggal) ──
                features = [
                    "Age", "Work_Stress_Level", "Financial_Stress", "Mood_Swings",
                    "Loneliness", "Sleep_Hours_Night", "Screen_Time_Hours_Day",
                    "Social_Media_Hours_Day", "Work_Hours_Per_Week",
                    "Previously_Diagnosed", "Trauma_History",
                    "Feeling_Sad_Down", "Anxious_Nervous", "Social_Support",
                ]
                gender_map = {"Male": 0, "Female": 1, "Other": 2}
                X_raw = [answers.get(f, 0) for f in features]
                X_raw.insert(1, gender_map.get(answers.get("Gender", "Male"), 0))
                X = np.array([X_raw], dtype=float)

                if _scaler_obj is not None:
                    X = _scaler_obj.transform(X)

                raw = model.predict(X, verbose=0)[0]
                ms = int((time.time() - t0) * 1000)
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
                        "High": ["Segera konsultasi profesional kesehatan mental."]}

                return {
                    "risk_level": level,
                    "risk_level_id": RISK_LABELS[level],
                    "risk_score": round(score, 2),
                    "confidence": round(float(raw[level_idx % 3]) * 100, 1),
                    "breakdown": {k: round(v, 2) for k, v in breakdown.items()},
                    "recommendations": recs[level],
                    "model_used": "screening_risk.keras",
                    "inference_ms": ms,
                    "is_mock": False,
                }
        except Exception:
            pass

    return _mock_predict_risk(answers)