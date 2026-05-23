# ============================================================
# RuangRasa — Konstanta Global
# ============================================================

# Brand Colors
PRIMARY_COLOR   = "#4464AD"
SECONDARY_COLOR = "#2E6DB4"
ACCENT_COLOR    = "#E74C3C"
SUCCESS_COLOR   = "#27AE60"
WARNING_COLOR   = "#F4D03F"
BG_DARK         = "#0F1923"
BG_CARD         = "#1A2535"

# Emotion Config
EMOTION_COLORS = {
    "Anger":   "#E74C3C",
    "Sadness": "#5DADE2",
    "Joy":     "#F4D03F",
    "Love":    "#E91E63",
    "Fear":    "#9B59B6",
    "Neutral": "#95A5A6",
}

EMOTION_ORDER = ["Anger", "Sadness", "Fear", "Neutral", "Joy", "Love"]

EMOTION_LABELS_ID = {
    "Anger":   "Marah",
    "Sadness": "Sedih",
    "Joy":     "Senang",
    "Love":    "Cinta",
    "Fear":    "Takut",
    "Neutral": "Netral",
}

EMOTION_EMOJI = {
    "Anger":   "😠",
    "Sadness": "😢",
    "Joy":     "😊",
    "Love":    "❤️",
    "Fear":    "😰",
    "Neutral": "😐",
}

EMOTION_COLORMAPS = {
    "Anger":   "Reds",
    "Sadness": "Blues",
    "Joy":     "YlOrRd",
    "Love":    "RdPu",
    "Fear":    "Purples",
    "Neutral": "Greys",
}

SENTIMENT_COLORS = {
    "Negative": "#E74C3C",
    "Positive": "#27AE60",
    "Neutral":  "#95A5A6",
}

RISK_COLORS = {
    "Low":    "#27AE60",
    "Medium": "#F4D03F",
    "High":   "#E74C3C",
}

RISK_LABELS = {
    "Low":    "Rendah",
    "Medium": "Sedang",
    "High":   "Tinggi",
}

# Hari
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAY_ORDER_ID = {
    "Monday":    "Senin",
    "Tuesday":   "Selasa",
    "Wednesday": "Rabu",
    "Thursday":  "Kamis",
    "Friday":    "Jumat",
    "Saturday":  "Sabtu",
    "Sunday":    "Minggu",
}

# Kolom Screening
STRESS_COLS = {
    "Work_Stress_Level": "Stres Kerja",
    "Financial_Stress":  "Stres Finansial",
    "Mood_Swings":       "Mood Swings",
    "Loneliness":        "Kesepian",
}

# Fitur 15 pertanyaan screening
SCREENING_FEATURES_15 = [
    "Age", "Gender", "Work_Stress_Level", "Financial_Stress",
    "Mood_Swings", "Loneliness", "Sleep_Hours_Night",
    "Screen_Time_Hours_Day", "Social_Media_Hours_Day",
    "Work_Hours_Per_Week", "Previously_Diagnosed",
    "Trauma_History", "Feeling_Sad_Down", "Anxious_Nervous", "Social_Support",
]

# Metadata notebook (hardcoded dari hasil analisis)
NB_CLASS_POS       = 9216
NB_CLASS_NEG       = 784
NB_RATIO           = "11.76"
NB_METRICS = {
    "Accuracy":                    91.8,
    "Keandalan Deteksi (Recall)":  93.2,
    "Precision":                   90.7,
    "F1-Score":                    91.8,
}
NB_AKURASI_FULL  = 91.8
NB_AKURASI_SEL   = 91.4
NB_PENGURANGAN   = 70
NB_FITUR_AWAL    = 50
NB_FITUR_FINAL   = 15

# Validasi empatik per emosi
EMPATHY_RESPONSES = {
    "Anger": [
        "Rasa marah adalah sinyal valid bahwa ada sesuatu yang tidak sesuai harapan. Perasaanmu sah.",
        "Wajar merasa marah. Coba tarik napas dalam, biarkan emosi itu ada tanpa menghakimi diri.",
        "Marah bukan kelemahan — itu tanda kamu peduli. Apa yang sesungguhnya menyakitimu hari ini?",
    ],
    "Sadness": [
        "Sedih itu bukan lemah — itu manusiawi. Izinkan dirimu merasakannya sepenuhnya.",
        "Kamu tidak harus baik-baik saja. Sedih juga butuh ruang untuk diakui.",
        "Kesedihan sering hadir ketika kita kehilangan sesuatu yang penting. Kamu tidak sendirian.",
    ],
    "Joy": [
        "Senang mendengar ada hal baik yang kamu rasakan! Momen seperti ini layak dirayakan.",
        "Kegembiraan itu menular — simpan energi positif ini untuk hari-hari yang lebih berat.",
        "Luar biasa! Semoga kebahagiaan ini terus menemanimu.",
    ],
    "Love": [
        "Cinta dalam segala bentuknya adalah kekuatan terbesar. Syukuri hubungan yang membuatmu merasa penuh.",
        "Indah sekali merasakan cinta. Semoga rasa ini selalu membuatmu kuat.",
        "Perasaan terhubung dengan orang lain adalah salah satu kebutuhan terdalam kita.",
    ],
    "Fear": [
        "Takut adalah mekanisme alami tubuhmu untuk melindungi dirimu. Kamu tidak sendiri.",
        "Rasa takut kadang hadir bukan untuk dihilangkan, tapi untuk didengarkan. Ada apa yang membebanimu?",
        "Mengakui ketakutan butuh keberanian. Kamu sudah selangkah lebih maju.",
    ],
    "Neutral": [
        "Hari-hari netral pun berharga. Tidak semua momen harus intens untuk bermakna.",
        "Kadang, tenang adalah hadiah. Nikmati kedamaian hari ini.",
        "Konsistensi dan rutinitas juga bentuk kesehatan mental yang sering terlupakan.",
    ],
}

RECOMMENDATIONS = {
    "Anger":   ["Coba teknik 4-7-8 breathing selama 5 menit.", "Tulis apa yang membuatmu marah tanpa filter selama 10 menit.", "Lakukan aktivitas fisik ringan — jalan kaki atau peregangan."],
    "Sadness": ["Hubungi satu orang yang kamu percaya hari ini.", "Tulis 3 hal kecil yang masih bisa kamu syukuri.", "Dengarkan musik yang kamu sukai, biarkan dirimu merasa."],
    "Joy":     ["Bagikan kebahagiaanmu ke orang terdekat.", "Catat momen ini agar bisa kamu ingat saat hari-hari sulit.", "Rayakan pencapaian kecil — kamu layak mendapat penghargaan."],
    "Love":    ["Luangkan waktu untuk orang yang kamu cintai hari ini.", "Ekspresikan rasa terima kasih kepada seseorang yang berarti.", "Cintai dirimu sendiri seperti kamu mencintai orang lain."],
    "Fear":    ["Coba tulis ketakutanmu secara spesifik — seringkali tulisan membantu.", "Bicara dengan seseorang yang kamu percaya tentang kekhawatiranmu.", "Fokus hanya pada satu langkah kecil yang bisa kamu lakukan sekarang."],
    "Neutral": ["Gunakan momen ini untuk istirahat yang berkualitas.", "Coba eksplorasi hobi baru atau aktivitas yang lama tidak dilakukan.", "Meditasi atau journaling bisa memperdalam kesadaran diri."],
}

HOTLINE_INFO = """
📞 **Butuh Bantuan Segera?**
- **Into The Light Indonesia**: 119 ext 8
- **Yayasan Pulih**: (021) 788-42580
- **RSJ Grhasia DIY**: (0274) 895231
- **Kemenkes (Sejiwa)**: 119 ext 8
"""
