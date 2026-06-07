LABEL_MAP = {0: "Rendah", 1: "Sedang", 2: "Tinggi"}

QUESTIONS = [
    {"key": "age", "label": "Berapa usia Anda?", "description": "Masukkan usia dalam tahun. Rentang: 10-60.", "min": 10, "max": 60, "step": 1, "default": 20},
    {"key": "gender", "label": "Jenis kelamin", "description": "Pilih jenis kelamin Anda.", "min": 0, "max": 1, "step": 1, "default": 1, "type": "select", "options": [{"value": 0, "label": "Perempuan"}, {"value": 1, "label": "Laki-laki"}]},
    {"key": "daily_screen_time_hours", "label": "Berapa jam rata-rata Anda menatap layar smartphone per hari?", "description": "Total waktu penggunaan smartphone (screen time) pada hari kerja/biasa. Rentang: 0-24 jam.", "min": 0, "max": 24, "step": 1, "default": 5},
    {"key": "social_media_hours", "label": "Berapa jam menggunakan media sosial per hari?", "description": "Instagram, TikTok, Twitter, dll. Rentang: 0-24 jam.", "min": 0, "max": 24, "step": 1, "default": 2},
    {"key": "gaming_hours", "label": "Berapa jam bermain game per hari?", "description": "Mobile game, console, dll. Rentang: 0-24 jam.", "min": 0, "max": 24, "step": 1, "default": 1},
    {"key": "work_study_hours", "label": "Berapa jam Anda menggunakan smartphone untuk kerja atau belajar per hari?", "description": "Contoh: mengerjakan tugas, meeting online, membaca materi, coding, atau pekerjaan kantor. Rentang: 0-24 jam.", "min": 0, "max": 24, "step": 1, "default": 4},
    {"key": "sleep_hours", "label": "Berapa jam tidur Anda per hari?", "description": "Rata-rata jam tidur per malam. Rentang: 0-12 jam.", "min": 0, "max": 12, "step": 1, "default": 7},
    {"key": "notifications_per_day", "label": "Berapa notifikasi yang Anda terima per hari?", "description": "Perkiraan jumlah notifikasi harian. Rentang: 0-500.", "min": 0, "max": 500, "step": 1, "default": 50},
    {"key": "app_opens_per_day", "label": "Berapa kali Anda membuka aplikasi per hari?", "description": "Total buka aplikasi apapun. Rentang: 0-500.", "min": 0, "max": 500, "step": 1, "default": 50},
    {"key": "weekend_screen_time", "label": "Berapa jam Anda menatap layar smartphone di hari libur?", "description": "Total waktu penggunaan smartphone (screen time) pada akhir pekan (Sabtu/Minggu) atau hari libur. Rentang: 0-24 jam.", "min": 0, "max": 24, "step": 1, "default": 6},
]

FEATURE_KEYS = [question["key"] for question in QUESTIONS]

MODEL_FILES = {
    "Decision Tree": "dt2_classifier.pkl",
    "K-Nearest Neighbors": "knn2_classifier.pkl",
    "Neural Network": "nn2_classifier.pkl",
    "Support Vector Machine": "svm2_classifier.pkl",
}

LEGACY_MODEL_FILES = {
    "Decision Tree": ["dt2_classifier.pkl", "dt_classifier.pkl"],
    "K-Nearest Neighbors": ["knn2_classifier.pkl", "knn_classifier.pkl"],
    "Neural Network": ["nn2_classifier.pkl", "nn_classifier.pkl"],
    "Support Vector Machine": ["svm2_classifier.pkl", "svm_classifier.pkl"],
}

SCALER_FILE = "scaler.pkl"
SCALER_FILE_CANDIDATES = ["scaler.pkl", "scaler_backup.pkl"]

RETRAIN_STEP_PLAN = [
    'Insert data session ke dataset',
    'Load library',
    'Load dataset',
    'Label encoding kolom kategorikal',
    'Visualisasi sederhana EDA',
    'Ambil feature dan label',
    'Split train/test 80/20',
    'SMOTE',
    'Feature scaling',
    'HPO SVM',
    'Evaluation',
    'HPO dan evaluasi k-NN',
    'HPO Decision Tree',
    'HPO Neural Network',
    'Tabel perbandingan sebelum HPO',
    'Tabel perbandingan setelah HPO',
    'Confusion matrix Decision Tree',
    'Deploy',
]