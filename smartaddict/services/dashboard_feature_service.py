from flask import url_for

from smartaddict.extensions import db
from smartaddict.models.dashboard_feature import DashboardFeature


DASHBOARD_FEATURE_TARGETS = [
    ("predict", "Halaman Predict"),
    ("history", "Halaman History"),
    ("about", "Halaman About"),
    ("about_algoritma", "About - Algoritma ML"),
    ("about_tips", "About - Tips & Ciri"),
    ("about_edukasi", "About - Artikel Edukasi"),
    ("dashboard", "Dashboard"),
]


DEFAULT_DASHBOARD_FEATURES = [
    {
        "slot_key": "prediction",
        "icon": "📊",
        "title": "Prediksi Level Kecanduan",
        "description": "Gunakan algoritma machine learning untuk mengukur level kecanduan berdasarkan 10 parameter penggunaan harian.",
        "target_key": "predict",
        "sort_order": 1,
    },
    {
        "slot_key": "input",
        "icon": "📁",
        "title": "Input Manual & CSV",
        "description": "Masukkan data secara langsung melalui form interaktif atau upload file CSV batch untuk evaluasi cepat.",
        "target_key": "predict",
        "sort_order": 2,
    },
    {
        "slot_key": "history",
        "icon": "🕐",
        "title": "History Prediksi",
        "description": "Akses riwayat diagnosis, detail input, model yang dipakai, serta unduh data hasil prediksi.",
        "target_key": "history",
        "sort_order": 3,
    },
    {
        "slot_key": "algorithms",
        "icon": "🤖",
        "title": "4 Algoritma ML",
        "description": "Decision Tree, K-Nearest Neighbors, Neural Network, dan Support Vector Machine tersedia untuk perbandingan hasil.",
        "target_key": "about_algoritma",
        "sort_order": 4,
    },
    {
        "slot_key": "tips",
        "icon": "⚡",
        "title": "Tips & Ciri Kecanduan",
        "description": "Kenali tanda penggunaan smartphone bermasalah dan langkah sederhana untuk menjaga fokus, tidur, dan kontrol diri.",
        "target_key": "about_tips",
        "sort_order": 5,
    },
    {
        "slot_key": "education",
        "icon": "🎓",
        "title": "Artikel Edukasi Digital",
        "description": "Baca artikel singkat tentang screen time, kualitas tidur, performa akademik, dan kebiasaan digital sehat.",
        "target_key": "about_edukasi",
        "sort_order": 6,
    },
]


def ensure_dashboard_features():
    existing_keys = {
        feature.slot_key for feature in DashboardFeature.query.with_entities(DashboardFeature.slot_key).all()
    }
    created = False
    for item in DEFAULT_DASHBOARD_FEATURES:
        if item["slot_key"] not in existing_keys:
            db.session.add(DashboardFeature(**item))
            created = True
    if created:
        db.session.commit()


def get_dashboard_features(include_inactive=False):
    ensure_dashboard_features()
    query = DashboardFeature.query
    if not include_inactive:
        query = query.filter_by(is_active=True)
    return query.order_by(DashboardFeature.sort_order.asc(), DashboardFeature.id.asc()).all()


def resolve_dashboard_feature_url(target_key):
    target_map = {
        "predict": url_for("user.predict"),
        "history": url_for("user.history_page"),
        "about": url_for("user.about"),
        "about_algoritma": url_for("user.about") + "#algoritma-ml",
        "about_tips": url_for("user.about") + "#tips-kecanduan",
        "about_edukasi": url_for("user.about") + "#edukasi-digital",
        "dashboard": url_for("user.dashboard"),
    }
    return target_map.get(target_key, url_for("user.dashboard"))


def get_dashboard_feature_cards():
    cards = []
    for feature in get_dashboard_features(include_inactive=False):
        cards.append({
            "id": feature.id,
            "icon": feature.icon,
            "title": feature.title,
            "description": feature.description,
            "url": resolve_dashboard_feature_url(feature.target_key),
        })
    return cards
