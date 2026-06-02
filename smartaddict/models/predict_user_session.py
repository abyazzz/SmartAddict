from datetime import datetime

from smartaddict.extensions import db


class PredictUserSession(db.Model):
    __tablename__ = "predict_user_session"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.Integer, nullable=False)
    daily_screen_time_hours = db.Column(db.Float, nullable=False)
    social_media_hours = db.Column(db.Float, nullable=False)
    gaming_hours = db.Column(db.Float, nullable=False)
    work_study_hours = db.Column(db.Float, nullable=False)
    sleep_hours = db.Column(db.Float, nullable=False)
    notifications_per_day = db.Column(db.Integer, nullable=False)
    app_opens_per_day = db.Column(db.Integer, nullable=False)
    weekend_screen_time = db.Column(db.Float, nullable=False)
    result = db.Column(db.String(20), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)