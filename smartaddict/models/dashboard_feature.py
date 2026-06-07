from datetime import datetime

from smartaddict.extensions import db


class DashboardFeature(db.Model):
    __tablename__ = "dashboard_features"

    id = db.Column(db.Integer, primary_key=True)
    slot_key = db.Column(db.String(40), unique=True, nullable=False)
    icon = db.Column(db.String(32), nullable=False, default="")
    title = db.Column(db.String(90), nullable=False)
    description = db.Column(db.Text, nullable=False)
    target_key = db.Column(db.String(40), nullable=False, default="dashboard")
    sort_order = db.Column(db.Integer, nullable=False, default=0)
    is_active = db.Column(db.Boolean, nullable=False, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
