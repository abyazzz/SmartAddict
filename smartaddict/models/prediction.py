import json
from datetime import datetime

from smartaddict.extensions import db


class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    model_name = db.Column(db.String(50), nullable=False)
    input_values = db.Column(db.Text, nullable=False)
    result = db.Column(db.String(20), nullable=False)
    prediction_raw = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    @property
    def input_list(self):
        return json.loads(self.input_values)