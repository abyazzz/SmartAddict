import argparse
import warnings

from flask import Flask, flash, redirect, render_template, url_for
from flask_login import LoginManager, current_user, login_required

from smartaddict.config import Config
from smartaddict.extensions import db, login_manager
from smartaddict.models.dashboard_feature import DashboardFeature
from smartaddict.models.prediction import Prediction
from smartaddict.models.predict_user_session import PredictUserSession
from smartaddict.models.user import User
from smartaddict.runtime import init_active_model
from smartaddict.routes import api_routes, auth_routes, admin_routes, user_routes

warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)
app.config.from_object(Config)

@app.after_request
def add_no_cache(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

db.init_app(app)
login_manager.init_app(app)
login_manager.login_view = 'auth.login'
login_manager.login_message = 'Silakan login terlebih dahulu.'
login_manager.login_message_category = 'warning'

@login_manager.user_loader
def load_user(user_id):
    try:
        return User.query.get(int(user_id))
    except Exception:
        return None

init_active_model()

app.register_blueprint(auth_routes.auth_bp)
app.register_blueprint(user_routes.user_bp)
app.register_blueprint(admin_routes.admin_bp)
app.register_blueprint(api_routes.api_bp)

@app.errorhandler(404)
def page_not_found(error):
    return render_template("404.html"), 404

with app.app_context():
    db.create_all()
    if not User.query.filter_by(role='admin').first():
        admin = User(username='admin', role='admin')
        admin.set_password('admin123')
        db.session.add(admin)
        db.session.commit()
        print("[OK] Default admin created: admin / admin123")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5000)

    args = parser.parse_args()

    app.run(debug=True, port=args.port)
