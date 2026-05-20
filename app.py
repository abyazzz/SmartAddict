from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from joblib import load
from functools import wraps
import argparse
import warnings
import json
import os
from datetime import datetime

warnings.filterwarnings("ignore", category=UserWarning)

# ═══════ APP CONFIG ═══════
app = Flask(__name__)
app.secret_key = 'smartaddict-ml-secret-2025'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///smartaddict.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['TEMPLATES_AUTO_RELOAD'] = True

@app.after_request
def add_no_cache(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Silakan login terlebih dahulu.'
login_manager.login_message_category = 'warning'

# ═══════ DATABASE MODELS ═══════
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(10), default='user')  # 'user' or 'admin'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    predictions = db.relationship('Prediction', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    @property
    def is_admin(self):
        return self.role == 'admin'

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    model_name = db.Column(db.String(50), nullable=False)
    input_values = db.Column(db.Text, nullable=False)  # JSON string
    result = db.Column(db.String(20), nullable=False)
    prediction_raw = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    @property
    def input_list(self):
        return json.loads(self.input_values)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def admin_required(f):
    @wraps(f)
    @login_required
    def decorated(*args, **kwargs):
        if not current_user.is_admin:
            flash('Akses ditolak. Hanya admin yang bisa mengakses halaman ini.', 'error')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated

# ═══════ ML CONFIG ═══════
MODEL_FILES = {
    "Decision Tree": "dt2_classifier.pkl",
    "K-Nearest Neighbors": "knn2_classifier.pkl",
    "Neural Network": "nn2_classifier.pkl",
    "Support Vector Machine": "svm2_classifier.pkl",
}

LABEL_MAP = { 0: "Rendah", 1: "Sedang", 2: "Tinggi" }

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

def load_ml_models():
    models = {}
    for name, filename in MODEL_FILES.items():
        try:
            models[name] = load(filename)
        except Exception as exc:
            models[name] = None
            app.logger.error(f"Gagal memuat model {name}: {exc}")
    return models

ml_models = load_ml_models()

# ═══════ AUTH ROUTES ═══════
@app.route("/")
def index():
    return redirect(url_for('dashboard'))

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            flash(f"Selamat datang, {user.username}! 👋", "success")
            next_page = request.args.get('next')
            if user.is_admin:
                return redirect(next_page or url_for('admin_dashboard'))
            return redirect(next_page or url_for('dashboard'))
        else:
            flash("Username atau password salah.", "error")
    return render_template("auth/login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")
        if len(username) < 3:
            flash("Username minimal 3 karakter.", "error")
        elif len(password) < 6:
            flash("Password minimal 6 karakter.", "error")
        elif password != confirm:
            flash("Konfirmasi password tidak cocok.", "error")
        elif User.query.filter_by(username=username).first():
            flash("Username sudah dipakai.", "error")
        else:
            user = User(username=username, role='user')
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            login_user(user)
            flash("Registrasi berhasil! Selamat datang! 🎉", "success")
            return redirect(url_for('dashboard'))
    return render_template("auth/register.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Berhasil logout.", "success")
    return redirect(url_for('login'))

# ═══════ USER ROUTES ═══════
@app.route("/dashboard")
def dashboard():
    user_preds = []
    stats = {'Rendah': 0, 'Sedang': 0, 'Tinggi': 0}
    if current_user.is_authenticated:
        user_preds = Prediction.query.filter_by(user_id=current_user.id).all()
        for p in user_preds:
            if p.result in stats:
                stats[p.result] += 1
    return render_template("dashboard.html", active_page='dashboard', predictions=user_preds, stats=stats)

@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    prediction = None
    diagnosis = None
    model_name = None
    errors = []
    selected_model = "Decision Tree"

    if request.method == "POST":
        selected_model = request.form.get("model") or selected_model
        values = []

        if 'manual_submit' in request.form:
            for question in QUESTIONS:
                raw = request.form.get(question["key"])
                if raw is None or raw == "":
                    errors.append(f"Pertanyaan '{question['label']}' harus diisi.")
                    continue
                try:
                    val = float(raw)
                    q_min = question.get("min", 0)
                    q_max = question.get("max", 999)
                    if not (q_min <= val <= q_max):
                        errors.append(f"Nilai untuk '{question['label']}' harus antara {q_min}-{q_max}.")
                        continue
                    values.append(val)
                except ValueError:
                    errors.append(f"Nilai untuk '{question['label']}' harus berupa angka.")

        elif 'csv_submit' in request.form:
            file = request.files.get('csv_file')
            if file and file.filename.endswith('.csv'):
                try:
                    import pandas as pd
                    df_raw = pd.read_csv(file, header=None)
                    first_row = df_raw.iloc[0]
                    has_header = False
                    try:
                        [float(v) for v in first_row]
                    except (ValueError, TypeError):
                        has_header = True
                    df = df_raw.iloc[1:].reset_index(drop=True) if has_header else df_raw
                    num_cols = len(df.columns)
                    if num_cols == 11:
                        df = df.iloc[:, :10]
                    elif num_cols != 10:
                        errors.append(f"CSV harus memiliki 10 kolom fitur (ditemukan {num_cols} kolom).")
                    if not errors:
                        if len(df) == 0:
                            errors.append("CSV tidak memiliki baris data.")
                        else:
                            raw_values = df.iloc[0].values.tolist()
                            for i, val in enumerate(raw_values):
                                try:
                                    values.append(float(val))
                                except (ValueError, TypeError):
                                    errors.append(f"Kolom {i+1} harus berupa angka.")
                                    values = []
                                    break
                except Exception as e:
                    errors.append(f"Error membaca CSV: {str(e)}")
            else:
                errors.append("Harap upload file CSV yang valid.")

        if not errors and values:
            model = ml_models.get(selected_model)
            if model is None:
                errors.append(f"Model {selected_model} tidak tersedia.")
            else:
                try:
                    prediction = model.predict([values])[0]
                    diagnosis = LABEL_MAP.get(int(prediction), "Tidak diketahui")
                    model_name = selected_model

                    comparison = []
                    for mname, mobj in ml_models.items():
                        if mobj is not None:
                            try:
                                p = mobj.predict([values])[0]
                                comparison.append({"model": mname, "prediction_raw": int(p), "diagnosis": LABEL_MAP.get(int(p), "?")})
                            except:
                                comparison.append({"model": mname, "prediction_raw": -1, "diagnosis": "Error"})

                    # Save to database
                    pred_entry = Prediction(
                        user_id=current_user.id,
                        model_name=model_name,
                        input_values=json.dumps(values),
                        result=diagnosis,
                        prediction_raw=int(prediction),
                    )
                    db.session.add(pred_entry)
                    db.session.commit()

                    session['last_prediction'] = {
                        "values": values,
                        "labels": [q['key'] for q in QUESTIONS],
                        "diagnosis": diagnosis,
                        "model": model_name,
                        "prediction_raw": int(prediction),
                        "comparison": comparison,
                    }
                    flash("Prediksi berhasil!", "success")
                    return redirect(url_for('thanks'))
                except Exception as exc:
                    errors.append(f"Terjadi kesalahan saat memprediksi: {exc}")

    return render_template("predict.html", questions=QUESTIONS, models=list(MODEL_FILES.keys()),
        selected_model=selected_model, errors=errors, active_page='predict')

@app.route("/history")
@login_required
def history_page():
    user_preds = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.timestamp.desc()).all()
    return render_template("history.html", predictions=user_preds, active_page='history')

def get_feature_averages():
    all_preds = Prediction.query.all()
    num_features = len(QUESTIONS)
    sums = [0.0] * num_features
    counts = [0] * num_features
    for p in all_preds:
        try:
            vals = json.loads(p.input_values)
            for i in range(min(len(vals), num_features)):
                sums[i] += vals[i]
                counts[i] += 1
        except Exception:
            pass
    averages = []
    for i in range(num_features):
        avg = sums[i] / counts[i] if counts[i] > 0 else QUESTIONS[i]['default']
        averages.append(round(avg, 2))
    return averages

@app.route("/thanks")
@login_required
def thanks():
    last = session.pop('last_prediction', None)
    averages = get_feature_averages()
    return render_template("thanks.html", result=last, questions=QUESTIONS, averages=averages, active_page='thanks')

@app.route("/about")
def about():
    return render_template("about.html", active_page='about')

@app.route("/delete-prediction/<int:pred_id>", methods=["POST"])
@login_required
def delete_prediction(pred_id):
    pred = Prediction.query.get_or_404(pred_id)
    if pred.user_id != current_user.id and not current_user.is_admin:
        flash("Akses ditolak.", "error")
        return redirect(url_for('history_page'))
    db.session.delete(pred)
    db.session.commit()
    flash("Prediksi berhasil dihapus!", "success")
    if current_user.is_admin and request.referrer and 'admin' in request.referrer:
        return redirect(url_for('admin_history'))
    return redirect(url_for('history_page'))

@app.route("/clear-my-history", methods=["POST"])
@login_required
def clear_my_history():
    Prediction.query.filter_by(user_id=current_user.id).delete()
    db.session.commit()
    flash("History berhasil dihapus!", "success")
    return redirect(url_for('history_page'))

# ═══════ ADMIN ROUTES ═══════
@app.route("/admin")
@admin_required
def admin_dashboard():
    total_users = User.query.filter_by(role='user').count()
    total_preds = Prediction.query.count()
    all_preds = Prediction.query.all()
    stats = {'Rendah': 0, 'Sedang': 0, 'Tinggi': 0}
    model_usage = {}
    for p in all_preds:
        if p.result in stats:
            stats[p.result] += 1
        model_usage[p.model_name] = model_usage.get(p.model_name, 0) + 1
    recent = Prediction.query.order_by(Prediction.timestamp.desc()).limit(8).all()
    return render_template("admin/dashboard.html", active_page='admin_dashboard',
        total_users=total_users, total_preds=total_preds, stats=stats,
        model_usage=model_usage, recent=recent)

@app.route("/admin/history")
@admin_required
def admin_history():
    all_preds = Prediction.query.order_by(Prediction.timestamp.desc()).all()
    return render_template("admin/all_history.html", predictions=all_preds, active_page='admin_history')

@app.route("/admin/users")
@admin_required
def admin_users():
    users = User.query.order_by(User.created_at.desc()).all()
    return render_template("admin/manage_users.html", users=users, active_page='admin_users')

@app.route("/admin/delete-user/<int:user_id>", methods=["POST"])
@admin_required
def admin_delete_user(user_id):
    user = User.query.get_or_404(user_id)
    if user.is_admin:
        flash("Tidak bisa menghapus akun admin.", "error")
    else:
        Prediction.query.filter_by(user_id=user.id).delete()
        db.session.delete(user)
        db.session.commit()
        flash(f"User '{user.username}' berhasil dihapus.", "success")
    return redirect(url_for('admin_users'))

@app.route("/admin/clear-all-history", methods=["POST"])
@admin_required
def admin_clear_all():
    Prediction.query.delete()
    db.session.commit()
    flash("Semua history berhasil dihapus!", "success")
    return redirect(url_for('admin_history'))

@app.errorhandler(404)
def page_not_found(error):
    return render_template("404.html"), 404

# ═══════ INIT DB & DEFAULT ADMIN ═══════
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