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
import subprocess
from datetime import datetime
import threading
import csv

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

class PredictUserSession(db.Model):
    __tablename__ = 'predict_user_session'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.Integer, nullable=False)  # 0=F, 1=M
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

LEGACY_MODEL_FILES = {
    "Decision Tree": ["dt2_classifier.pkl", "dt_classifier.pkl"],
    "K-Nearest Neighbors": ["knn2_classifier.pkl", "knn_classifier.pkl"],
    "Neural Network": ["nn2_classifier.pkl", "nn_classifier.pkl"],
    "Support Vector Machine": ["svm2_classifier.pkl", "svm_classifier.pkl"],
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

FEATURE_KEYS = [question["key"] for question in QUESTIONS]
SCALER_FILE = "scaler.pkl"
SCALER_FILE_CANDIDATES = ["scaler.pkl", "scaler_backup.pkl"]


def get_venv_python_executable():
    project_root = os.path.dirname(os.path.abspath(__file__))
    venv_python = os.path.join(project_root, "venv", "Scripts", "python.exe")
    if os.path.exists(venv_python):
        return venv_python
    return None


# Active model config persistence
CONFIG_PATH = os.path.join("instance", "model_config.json")
ACTIVE_MODEL_VERSION = "model_default"

def get_active_version_from_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("active_model")
        except Exception:
            pass
    return None

def save_active_version_to_config(version_name):
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump({"active_model": version_name}, f, indent=2)
    except Exception as e:
        app.logger.error(f"Gagal menyimpan config versi model: {e}")

def get_available_retrain_versions():
    versions = []
    model_dir = "model"
    if not os.path.exists(model_dir):
        return []
    for item in os.listdir(model_dir):
        item_path = os.path.join(model_dir, item)
        if os.path.isdir(item_path) and item.startswith("model_") and item != "model_default":
            metrics = {"dt": 0.0, "knn": 0.0, "nn": 0.0, "svm": 0.0}
            avg_accuracy = 0.0

            metrics_sources = [
                os.path.join(item_path, "metrics.json"),
                os.path.join(item_path, "metadata.json"),
            ]
            for metrics_path in metrics_sources:
                if not os.path.exists(metrics_path):
                    continue
                try:
                    with open(metrics_path, "r", encoding="utf-8") as f:
                        raw_metrics = json.load(f)

                    if isinstance(raw_metrics, dict) and all(key in raw_metrics for key in ["dt", "knn", "nn", "svm"]):
                        metrics = {
                            "dt": float(raw_metrics.get("dt", 0) or 0),
                            "knn": float(raw_metrics.get("knn", 0) or 0),
                            "nn": float(raw_metrics.get("nn", 0) or 0),
                            "svm": float(raw_metrics.get("svm", 0) or 0),
                        }
                    elif isinstance(raw_metrics, dict) and isinstance(raw_metrics.get("model_metrics"), list):
                        legacy_map = {
                            "Decision Tree": "dt",
                            "decision tree": "dt",
                            "dt": "dt",
                            "K-Nearest Neighbors": "knn",
                            "k-NN": "knn",
                            "knn": "knn",
                            "Neural Network": "nn",
                            "nn": "nn",
                            "SVM": "svm",
                            "Support Vector Machine": "svm",
                            "svm": "svm",
                        }
                        for entry in raw_metrics.get("model_metrics", []):
                            if not isinstance(entry, dict):
                                continue
                            metric_key = legacy_map.get(str(entry.get("model", "")).strip())
                            if metric_key:
                                metrics[metric_key] = float(entry.get("accuracy", 0) or 0)
                    avg_accuracy = sum(metrics.values()) / len(metrics)
                    break
                except Exception as e:
                    app.logger.error(f"Gagal membaca metrics di {item} dari {os.path.basename(metrics_path)}: {e}")
            versions.append({
                "version_name": item,
                "average_accuracy": avg_accuracy,
                "metrics": metrics,
                "is_active": (item == ACTIVE_MODEL_VERSION)
            })
    versions.sort(key=lambda x: x["version_name"], reverse=True)
    return versions

def load_model_version(version_name):
    base_path = os.path.join("model", version_name)
    models = {}
    if not os.path.exists(base_path):
        return None, None, False
    loaded_any_model = False
    for name, filenames in LEGACY_MODEL_FILES.items():
        loaded_model = None
        last_error = None
        for filename in filenames:
            model_path = os.path.join(base_path, filename)
            if not os.path.exists(model_path):
                continue
            try:
                loaded_model = load(model_path)
                break
            except Exception as exc:
                last_error = exc
                app.logger.error(f"Gagal memuat model {name} dari {model_path}: {exc}")
        if loaded_model is None:
            if last_error is None:
                app.logger.error(f"File model {name} tidak ditemukan di {base_path}")
            continue
        models[name] = loaded_model
        loaded_any_model = True

    scaler_obj = None
    last_scaler_error = None
    for filename in SCALER_FILE_CANDIDATES:
        scaler_path = os.path.join(base_path, filename)
        if not os.path.exists(scaler_path):
            continue
        try:
            scaler_obj = load(scaler_path)
            break
        except Exception as exc:
            last_scaler_error = exc
            app.logger.error(f"Gagal memuat scaler dari {scaler_path}: {exc}")
    if scaler_obj is None:
        if last_scaler_error is None:
            app.logger.error(f"File scaler tidak ditemukan di {base_path}")
        return None, None, False
    if not loaded_any_model:
        return None, None, False
    return models, scaler_obj, True

ml_models = {}
scaler = None

def init_active_model():
    global ACTIVE_MODEL_VERSION, ml_models, scaler
    cfg_version = get_active_version_from_config()
    if cfg_version:
        models, scaler_obj, success = load_model_version(cfg_version)
        if success:
            ACTIVE_MODEL_VERSION = cfg_version
            ml_models = models
            scaler = scaler_obj
            return
    for version_info in get_available_retrain_versions():
        version_name = version_info["version_name"]
        models, scaler_obj, success = load_model_version(version_name)
        if success:
            ACTIVE_MODEL_VERSION = version_name
            ml_models = models
            scaler = scaler_obj
            save_active_version_to_config(version_name)
            return
    models, scaler_obj, success = load_model_version("model_default")
    if success:
        ACTIVE_MODEL_VERSION = "model_default"
        ml_models = models
        scaler = scaler_obj
        save_active_version_to_config("model_default")
        return
    ACTIVE_MODEL_VERSION = None
    ml_models = {}
    scaler = None


def predict_with_model(values, selected_model, include_comparison=True):
    model = ml_models.get(selected_model)
    if model is None:
        raise ValueError(f"Model {selected_model} tidak tersedia.")

    input_array = [values]
    if scaler is not None:
        try:
            input_array = scaler.transform(input_array)
        except Exception as exc:
            raise ValueError(f"Gagal melakukan transformasi scaler: {exc}")

    prediction_raw = int(model.predict(input_array)[0])
    diagnosis = LABEL_MAP.get(prediction_raw, "Tidak diketahui")

    comparison = []
    if include_comparison:
        for mname, mobj in ml_models.items():
            if mobj is not None:
                try:
                    p = int(mobj.predict(input_array)[0])
                    comparison.append({"model": mname, "prediction_raw": p, "diagnosis": LABEL_MAP.get(p, "?")})
                except Exception:
                    comparison.append({"model": mname, "prediction_raw": -1, "diagnosis": "Error"})

    return {
        "values": values,
        "diagnosis": diagnosis,
        "prediction_raw": prediction_raw,
        "model": selected_model,
        "comparison": comparison,
    }


def parse_csv_rows(file_obj):
    import pandas as pd

    df_raw = pd.read_csv(file_obj, header=None)
    if len(df_raw) == 0:
        return [], False

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
        raise ValueError(f"CSV harus memiliki 10 kolom fitur (ditemukan {num_cols} kolom).")

    if len(df) == 0:
        return [], False
    if len(df) > 20:
        raise ValueError("CSV maksimal berisi 20 baris data.")

    rows = []
    for row_index, (_, row) in enumerate(df.iterrows(), start=1):
        values = []
        for col_index, value in enumerate(row.values, start=1):
            if pd.isna(value):
                raise ValueError(f"Baris {row_index}, kolom {col_index} tidak boleh kosong.")
            values.append(float(value))
        rows.append(values)

    return rows, has_header


def average_rows(rows):
    if not rows:
        return []
    column_count = len(rows[0])
    return [sum(row[index] for row in rows) / len(rows) for index in range(column_count)]

RETRAIN_LOCK = threading.Lock()
IS_RETRAINING = False

def _execute_retrain_job(app_instance):
    global ACTIVE_MODEL_VERSION, ml_models, scaler
    with app_instance.app_context():
        app_instance.logger.info("Retrain pipeline: Mengambil baris dari predict_user_session...")
        rows = PredictUserSession.query.order_by(PredictUserSession.timestamp.asc()).all()

        # Jika ada baris, tambahkan ke CSV dan kosongkan database sesi
        if len(rows) > 0:
            csv_path = os.path.join("dataset-notebook", "Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv")
            app_instance.logger.info(f"Retrain pipeline: Menambahkan {len(rows)} baris ke {csv_path}...")

            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for idx, row in enumerate(rows):
                    # Generate transaction & user IDs
                    txn_id = f"TXN{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{idx}"
                    u_id = f"U{row.user_id:05d}"

                    # Konversi result dan gender
                    gender_str = "Male" if row.gender == 1 else "Female"
                    result_str = "Mild" if row.result == "Rendah" else ("Moderate" if row.result == "Sedang" else "Severe")
                    addicted_lbl = 0 if row.result == "Rendah" else 1

                    # Skema kolom CSV
                    writer.writerow([
                        txn_id, u_id, int(row.age), gender_str,
                        float(row.daily_screen_time_hours), float(row.social_media_hours),
                        float(row.gaming_hours), float(row.work_study_hours),
                        float(row.sleep_hours), int(row.notifications_per_day),
                        int(row.app_opens_per_day), float(row.weekend_screen_time),
                        "Medium", "Yes", result_str, addicted_lbl
                    ])

            # Kosongkan tabel sesi
            app_instance.logger.info("Retrain pipeline: Mengosongkan tabel predict_user_session...")
            PredictUserSession.query.delete()
            db.session.commit()

        # Jalankan Papermill untuk retraining notebook
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_version = f"model_{timestamp}"
        output_dir = os.path.join("model", output_version)
        os.makedirs(output_dir, exist_ok=True)

        notebook_in = os.path.join("dataset-notebook", "Tubes_FIX.ipynb")
        notebook_out = os.path.join("scratch", f"executed_{output_version}.ipynb")

        os.makedirs(os.path.dirname(notebook_out), exist_ok=True)

        papermill_python = get_venv_python_executable() or os.environ.get("PYTHON_EXECUTABLE") or "python"
        papermill_script = (
            "import papermill as pm; "
            f"pm.execute_notebook(r'{notebook_in}', r'{notebook_out}', parameters={{'output_model_dir': r'{output_dir}'}})"
        )

        app_instance.logger.info(
            f"Retrain pipeline: Mengeksekusi papermill via {papermill_python} ({notebook_in} -> {output_dir})..."
        )
        completed = subprocess.run(
            [papermill_python, "-c", papermill_script],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.stdout:
            app_instance.logger.info(f"Retrain pipeline stdout: {completed.stdout.strip()}")
        if completed.stderr:
            app_instance.logger.error(f"Retrain pipeline stderr: {completed.stderr.strip()}")
        if completed.returncode != 0:
            raise RuntimeError(f"Papermill gagal dijalankan dengan kode {completed.returncode}.")

        # Bersihkan file temp executed notebook
        if os.path.exists(notebook_out):
            try:
                os.remove(notebook_out)
            except Exception:
                pass

        # Muat model yang baru selesai ditraining
        new_models, new_scaler, success = load_model_version(output_version)
        if success:
            ACTIVE_MODEL_VERSION = output_version
            ml_models = new_models
            scaler = new_scaler
            save_active_version_to_config(output_version)
            app_instance.logger.info(f"Retrain pipeline: Berhasil melatih model baru dan mengaktifkannya: {output_version}")
            return output_version

        app_instance.logger.error("Retrain pipeline: Gagal memuat model baru pasca training.")
        return None


def run_retrain_pipeline(app_instance):
    global IS_RETRAINING

    def job():
        global IS_RETRAINING
        try:
            _execute_retrain_job(app_instance)
        except Exception as e:
            app_instance.logger.error(f"Retrain pipeline: Terjadi error saat retraining: {e}")
        finally:
            IS_RETRAINING = False
            if RETRAIN_LOCK.locked():
                try:
                    RETRAIN_LOCK.release()
                except RuntimeError:
                    pass

    if RETRAIN_LOCK.acquire(blocking=False):
        IS_RETRAINING = True
        thread = threading.Thread(target=job)
        thread.daemon = True
        thread.start()
        return True
    return False

# Inisialisasi model aktif saat startup aplikasi
init_active_model()

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

@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    prediction_count = Prediction.query.filter_by(user_id=current_user.id).count()
    latest_prediction = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.timestamp.desc()).first()

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        current_password = request.form.get("current_password", "")
        new_password = request.form.get("new_password", "")
        confirm_password = request.form.get("confirm_password", "")

        errors = []
        next_username = None
        change_password = False

        if username != current_user.username:
            if len(username) < 3:
                errors.append("Username minimal 3 karakter.")
            elif User.query.filter(User.username == username, User.id != current_user.id).first():
                errors.append("Username sudah dipakai akun lain.")
            else:
                next_username = username

        password_touched = any([current_password, new_password, confirm_password])
        if password_touched:
            if not current_password or not current_user.check_password(current_password):
                errors.append("Password saat ini tidak valid.")
            elif len(new_password) < 6:
                errors.append("Password baru minimal 6 karakter.")
            elif new_password != confirm_password:
                errors.append("Konfirmasi password baru tidak cocok.")
            else:
                change_password = True

        if errors:
            for error in errors:
                flash(error, "error")
        else:
            changed = False
            if next_username:
                current_user.username = next_username
                changed = True
            if change_password:
                current_user.set_password(new_password)
                changed = True

            if changed:
                db.session.commit()
                flash("Profile berhasil diperbarui.", "success")
                return redirect(url_for('profile'))
            flash("Tidak ada perubahan profile.", "warning")

    return render_template(
        "profile.html",
        active_page='profile',
        prediction_count=prediction_count,
        latest_prediction=latest_prediction
    )

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
                    csv_rows, _ = parse_csv_rows(file)
                    if not csv_rows:
                        errors.append("CSV tidak memiliki baris data.")
                    elif len(csv_rows) == 1:
                        values = csv_rows[0]
                    else:
                        batch_rows = []
                        for row_number, row_values in enumerate(csv_rows, start=1):
                            row_result = predict_with_model(row_values, selected_model, include_comparison=False)
                            batch_rows.append({
                                "row_number": row_number,
                                "values": row_values,
                                "diagnosis": row_result["diagnosis"],
                                "prediction_raw": row_result["prediction_raw"],
                            })

                        values = average_rows(csv_rows)
                        aggregate_result = predict_with_model(values, selected_model, include_comparison=True)
                        distribution = {"Rendah": 0, "Sedang": 0, "Tinggi": 0}
                        for row in batch_rows:
                            if row["diagnosis"] in distribution:
                                distribution[row["diagnosis"]] += 1

                        pred_entry = Prediction(
                            user_id=current_user.id,
                            model_name=selected_model,
                            input_values=json.dumps(values),
                            result=aggregate_result["diagnosis"],
                            prediction_raw=aggregate_result["prediction_raw"],
                        )
                        db.session.add(pred_entry)
                        db.session.commit()

                        session['last_prediction'] = {
                            "values": values,
                            "labels": FEATURE_KEYS,
                            "diagnosis": aggregate_result["diagnosis"],
                            "model": selected_model,
                            "prediction_raw": aggregate_result["prediction_raw"],
                            "comparison": aggregate_result["comparison"],
                            "batch_mode": True,
                            "batch_count": len(batch_rows),
                            "batch_rows": batch_rows,
                            "distribution": distribution,
                            "feature_averages": values,
                        }
                        flash("Prediksi batch berhasil!", "success")
                        return redirect(url_for('thanks'))
                except Exception as e:
                    errors.append(f"Error membaca CSV: {str(e)}")
            else:
                errors.append("Harap upload file CSV yang valid.")

        if not errors and values:
            try:
                result_payload = predict_with_model(values, selected_model, include_comparison=True)
                prediction = result_payload["prediction_raw"]
                diagnosis = result_payload["diagnosis"]
                model_name = selected_model

                pred_entry = Prediction(
                    user_id=current_user.id,
                    model_name=model_name,
                    input_values=json.dumps(values),
                    result=diagnosis,
                    prediction_raw=int(prediction),
                )
                db.session.add(pred_entry)
                db.session.commit()

                # ═══════ AUTO RETRAIN SESSION BUFFER ═══════
                try:
                    session_count = PredictUserSession.query.count()
                    
                    new_session = PredictUserSession(
                        user_id=current_user.id,
                        age=int(values[0]),
                        gender=int(values[1]),
                        daily_screen_time_hours=float(values[2]),
                        social_media_hours=float(values[3]),
                        gaming_hours=float(values[4]),
                        work_study_hours=float(values[5]),
                        sleep_hours=float(values[6]),
                        notifications_per_day=int(values[7]),
                        app_opens_per_day=int(values[8]),
                        weekend_screen_time=float(values[9]),
                        result=diagnosis
                    )
                    db.session.add(new_session)
                    db.session.commit()
                    
                    if session_count >= 49:
                        from flask import current_app
                        app_obj = current_app._get_current_object()
                        triggered = run_retrain_pipeline(app_obj)
                        if triggered:
                            flash("Retraining otomatis berjalan di background (50 data terpenuhi)!", "info")
                except Exception as db_err:
                    app.logger.error(f"Gagal mencatat sesi prediksi ke database: {db_err}")

                session['last_prediction'] = {
                    "values": values,
                    "labels": FEATURE_KEYS,
                    "diagnosis": diagnosis,
                    "model": model_name,
                    "prediction_raw": int(prediction),
                    "comparison": result_payload["comparison"],
                    "batch_mode": False,
                    "feature_averages": values,
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
    averages = last.get('feature_averages') if last and last.get('feature_averages') else get_feature_averages()
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
    total_predict_session = PredictUserSession.query.count()
    all_preds = Prediction.query.all()
    stats = {'Rendah': 0, 'Sedang': 0, 'Tinggi': 0}
    model_usage = {}
    for p in all_preds:
        if p.result in stats:
            stats[p.result] += 1
        model_usage[p.model_name] = model_usage.get(p.model_name, 0) + 1
    recent = Prediction.query.order_by(Prediction.timestamp.desc()).limit(8).all()
    
    # Versi retraining dinamis
    versions = get_available_retrain_versions()
    total_retrains = len(versions)
    
    return render_template("admin/dashboard.html", active_page='admin_dashboard',
        total_users=total_users, total_preds=total_preds, stats=stats,
        model_usage=model_usage, recent=recent,
        total_predict_session=total_predict_session,
        total_retrains=total_retrains, retrain_versions=versions,
        active_model_version=ACTIVE_MODEL_VERSION)

@app.route("/admin/retrain-manual", methods=["POST"])
@admin_required
def admin_retrain_manual():
    global IS_RETRAINING
    if IS_RETRAINING:
        flash("Proses retraining sedang berjalan. Harap tunggu hingga selesai.", "warning")
        return redirect(url_for('admin_dashboard'))

    from flask import current_app
    app_obj = current_app._get_current_object()
    if not RETRAIN_LOCK.acquire(blocking=False):
        flash("Proses retraining sedang berjalan. Harap tunggu hingga selesai.", "warning")
        return redirect(url_for('admin_dashboard'))

    IS_RETRAINING = True
    try:
        output_version = _execute_retrain_job(app_obj)
        if output_version:
            flash(f"Retraining manual selesai dan model aktif berpindah ke {output_version}.", "success")
        else:
            flash("Retraining manual selesai, tetapi model baru gagal dimuat.", "warning")
    except Exception as exc:
        app_obj.logger.error(f"Retrain manual gagal: {exc}")
        flash(f"Retraining manual gagal: {exc}", "error")
    finally:
        IS_RETRAINING = False
        if RETRAIN_LOCK.locked():
            try:
                RETRAIN_LOCK.release()
            except RuntimeError:
                pass
    return redirect(url_for('admin_dashboard'))

@app.route("/admin/clear-retrains", methods=["POST"])
@admin_required
def admin_clear_retrains():
    import shutil
    global ACTIVE_MODEL_VERSION, ml_models, scaler
    
    versions = get_available_retrain_versions()
    deleted_count = 0
    for ver in versions:
        version_dir = os.path.join("model", ver["version_name"])
        try:
            if os.path.exists(version_dir):
                shutil.rmtree(version_dir)
                deleted_count += 1
        except Exception as e:
            app.logger.error(f"Gagal menghapus folder {version_dir}: {e}")
            
    # Kembalikan ke model_default
    models, scaler_obj, success = load_model_version("model_default")
    if success:
        ACTIVE_MODEL_VERSION = "model_default"
        ml_models = models
        scaler = scaler_obj
        save_active_version_to_config("model_default")
        flash(f"Berhasil menghapus {deleted_count} model retrain. Sistem kembali menggunakan model_default.", "success")
    else:
        ACTIVE_MODEL_VERSION = None
        ml_models = {}
        scaler = None
        flash("Semua model retrain dihapus, namun model_default gagal dimuat.", "error")
        
    return redirect(url_for('admin_dashboard'))

@app.route("/admin/use-retrain/<version_name>", methods=["POST"])
@admin_required
def admin_use_retrain(version_name):
    global ACTIVE_MODEL_VERSION, ml_models, scaler
    
    models, scaler_obj, success = load_model_version(version_name)
    if success:
        ACTIVE_MODEL_VERSION = version_name
        ml_models = models
        scaler = scaler_obj
        save_active_version_to_config(version_name)
        flash(f"Berhasil mengubah model aktif ke versi {version_name}!", "success")
    else:
        flash(f"Gagal memuat model dari versi {version_name}. Tetap menggunakan versi sebelumnya.", "error")
        
    return redirect(url_for('admin_dashboard'))

@app.route("/admin/delete-retrain/<version_name>", methods=["POST"])
@admin_required
def admin_delete_retrain(version_name):
    import shutil
    global ACTIVE_MODEL_VERSION
    
    if version_name == "model_default":
        flash("Model default bawaan tidak boleh dihapus.", "error")
        return redirect(url_for('admin_dashboard'))
        
    version_dir = os.path.join("model", version_name)
    try:
        if os.path.exists(version_dir):
            shutil.rmtree(version_dir)
            flash(f"Versi model {version_name} berhasil dihapus.", "success")
        else:
            flash(f"Direktori versi model {version_name} tidak ditemukan.", "error")
    except Exception as e:
        flash(f"Gagal menghapus folder versi model: {e}", "error")
        
    # Inisialisasi ulang model aktif
    init_active_model()
    return redirect(url_for('admin_dashboard'))

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
