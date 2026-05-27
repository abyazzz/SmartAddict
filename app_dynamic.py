from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from joblib import load
from functools import wraps
import argparse
import csv
import warnings
import json
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from datetime import datetime
import sys

warnings.filterwarnings("ignore", category=UserWarning)

# â•â•â•â•â•â•â• APP CONFIG â•â•â•â•â•â•â•
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

BASE_DIR = Path(__file__).resolve().parent
INSTANCE_DIR = BASE_DIR / 'instance'
MODEL_ROOT_DIR = BASE_DIR / 'model'
DATASET_PATH = BASE_DIR / 'notebook-dataset' / 'Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv'
NOTEBOOK_PATH = BASE_DIR / 'notebook-dataset' / 'Tubes_FIX.ipynb'
INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_ROOT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILES = {
    'Decision Tree': 'dt_classifier.pkl',
    'K-Nearest Neighbors': 'knn_classifier.pkl',
    'Neural Network': 'nn_classifier.pkl',
    'Support Vector Machine': 'svm_classifier.pkl',
}

LEGACY_MODEL_FILES = {
    'Decision Tree': 'dt2_classifier.pkl',
    'K-Nearest Neighbors': 'knn2_classifier.pkl',
    'Neural Network': 'nn2_classifier.pkl',
    'Support Vector Machine': 'svm2_classifier.pkl',
}

RETRAIN_BATCH_SIZE = 50
RETRAIN_LOCK = threading.Lock()
ACTIVE_MODEL_VERSION = None
ACTIVE_MODEL_DIR = None

# â•â•â•â•â•â•â• DATABASE MODELS â•â•â•â•â•â•â•
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


class PredictUser(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    model_name = db.Column(db.String(50), nullable=False)
    input_values = db.Column(db.Text, nullable=False)
    diagnosis = db.Column(db.String(20), nullable=False)
    prediction_raw = db.Column(db.Integer, nullable=False)
    batch_tag = db.Column(db.String(80), nullable=True)
    is_processed = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class RetrainingRun(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    model_version = db.Column(db.String(120), nullable=False, unique=True)
    model_dir = db.Column(db.String(255), nullable=False)
    status = db.Column(db.String(20), nullable=False, default='running')
    row_count = db.Column(db.Integer, nullable=False, default=0)
    average_accuracy = db.Column(db.Float, nullable=True)
    metrics_json = db.Column(db.Text, nullable=False, default='{}')
    error_message = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    finished_at = db.Column(db.DateTime, nullable=True)

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

def admin_required(f):
    @wraps(f)
    @login_required
    def decorated(*args, **kwargs):
        if not current_user.is_admin:
            flash('Akses ditolak. Hanya admin yang bisa mengakses halaman ini.', 'error')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated


def ensure_default_admin():
    admin = User.query.filter_by(username='admin').first()
    if admin:
        if admin.role != 'admin':
            admin.role = 'admin'
            if not admin.password_hash:
                admin.set_password('admin123')
            db.session.commit()
        return admin

    admin = User(username='admin', role='admin')
    admin.set_password('admin123')
    db.session.add(admin)
    db.session.commit()
    print("[OK] Default admin created: admin / admin123")
    return admin

# â•â•â•â•â•â•â• ML CONFIG â•â•â•â•â•â•â•
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


def predict_with_model(values, selected_model, include_comparison=True):
    model = ml_models.get(selected_model)
    if model is None:
        raise ValueError(f"Model {selected_model} tidak tersedia.")

    prediction_raw = int(model.predict([values])[0])
    diagnosis = LABEL_MAP.get(prediction_raw, "Tidak diketahui")

    comparison = []
    if include_comparison:
        for mname, mobj in ml_models.items():
            if mobj is not None:
                try:
                    p = int(mobj.predict([values])[0])
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


def get_pending_predict_user_count():
    return PredictUser.query.filter_by(is_processed=False).count()


def _read_json_file(file_path):
    try:
        if file_path.exists():
            return json.loads(file_path.read_text(encoding='utf-8'))
    except Exception:
        pass
    return {}


def _write_json_file(file_path, payload):
    file_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')


def _resolve_metadata_file(model_dir):
    return Path(model_dir) / 'metadata.json'


def _get_timestamp_slug():
    return datetime.utcnow().strftime('%Y%m%d_%H%M%S')


def discover_model_versions():
    versions = []
    if not MODEL_ROOT_DIR.exists():
        return versions

    for folder in sorted((p for p in MODEL_ROOT_DIR.iterdir() if p.is_dir()), key=lambda item: item.name, reverse=True):
        metadata = _read_json_file(_resolve_metadata_file(folder))
        model_metrics = metadata.get('model_metrics') or metadata.get('model_accuracies') or []
        average_accuracy = metadata.get('average_accuracy')
        if average_accuracy is None and model_metrics:
            accuracies = [item.get('accuracy') for item in model_metrics if isinstance(item.get('accuracy'), (int, float))]
            if accuracies:
                average_accuracy = sum(accuracies) / len(accuracies)

        versions.append({
            'version_name': folder.name,
            'display_name': folder.name.replace('model_', ''),
            'model_dir': str(folder),
            'model_dir_path': folder,
            'metadata': metadata,
            'model_metrics': model_metrics,
            'average_accuracy': round(float(average_accuracy), 4) if average_accuracy is not None else None,
            'status': metadata.get('status', 'completed'),
            'row_count': metadata.get('row_count', 0),
            'created_at': metadata.get('created_at', folder.name),
            'is_active': False,
        })

    versions.sort(key=lambda item: item['version_name'], reverse=True)
    return versions


def _set_active_model_state(version_name=None, model_dir=None):
    global ACTIVE_MODEL_VERSION, ACTIVE_MODEL_DIR
    ACTIVE_MODEL_VERSION = version_name
    ACTIVE_MODEL_DIR = Path(model_dir) if model_dir else None


def _is_valid_ml_model(model_obj):
    return model_obj is not None and callable(getattr(model_obj, 'predict', None))


def load_ml_models(model_dir=None):
    models = {}
    base_dir = Path(model_dir) if model_dir else BASE_DIR
    file_map = MODEL_FILES if model_dir else LEGACY_MODEL_FILES

    for name, filename in file_map.items():
        target_file = base_dir / filename
        try:
            loaded_model = load(target_file)
            if not _is_valid_ml_model(loaded_model):
                raise TypeError(f'Model {name} tidak valid: objek {type(loaded_model).__name__} tidak punya method predict.')
            models[name] = loaded_model
        except Exception as exc:
            models[name] = None
            app.logger.error(f'Gagal memuat model {name} dari {target_file}: {exc}')
    return models


def initialize_active_model_state():
    versions = discover_model_versions()
    for active in versions:
        _set_active_model_state(active['version_name'], active['model_dir_path'])
        active_models = load_ml_models(ACTIVE_MODEL_DIR)
        if all(model is not None for model in active_models.values()):
            active['is_active'] = True
            return active_models

    _set_active_model_state(None, None)
    return load_ml_models(None)


def set_active_model_version(version_name):
    target_dir = MODEL_ROOT_DIR / version_name
    if not target_dir.exists():
        raise FileNotFoundError(f'Folder model {version_name} tidak ditemukan.')

    loaded_models = load_ml_models(target_dir)
    missing_models = [name for name, model in loaded_models.items() if model is None]
    if missing_models:
        raise ValueError(f'Model berikut gagal dimuat: {", ".join(missing_models)}')

    _set_active_model_state(version_name, target_dir)
    return loaded_models


def delete_all_model_versions():
    removed_versions = []
    if MODEL_ROOT_DIR.exists():
        for folder in MODEL_ROOT_DIR.iterdir():
            if folder.is_dir():
                shutil.rmtree(folder)
                removed_versions.append(folder.name)

    global ml_models
    ml_models = {}
    _set_active_model_state(None, None)
    return removed_versions


def _run_manual_retrain_job():
    with app.app_context():
        try:
            pending_count = get_pending_predict_user_count()
            force_retrain_pending_predict_users()
            app.logger.info(f'Retrain manual selesai di background. {pending_count} data diproses.')
        except Exception as exc:
            app.logger.exception(f'Gagal menjalankan retrain manual di background: {exc}')


def _safe_display(*args, **kwargs):
    if args:
        print(*args)


def queue_predict_user(user_id, model_name, values, diagnosis, prediction_raw, batch_tag=None):
    queued_row = PredictUser(
        user_id=user_id,
        model_name=model_name,
        input_values=json.dumps(values),
        diagnosis=diagnosis,
        prediction_raw=int(prediction_raw),
        batch_tag=batch_tag,
    )
    db.session.add(queued_row)
    return queued_row


def append_rows_to_dataset(rows):
    header = FEATURE_KEYS + ['addiction_level']
    file_exists = DATASET_PATH.exists()
    with DATASET_PATH.open('a', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def execute_training_notebook(output_dir):
    project_python = BASE_DIR / 'venv' / 'Scripts' / 'python.exe'
    runner = project_python if project_python.exists() else Path(sys.executable)
    notebook_runner = '\n'.join([
        'import json',
        'import os',
        'from pathlib import Path',
        '',
        'try:',
        "    matplotlib = __import__('matplotlib')",
        "    matplotlib.use('Agg')",
        'except Exception:',
        '    pass',
        '',
        f'project_root = Path({str(BASE_DIR)!r})',
        f'notebook_path = Path({str(NOTEBOOK_PATH)!r})',
        f'output_dir = Path({str(Path(output_dir))!r})',
        "os.environ['SMARTADDICT_PROJECT_ROOT'] = str(project_root)",
        f"os.environ['SMARTADDICT_DATASET_PATH'] = str(Path({str(DATASET_PATH)!r}))",
        "os.environ['SMARTADDICT_MODEL_OUTPUT_DIR'] = str(output_dir)",
        '',
        "namespace = {'__name__': '__main__', 'display': print}",
        '',
        "with notebook_path.open('r', encoding='utf-8') as notebook_file:",
        '    notebook = json.load(notebook_file)',
        '',
        "for cell in notebook.get('cells', []):",
        "    if cell.get('cell_type') != 'code':",
        '        continue',
        '',
        "    source = cell.get('source', [])",
        '    if isinstance(source, list):',
        "        source = ''.join(source)",
        '',
        '    cleaned_lines = []',
        '    for line in source.splitlines():',
        '        stripped = line.lstrip()',
        "        if stripped.startswith('%') or stripped.startswith('!'):",
        '            continue',
        "        if 'from google.colab import drive' in line or 'drive.mount(' in line:",
        '            continue',
        '        cleaned_lines.append(line)',
        '',
        "    code = '\\n'.join(cleaned_lines).strip()",
        '    if not code:',
        '        continue',
        '',
        "    exec(compile(code, str(notebook_path), 'exec'), namespace)",
    ])

    temp_runner_path = None
    try:
        with tempfile.NamedTemporaryFile('w', delete=False, suffix='.py', encoding='utf-8') as temp_runner:
            temp_runner.write(notebook_runner)
            temp_runner_path = temp_runner.name

        result = subprocess.run(
            [str(runner), temp_runner_path],
            capture_output=True,
            text=True,
            cwd=str(BASE_DIR),
        )
    finally:
        if temp_runner_path and Path(temp_runner_path).exists():
            try:
                Path(temp_runner_path).unlink()
            except Exception:
                pass

    if result.stdout:
        print(result.stdout, end='')
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or 'Notebook training failed.')

    if result.stderr:
        app.logger.warning(result.stderr.strip())

    return True


def build_run_metadata(namespace, model_version, row_count, output_dir):
    model_metrics = []
    metric_map = [
        ('Support Vector Machine', 'svm_accuracy'),
        ('K-Nearest Neighbors', 'knn_accuracy'),
        ('Decision Tree', 'dt_accuracy'),
        ('Neural Network', 'mlp_accuracy'),
    ]

    for model_name, variable_name in metric_map:
        value = namespace.get(variable_name)
        if isinstance(value, (int, float)):
            model_metrics.append({
                'model': model_name,
                'accuracy': round(float(value), 6),
            })

    average_accuracy = None
    if model_metrics:
        average_accuracy = sum(item['accuracy'] for item in model_metrics) / len(model_metrics)

    return {
        'version_name': model_version,
        'created_at': datetime.utcnow().isoformat(),
        'row_count': row_count,
        'status': 'completed',
        'average_accuracy': round(float(average_accuracy), 6) if average_accuracy is not None else None,
        'model_metrics': model_metrics,
        'model_dir': str(output_dir),
        'dataset_path': str(DATASET_PATH),
    }


def _create_retraining_run(model_version, model_dir, row_count):
    run = RetrainingRun(
        model_version=model_version,
        model_dir=str(model_dir),
        status='running',
        row_count=row_count,
        metrics_json='{}',
    )
    db.session.add(run)
    db.session.commit()
    return run


def _finalize_retraining_run(run_id, metadata, status='completed', error_message=None):
    run = db.session.get(RetrainingRun, run_id)
    if not run:
        return
    run.status = status
    run.finished_at = datetime.utcnow()
    run.error_message = error_message
    run.average_accuracy = metadata.get('average_accuracy')
    run.metrics_json = json.dumps(metadata, ensure_ascii=False)
    db.session.commit()


def retrain_one_batch(pending_rows):
    model_version = f'model_{_get_timestamp_slug()}'
    model_dir = MODEL_ROOT_DIR / model_version
    model_dir.mkdir(parents=True, exist_ok=True)
    backup_path = DATASET_PATH.with_suffix(f'{DATASET_PATH.suffix}.bak_{_get_timestamp_slug()}')
    shutil.copy2(DATASET_PATH, backup_path)

    run = _create_retraining_run(model_version, model_dir, len(pending_rows))

    try:
        dataset_rows = []
        for row in pending_rows:
            values = json.loads(row.input_values)
            dataset_rows.append({
                'age': values[0],
                'gender': values[1],
                'daily_screen_time_hours': values[2],
                'social_media_hours': values[3],
                'gaming_hours': values[4],
                'work_study_hours': values[5],
                'sleep_hours': values[6],
                'notifications_per_day': values[7],
                'app_opens_per_day': values[8],
                'weekend_screen_time': values[9],
                'addiction_level': int(row.prediction_raw),
            })

        append_rows_to_dataset(dataset_rows)
        execute_training_notebook(model_dir)
        metadata = _read_json_file(_resolve_metadata_file(model_dir))
        if not metadata:
            raise RuntimeError('metadata.json tidak dibuat oleh notebook training.')

        new_models = load_ml_models(model_dir)
        missing_models = [name for name, model in new_models.items() if model is None]
        if missing_models:
            raise ValueError(f'Output model belum lengkap: {", ".join(missing_models)}')

        for row in pending_rows:
            row.is_processed = True
            db.session.delete(row)

        db.session.commit()
        global ml_models
        ml_models = new_models
        _set_active_model_state(model_version, model_dir)
        _finalize_retraining_run(run.id, metadata, status='completed')
        app.logger.info(f'Retraining selesai: {model_version}')
        return metadata
    except Exception as exc:
        if backup_path.exists():
            shutil.copy2(backup_path, DATASET_PATH)
        db.session.rollback()
        _finalize_retraining_run(run.id, {'average_accuracy': None, 'model_metrics': []}, status='failed', error_message=str(exc))
        raise


def process_retraining_queue():
    if not RETRAIN_LOCK.acquire(blocking=False):
        app.logger.info('Retraining sedang berjalan, permintaan baru di-skip.')
        return False

    try:
        pending_rows = PredictUser.query.filter_by(is_processed=False).order_by(PredictUser.created_at.asc(), PredictUser.id.asc()).all()
        if len(pending_rows) < RETRAIN_BATCH_SIZE:
            return False

        retrain_one_batch(pending_rows)
        return True
    finally:
        RETRAIN_LOCK.release()


def force_retrain_pending_predict_users():
    if not RETRAIN_LOCK.acquire(blocking=False):
        app.logger.info('Retraining sedang berjalan, permintaan manual di-skip.')
        return False

    try:
        pending_rows = PredictUser.query.filter_by(is_processed=False).order_by(PredictUser.created_at.asc(), PredictUser.id.asc()).all()
        retrain_one_batch(pending_rows)
        return True
    finally:
        RETRAIN_LOCK.release()


def format_model_timestamp(version_name):
    return version_name.replace('model_', '', 1)


ml_models = initialize_active_model_state()

# â•â•â•â•â•â•â• AUTH ROUTES â•â•â•â•â•â•â•
@app.route("/")
def index():
    return redirect(url_for('dashboard'))

@app.route("/login", methods=["GET", "POST"])
def login():
    ensure_default_admin()
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            flash(f"Selamat datang, {user.username}! ðŸ‘‹", "success")
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
            flash("Registrasi berhasil! Selamat datang! ðŸŽ‰", "success")
            return redirect(url_for('dashboard'))
    return render_template("auth/register.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Berhasil logout.", "success")
    return redirect(url_for('login'))

# â•â•â•â•â•â•â• USER ROUTES â•â•â•â•â•â•â•
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
    available_models = list(ml_models.keys())
    selected_model = available_models[0] if available_models else None

    if not available_models:
        errors.append("Tidak ada model aktif di folder model/. Upload atau pilih versi model dulu dari halaman admin.")

    if request.method == "POST":
        selected_model = request.form.get("model") or selected_model
        values = []

        if not available_models:
            return render_template(
                "predict.html",
                questions=QUESTIONS,
                models=[],
                selected_model=None,
                errors=errors,
                active_page='predict'
            )

        if selected_model not in available_models:
            errors.append("Model yang dipilih tidak tersedia di folder model/.")

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
                        batch_tag = f'batch-{current_user.id}-{_get_timestamp_slug()}'
                        dataset_rows = []

                        for row_number, row_values in enumerate(csv_rows, start=1):
                            row_result = predict_with_model(row_values, selected_model, include_comparison=False)
                            batch_rows.append({
                                "row_number": row_number,
                                "values": row_values,
                                "diagnosis": row_result["diagnosis"],
                                "prediction_raw": row_result["prediction_raw"],
                            })
                            dataset_rows.append({
                                'age': row_values[0],
                                'gender': row_values[1],
                                'daily_screen_time_hours': row_values[2],
                                'social_media_hours': row_values[3],
                                'gaming_hours': row_values[4],
                                'work_study_hours': row_values[5],
                                'sleep_hours': row_values[6],
                                'notifications_per_day': row_values[7],
                                'app_opens_per_day': row_values[8],
                                'weekend_screen_time': row_values[9],
                                'addiction_level': int(row_result["prediction_raw"]),
                            })
                            pred_entry = Prediction(
                                user_id=current_user.id,
                                model_name=selected_model,
                                input_values=json.dumps(row_values),
                                result=row_result["diagnosis"],
                                prediction_raw=row_result["prediction_raw"],
                            )
                            db.session.add(pred_entry)
                            queue_predict_user(
                                current_user.id,
                                selected_model,
                                row_values,
                                row_result["diagnosis"],
                                row_result["prediction_raw"],
                                batch_tag=batch_tag,
                            )

                        db.session.commit()
                        values = average_rows(csv_rows)
                        aggregate_result = predict_with_model(values, selected_model, include_comparison=True)
                        distribution = {"Rendah": 0, "Sedang": 0, "Tinggi": 0}
                        for row in batch_rows:
                            if row["diagnosis"] in distribution:
                                distribution[row["diagnosis"]] += 1

                        append_rows_to_dataset(dataset_rows)

                        try:
                            process_retraining_queue()
                        except Exception as retrain_error:
                            app.logger.error(f'Auto retraining gagal setelah batch prediction: {retrain_error}')

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
                queue_predict_user(current_user.id, model_name, values, diagnosis, prediction)
                db.session.commit()

                try:
                    process_retraining_queue()
                except Exception as retrain_error:
                    app.logger.error(f'Auto retraining gagal setelah manual prediction: {retrain_error}')

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

    return render_template("predict.html", questions=QUESTIONS, models=available_models,
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

# â•â•â•â•â•â•â• ADMIN ROUTES â•â•â•â•â•â•â•
@app.route("/admin")
@admin_required
def admin_dashboard():
    ensure_default_admin()
    total_users = User.query.filter_by(role='user').count()
    total_preds = Prediction.query.count()
    pending_predict_users = get_pending_predict_user_count()
    if pending_predict_users == 0:
        queue_status = 'Queue kosong. Tidak ada data di predict_users.'
        queue_status_tone = 'ready'
    elif pending_predict_users < RETRAIN_BATCH_SIZE:
        queue_status = f'Queue masih menunggu retrain otomatis. Saat ini ada {pending_predict_users} data, belum cukup untuk trigger otomatis {RETRAIN_BATCH_SIZE}.'
        queue_status_tone = 'waiting'
    else:
        queue_status = f'Queue siap retrain otomatis. Ada {pending_predict_users} data di predict_users dan akan diproses saat retrain otomatis atau tombol manual ditekan.'
        queue_status_tone = 'trigger'
    all_preds = Prediction.query.all()
    stats = {'Rendah': 0, 'Sedang': 0, 'Tinggi': 0}
    model_usage = {}
    for p in all_preds:
        if p.result in stats:
            stats[p.result] += 1
        model_usage[p.model_name] = model_usage.get(p.model_name, 0) + 1
    recent = Prediction.query.order_by(Prediction.timestamp.desc()).limit(8).all()
    retraining_runs = RetrainingRun.query.order_by(RetrainingRun.created_at.desc()).limit(8).all()
    model_versions = discover_model_versions()
    for version in model_versions:
        version['is_active'] = version['version_name'] == ACTIVE_MODEL_VERSION
    return render_template("admin/dashboard.html", active_page='admin_dashboard',
        total_users=total_users, total_preds=total_preds, stats=stats,
        model_usage=model_usage, recent=recent,
        retraining_runs=retraining_runs,
        model_versions=model_versions,
        active_model_version=ACTIVE_MODEL_VERSION,
        active_model_dir=str(ACTIVE_MODEL_DIR) if ACTIVE_MODEL_DIR else None,
        pending_predict_users=pending_predict_users,
        queue_status=queue_status,
        queue_status_tone=queue_status_tone)


@app.route("/admin/model-version", methods=["POST"])
@admin_required
def admin_set_model_version():
    global ml_models
    version_name = request.form.get('model_version', '').strip()
    if not version_name:
        flash('Versi model tidak ditemukan.', 'error')
        return redirect(url_for('admin_dashboard'))

    try:
        ml_models = set_active_model_version(version_name)
        flash(f'Model aktif berhasil diganti ke {version_name}.', 'success')
    except Exception as exc:
        flash(f'Gagal mengganti model aktif: {exc}', 'error')
    return redirect(url_for('admin_dashboard'))


@app.route("/admin/retrain-now", methods=["POST"])
@admin_required
def admin_manual_retrain():
    try:
        worker = threading.Thread(target=_run_manual_retrain_job, daemon=True)
        worker.start()
        flash('Retrain manual dimulai. Halaman akan tetap responsif, hasil model baru akan muncul setelah proses selesai.', 'success')
    except Exception as exc:
        flash(f'Gagal menjalankan retrain manual: {exc}', 'error')
    return redirect(url_for('admin_dashboard'))


@app.route("/admin/delete-models", methods=["POST"])
@admin_required
def admin_delete_models():
    try:
        removed_versions = delete_all_model_versions()
        flash(f'Semua folder model berhasil dihapus: {len(removed_versions)} versi.', 'success')
    except Exception as exc:
        flash(f'Gagal menghapus folder model: {exc}', 'error')
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

# â•â•â•â•â•â•â• INIT DB & DEFAULT ADMIN â•â•â•â•â•â•â•
with app.app_context():
    db.create_all()
    ensure_default_admin()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5000)

    args = parser.parse_args()

    app.run(debug=True, port=args.port)
