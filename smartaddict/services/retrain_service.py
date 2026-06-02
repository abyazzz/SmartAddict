import importlib
import json
import logging
import subprocess
import sys
import tempfile
import threading
import uuid
from datetime import datetime
from pathlib import Path

from flask import current_app, has_app_context

from smartaddict.config import DATASET_PATH, MODEL_ROOT_DIR, NOTEBOOK_PATH, STATUS_DIR
from smartaddict.extensions import db
from smartaddict.models.predict_user_session import PredictUserSession
from smartaddict.services.model_service import get_venv_python_executable, load_model_version, save_active_version_to_config
from smartaddict.utils.constants import RETRAIN_STEP_PLAN


RETRAIN_LOCK = threading.Lock()
IS_RETRAINING = False
RETRAIN_JOB_ID = None


def _get_logger():
    if has_app_context():
        return current_app.logger
    return logging.getLogger(__name__)


def build_retrain_steps(current_step=None, completed_steps=None, finished=False):
    completed_steps = set(completed_steps or [])
    steps = []
    for index, name in enumerate(RETRAIN_STEP_PLAN):
        if finished or name in completed_steps:
            status = "done"
        elif current_step == name:
            status = "running"
        else:
            status = "pending"
        steps.append({"name": name, "status": status, "index": index, "is_current": name == current_step})
    return steps


def _default_retrain_status(job_id=None):
    return {
        "job_id": job_id,
        "triggered_at": None,
        "started_at": None,
        "finished_at": None,
        "status": "idle",
        "progress": 0,
        "current_step": None,
        "steps": build_retrain_steps(),
        "logs": [],
        "metrics": {},
        "model_artifact": None,
    }


def _status_path(job_id):
    return STATUS_DIR / f"{job_id}.json"


def write_status(job_id, payload):
    payload = dict(payload or {})
    payload.setdefault("job_id", job_id)
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    with _status_path(job_id).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def read_status(job_id):
    try:
        with _status_path(job_id).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def get_current_retrain_status():
    if RETRAIN_JOB_ID:
        current = read_status(RETRAIN_JOB_ID)
        if current and current.get("status") in ("pending", "running"):
            return current
    return _default_retrain_status()


def get_latest_retrain_status():
    statuses = list_statuses()
    return statuses[0] if statuses else None


def list_statuses():
    items = []
    if not STATUS_DIR.exists():
        return items
    for fname in STATUS_DIR.iterdir():
        if fname.suffix != ".json":
            continue
        try:
            with fname.open("r", encoding="utf-8") as handle:
                items.append(json.load(handle))
        except Exception:
            pass
    items.sort(key=lambda item: item.get("started_at") or item.get("triggered_at") or "", reverse=True)
    return items


def list_statuses_paginated(page=1, per_page=20):
    all_items = list_statuses()
    total = len(all_items)
    try:
        page = int(page)
    except Exception:
        page = 1
    try:
        per_page = int(per_page)
    except Exception:
        per_page = 20
    if per_page <= 0:
        per_page = 20
    start = (page - 1) * per_page
    end = start + per_page
    items = all_items[start:end]
    total_pages = (total + per_page - 1) // per_page
    return {"items": items, "total": total, "page": page, "per_page": per_page, "total_pages": total_pages}


def cleanup_statuses(retain_days=30, max_entries=200):
    files = []
    if not STATUS_DIR.exists():
        return 0
    for fname in STATUS_DIR.iterdir():
        if not fname.name.endswith(".json"):
            continue
        try:
            files.append((fname.stat().st_mtime, fname))
        except Exception:
            pass
    files.sort(reverse=True)
    cutoff = None
    if retain_days is not None and retain_days > 0:
        cutoff = datetime.utcnow().timestamp() - (retain_days * 86400)
    removed = 0
    for mtime, path in list(files):
        if cutoff and mtime < cutoff:
            try:
                path.unlink()
                files.remove((mtime, path))
                removed += 1
            except Exception:
                pass
    if max_entries is not None and max_entries > 0 and len(files) > max_entries:
        for mtime, path in files[max_entries:]:
            try:
                path.unlink()
                removed += 1
            except Exception:
                pass
    return removed


def append_log(job_id, level, message):
    status = read_status(job_id) or {}
    logs = status.get("logs", [])
    logs.append({"ts": datetime.utcnow().isoformat() + "Z", "level": level, "message": message})
    status["logs"] = logs
    write_status(job_id, status)


def update_progress(job_id, **kwargs):
    status = read_status(job_id) or {}
    status.update(kwargs)
    write_status(job_id, status)


def finish_retrain_job(job_id, model_artifact=None, metrics=None):
    status = read_status(job_id) or _default_retrain_status(job_id)
    status["status"] = "success"
    status["finished_at"] = datetime.utcnow().isoformat() + "Z"
    status["progress"] = 100
    status["steps"] = build_retrain_steps(finished=True)
    if model_artifact:
        status["model_artifact"] = model_artifact
    if metrics is not None:
        status["metrics"] = metrics
    logs = status.get("logs", [])
    logs.append({"ts": datetime.utcnow().isoformat() + "Z", "level": "INFO", "message": "RETRAIN SELESAI"})
    status["logs"] = logs
    write_status(job_id, status)


def _get_timestamp_slug():
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _read_json_file(file_path):
    try:
        if file_path.exists():
            return json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _resolve_metadata_file(model_dir):
    return Path(model_dir) / "metadata.json"


def execute_training_notebook(output_dir, job_id=None):
    project_root = Path(__file__).resolve().parent.parent.parent
    runner = Path(get_venv_python_executable() or sys.executable)
    notebook_path = Path(NOTEBOOK_PATH)
    dataset_path = Path(DATASET_PATH)
    output_dir = Path(output_dir)

    status_file = _status_path(job_id) if job_id else None

    notebook_runner = "\n".join([
        "import json",
        "import os",
        "from pathlib import Path",
        "",
        "try:",
        "    matplotlib = __import__('matplotlib')",
        "    matplotlib.use('Agg')",
        "except Exception:",
        "    pass",
        "",
        f"project_root = Path({str(project_root)!r})",
        f"notebook_path = Path({str(notebook_path)!r})",
        f"output_dir = Path({str(output_dir)!r})",
        f"dataset_path = Path({str(dataset_path)!r})",
        f"job_id = {job_id!r}",
        f"status_file_path = {str(status_file)!r}",
        "os.environ['SMARTADDICT_PROJECT_ROOT'] = str(project_root)",
        "os.environ['SMARTADDICT_DATASET_PATH'] = str(dataset_path)",
        "os.environ['OUTPUT_MODEL_DIR'] = str(output_dir)",
        "os.environ['SMARTADDICT_MODEL_OUTPUT_DIR'] = str(output_dir)",
        "if job_id:",
        "    os.environ['RETRAIN_JOB_ID'] = str(job_id)",
        "if status_file_path:",
        "    os.environ['RETRAIN_STATUS_FILE'] = str(status_file_path)",
        "",
        "namespace = {",
        "    '__name__': '__main__',",
        "    'display': print,",
        "    'output_model_dir': str(output_dir),",
        "    'job_id': job_id,",
        "    'status_file_path': status_file_path,",
        "}",
        "",
        "with notebook_path.open('r', encoding='utf-8') as notebook_file:",
        "    notebook = json.load(notebook_file)",
        "",
        "for cell in notebook.get('cells', []):",
        "    if cell.get('cell_type') != 'code':",
        "        continue",
        "",
        "    source = cell.get('source', [])",
        "    if isinstance(source, list):",
        "        source = ''.join(source)",
        "",
        "    cleaned_lines = []",
        "    for line in source.splitlines():",
        "        stripped = line.lstrip()",
        "        if stripped.startswith('%') or stripped.startswith('!'):",
        "            continue",
        "        if 'from google.colab import drive' in line or 'drive.mount(' in line:",
        "            continue",
        "        cleaned_lines.append(line)",
        "",
        "    code = '\\n'.join(cleaned_lines).strip()",
        "    if not code:",
        "        continue",
        "",
        "    exec(compile(code, str(notebook_path), 'exec'), namespace)",
    ])

    temp_runner_path = None
    result = None
    try:
        with tempfile.NamedTemporaryFile('w', delete=False, suffix='.py', encoding='utf-8') as temp_runner:
            temp_runner.write(notebook_runner)
            temp_runner_path = temp_runner.name

        result = subprocess.run([str(runner), temp_runner_path], capture_output=True, text=True, cwd=str(project_root))
    finally:
        if temp_runner_path:
            try:
                Path(temp_runner_path).unlink(missing_ok=True)
            except Exception:
                pass

    if result and result.stdout:
        _get_logger().info(result.stdout.strip())
    if not result or result.returncode != 0:
        raise RuntimeError((result.stderr.strip() if result and result.stderr else "") or (result.stdout.strip() if result and result.stdout else "") or "Notebook training failed.")
    if result.stderr:
        _get_logger().warning(result.stderr.strip())
    return True


def _refresh_app_model_state(version_name):
    try:
        app_module = importlib.import_module("app")
        if hasattr(app_module, "init_active_model"):
            app_module.init_active_model()
            return
        models, scaler_obj, success = load_model_version(version_name)
        if success:
            if hasattr(app_module, "ml_models"):
                app_module.ml_models = models
            if hasattr(app_module, "scaler"):
                app_module.scaler = scaler_obj
    except Exception:
        _get_logger().exception("Gagal memuat ulang state model aktif setelah retrain")


def _reset_predict_user_sessions(app_instance):
    with app_instance.app_context():
        removed = PredictUserSession.query.delete(synchronize_session=False)
        db.session.commit()
    return removed


def _execute_retrain_job(app_instance, job_id):
    version_name = f"model_{_get_timestamp_slug()}"
    model_dir = MODEL_ROOT_DIR / version_name
    model_dir.mkdir(parents=True, exist_ok=True)

    update_progress(job_id, status="running", started_at=datetime.utcnow().isoformat() + "Z", progress=5, current_step="Load library", steps=build_retrain_steps(current_step="Load library"))
    append_log(job_id, "INFO", "Memulai retrain notebook")

    try:
        update_progress(job_id, progress=15, current_step="Load dataset", steps=build_retrain_steps(current_step="Load dataset"))
        execute_training_notebook(model_dir, job_id=job_id)

        update_progress(job_id, progress=85, current_step="Deploy", steps=build_retrain_steps(current_step="Deploy"))
        metadata = _read_json_file(_resolve_metadata_file(model_dir))
        metrics_payload = _read_json_file(model_dir / "metrics.json")
        metrics_for_status = metadata.get("model_metrics") or metadata.get("metrics") or metrics_payload.get("metrics") or metrics_payload
        models, scaler_obj, success = load_model_version(version_name)
        if not success:
            raise RuntimeError("Model hasil retrain belum lengkap atau gagal dimuat.")

        save_active_version_to_config(version_name)
        _refresh_app_model_state(version_name)
        removed_sessions = _reset_predict_user_sessions(app_instance)
        app_instance.logger.info("Reset tabel predict_user_session: %s baris dihapus", removed_sessions)

        finish_retrain_job(job_id, model_artifact=str(model_dir), metrics=metrics_for_status or {})
        app_instance.logger.info("Retrain pipeline selesai: %s", version_name)
        return version_name
    except Exception as exc:
        try:
            if model_dir.exists() and not any(model_dir.iterdir()):
                model_dir.rmdir()
        except Exception:
            pass
        update_progress(job_id, status="failed", finished_at=datetime.utcnow().isoformat() + "Z", progress=100, current_step=None)
        append_log(job_id, "ERROR", f"Retrain gagal: {exc}")
        app_instance.logger.exception("Retrain pipeline gagal")
        raise


def run_retrain_pipeline(app_instance):
    global IS_RETRAINING, RETRAIN_JOB_ID

    if not RETRAIN_LOCK.acquire(blocking=False):
        _get_logger().info("Retraining sedang berjalan, permintaan baru di-skip.")
        return None

    IS_RETRAINING = True
    job_id = str(uuid.uuid4())
    RETRAIN_JOB_ID = job_id
    write_status(job_id, _default_retrain_status(job_id))
    append_log(job_id, "INFO", "Retrain job dimulai")

    def job():
        global IS_RETRAINING, RETRAIN_JOB_ID
        try:
            _execute_retrain_job(app_instance, job_id)
        except Exception:
            pass
        finally:
            IS_RETRAINING = False
            if RETRAIN_JOB_ID == job_id:
                RETRAIN_JOB_ID = None
            try:
                RETRAIN_LOCK.release()
            except RuntimeError:
                pass

    thread = threading.Thread(target=job, daemon=True)
    thread.start()
    return job_id
