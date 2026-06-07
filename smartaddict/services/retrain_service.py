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


def should_trigger_retrain(session_count_before, inserted_count, threshold=50):
    try:
        session_count_before = int(session_count_before)
        inserted_count = int(inserted_count)
        threshold = int(threshold)
    except Exception:
        return False
    if inserted_count <= 0 or threshold <= 0:
        return False
    return session_count_before < threshold <= (session_count_before + inserted_count)


def maybe_trigger_retrain(app_instance, session_count_before, inserted_count, threshold=50):
    """
    Check if retrain should be triggered based on threshold.
    NOTE: Auto-trigger is DISABLED. This function now only logs and returns None.
    Admin must manually trigger retrain from admin panel after reviewing data.
    """
    if should_trigger_retrain(session_count_before, inserted_count, threshold=threshold):
        app_instance.logger.info(
            "Data mencapai threshold retrain: before=%s added=%s threshold=%s. "
            "Admin perlu review data di halaman retrain dan jalankan retrain manual.",
            session_count_before,
            inserted_count,
            threshold,
        )
    # Return None - auto-trigger is disabled
    return None


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
    """
    Get current retrain status, with stuck job detection.
    If a job has been running for more than 2 hours, it's considered stuck/failed.
    """
    if RETRAIN_JOB_ID:
        current = read_status(RETRAIN_JOB_ID)
        if current and current.get("status") in ("pending", "running"):
            # Check if job is stuck (running > 2 hours)
            started_at = current.get("started_at")
            if started_at:
                try:
                    from datetime import datetime
                    start_time = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
                    now = datetime.utcnow()
                    elapsed_hours = (now - start_time.replace(tzinfo=None)).total_seconds() / 3600
                    
                    if elapsed_hours > 2:
                        # Mark job as failed due to timeout
                        _get_logger().warning(f"Job {RETRAIN_JOB_ID} stuck for {elapsed_hours:.1f} hours, marking as failed")
                        current["status"] = "failed"
                        current["finished_at"] = datetime.utcnow().isoformat() + "Z"
                        append_log(RETRAIN_JOB_ID, "ERROR", f"Job timeout setelah {elapsed_hours:.1f} jam")
                        write_status(RETRAIN_JOB_ID, current)
                        return _default_retrain_status()
                except Exception as e:
                    _get_logger().error(f"Error checking job timeout: {e}")
            
            return current
    
    # Also check latest status file in case app was restarted
    latest = get_latest_retrain_status()
    if latest and latest.get("status") in ("pending", "running"):
        started_at = latest.get("started_at")
        if started_at:
            try:
                from datetime import datetime
                start_time = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
                now = datetime.utcnow()
                elapsed_hours = (now - start_time.replace(tzinfo=None)).total_seconds() / 3600
                
                if elapsed_hours > 2:
                    # Mark stuck job as failed
                    job_id = latest.get("job_id")
                    _get_logger().warning(f"Found stuck job {job_id} from {elapsed_hours:.1f} hours ago, marking as failed")
                    latest["status"] = "failed"
                    latest["finished_at"] = datetime.utcnow().isoformat() + "Z"
                    append_log(job_id, "ERROR", f"Job timeout setelah {elapsed_hours:.1f} jam (detected after app restart)")
                    write_status(job_id, latest)
                    return _default_retrain_status()
                else:
                    return latest
            except Exception as e:
                _get_logger().error(f"Error checking stuck job: {e}")
    
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
    """Reload runtime model state after successful retrain"""
    try:
        # Import runtime module and force reload
        import smartaddict.runtime as runtime
        
        # Force reload the model state
        runtime.init_active_model()
        
        # Verify the reload worked
        if runtime.ACTIVE_MODEL_VERSION == version_name:
            _get_logger().info(f"Runtime model state berhasil di-reload ke {version_name}")
            _get_logger().info(f"Available models: {list(runtime.ml_models.keys())}")
            return True
        else:
            _get_logger().error(
                f"Runtime reload failed: expected {version_name}, got {runtime.ACTIVE_MODEL_VERSION}"
            )
            return False
    except Exception as exc:
        _get_logger().exception(f"Gagal memuat ulang state model aktif setelah retrain: {exc}")
        return False


def _append_session_data_to_csv(app_instance):
    """
    Append all data from predict_user_session table to CSV dataset before training.
    Returns tuple: (rows_added, validation_passed, message)
    """
    import csv
    import pandas as pd
    from pathlib import Path
    
    try:
        with app_instance.app_context():
            session_data = PredictUserSession.query.all()
            session_count = len(session_data)
            
            _get_logger().info(f"Found {session_count} rows in predict_user_session table")
            
            if session_count == 0:
                msg = "Tidak ada data di predict_user_session untuk ditambahkan ke dataset"
                _get_logger().info(msg)
                return (0, True, msg)
            
            csv_path = Path(DATASET_PATH)
            _get_logger().info(f"CSV path: {csv_path}")
            
            # STEP 1: Count rows BEFORE insert
            try:
                df_before = pd.read_csv(csv_path)
                rows_before = len(df_before)
                columns = df_before.columns.tolist()
                _get_logger().info(f"CSV sebelum insert: {rows_before} rows, columns: {columns}")
            except Exception as e:
                error_msg = f"Gagal membaca CSV dataset: {e}"
                _get_logger().error(error_msg)
                return (0, False, error_msg)
            
            # STEP 2: Prepare new rows from session data
            new_rows = []
            for i, data in enumerate(session_data):
                try:
                    # Generate transaction ID with timestamp
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    transaction_id = f"TXN{timestamp}_{i+1:02d}"
                    
                    # Get user_id from data or default
                    user_id = f"U{data.user_id:05d}" if hasattr(data, 'user_id') and data.user_id else "U00000"
                    
                    # Convert gender: 1=Male, 0=Female
                    gender_text = "Male" if data.gender == 1 else "Female"
                    
                    # Convert addiction_level text to number: Rendah=0, Sedang=1, Tinggi=2
                    addiction_map = {"Rendah": 0, "Sedang": 1, "Tinggi": 2}
                    addicted_label = addiction_map.get(data.result, 1)  # Default 1 if not found
                    
                    # Build row matching CSV column order
                    row = {
                        'transaction_id': transaction_id,
                        'user_id': user_id,
                        'age': int(data.age),
                        'gender': gender_text,
                        'daily_screen_time_hours': float(data.daily_screen_time_hours),
                        'social_media_hours': float(data.social_media_hours),
                        'gaming_hours': float(data.gaming_hours),
                        'work_study_hours': float(data.work_study_hours),
                        'sleep_hours': float(data.sleep_hours),
                        'notifications_per_day': int(data.notifications_per_day),
                        'app_opens_per_day': int(data.app_opens_per_day),
                        'weekend_screen_time': float(data.weekend_screen_time),
                        'stress_level': 'Medium',  # Default value
                        'academic_work_impact': 'Yes',  # Default value
                        'addiction_level': data.result,  # "Rendah", "Sedang", "Tinggi"
                        'addicted_label': addicted_label  # 0, 1, or 2
                    }
                    
                    new_rows.append(row)
                except Exception as e:
                    _get_logger().error(f"Error converting row {i}: {e}")
                    return (0, False, f"Error converting data row {i}: {e}")
            
            _get_logger().info(f"Prepared {len(new_rows)} rows for insert")
            
            # STEP 3: Append to CSV file
            try:
                with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=columns)
                    for row in new_rows:
                        writer.writerow(row)
                
                _get_logger().info(f"Berhasil write {len(new_rows)} rows ke CSV")
            except Exception as e:
                error_msg = f"Gagal menulis ke CSV: {e}"
                _get_logger().error(error_msg)
                return (0, False, error_msg)
            
            # STEP 4: Count rows AFTER insert
            try:
                df_after = pd.read_csv(csv_path)
                rows_after = len(df_after)
                _get_logger().info(f"CSV setelah insert: {rows_after} rows")
            except Exception as e:
                error_msg = f"Gagal membaca CSV setelah insert: {e}"
                _get_logger().error(error_msg)
                return (0, False, error_msg)
            
            # STEP 5: VALIDATE - Selisih harus sama dengan jumlah session data
            rows_diff = rows_after - rows_before
            validation_passed = (rows_diff == session_count)
            
            if validation_passed:
                msg = f"✅ VALID: Dataset bertambah {rows_diff} rows (sesuai {session_count} data session). Total dataset: {rows_after} rows"
                _get_logger().info(msg)
                return (rows_diff, True, msg)
            else:
                msg = f"❌ INVALID: Dataset bertambah {rows_diff} rows, tapi data session ada {session_count} rows. Tidak match!"
                _get_logger().error(msg)
                return (rows_diff, False, msg)
                
    except Exception as e:
        error_msg = f"Exception in _append_session_data_to_csv: {str(e)}"
        _get_logger().exception(error_msg)
        return (0, False, error_msg)


def _reset_predict_user_sessions(app_instance):
    with app_instance.app_context():
        removed = PredictUserSession.query.delete(synchronize_session=False)
        db.session.commit()
    return removed


def _execute_retrain_job(app_instance, job_id):
    version_name = f"model_{_get_timestamp_slug()}"
    model_dir = MODEL_ROOT_DIR / version_name
    model_dir.mkdir(parents=True, exist_ok=True)

    update_progress(job_id, status="running", started_at=datetime.utcnow().isoformat() + "Z", progress=3, current_step="Insert data session ke dataset", steps=build_retrain_steps(current_step="Insert data session ke dataset"))
    append_log(job_id, "INFO", "Memulai retrain pipeline")

    try:
        # STEP 0: Insert session data to CSV dataset
        append_log(job_id, "INFO", "📊 STEP 1: Menambahkan data dari predict_user_session ke dataset CSV...")
        
        rows_added, validation_passed, validation_msg = _append_session_data_to_csv(app_instance)
        
        append_log(job_id, "INFO", validation_msg)
        app_instance.logger.info(f"Retrain CSV insert: {validation_msg}")
        
        # If validation failed, stop retrain
        if not validation_passed:
            error_msg = f"Validasi insert data GAGAL! {validation_msg}"
            append_log(job_id, "ERROR", error_msg)
            # Mark step as failed and stop
            update_progress(
                job_id, 
                status="failed", 
                finished_at=datetime.utcnow().isoformat() + "Z", 
                progress=5, 
                current_step=None
            )
            raise RuntimeError(error_msg)
        
        # Mark insert step as DONE
        completed_steps = ["Insert data session ke dataset"]
        update_progress(
            job_id, 
            progress=7, 
            current_step="Load library", 
            steps=build_retrain_steps(current_step="Load library", completed_steps=completed_steps)
        )
        append_log(job_id, "INFO", "✅ Insert data session ke dataset: SELESAI")
        
        # STEP 1: Load library and run notebook
        update_progress(job_id, progress=8, current_step="Load library", steps=build_retrain_steps(current_step="Load library"))
        append_log(job_id, "INFO", "📚 STEP 2: Menjalankan notebook training dengan dataset yang sudah diupdate...")
        
        update_progress(job_id, progress=15, current_step="Load dataset", steps=build_retrain_steps(current_step="Load dataset"))
        execute_training_notebook(model_dir, job_id=job_id)

        # STEP 2: Deploy model
        update_progress(job_id, progress=85, current_step="Deploy", steps=build_retrain_steps(current_step="Deploy"))
        append_log(job_id, "INFO", f"🚀 STEP 3: Notebook selesai, deploy model ke {version_name}")
        
        metadata = _read_json_file(_resolve_metadata_file(model_dir))
        metrics_payload = _read_json_file(model_dir / "metrics.json")
        metrics_for_status = metadata.get("model_metrics") or metadata.get("metrics") or metrics_payload.get("metrics") or metrics_payload
        
        append_log(job_id, "INFO", "Memuat dan validasi model hasil retrain...")
        models, scaler_obj, success = load_model_version(version_name)
        if not success:
            raise RuntimeError("Model hasil retrain belum lengkap atau gagal dimuat.")

        append_log(job_id, "INFO", f"Model berhasil dimuat: {list(models.keys())}")
        
        # STEP 3: Auto-select model with highest accuracy
        append_log(job_id, "INFO", "🎯 STEP 4: Memilih model dengan akurasi tertinggi...")
        from smartaddict.services.model_service import select_and_activate_best_model
        
        best_model_name, best_accuracy = select_and_activate_best_model()
        if best_model_name:
            append_log(
                job_id, 
                "INFO", 
                f"Model terbaik dipilih dan diaktifkan: {best_model_name} (akurasi: {best_accuracy * 100:.2f}%)"
            )
        else:
            # Fallback: activate the newly created model
            append_log(job_id, "INFO", f"Fallback: Mengaktifkan model baru: {version_name}")
            save_active_version_to_config(version_name)
        
        append_log(job_id, "INFO", "Reload runtime model state...")
        reload_success = _refresh_app_model_state(best_model_name or version_name)
        if not reload_success:
            app_instance.logger.warning("Runtime reload gagal atau tidak optimal, tapi model sudah tersimpan")
        
        # STEP 4: Reset predict_user_session table
        append_log(job_id, "INFO", "🗑️ STEP 5: Reset tabel predict_user_session...")
        removed_sessions = _reset_predict_user_sessions(app_instance)
        app_instance.logger.info("Reset tabel predict_user_session: %s baris dihapus", removed_sessions)
        append_log(job_id, "INFO", f"Tabel predict_user_session di-reset: {removed_sessions} baris dihapus")

        finish_retrain_job(job_id, model_artifact=str(model_dir), metrics=metrics_for_status or {})
        app_instance.logger.info("Retrain pipeline selesai: %s", version_name)
        append_log(job_id, "INFO", f"✅ Retrain pipeline SELESAI: {version_name}")
        return version_name
    except Exception as exc:
        try:
            if model_dir.exists() and not any(model_dir.iterdir()):
                model_dir.rmdir()
        except Exception:
            pass
        update_progress(job_id, status="failed", finished_at=datetime.utcnow().isoformat() + "Z", progress=100, current_step=None)
        append_log(job_id, "ERROR", f"❌ Retrain gagal: {exc}")
        app_instance.logger.exception("Retrain pipeline gagal")
        raise


def run_retrain_pipeline(app_instance):
    global IS_RETRAINING, RETRAIN_JOB_ID

    # Check if there's already a running job
    if IS_RETRAINING:
        _get_logger().warning("Retraining already in progress (IS_RETRAINING=True), request skipped.")
        return None

    # Try to acquire lock, with check for stuck jobs
    if not RETRAIN_LOCK.acquire(blocking=False):
        # Check if the current job is stuck
        current = get_current_retrain_status()
        if current and current.get("status") == "idle":
            # The lock is held but no job is actually running, force release
            _get_logger().warning("RETRAIN_LOCK held but no active job, attempting force release...")
            try:
                RETRAIN_LOCK.release()
                # Try to acquire again
                if not RETRAIN_LOCK.acquire(blocking=False):
                    _get_logger().error("Failed to acquire lock even after force release")
                    return None
            except RuntimeError:
                _get_logger().error("Failed to release stuck lock")
                return None
        else:
            _get_logger().info("Retraining sedang berjalan, permintaan baru di-skip.")
            return None

    IS_RETRAINING = True
    job_id = str(uuid.uuid4())
    RETRAIN_JOB_ID = job_id
    
    # Initialize status file
    initial_status = _default_retrain_status(job_id)
    initial_status["triggered_at"] = datetime.utcnow().isoformat() + "Z"
    write_status(job_id, initial_status)
    append_log(job_id, "INFO", "Retrain job dimulai")

    def job():
        global IS_RETRAINING, RETRAIN_JOB_ID
        try:
            _execute_retrain_job(app_instance, job_id)
        except Exception as exc:
            _get_logger().exception(f"Retrain job {job_id} failed with exception: {exc}")
        finally:
            IS_RETRAINING = False
            if RETRAIN_JOB_ID == job_id:
                RETRAIN_JOB_ID = None
            try:
                RETRAIN_LOCK.release()
            except RuntimeError:
                _get_logger().warning(f"Lock already released for job {job_id}")

    thread = threading.Thread(target=job, daemon=True)
    thread.start()
    _get_logger().info(f"Retrain job {job_id} started in background thread")
    return job_id
