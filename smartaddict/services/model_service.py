import json
import os
import logging
from pathlib import Path

from flask import has_app_context, current_app
from joblib import load

from smartaddict.config import CONFIG_PATH, MODEL_ROOT_DIR
from smartaddict.utils.constants import LEGACY_MODEL_FILES, SCALER_FILE_CANDIDATES


def _get_logger():
    if has_app_context():
        return current_app.logger
    return logging.getLogger(__name__)


def get_venv_python_executable():
    project_root = Path(__file__).resolve().parent.parent.parent
    venv_python = project_root / "venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    return None


def get_active_version_from_config():
    if CONFIG_PATH.exists():
        try:
            with CONFIG_PATH.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
                return data.get("active_model")
        except Exception:
            pass
    return None


def save_active_version_to_config(version_name):
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with CONFIG_PATH.open("w", encoding="utf-8") as handle:
            json.dump({"active_model": version_name}, handle, indent=2)
    except Exception as exc:
        _get_logger().error(f"Gagal menyimpan config versi model: {exc}")


def get_available_retrain_versions(active_model_version=None):
    versions = []
    if not MODEL_ROOT_DIR.exists():
        return []

    for item in sorted(MODEL_ROOT_DIR.iterdir(), key=lambda path: path.name, reverse=True):
        if not item.is_dir() or not item.name.startswith("model_") or item.name == "model_default":
            continue

        metrics = {"dt": 0.0, "knn": 0.0, "nn": 0.0, "svm": 0.0}
        avg_accuracy = 0.0
        metrics_sources = [item / "metrics.json", item / "metadata.json"]

        for metrics_path in metrics_sources:
            if not metrics_path.exists():
                continue
            try:
                with metrics_path.open("r", encoding="utf-8") as handle:
                    raw_metrics = json.load(handle)

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
            except Exception as exc:
                _get_logger().error(
                    f"Gagal membaca metrics di {item.name} dari {metrics_path.name}: {exc}"
                )

        versions.append(
            {
                "version_name": item.name,
                "average_accuracy": avg_accuracy,
                "metrics": metrics,
                "is_active": item.name == active_model_version,
            }
        )

    return versions


def load_model_version(version_name):
    base_path = MODEL_ROOT_DIR / version_name
    models = {}
    if not base_path.exists():
        return None, None, False

    loaded_any_model = False
    for name, filenames in LEGACY_MODEL_FILES.items():
        loaded_model = None
        last_error = None
        for filename in filenames:
            model_path = base_path / filename
            if not model_path.exists():
                continue
            try:
                loaded_model = load(model_path)
                break
            except Exception as exc:
                last_error = exc
                _get_logger().error(f"Gagal memuat model {name} dari {model_path}: {exc}")
        if loaded_model is None:
            if last_error is None:
                _get_logger().error(f"File model {name} tidak ditemukan di {base_path}")
            continue
        models[name] = loaded_model
        loaded_any_model = True

    scaler_obj = None
    last_scaler_error = None
    for filename in SCALER_FILE_CANDIDATES:
        scaler_path = base_path / filename
        if not scaler_path.exists():
            continue
        try:
            scaler_obj = load(scaler_path)
            break
        except Exception as exc:
            last_scaler_error = exc
            _get_logger().error(f"Gagal memuat scaler dari {scaler_path}: {exc}")

    if scaler_obj is None:
        if last_scaler_error is None:
            _get_logger().error(f"File scaler tidak ditemukan di {base_path}")
        return None, None, False
    if not loaded_any_model:
        return None, None, False
    return models, scaler_obj, True