import json
import os
import csv
from io import StringIO

from flask import Blueprint, Response, current_app, flash, redirect, render_template, request, send_file, session, url_for
from flask_login import current_user, login_required

import smartaddict.runtime as runtime
from smartaddict.extensions import db
from smartaddict.models.prediction import Prediction
from smartaddict.models.predict_user_session import PredictUserSession
from smartaddict.models.user import User
from smartaddict.services.csv_service import average_rows, parse_csv_rows
from smartaddict.services.dashboard_feature_service import get_dashboard_feature_cards
from smartaddict.services.retrain_service import maybe_trigger_retrain
from smartaddict.services.prediction_service import process_batch, process_single
from smartaddict.utils.constants import FEATURE_KEYS, MODEL_FILES, QUESTIONS

user_bp = Blueprint('user', __name__)


def _commit_prediction_payload(prediction_entries, session_entries):
    db.session.add_all(prediction_entries)
    db.session.add_all(session_entries)
    db.session.commit()


def _remember_prediction_input(values, selected_model, input_method, batch_mode=False, batch_rows=None):
    payload = {
        "values": values,
        "model": selected_model,
        "input_method": input_method,
        "batch_mode": batch_mode,
    }
    if batch_mode and batch_rows:
        payload["batch_rows"] = [
            {
                "row_number": row.get("row_number"),
                "values": row.get("values", []),
            }
            for row in batch_rows
        ]
    session["last_prediction_input"] = payload
    session.modified = True


def _build_batch_session_payload(result, selected_model, values, aggregate_result):
    return {
        "values": values,
        "labels": FEATURE_KEYS,
        "diagnosis": aggregate_result["diagnosis"],
        "model": selected_model,
        "prediction_raw": aggregate_result["prediction_raw"],
        "comparison": aggregate_result.get("comparison"),
        "batch_mode": True,
        "batch_count": result["batch_count"],
        "batch_rows": result["batch_rows"],
        "distribution": result["distribution"],
        "input_method": "csv",
        "batch_stats": {
            "male_count": result["male_count"],
            "female_count": result["female_count"],
            "avg_age": values[0],
            "avg_screen_time": values[2],
            "avg_social_media": values[3],
            "avg_gaming": values[4],
            "avg_sleep": values[6],
            "avg_notifications": values[7],
            "avg_app_opens": values[8],
        },
        "feature_averages": values,
    }


@user_bp.route("/profile", methods=["GET", "POST"], endpoint='profile')
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

        # Removed error flash messages - validation errors no longer shown
        if not errors:
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
                return redirect(url_for('user.profile'))
            # Removed warning flash for no changes

    return render_template(
        "profile.html",
        active_page='profile',
        prediction_count=prediction_count,
        latest_prediction=latest_prediction,
    )


@user_bp.route("/dashboard", endpoint='dashboard')
def dashboard():
    averages = runtime.get_feature_averages()
    recent_preds = Prediction.query.order_by(Prediction.timestamp.desc()).limit(6).all()
    feature_cards = get_dashboard_feature_cards()
    return render_template(
        "dashboard.html",
        active_page='dashboard',
        averages=averages,
        recent_preds=recent_preds,
        feature_cards=feature_cards,
    )


@user_bp.route("/predict", methods=["GET", "POST"], endpoint='predict')
@login_required
def predict():
    if request.args.get("reset") == "1":
        session.pop("last_prediction_input", None)

    remembered_input = session.get("last_prediction_input") or {}
    selected_model = remembered_input.get("model", "Decision Tree") if request.method == "GET" else "Decision Tree"
    errors = []

    if request.method == "POST":
        selected_model = request.form.get("model") or selected_model
        values = []
        input_method = "manual"

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

        elif 'reuse_batch_submit' in request.form:
            remembered_rows = (session.get("last_prediction_input") or {}).get("batch_rows") or []
            csv_rows = [row.get("values", []) for row in remembered_rows if row.get("values")]
            if not csv_rows:
                errors.append("Tidak ada file CSV terakhir yang bisa diproses ulang.")
            else:
                try:
                    result = process_batch(csv_rows, selected_model, current_user.id, trigger_app=current_app._get_current_object())
                    values = result["values"]
                    aggregate_result = result["aggregate"]
                    last_payload = _build_batch_session_payload(result, selected_model, values, aggregate_result)
                    session['last_prediction'] = last_payload
                    _remember_prediction_input(values, selected_model, "csv", batch_mode=True, batch_rows=result["batch_rows"])
                    flash(f"Prediksi ulang CSV berhasil! {result['batch_count']} baris diproses dengan model {selected_model}.", "success")
                    return redirect(url_for('user.thanks'))
                except Exception as e:
                    errors.append(f"Error memproses ulang CSV terakhir: {str(e)}")

        elif 'csv_submit' in request.form:
            file = request.files.get('csv_file')
            if file and file.filename.endswith('.csv'):
                try:
                    csv_rows, _ = parse_csv_rows(file)
                    if not csv_rows:
                        errors.append("CSV tidak memiliki baris data.")
                    elif len(csv_rows) == 1:
                        input_method = "csv"
                        values = csv_rows[0]
                    else:
                        result = process_batch(csv_rows, selected_model, current_user.id, trigger_app=current_app._get_current_object())
                        values = result["values"]
                        aggregate_result = result["aggregate"]
                        last_payload = _build_batch_session_payload(result, selected_model, values, aggregate_result)
                        session['last_prediction'] = last_payload
                        _remember_prediction_input(values, selected_model, "csv", batch_mode=True, batch_rows=result["batch_rows"])
                        # Flash message removed - silent success
                        return redirect(url_for('user.thanks'))
                except Exception as e:
                    errors.append(f"Error membaca CSV: {str(e)}")
            else:
                errors.append("Harap upload file CSV yang valid.")

        if not errors and values:
            try:
                result = process_single(values, selected_model, current_user.id, trigger_app=current_app._get_current_object())
                session['last_prediction'] = {
                    "values": values,
                    "labels": FEATURE_KEYS,
                    "diagnosis": result.get("diagnosis"),
                    "model": selected_model,
                    "prediction_raw": int(result.get("prediction_raw")) if result.get("prediction_raw") is not None else None,
                    "comparison": result.get("comparison", []),
                    "batch_mode": False,
                    "input_method": input_method,
                }
                _remember_prediction_input(values, selected_model, input_method, batch_mode=False)
                # Flash message removed - silent success
                return redirect(url_for('user.thanks'))
            except Exception as exc:
                current_app.logger.error(f"Terjadi kesalahan saat memprediksi: {exc}")
                errors.append(f"Terjadi kesalahan saat memprediksi: {exc}")

    # Get available models from runtime (only models that are actually loaded)
    available_models = list(runtime.ml_models.keys()) if runtime.ml_models else list(MODEL_FILES.keys())
    if available_models and selected_model not in available_models:
        selected_model = available_models[0]
    prefill_values = {}
    remembered_values = remembered_input.get("values") if request.method == "GET" else None
    if remembered_values:
        prefill_values = {
            key: remembered_values[index]
            for index, key in enumerate(FEATURE_KEYS)
            if index < len(remembered_values)
        }
    active_input_method = request.form.get("input_method")
    if not active_input_method:
        active_input_method = "file" if remembered_input.get("batch_mode") else "manual"
    
    return render_template(
        "predict.html",
        questions=QUESTIONS,
        models=available_models,
        selected_model=selected_model,
        errors=errors,
        prefill_values=prefill_values,
        reusable_input=remembered_input,
        active_input_method=active_input_method,
        active_page='predict',
    )


@user_bp.route("/history", endpoint='history_page')
@login_required
def history_page():
    page = request.args.get('page', 1, type=int)
    per_page = 15
    pagination = Prediction.query.filter_by(user_id=current_user.id).order_by(
        Prediction.timestamp.desc()
    ).paginate(page=page, per_page=per_page, error_out=False)
    return render_template(
        "history.html",
        predictions=pagination.items,
        pagination=pagination,
        active_page='history',
    )


@user_bp.route("/history/download", endpoint='download_history')
@login_required
def download_history():
    predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.timestamp.desc()).all()
    buffer = StringIO()
    writer = csv.writer(buffer)
    writer.writerow([
        "tanggal",
        "username",
        "algoritma_machine_learning",
        "hasil_diagnosis",
        "prediction_raw",
        "age",
        "gender",
        "daily_screen_time_hours",
        "social_media_hours",
        "gaming_hours",
        "work_study_hours",
        "sleep_hours",
        "notifications_per_day",
        "app_opens_per_day",
        "weekend_screen_time",
    ])

    for prediction in predictions:
        values = prediction.input_list
        gender = ""
        if len(values) > 1:
            gender = "Laki-laki" if int(round(float(values[1]))) == 1 else "Perempuan"
        row = [
            prediction.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            current_user.username,
            prediction.model_name,
            prediction.result,
            prediction.prediction_raw,
        ]
        row.extend([
            values[0] if len(values) > 0 else "",
            gender,
            values[2] if len(values) > 2 else "",
            values[3] if len(values) > 3 else "",
            values[4] if len(values) > 4 else "",
            values[5] if len(values) > 5 else "",
            values[6] if len(values) > 6 else "",
            values[7] if len(values) > 7 else "",
            values[8] if len(values) > 8 else "",
            values[9] if len(values) > 9 else "",
        ])
        writer.writerow(row)

    filename = f"smartaddict_history_{current_user.username}.csv"
    return Response(
        buffer.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@user_bp.route("/thanks", endpoint='thanks')
@login_required
def thanks():
    last = session.get('last_prediction')
    averages = runtime.get_feature_averages()
    return render_template(
        "thanks.html",
        result=last,
        questions=QUESTIONS,
        averages=averages,
        active_model_version=runtime.ACTIVE_MODEL_VERSION,
        active_page='thanks',
    )


@user_bp.route("/download-csv-template", endpoint='download_csv_template')
@login_required
def download_csv_template():
    template_path = os.path.join(current_app.root_path, "static", "templates", "smart_addict_template.csv")
    return send_file(template_path, mimetype="text/csv", as_attachment=True, download_name="smart_addict_template.csv")


@user_bp.route("/about", endpoint='about')
def about():
    return render_template("about.html", active_page='about')


@user_bp.route("/delete-prediction/<int:pred_id>", methods=["POST"], endpoint='delete_prediction')
@login_required
def delete_prediction(pred_id):
    pred = Prediction.query.get_or_404(pred_id)
    if pred.user_id != current_user.id and not current_user.is_admin:
        # Removed "Akses ditolak" flash message
        return redirect(url_for('user.history_page'))
    db.session.delete(pred)
    db.session.commit()
    # Flash message removed - silent success
    if current_user.is_admin and request.referrer and 'admin' in request.referrer:
        return redirect(url_for('admin.admin_history'))
    return redirect(url_for('user.history_page'))


@user_bp.route("/clear-my-history", methods=["POST"], endpoint='clear_my_history')
@login_required
def clear_my_history():
    Prediction.query.filter_by(user_id=current_user.id).delete()
    db.session.commit()
    flash("History berhasil dihapus!", "success")
    return redirect(url_for('user.history_page'))
