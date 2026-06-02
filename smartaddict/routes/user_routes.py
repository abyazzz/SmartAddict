from flask import Blueprint, flash, redirect, render_template, request, url_for, session, send_file, current_app
from flask_login import current_user, login_required

import json
import os

import smartaddict.runtime as runtime
from smartaddict.extensions import db
from smartaddict.models.prediction import Prediction
from smartaddict.models.predict_user_session import PredictUserSession
from smartaddict.models.user import User
from smartaddict.services.csv_service import average_rows, parse_csv_rows
from smartaddict.services.retrain_service import run_retrain_pipeline
from smartaddict.utils.constants import FEATURE_KEYS, MODEL_FILES, QUESTIONS

user_bp = Blueprint('user', __name__)


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
                return redirect(url_for('user.profile'))
            flash("Tidak ada perubahan profile.", "warning")

    return render_template(
        "profile.html",
        active_page='profile',
        prediction_count=prediction_count,
        latest_prediction=latest_prediction
    )


@user_bp.route("/dashboard", endpoint='dashboard')
def dashboard():
    averages = runtime.get_feature_averages()
    recent_preds = Prediction.query.order_by(Prediction.timestamp.desc()).limit(6).all()
    return render_template("dashboard.html", active_page='dashboard', averages=averages, recent_preds=recent_preds)


@user_bp.route("/predict", methods=["GET", "POST"], endpoint='predict')
@login_required
def predict():
    selected_model = "Decision Tree"
    model_name = None
    errors = []
    selected_model = request.form.get("model") or selected_model if request.method == 'POST' else selected_model

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
                        prediction_entries = []
                        session_entries = []
                        distribution = {"Rendah": 0, "Sedang": 0, "Tinggi": 0}

                        for row_number, row_values in enumerate(csv_rows, start=1):
                            row_result = runtime.predict_with_model(row_values, selected_model, include_comparison=False)
                            row_diagnosis = row_result["diagnosis"]
                            row_prediction_raw = int(row_result["prediction_raw"])

                            batch_rows.append({
                                "row_number": row_number,
                                "values": row_values,
                                "diagnosis": row_diagnosis,
                                "prediction_raw": row_prediction_raw,
                            })

                            if row_diagnosis in distribution:
                                distribution[row_diagnosis] += 1

                            prediction_entries.append(Prediction(
                                user_id=current_user.id,
                                model_name=selected_model,
                                input_values=json.dumps(row_values),
                                result=row_diagnosis,
                                prediction_raw=row_prediction_raw,
                            ))

                            session_entries.append(PredictUserSession(
                                user_id=current_user.id,
                                age=int(row_values[0]),
                                gender=int(row_values[1]),
                                daily_screen_time_hours=float(row_values[2]),
                                social_media_hours=float(row_values[3]),
                                gaming_hours=float(row_values[4]),
                                work_study_hours=float(row_values[5]),
                                sleep_hours=float(row_values[6]),
                                notifications_per_day=int(row_values[7]),
                                app_opens_per_day=int(row_values[8]),
                                weekend_screen_time=float(row_values[9]),
                                result=row_diagnosis,
                            ))

                        values = average_rows(csv_rows)
                        aggregate_result = runtime.predict_with_model(values, selected_model, include_comparison=True)
                        male_count = sum(1 for row in csv_rows if int(row[1]) == 1)
                        female_count = len(csv_rows) - male_count

                        try:
                            db.session.add_all(prediction_entries)
                            db.session.add_all(session_entries)
                            db.session.commit()
                        except Exception:
                            db.session.rollback()
                            raise

                        try:
                            session_count = PredictUserSession.query.count()
                            if session_count >= 50:
                                app_obj = current_app._get_current_object()
                                triggered = run_retrain_pipeline(app_obj)
                                if triggered:
                                    flash("Retraining otomatis berjalan di background (50 data terpenuhi)!", "info")
                        except Exception as db_err:
                            current_app.logger.error(f"Gagal memeriksa trigger retraining batch: {db_err}")

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
                            "batch_stats": {
                                "male_count": male_count,
                                "female_count": female_count,
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
                        flash(f"Prediksi batch berhasil! {len(batch_rows)} baris diproses dan disimpan.", "success")
                        return redirect(url_for('user.thanks'))
                except Exception as e:
                    errors.append(f"Error membaca CSV: {str(e)}")
            else:
                errors.append("Harap upload file CSV yang valid.")

        if not errors and values:
            try:
                result_payload = runtime.predict_with_model(values, selected_model, include_comparison=True)
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
                try:
                    db.session.add(pred_entry)
                    db.session.add(new_session)
                    db.session.commit()
                except Exception as db_err:
                    db.session.rollback()
                    current_app.logger.error(f"Gagal mencatat sesi prediksi ke database: {db_err}")

                # Trigger retraining when session buffer has reached minimum threshold.
                try:
                    session_count = PredictUserSession.query.count()
                    if session_count >= 50:
                        app_obj = current_app._get_current_object()
                        triggered = run_retrain_pipeline(app_obj)
                        if triggered:
                            flash("Retraining otomatis berjalan di background (50 data terpenuhi)!", "info")
                except Exception as db_err:
                    current_app.logger.error(f"Gagal memeriksa trigger retraining: {db_err}")

                session['last_prediction'] = {
                    "values": values,
                    "labels": FEATURE_KEYS,
                    "diagnosis": diagnosis,
                    "model": model_name,
                    "prediction_raw": int(prediction),
                    "comparison": result_payload.get("comparison"),
                    "batch_mode": False,
                    "feature_averages": values,
                }
                flash("Prediksi berhasil!", "success")
                return redirect(url_for('user.thanks'))
            except Exception as exc:
                current_app.logger.error(f"Terjadi kesalahan saat memprediksi: {exc}")
                errors.append(f"Terjadi kesalahan saat memprediksi: {exc}")

    return render_template("predict.html", questions=QUESTIONS, models=list(MODEL_FILES.keys()),
        selected_model=selected_model, errors=errors, active_page='predict')


@user_bp.route("/history", endpoint='history_page')
@login_required
def history_page():
    user_preds = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.timestamp.desc()).all()
    return render_template("history.html", predictions=user_preds, active_page='history')


@user_bp.route("/thanks", endpoint='thanks')
@login_required
def thanks():
    last = session.pop('last_prediction', None)
    averages = last.get('feature_averages') if last and last.get('feature_averages') else runtime.get_feature_averages()
    return render_template("thanks.html", result=last, questions=QUESTIONS, averages=averages, active_page='thanks')


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
        flash("Akses ditolak.", "error")
        return redirect(url_for('user.history_page'))
    db.session.delete(pred)
    db.session.commit()
    flash("Prediksi berhasil dihapus!", "success")
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
