from functools import wraps
import csv
from io import StringIO

from flask import Blueprint, Response, flash, redirect, render_template, request, url_for, jsonify, current_app
from flask_login import current_user, login_required
import os

import smartaddict.runtime as runtime
from smartaddict.extensions import db
from smartaddict.models.dashboard_feature import DashboardFeature
from smartaddict.models.prediction import Prediction
from smartaddict.models.predict_user_session import PredictUserSession
from smartaddict.models.user import User
from smartaddict.services.dashboard_feature_service import (
    DASHBOARD_FEATURE_TARGETS,
    get_dashboard_features,
)
from smartaddict.services.model_service import activate_model_version, get_available_retrain_versions
from smartaddict.services.retrain_service import cleanup_statuses, run_retrain_pipeline

admin_bp = Blueprint('admin', __name__)


def admin_required(view_func):
    @wraps(view_func)
    @login_required
    def wrapped(*args, **kwargs):
        if not current_user.is_admin:
            # Removed "Akses ditolak" flash message
            return redirect(url_for('user.dashboard'))
        return view_func(*args, **kwargs)

    return wrapped


@admin_bp.route("/admin", endpoint='admin_dashboard')
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

    versions = get_available_retrain_versions(runtime.ACTIVE_MODEL_VERSION)
    total_retrains = len(versions)
    dashboard_features = get_dashboard_features(include_inactive=True)

    return render_template("admin/dashboard.html", active_page='admin_dashboard',
        total_users=total_users, total_preds=total_preds, stats=stats,
        model_usage=model_usage, recent=recent,
        total_predict_session=total_predict_session,
        total_retrains=total_retrains, retrain_versions=versions,
        dashboard_features=dashboard_features,
        dashboard_feature_targets=DASHBOARD_FEATURE_TARGETS,
        active_model_version=runtime.ACTIVE_MODEL_VERSION)


@admin_bp.route("/admin/dashboard-features/update", methods=["POST"], endpoint='admin_update_dashboard_features')
@admin_required
def admin_update_dashboard_features():
    valid_targets = {target for target, _ in DASHBOARD_FEATURE_TARGETS}
    features = get_dashboard_features(include_inactive=True)

    for feature in features:
        prefix = f"feature_{feature.id}_"
        title = request.form.get(prefix + "title", "").strip()
        description = request.form.get(prefix + "description", "").strip()
        icon = request.form.get(prefix + "icon", "").strip()
        target_key = request.form.get(prefix + "target", "dashboard")
        sort_order_raw = request.form.get(prefix + "sort_order", feature.sort_order)

        if not title or not description:
            # Removed validation error flash message
            return redirect(url_for('admin.admin_dashboard'))
        if target_key not in valid_targets:
            target_key = "dashboard"

        try:
            sort_order = int(sort_order_raw)
        except (TypeError, ValueError):
            sort_order = feature.sort_order

        feature.title = title[:90]
        feature.description = description
        feature.icon = icon[:32] or feature.icon
        feature.target_key = target_key
        feature.sort_order = sort_order
        feature.is_active = request.form.get(prefix + "is_active") == "on"

    db.session.commit()
    flash("Konten 6 kotak dashboard berhasil diperbarui.", "success")
    return redirect(url_for('admin.admin_dashboard'))


@admin_bp.route("/admin/retrain-manual", methods=["POST"], endpoint='admin_retrain_manual')
@admin_required
def admin_retrain_manual():
    from flask import current_app
    app_obj = current_app._get_current_object()
    job_id = run_retrain_pipeline(app_obj)
    if job_id:
        flash(f"Retraining dimulai di background (job_id={job_id}). Pantau status di halaman retrain.", "info")
    else:
        # Removed error flash message
        pass
    return redirect(url_for('admin.admin_retrain_status'))


@admin_bp.route("/admin/clear-retrains", methods=["POST"], endpoint='admin_clear_retrains')
@admin_required
def admin_clear_retrains():
    import shutil
    versions = get_available_retrain_versions(runtime.ACTIVE_MODEL_VERSION)
    deleted_count = 0
    for ver in versions:
        version_dir = os.path.join("model", ver["version_name"])
        try:
            if os.path.exists(version_dir):
                shutil.rmtree(version_dir)
                deleted_count += 1
        except Exception as e:
            current_app.logger.error(f"Gagal menghapus folder {version_dir}: {e}")

    if activate_model_version("model_default"):
        flash(f"Berhasil menghapus {deleted_count} model retrain. Sistem kembali menggunakan model_default.", "success")
    else:
        # Removed error flash message
        pass

    if activate_model_version("model_default"):
        flash(f"Berhasil menghapus {deleted_count} model retrain. Sistem kembali menggunakan model_default.", "success")
    else:
        # Removed error flash message
        pass

    return redirect(url_for('admin.admin_dashboard'))


@admin_bp.route("/admin/use-retrain/<version_name>", methods=["POST"], endpoint='admin_use_retrain')
@admin_required
def admin_use_retrain(version_name):
    if activate_model_version(version_name):
        flash(f"Berhasil mengubah model aktif ke versi {version_name}!", "success")
    else:
        # Removed error flash message
        pass

    return redirect(url_for('admin.admin_dashboard'))


@admin_bp.route("/admin/delete-retrain/<version_name>", methods=["POST"], endpoint='admin_delete_retrain')
@admin_required
def admin_delete_retrain(version_name):
    import shutil

    if version_name == "model_default":
        # Removed error flash message
        return redirect(url_for('admin.admin_retrain_page'))

    version_dir = os.path.join("model", version_name)
    try:
        if os.path.exists(version_dir):
            shutil.rmtree(version_dir)
            flash(f"Versi model {version_name} berhasil dihapus.", "success")
        else:
            # Removed error flash message for directory not found
            pass
    except Exception as e:
        # Removed error flash message
        pass

    runtime.init_active_model()
    return redirect(url_for('admin.admin_retrain_page'))


@admin_bp.route("/admin/history", endpoint='admin_history')
@admin_required
def admin_history():
    # Pagination: 15 items per page
    page = request.args.get('page', 1, type=int)
    username = request.args.get('username', '').strip()
    per_page = 15
    
    query = Prediction.query
    if username:
        query = query.join(User).filter(User.username == username)
        
    pagination = query.order_by(Prediction.timestamp.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )

    all_users = User.query.filter_by(role='user').order_by(User.username.asc()).all()

    return render_template(
        "admin/all_history.html",
        predictions=pagination.items,
        pagination=pagination,
        all_users=all_users,
        selected_username=username,
        active_page='admin_history'
    )


@admin_bp.route("/admin/history/download", endpoint='admin_history_download')
@admin_required
def admin_history_download():
    username = request.args.get('username', '').strip()
    query = Prediction.query
    if username:
        query = query.join(User).filter(User.username == username)
    predictions = query.order_by(Prediction.timestamp.desc()).all()
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
        writer.writerow([
            prediction.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            prediction.user.username if prediction.user else "",
            prediction.model_name,
            prediction.result,
            prediction.prediction_raw,
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

    return Response(
        buffer.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=smartaddict_admin_history.csv"},
    )

@admin_bp.route("/admin/retrain-status", endpoint='admin_retrain_status')
@admin_required
def admin_retrain_status():
    return redirect(url_for('admin.admin_dashboard'))


@admin_bp.route('/admin/retrain-status/cleanup', methods=['POST'], endpoint='admin_retrain_status_cleanup')
@admin_required
def admin_retrain_status_cleanup():
    retain_days = request.form.get('retain_days') or request.args.get('retain_days') or 30
    max_entries = request.form.get('max_entries') or request.args.get('max_entries') or 200
    try:
        removed = cleanup_statuses(retain_days=int(retain_days), max_entries=int(max_entries))
        flash(f"Cleanup selesai. Menghapus {removed} file status.", 'success')
    except Exception as e:
        current_app.logger.exception('Cleanup retrain status gagal')
        # Removed error flash message
        pass
    return redirect(url_for('admin.admin_retrain_status'))


@admin_bp.route("/admin/users", endpoint='admin_users')
@admin_required
def admin_users():
    users = User.query.order_by(User.created_at.desc()).all()
    return render_template("admin/manage_users.html", users=users, active_page='admin_users')


@admin_bp.route("/admin/delete-user/<int:user_id>", methods=["POST"], endpoint='admin_delete_user')
@admin_required
def admin_delete_user(user_id):
    user = User.query.get_or_404(user_id)
    if user.is_admin:
        # Removed error flash message for admin account
        pass
    else:
        Prediction.query.filter_by(user_id=user.id).delete()
        db.session.delete(user)
        db.session.commit()
        flash(f"User '{user.username}' berhasil dihapus.", "success")
    return redirect(url_for('admin.admin_users'))


@admin_bp.route("/admin/clear-all-history", methods=["POST"], endpoint='admin_clear_all')
@admin_required
def admin_clear_all():
    Prediction.query.delete()
    db.session.commit()
    flash("Semua history berhasil dihapus!", "success")
    return redirect(url_for('admin.admin_history'))


@admin_bp.route("/admin/retrain", endpoint='admin_retrain_page')
@admin_required
def admin_retrain_page():
    """Main retrain page where admin can review predict_users_session data and trigger manual retrain"""
    # Get all session data for review
    session_data = PredictUserSession.query.order_by(PredictUserSession.timestamp.desc()).all()
    total_session_data = len(session_data)
    
    # Get retrain versions info
    versions = get_available_retrain_versions(runtime.ACTIVE_MODEL_VERSION)
    total_retrains = len(versions)
    
    # Check if threshold reached for notification
    threshold_reached = total_session_data >= 50
    
    return render_template(
        "admin/retrain.html",
        active_page='admin_retrain',
        session_data=session_data,
        total_session_data=total_session_data,
        total_retrains=total_retrains,
        threshold_reached=threshold_reached,
        retrain_versions=versions,
        active_model_version=runtime.ACTIVE_MODEL_VERSION
    )


@admin_bp.route("/admin/retrain/delete-session/<int:session_id>", methods=["POST"], endpoint='admin_delete_session_data')
@admin_required
def admin_delete_session_data(session_id):
    """Delete a specific predict_users_session entry"""
    session = PredictUserSession.query.get_or_404(session_id)
    db.session.delete(session)
    db.session.commit()
    return jsonify({"success": True, "message": "Data berhasil dihapus"})


@admin_bp.route("/admin/retrain/start", methods=["POST"], endpoint='admin_start_retrain')
@admin_required
def admin_start_retrain():
    """Start manual retrain process"""
    from flask import current_app
    app_obj = current_app._get_current_object()
    job_id = run_retrain_pipeline(app_obj)
    if job_id:
        return jsonify({
            "success": True, 
            "job_id": job_id,
            "message": f"Retrain dimulai dengan job_id: {job_id}"
        })
    else:
        return jsonify({
            "success": False,
            "message": "Gagal memulai retrain. Mungkin ada proses yang sedang berjalan."
        }), 400
