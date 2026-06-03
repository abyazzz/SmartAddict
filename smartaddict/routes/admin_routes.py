from functools import wraps

from flask import Blueprint, flash, redirect, render_template, request, url_for, jsonify, current_app
from flask_login import current_user, login_required
import os

import smartaddict.runtime as runtime
from smartaddict.extensions import db
from smartaddict.models.prediction import Prediction
from smartaddict.models.predict_user_session import PredictUserSession
from smartaddict.models.user import User
from smartaddict.services.model_service import activate_model_version, get_available_retrain_versions
from smartaddict.services.retrain_service import cleanup_statuses, run_retrain_pipeline

admin_bp = Blueprint('admin', __name__)


def admin_required(view_func):
    @wraps(view_func)
    @login_required
    def wrapped(*args, **kwargs):
        if not current_user.is_admin:
            flash('Akses ditolak. Hanya admin yang bisa mengakses halaman ini.', 'error')
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

    return render_template("admin/dashboard.html", active_page='admin_dashboard',
        total_users=total_users, total_preds=total_preds, stats=stats,
        model_usage=model_usage, recent=recent,
        total_predict_session=total_predict_session,
        total_retrains=total_retrains, retrain_versions=versions,
        active_model_version=runtime.ACTIVE_MODEL_VERSION)


@admin_bp.route("/admin/retrain-manual", methods=["POST"], endpoint='admin_retrain_manual')
@admin_required
def admin_retrain_manual():
    from flask import current_app
    app_obj = current_app._get_current_object()
    job_id = run_retrain_pipeline(app_obj)
    if job_id:
        flash(f"Retraining dimulai di background (job_id={job_id}). Pantau status di halaman retrain.", "info")
    else:
        flash("Gagal memulai proses retraining. Mungkin ada proses yang sedang berjalan.", "error")
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
        flash("Semua model retrain dihapus, namun model_default gagal dimuat.", "error")

    return redirect(url_for('admin.admin_dashboard'))


@admin_bp.route("/admin/use-retrain/<version_name>", methods=["POST"], endpoint='admin_use_retrain')
@admin_required
def admin_use_retrain(version_name):
    if activate_model_version(version_name):
        flash(f"Berhasil mengubah model aktif ke versi {version_name}!", "success")
    else:
        flash(f"Gagal memuat model dari versi {version_name}. Tetap menggunakan versi sebelumnya.", "error")

    return redirect(url_for('admin.admin_dashboard'))


@admin_bp.route("/admin/delete-retrain/<version_name>", methods=["POST"], endpoint='admin_delete_retrain')
@admin_required
def admin_delete_retrain(version_name):
    import shutil

    if version_name == "model_default":
        flash("Model default bawaan tidak boleh dihapus.", "error")
        return redirect(url_for('admin.admin_dashboard'))

    version_dir = os.path.join("model", version_name)
    try:
        if os.path.exists(version_dir):
            shutil.rmtree(version_dir)
            flash(f"Versi model {version_name} berhasil dihapus.", "success")
        else:
            flash(f"Direktori versi model {version_name} tidak ditemukan.", "error")
    except Exception as e:
        flash(f"Gagal menghapus folder versi model: {e}", "error")

    runtime.init_active_model()
    return redirect(url_for('admin.admin_dashboard'))


@admin_bp.route("/admin/history", endpoint='admin_history')
@admin_required
def admin_history():
    # Pagination: 15 items per page
    page = request.args.get('page', 1, type=int)
    per_page = 15
    
    pagination = Prediction.query.order_by(Prediction.timestamp.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    return render_template(
        "admin/all_history.html", 
        predictions=pagination.items,
        pagination=pagination,
        active_page='admin_history'
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
        flash(f"Cleanup gagal: {e}", 'error')
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
        flash("Tidak bisa menghapus akun admin.", "error")
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
