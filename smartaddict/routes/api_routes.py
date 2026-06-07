from flask import Blueprint, jsonify, request

from smartaddict.services.retrain_service import get_current_retrain_status, list_statuses_paginated, read_status

api_bp = Blueprint('api', __name__)


@api_bp.route("/api/retrain-status", endpoint='api_retrain_status_list')
def api_retrain_status_list():
    page = request.args.get('page', 1)
    per_page = request.args.get('per_page', 20)
    data = list_statuses_paginated(page=page, per_page=per_page)
    return jsonify(data)


@api_bp.route("/api/retrain-status/<job_id>", endpoint='api_retrain_status_detail')
def api_retrain_status_detail(job_id):
    s = read_status(job_id)
    if not s:
        return jsonify({"error": "not found"}), 404
    return jsonify(s)


@api_bp.route("/api/retrain-status/current", endpoint='api_retrain_status_current')
def api_retrain_status_current():
    current = get_current_retrain_status()
    if not current:
        current = {
            'job_id': None,
            'status': 'idle'
        }
    return jsonify(current)


@api_bp.route("/api/retrain-status/force-reset", methods=['POST'], endpoint='api_retrain_force_reset')
def api_retrain_force_reset():
    """Force reset a stuck retrain job"""
    from flask_login import login_required, current_user
    from functools import wraps
    
    @wraps(api_retrain_force_reset)
    @login_required
    def check_admin():
        if not current_user.is_admin:
            return jsonify({"error": "Unauthorized"}), 403
        
        try:
            from smartaddict.services.retrain_service import RETRAIN_LOCK, get_latest_retrain_status
            from datetime import datetime
            
            # Get latest running/pending job
            latest = get_latest_retrain_status()
            if latest and latest.get("status") in ("pending", "running"):
                job_id = latest.get("job_id")
                
                # Mark as failed
                latest["status"] = "failed"
                latest["finished_at"] = datetime.utcnow().isoformat() + "Z"
                from smartaddict.services.retrain_service import append_log, write_status
                append_log(job_id, "ERROR", "Job di-reset paksa oleh admin")
                write_status(job_id, latest)
                
                # Try to release lock
                try:
                    RETRAIN_LOCK.release()
                except RuntimeError:
                    pass  # Lock was not acquired
                
                return jsonify({
                    "success": True,
                    "message": f"Job {job_id} berhasil di-reset",
                    "job_id": job_id
                })
            else:
                return jsonify({
                    "success": False,
                    "message": "Tidak ada job yang perlu di-reset"
                }), 400
        except Exception as e:
            return jsonify({
                "success": False,
                "message": f"Error: {str(e)}"
            }), 500
    
    return check_admin()