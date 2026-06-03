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