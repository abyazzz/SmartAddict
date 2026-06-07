import json
from typing import List, Dict, Any

from flask import current_app

import smartaddict.runtime as runtime
from smartaddict.extensions import db
from smartaddict.models.prediction import Prediction
from smartaddict.models.predict_user_session import PredictUserSession
from smartaddict.services.retrain_service import maybe_trigger_retrain


def _build_entries_from_row(row_values, selected_model, user_id):
    row_result = runtime.predict_with_model(row_values, selected_model, include_comparison=False)
    diagnosis = row_result["diagnosis"]
    prediction_raw = int(row_result["prediction_raw"])

    pred = Prediction(
        user_id=user_id,
        model_name=selected_model,
        input_values=json.dumps(row_values),
        result=diagnosis,
        prediction_raw=prediction_raw,
    )

    sess = PredictUserSession(
        user_id=user_id,
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
        result=diagnosis,
    )

    return pred, sess, diagnosis, prediction_raw


def process_batch(csv_rows: List[List[Any]], selected_model: str, user_id: int, trigger_app=None) -> Dict[str, Any]:
    batch_rows = []
    prediction_entries = []
    session_entries = []
    distribution = {"Rendah": 0, "Sedang": 0, "Tinggi": 0}

    for row_number, row_values in enumerate(csv_rows, start=1):
        pred, sess, diagnosis, prediction_raw = _build_entries_from_row(row_values, selected_model, user_id)

        batch_rows.append({
            "row_number": row_number,
            "values": row_values,
            "diagnosis": diagnosis,
            "prediction_raw": prediction_raw,
        })

        if diagnosis in distribution:
            distribution[diagnosis] += 1

        prediction_entries.append(pred)
        session_entries.append(sess)

    # compute aggregates
    values = [
        sum(float(row[i]) for row in csv_rows) / len(csv_rows) for i in range(len(csv_rows[0]))
    ]
    aggregate_result = runtime.predict_with_model(values, selected_model, include_comparison=True)
    male_count = sum(1 for row in csv_rows if int(row[1]) == 1)
    female_count = len(csv_rows) - male_count

    session_count_before = PredictUserSession.query.count()

    try:
        db.session.add_all(prediction_entries)
        db.session.add_all(session_entries)
        db.session.commit()
    except Exception:
        db.session.rollback()
        raise

    triggered = None
    if trigger_app is not None:
        try:
            triggered = maybe_trigger_retrain(trigger_app, session_count_before, len(session_entries))
        except Exception:
            current_app.logger.exception("Gagal memeriksa trigger retraining batch")

    result = {
        "values": values,
        "aggregate": aggregate_result,
        "batch_rows": batch_rows,
        "distribution": distribution,
        "male_count": male_count,
        "female_count": female_count,
        "batch_count": len(batch_rows),
        "triggered": triggered,
    }
    return result


def process_single(values: List[Any], selected_model: str, user_id: int, trigger_app=None) -> Dict[str, Any]:
    result_payload = runtime.predict_with_model(values, selected_model, include_comparison=True)
    prediction_raw = int(result_payload["prediction_raw"]) if result_payload.get("prediction_raw") is not None else None
    diagnosis = result_payload.get("diagnosis")

    pred_entry = Prediction(
        user_id=user_id,
        model_name=selected_model,
        input_values=json.dumps(values),
        result=diagnosis,
        prediction_raw=int(prediction_raw) if prediction_raw is not None else None,
    )
    new_session = PredictUserSession(
        user_id=user_id,
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
        result=diagnosis,
    )

    session_count_before = PredictUserSession.query.count()

    try:
        db.session.add(pred_entry)
        db.session.add(new_session)
        db.session.commit()
    except Exception:
        db.session.rollback()
        raise

    triggered = None
    if trigger_app is not None:
        try:
            triggered = maybe_trigger_retrain(trigger_app, session_count_before, 1)
        except Exception:
            current_app.logger.exception("Gagal memeriksa trigger retraining")

    return {
        "values": values,
        "diagnosis": diagnosis,
        "prediction_raw": prediction_raw,
        "comparison": result_payload.get("comparison", []),
        "triggered": triggered,
    }
