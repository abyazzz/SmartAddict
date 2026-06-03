import json

from smartaddict.models.prediction import Prediction
from smartaddict.services.model_service import get_active_version_from_config, load_model_version
from smartaddict.utils.constants import FEATURE_KEYS, LABEL_MAP, MODEL_FILES, QUESTIONS


ACTIVE_MODEL_VERSION = None
ml_models = {}
scaler = None


def init_active_model():
    global ACTIVE_MODEL_VERSION, ml_models, scaler
    
    import logging
    logger = logging.getLogger(__name__)

    for candidate in (get_active_version_from_config(), "model_default"):
        if not candidate:
            continue
        try:
            models, scaler_obj, success = load_model_version(candidate)
        except Exception as e:
            logger.error(f"Error loading model version {candidate}: {e}")
            continue
        if success:
            ACTIVE_MODEL_VERSION = candidate
            ml_models = models
            scaler = scaler_obj
            logger.info(f"Active model version: {ACTIVE_MODEL_VERSION}")
            logger.info(f"Loaded models: {list(ml_models.keys())}")
            logger.info(f"Scaler available: {scaler is not None}")
            return

    logger.error("CRITICAL: No model version could be loaded!")
    ACTIVE_MODEL_VERSION = None
    ml_models = {}
    scaler = None


def predict_with_model(values, selected_model, include_comparison=True):
    import logging
    logger = logging.getLogger(__name__)
    
    if not ml_models:
        logger.warning("Models not loaded, attempting to initialize...")
        init_active_model()

    if not ml_models:
        raise ValueError("Tidak ada model yang tersedia. Silakan cek konfigurasi model.")

    model = ml_models.get(selected_model)
    if model is None:
        available_models = list(ml_models.keys())
        logger.error(f"Model {selected_model} tidak ditemukan. Available: {available_models}")
        raise ValueError(
            f"Model '{selected_model}' tidak tersedia. "
            f"Model yang tersedia: {', '.join(available_models) if available_models else 'Tidak ada'}"
        )

    # Scale input values using the loaded scaler
    scaled_values = values
    if scaler is not None:
        try:
            scaled_values = scaler.transform([values])[0]
            logger.debug(f"Input scaled successfully for {selected_model}")
        except Exception as e:
            logger.warning(f"Scaler failed, using unscaled values: {e}")
            # Fallback to unscaled if there's any dimension mismatch/error
            pass
    else:
        logger.warning(f"No scaler available, using raw values for {selected_model}")

    prediction_raw = int(model.predict([scaled_values])[0])
    diagnosis = LABEL_MAP.get(prediction_raw, "Tidak diketahui")

    comparison = []
    if include_comparison:
        for model_name, model_obj in ml_models.items():
            if model_obj is None:
                continue
            try:
                model_prediction = int(model_obj.predict([scaled_values])[0])
                comparison.append({
                    "model": model_name,
                    "prediction_raw": model_prediction,
                    "diagnosis": LABEL_MAP.get(model_prediction, "?"),
                })
            except Exception as e:
                logger.error(f"Error predicting with {model_name}: {e}")
                comparison.append({
                    "model": model_name,
                    "prediction_raw": -1,
                    "diagnosis": "Error",
                })

    return {
        "values": values,
        "diagnosis": diagnosis,
        "prediction_raw": prediction_raw,
        "model": selected_model,
        "comparison": comparison,
    }


def get_feature_averages():
    all_preds = Prediction.query.all()
    num_features = len(QUESTIONS)
    sums = [0.0] * num_features
    counts = [0] * num_features

    for prediction in all_preds:
        try:
            values = json.loads(prediction.input_values)
            for index in range(min(len(values), num_features)):
                sums[index] += values[index]
                counts[index] += 1
        except Exception:
            pass

    averages = []
    for index in range(num_features):
        average = sums[index] / counts[index] if counts[index] > 0 else QUESTIONS[index]["default"]
        averages.append(round(average, 2))
    return averages