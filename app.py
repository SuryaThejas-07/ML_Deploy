
from flask import Flask, request, jsonify
from pathlib import Path
import pickle
import logging
import numpy as np

# --------------------------------------------------
# 1. Initialize Flask application
# --------------------------------------------------
app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1 MB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "logistic_regression_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"


# --------------------------------------------------
# 2. Load trained model and scaler
# --------------------------------------------------
def load_artifact(path):
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path.name}")

    with path.open("rb") as file:
        return pickle.load(file)


model = load_artifact(MODEL_PATH)
scaler = load_artifact(SCALER_PATH)

# Number of features expected by the model
EXPECTED_FEATURES = getattr(
    scaler,
    "n_features_in_",
    getattr(model, "n_features_in_", None)
)

if EXPECTED_FEATURES is None:
    raise RuntimeError("Cannot determine the expected number of features.")

logger.info("Model and scaler loaded successfully.")
logger.info("Expected number of features: %s", EXPECTED_FEATURES)


# --------------------------------------------------
# 3. Health-check endpoint
# --------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Logistic Regression Prediction API",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict"
        }
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "expected_features": int(EXPECTED_FEATURES)
    })


# --------------------------------------------------
# 4. Prediction endpoint
# --------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Require valid JSON
        if not request.is_json:
            return jsonify({
                "error": "Content-Type must be application/json."
            }), 415

        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "error": "Request body must be a JSON object."
            }), 400

        features = data.get("features")

        if not isinstance(features, list):
            return jsonify({
                "error": "'features' must be a list of numbers."
            }), 400

        if len(features) != EXPECTED_FEATURES:
            return jsonify({
                "error": "Incorrect number of features.",
                "expected": int(EXPECTED_FEATURES),
                "received": len(features)
            }), 400

        # Validate numeric, finite input values
        try:
            features_array = np.asarray(features, dtype=np.float64)
        except (ValueError, TypeError, OverflowError):
            return jsonify({
                "error": "All features must be valid numbers."
            }), 400

        if not np.all(np.isfinite(features_array)):
            return jsonify({
                "error": "Features cannot contain NaN or infinity."
            }), 400

        features_array = features_array.reshape(1, -1)

        # Apply the same preprocessing used during training
        scaled_features = scaler.transform(features_array)

        # Generate prediction
        prediction = model.predict(scaled_features)[0]

        result = {
            "prediction": prediction.item()
            if isinstance(prediction, np.generic)
            else prediction
        }

        # Include probabilities when supported by the model
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(scaled_features)[0]
            classes = model.classes_

            result["probabilities"] = {
                str(
                    cls.item()
                    if isinstance(cls, np.generic)
                    else cls
                ): float(prob)
                for cls, prob in zip(classes, probabilities)
            }

            result["confidence"] = float(np.max(probabilities))

        return jsonify(result), 200

    except Exception:
        logger.exception("Prediction failed")

        return jsonify({
            "error": "Prediction failed due to an internal server error."
        }), 500


# --------------------------------------------------
# 5. Run development server
# --------------------------------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
