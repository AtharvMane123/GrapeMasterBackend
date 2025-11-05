import os
from flask import Flask, request, jsonify
from ultralytics import YOLO
import torch

app = Flask(__name__)

# ==============================
# Load Model (Auto-detect type)
# ==============================
MODEL_PATH = r"C:\Users\manea\Downloads\QCommerce Papers\grape_project_data\grape_project_data\runs\classify\grape_classifier_10_epoch_test\weights\best.engine"

print(f"Loading model from: {MODEL_PATH}")
if MODEL_PATH.endswith(".engine"):
    print("Detected TensorRT Engine model ✅")
else:
    print("Detected PyTorch model ✅")

# Load YOLO model (works for both .pt and .engine)
model = YOLO(MODEL_PATH)

# ==============================
# Prediction Endpoint
# ==============================
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        # Save temporary uploaded file
        image_path = os.path.join("uploads", file.filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(image_path)

        # TensorRT → must use batch=1
        if MODEL_PATH.endswith(".engine"):
            print("Running TensorRT inference...")
            results = list(model(source=image_path, batch=1))
        else:
            print("Running PyTorch inference...")
            results = list(model(source=image_path))

        # Extract predicted class
        pred_label = results[0].names[int(results[0].probs.top1)]
        confidence = float(results[0].probs.top1conf)

        # Cleanup
        os.remove(image_path)

        return jsonify({
            "prediction": pred_label,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500


# ==============================
# Run Flask app
# ==============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
