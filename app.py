from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import os

from preprocessing.feature_engineering import encode_features

app = Flask(__name__)

# Load trained model
MODEL_PATH = "saved_models/dnn_final_model.h5"
model = load_model(MODEL_PATH)

# Global scaler (same logic as training)
scaler = StandardScaler()


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]

        if file.filename == "":
            return render_template("index.html", error="No file selected")

        # Read uploaded CSV
        df = pd.read_csv(file)

        # Feature engineering
        X, _ = encode_features(df)

        # Scale features
        X_scaled = scaler.fit_transform(X)
        X_scaled = X_scaled.astype(np.float32)

        # Predict
        predictions = model.predict(X_scaled)
        predictions = (predictions >= 0.5).astype(int)

        # Summary
        total = len(predictions)
        attacks = int(np.sum(predictions))
        normal = total - attacks

        return render_template(
            "result.html",
            total=total,
            attacks=attacks,
            normal=normal
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
