from flask import Flask, render_template, request, redirect, session
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from flask import send_file

import os

from preprocessing.feature_engineering import encode_features

app = Flask(__name__)
app.secret_key = "deepnids_secret"

# Load trained model
MODEL_PATH = "saved_models/dnn_final_model.h5"
model = load_model(MODEL_PATH)

# Global scaler (same logic as training)
scaler = StandardScaler()


@app.route("/detect", methods=["GET", "POST"])
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
        session["total"] = total
        session["attacks"] = attacks
        session["normal"] = normal

        return render_template(
            "result.html",
            total=total,
            attacks=attacks,
            normal=normal
        )

    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = request.form["username"]
        pwd = request.form["password"]

        if user == "admin" and pwd == "admin123":
            session["user"] = "admin"
            return redirect("/")
        else:
            return render_template("login.html", error="Invalid Credentials")

    return render_template("login.html")

@app.route("/", methods=["GET", "POST"])
def home():
    if "user" not in session:
        return redirect("/login")

    # existing prediction code remains unchanged
    return index()

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

@app.route("/admin")
def admin():
    if session.get("user") != "admin":
        return redirect("/login")

    return render_template("admin.html")

@app.route("/download")
def download_report():
    file_path = "intrusion_report.pdf"

    c = canvas.Canvas(file_path, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Network Intrusion Detection Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, height - 100, "Detection Summary:")
    c.drawString(50, height - 130, f"Total Records: {session.get('total')}")
    c.drawString(50, height - 160, f"Normal Traffic: {session.get('normal')}")
    c.drawString(50, height - 190, f"Attack Traffic: {session.get('attacks')}")

    c.drawString(50, height - 240, "Model: Deep Neural Network (DNN)")
    c.drawString(50, height - 270, "Dataset: NSL-KDD")
    c.drawString(50, height - 300, "By: DeepNIDS Team")

    c.save()

    return send_file(file_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
