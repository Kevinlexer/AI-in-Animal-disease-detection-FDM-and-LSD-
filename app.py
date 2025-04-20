from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session
import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # for session management

# Load the trained model
MODEL_PATH = "foot_and_mouth_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels
class_labels = ["Foot_and_mouth", "Lumpy", "Normal"]

# Define the upload folder
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Path to the predictions log file
PREDICTIONS_FILE = "predictions.json"

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load existing predictions from file if it exists
if os.path.exists(PREDICTIONS_FILE):
    with open(PREDICTIONS_FILE, "r") as f:
        predictions_log = json.load(f)
else:
    predictions_log = []

# Image preprocessing function
def preprocess_image(img_path, img_size=(224, 224)):
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Prediction function
def predict_image(img_path):
    processed_img = preprocess_image(img_path)
    prediction = model.predict(processed_img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100
    return class_labels[predicted_class], confidence

# Homepage - Upload and predict
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]

        if file.filename == "":
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            predicted_class, confidence = predict_image(file_path)

            # Save prediction entry
            entry = {
                "filename": filename,
                "prediction": predicted_class,
                "confidence": f"{confidence:.2f}%"
            }
            predictions_log.append(entry)

            # Save to file
            with open(PREDICTIONS_FILE, "w") as f:
                json.dump(predictions_log, f, indent=4)

            return render_template("index.html",
                                   filename=filename,
                                   prediction=predicted_class,
                                   confidence=confidence)

    return render_template("index.html")

# Serve uploaded images
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# Admin Login
@app.route("/admin", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username == "admin" and password == "admin":
            session["admin_logged_in"] = True
            return redirect(url_for("admin_dashboard"))
        else:
            return render_template("admin/index.html", error="Invalid credentials.")
    return render_template("admin/index.html")

# Admin Dashboard
@app.route("/admin/dashboard")
def admin_dashboard():
    if not session.get("admin_logged_in"):
        return redirect(url_for("admin_login"))
    return render_template("admin/dashboard.html", logs=predictions_log)

# Admin Logout
@app.route("/admin/logout")
def admin_logout():
    session.pop("admin_logged_in", None)
    return redirect(url_for("admin_login"))

if __name__ == "__main__":
    app.run(debug=True)
