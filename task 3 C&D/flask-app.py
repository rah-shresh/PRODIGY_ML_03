import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import joblib

from flask import Flask, render_template, request, send_from_directory

app = Flask(__name__)

model = joblib.load("cat_dog_svm.pkl")

@app.route('/history/<filename>')

def history_file(filename):
    return send_from_directory('history', filename)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    filename = secure_filename(file.filename)

    upload_folder = os.path.join("static", "history")


    os.makedirs(upload_folder, exist_ok=True)

    import time


    filename = f"{int(time.time())}_{secure_filename(file.filename)}"

    upload_folder = "history"

    os.makedirs(upload_folder, exist_ok=True)

    upload_path = os.path.join(upload_folder, filename)

    file.save(upload_path)

    # Read image
    img = cv2.imread(upload_path)

    img = cv2.resize(img, (64, 64))

    img = img.flatten()

    img = np.array(img, dtype=np.float32) / 255.0

    img = img.reshape(1, -1)

    prediction = model.predict(img)

    result = "Dog 🐶" if prediction[0] == 1 else "Cat 🐱"

    return render_template(
    "index.html",
    prediction=result,
    image_path=filename
)

if __name__ == "__main__":
    app.run(debug=True)