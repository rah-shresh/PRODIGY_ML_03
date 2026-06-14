# 🐱🐶 Cats vs Dogs Image Classifier using SVM

## 📌 Project Overview

This project implements a **Support Vector Machine (SVM)** model to classify images of **cats** and **dogs**. The model is trained on the Kaggle Dogs vs Cats dataset and deployed using **Flask** with a simple web interface.

Users can upload an image through the web application and receive a prediction indicating whether the image contains a cat or a dog.

---

## 🚀 Features

* Image classification using Support Vector Machine (SVM)
* Image preprocessing with OpenCV
* Image resizing and normalization
* Flask web application
* Upload image and get instant prediction
* Uploaded image preview
* Image history storage
* Clean and responsive user interface

---

## 🛠️ Technologies Used

* Python
* OpenCV
* NumPy
* Scikit-learn
* Flask
* HTML
* CSS

---

## 📂 Project Structure

```text
task 3 C&D
│
├── history/
│   └── Uploaded Images
│
├── static/
│   └── style.css
│
├── templates/
│   └── index.html
│
├── cat_dog_svm.pkl
├── flask_app.py
├── train_model.py
└── README.md
```

---

## 📊 Dataset

Dataset used:

**Kaggle Dogs vs Cats Dataset**

https://www.kaggle.com/c/dogs-vs-cats

---

## ⚙️ Preprocessing Steps

1. Load images using OpenCV
2. Resize images to 64 × 64 pixels
3. Flatten image into a feature vector
4. Normalize pixel values to range [0,1]
5. Assign labels:

   * Cat → 0
   * Dog → 1

---

## 🤖 Model Training

* Algorithm: Support Vector Machine (SVM)
* Kernel: RBF
* Train-Test Split: 80% / 20%

### Training Results

* Dataset Size: 5000 Images
* Accuracy: **61.10%**

---

## ▶️ How to Run

### Install Dependencies

```bash
pip install flask numpy opencv-python scikit-learn joblib
```

### Start Flask Application

```bash
python flask_app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

## 📸 Web Application

1. Upload an image.
2. Click Predict.
3. View prediction result.
4. Uploaded image is displayed on the page.
5. Images are saved in the history folder.

---

## 🎯 Future Improvements

* CNN-based image classification
* Higher accuracy using deep learning
* Drag-and-drop image upload
* Prediction confidence score
* Image gallery for upload history
* Deploy on Render or Railway

---

## 👨‍💻 Author

Shresh Rahangdale

Machine Learning & Python Enthusiast
