import os
import random
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Dataset path
DATASET_PATH = "data/train"

images = []
labels = []

# Get all files
files = os.listdir(DATASET_PATH)

# Shuffle files randomly
random.shuffle(files)

# Use only first 500  images for faster training
files = files[:500]

for file in files:

    img_path = os.path.join(DATASET_PATH, file)

    img = cv2.imread(img_path)

    if img is None:
        continue

    # Resize image
    img = cv2.resize(img, (64, 64))

    # Flatten image
    img = img.flatten()

    # Store image
    images.append(img)

    # Store label
    if file.startswith("cat"):
        labels.append(0)
    else:
        labels.append(1)

print("Cats:", labels.count(0))
print("Dogs:", labels.count(1))

# Convert to NumPy arrays
X = X = np.array(images, dtype=np.float32) / 255.0
y = np.array(labels)

print("X shape:", X.shape)
print("y shape:", y.shape)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Unique y:", np.unique(y))
print("Unique y_train:", np.unique(y_train))

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

# Train SVM
model = SVC(kernel="rbf")

print("Training model...")
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

import joblib
joblib.dump(model, "cat_dog_svm.pkl", compress=9)
