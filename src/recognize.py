
import os
import cv2
import joblib
import numpy as np

MODEL_PATH = "models/knn_model.pkl"
SYMBOL_DIR = "dataset/processed/equation_01"


def recognize_digits():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("KNN model not found. Run src/train.py first.")

    model = joblib.load(MODEL_PATH)

    digits = []

    for symbol_number in [1, 3]:
        image_path = os.path.join(
            SYMBOL_DIR, f"symbol_{symbol_number}.png"
        )

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(
                f"Cannot find {image_path}. Run preprocessing first."
            )

        image = cv2.resize(image, (28, 28))
        features = image.flatten().reshape(1, -1)

        prediction = model.predict(features)[0]
        digits.append(int(prediction))

        print(f"Symbol {symbol_number}: Predicted digit = {prediction}")

    return digits[0], digits[1]
