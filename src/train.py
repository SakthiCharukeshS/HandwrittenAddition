
import os
import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)

DATASET_DIR = "dataset/processed/digits"
MODEL_PATH = "models/knn_model.pkl"


def load_dataset(split):

    images = []
    labels = []

    split_dir = os.path.join(DATASET_DIR, split)

    for digit in range(10):

        digit_dir = os.path.join(split_dir, str(digit))

        for filename in os.listdir(digit_dir):

            path = os.path.join(digit_dir, filename)

            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                continue

            # Flatten 28x28 into 784 features
            images.append(image.flatten())
            labels.append(digit)

    return np.array(images), np.array(labels)


def main():

    print("Loading dataset...")

    X_train, y_train = load_dataset("train")
    X_test, y_test = load_dataset("test")

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print("Features per image:", X_train.shape[1])

    # Create KNN classifier
    model = KNeighborsClassifier(
        n_neighbors=3,
        weights="distance",
        metric="euclidean",
        n_jobs=-1
    )

    # Train
    print("\nTraining KNN...")
    model.fit(X_train, y_train)

    # Evaluate
    print("\nEvaluating model...")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\nAccuracy:", round(accuracy * 100, 2), "%")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # Confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        cmap="Blues"
    )

    plt.title("KNN Digit Recognition")
    plt.tight_layout()
    plt.show()

    # Save model
    os.makedirs("models", exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    print("\nModel saved:", MODEL_PATH)


if __name__ == "__main__":
    main()
