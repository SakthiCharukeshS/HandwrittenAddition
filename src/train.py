
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import cv2
from normalization import normalize_digit

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)

DATA_PATH = "dataset/mnist/mnist.npz"
MODEL_PATH = "models/knn_model.pkl"


def main():
    print("Loading MNIST dataset...")

    data = np.load(DATA_PATH)

    x_train = data["x_train"]
    y_train = data["y_train"]

    x_test = data["x_test"]
    y_test = data["y_test"]

    # Select manageable subsets for KNN
    rng = np.random.default_rng(42)

    train_indices = rng.choice(
        len(x_train), 8000, replace=False
    )

    test_indices = rng.choice(
        len(x_test), 1000, replace=False
    )

    print("Normalizing MNIST images...")

    X_train = np.array([
        normalize_digit(img).flatten()
        for img in x_train[train_indices]
    ])

    Y_train = y_train[train_indices]

    X_test = np.array([
        normalize_digit(img).flatten()
        for img in x_test[test_indices]
    ])

    Y_test = y_test[test_indices]

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    model = KNeighborsClassifier(
        n_neighbors=3,
        weights="distance",
        metric="euclidean",
        n_jobs=-1
    )

    print("\nTraining KNN...")
    model.fit(X_train, Y_train)

    print("\nEvaluating model...")
    predictions = model.predict(X_test)

    accuracy = accuracy_score(Y_test, predictions)

    print("\nAccuracy:", round(accuracy * 100, 2), "%")
    print("\nClassification Report:")
    print(classification_report(Y_test, predictions))

    ConfusionMatrixDisplay.from_predictions(
        Y_test, predictions, cmap="Blues"
    )

    plt.title("MNIST KNN Digit Recognition")
    plt.tight_layout()
    plt.show()

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("\nModel saved:", MODEL_PATH)


if __name__ == "__main__":
    main()
