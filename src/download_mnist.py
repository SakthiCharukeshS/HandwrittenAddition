
import os
import gzip
import struct
import urllib.request
import numpy as np

DATA_DIR = "dataset/mnist"

BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"

FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]


def download_files():
    os.makedirs(DATA_DIR, exist_ok=True)

    for filename in FILES:
        path = os.path.join(DATA_DIR, filename)

        if not os.path.exists(path):
            print("Downloading:", filename)
            urllib.request.urlretrieve(BASE_URL + filename, path)
        else:
            print("Already downloaded:", filename)


def load_images(filename):
    path = os.path.join(DATA_DIR, filename)

    with gzip.open(path, "rb") as file:
        magic, count, rows, cols = struct.unpack(">IIII", file.read(16))
        images = np.frombuffer(file.read(), dtype=np.uint8)

    return images.reshape(count, rows, cols)


def load_labels(filename):
    path = os.path.join(DATA_DIR, filename)

    with gzip.open(path, "rb") as file:
        magic, count = struct.unpack(">II", file.read(8))
        labels = np.frombuffer(file.read(), dtype=np.uint8)

    return labels


def main():
    download_files()

    x_train = load_images(FILES[0])
    y_train = load_labels(FILES[1])

    x_test = load_images(FILES[2])
    y_test = load_labels(FILES[3])

    output_path = os.path.join(DATA_DIR, "mnist.npz")

    np.savez_compressed(
        output_path,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test
    )

    print("\nMNIST downloaded successfully!")
    print("Training:", x_train.shape)
    print("Testing:", x_test.shape)
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
