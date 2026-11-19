
from src.preprocess import preprocess_image
from src.recognize import recognize_digits
from src.calculate import calculate_addition


def main():
    print("=== Handwritten Addition Recognition ===\n")

    # Step 1: Preprocess input image
    print("[1] Preprocessing image...")
    preprocess_image()

    # Step 2: Recognize digits using KNN
    print("\n[2] Recognizing digits...")
    first, second = recognize_digits()

    # Step 3: Calculate result
    print("\n[3] Calculating result...")
    calculate_addition(first, second)


if __name__ == "__main__":
    main()
