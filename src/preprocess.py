
import cv2
import os
from src.normalization import normalize_digit

# Project paths
IMAGE_PATH = "dataset/raw/equation.jpeg"
OUTPUT_DIR = "dataset/processed/equation_01"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def preprocess_image():

    # Read image
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        raise FileNotFoundError(IMAGE_PATH)

    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Noise removal
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Binary thresholding
    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Find contours
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Extract bounding boxes
    boxes = []

    for contour in contours:

        if cv2.contourArea(contour) > 20:

            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, w, h))

    # Group disconnected parts of symbols
    changed = True

    while changed:

        changed = False

        for i in range(len(boxes)):

            if changed:
                break

            for j in range(i + 1, len(boxes)):

                x1, y1, w1, h1 = boxes[i]
                x2, y2, w2, h2 = boxes[j]

                overlap = max(
                    0,
                    min(x1 + w1, x2 + w2) - max(x1, x2)
                )

                overlap_ratio = overlap / min(w1, w2)

                vertical_gap = max(
                    0,
                    max(y1, y2) - min(y1 + h1, y2 + h2)
                )

                if (
                    overlap_ratio >= 0.6
                    and vertical_gap <= 1.5 * min(h1, h2)
                ):

                    new_x = min(x1, x2)
                    new_y = min(y1, y2)

                    new_w = max(x1+w1, x2+w2) - new_x
                    new_h = max(y1+h1, y2+h2) - new_y

                    boxes[i] = (new_x, new_y, new_w, new_h)

                    boxes.pop(j)

                    changed = True
                    break

    # Sort left to right
    boxes.sort(key=lambda box: box[0])

    # Draw detected components
    output = image.copy()

    for index, (x, y, w, h) in enumerate(boxes):

        cv2.rectangle(
            output,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        # Extract character
        crop = binary[y:y+h, x:x+w]

        # Normalize to 28x28
        normalized = normalize_digit(crop)

        if normalized is not None:

            filename = f"symbol_{index+1}.png"

            cv2.imwrite(
                os.path.join(OUTPUT_DIR, filename),
                normalized
            )

    # Save outputs
    cv2.imwrite(
        os.path.join(OUTPUT_DIR, "binary.png"),
        binary
    )

    cv2.imwrite(
        os.path.join(OUTPUT_DIR, "detection.png"),
        output
    )

    print("Detected symbols:", len(boxes))

    for i, box in enumerate(boxes):
        print(f"Symbol {i+1}: {box}")

    print("\nNormalized images saved in:", OUTPUT_DIR)

    # Display
    cv2.imshow("Original", image)
    cv2.imshow("Binary", binary)
    cv2.imshow("Detection", output)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    preprocess_image()
