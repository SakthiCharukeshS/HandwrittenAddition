
import cv2
import numpy as np


def normalize_digit(image):
    # Convert foreground pixels to white and background to black
    _, binary = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY
    )

    points = cv2.findNonZero(binary)

    if points is None:
        return np.zeros((28, 28), dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(points)
    digit = binary[y:y+h, x:x+w]

    # Preserve aspect ratio; fit inside 20x20
    scale = 20.0 / max(w, h)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(
        digit, (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    # Place digit at the center of a 28x28 canvas
    canvas = np.zeros((28, 28), dtype=np.uint8)

    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    canvas[
        y_offset:y_offset+new_h,
        x_offset:x_offset+new_w
    ] = resized

    return canvas
