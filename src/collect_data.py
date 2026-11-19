
import os
import random
import cv2
import numpy as np

from PIL import Image, ImageDraw, ImageFont


# Configuration
SAMPLES_PER_DIGIT = 300
IMAGE_SIZE = 28
OUTPUT_DIR = "dataset/processed/digits"

random.seed(42)
np.random.seed(42)


# Windows handwriting-style fonts
FONT_DIR = r"C:\Windows\Fonts"

FONT_NAMES = [
    "segoepr.ttf",
    "segoeprb.ttf",
    "comic.ttf",
    "comicbd.ttf",
    "inkfree.ttf",
    "bradhitc.ttf",
    "mvboli.ttf",
    "gabriola.ttf",
    "arial.ttf"
]

FONTS = [
    os.path.join(FONT_DIR, name)
    for name in FONT_NAMES
    if os.path.exists(os.path.join(FONT_DIR, name))
]

if not FONTS:
    raise FileNotFoundError("No suitable Windows fonts found.")

print("Available fonts:", len(FONTS))


def normalize(image):
    """Resize digit while preserving its aspect ratio."""

    image = np.array(image)

    points = cv2.findNonZero(image)

    if points is None:
        return np.zeros((28, 28), dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(points)
    digit = image[y:y+h, x:x+w]

    scale = 20 / max(w, h)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    digit = cv2.resize(
        digit,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.zeros((28, 28), dtype=np.uint8)

    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    canvas[
        y_offset:y_offset+new_h,
        x_offset:x_offset+new_w
    ] = digit

    return canvas


def generate_digit(digit):
    """Generate one synthetic digit image."""

    # Random font and size
    font_path = random.choice(FONTS)
    font_size = random.randint(34, 48)

    font = ImageFont.truetype(font_path, font_size)

    # White background
    image = Image.new("L", (64, 64), 255)
    draw = ImageDraw.Draw(image)

    # Random stroke thickness
    stroke = random.randint(0, 2)

    # Draw digit
    draw.text(
        (32, 32),
        str(digit),
        font=font,
        fill=0,
        anchor="mm",
        stroke_width=stroke,
        stroke_fill=0
    )

    # Random rotation
    angle = random.uniform(-15, 15)

    image = image.rotate(
        angle,
        fillcolor=255,
        resample=Image.Resampling.BICUBIC
    )

    # Convert black handwriting to white foreground
    image = 255 - np.array(image)

    # Normalize
    image = normalize(image)

    # Add slight noise
    noise = np.random.normal(0, 5, image.shape)

    image = np.clip(
        image.astype(float) + noise,
        0,
        255
    ).astype(np.uint8)

    return image


def main():

    for digit in range(10):

        samples = []

        for _ in range(SAMPLES_PER_DIGIT):
            samples.append(generate_digit(digit))

        # Shuffle samples
        random.shuffle(samples)

        # 80% training, 20% testing
        split = int(0.8 * len(samples))

        train_samples = samples[:split]
        test_samples = samples[split:]

        for split_name, data in [
            ("train", train_samples),
            ("test", test_samples)
        ]:

            folder = os.path.join(
                OUTPUT_DIR,
                split_name,
                str(digit)
            )

            os.makedirs(folder, exist_ok=True)

            for index, sample in enumerate(data):

                path = os.path.join(
                    folder,
                    f"{index}.png"
                )

                cv2.imwrite(path, sample)

        print(
            f"Digit {digit}: "
            f"{len(train_samples)} training, "
            f"{len(test_samples)} testing"
        )

    print("\nDataset generation completed!")
    print("Total samples:", SAMPLES_PER_DIGIT * 10)


if __name__ == "__main__":
    main()
