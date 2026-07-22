"""
visualize_dataset.py

Visualize YOLO dataset annotations.

Examples
--------
Random 5 images:
    python utils/visualize_dataset.py

Random 20 validation images:
    python utils/visualize_dataset.py --split val --num 20

Specific image:
    python utils/visualize_dataset.py --image IMG_123
"""

import argparse
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2

import config


MAX_DISPLAY_WIDTH = 1600
MAX_DISPLAY_HEIGHT = 900


def parse_arguments():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Visualize YOLO dataset annotations."
    )

    parser.add_argument(
        "--num",
        type=int,
        default=5,
        help="Number of random images to visualize.",
    )

    parser.add_argument(
        "--split",
        choices=["all", *config.DATASET_SPLITS],
        default="all",
        help="Dataset split to visualize.",
    )

    parser.add_argument(
        "--image",
        type=str,
        help="Specific image filename (with or without extension).",
    )

    return parser.parse_args()


def get_image_list(split):
    """Return a list of image paths."""

    image_paths = []

    splits = (
        config.DATASET_SPLITS.keys()
        if split == "all"
        else [split]
    )

    for current_split in splits:

        image_directory = (
            config.YOLO_IMAGE_DIRECTORY / current_split
        )

        for extension in config.SUPPORTED_IMAGE_EXTENSIONS:
            image_paths.extend(
                image_directory.glob(f"*{extension}")
            )

    return image_paths


def find_image(image_name):
    """Find a specific image across all dataset splits."""

    image_path = Path(image_name)

    if image_path.suffix:

        candidates = [image_path.name]

    else:

        candidates = [
            image_name + extension
            for extension in config.SUPPORTED_IMAGE_EXTENSIONS
        ]

    for split in config.DATASET_SPLITS:

        image_directory = (
            config.YOLO_IMAGE_DIRECTORY / split
        )

        for candidate in candidates:

            path = image_directory / candidate

            if path.exists():
                return path

    return None


def draw_annotations(image, label_path):
    """Draw YOLO annotations onto an image."""

    image_height, image_width = image.shape[:2]

    object_count = 0

    if not label_path.exists():
        return image, object_count

    with label_path.open(
            mode="r",
            encoding="utf-8",
    ) as file:

        for line in file:

            values = line.strip().split()

            if len(values) != 5:
                continue

            class_id = int(values[0])

            x_center = float(values[1]) * image_width
            y_center = float(values[2]) * image_height
            width = float(values[3]) * image_width
            height = float(values[4]) * image_height

            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            class_name = config.OBJECT_CLASSES.get(
                class_id,
                str(class_id),
            )

            (text_width, text_height), baseline = cv2.getTextSize(
                class_name,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2,
            )

            cv2.rectangle(
                image,
                (x1, y1 - text_height - 8),
                (x1 + text_width + 6, y1),
                (0, 255, 0),
                -1,
            )

            cv2.putText(
                image,
                class_name,
                (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

            object_count += 1

    return image, object_count


def visualize_images(image_paths):
    """Display images one by one."""

    total_images = len(image_paths)

    for index, image_path in enumerate(image_paths, start=1):

        split = image_path.parent.name

        label_path = (
            config.YOLO_LABEL_DIRECTORY
            / split
            / f"{image_path.stem}.txt"
        )

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        image, object_count = draw_annotations(
            image,
            label_path,
        )

        info = [
            f"Image :   {index}/{total_images}",
            f"Split :   {split}",
            f"File  :   {image_path.name}",
            f"Objects : {object_count}",
        ]

        y = 30

        for line in info:

            cv2.putText(
                image,
                line,
                (15, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            y += 30

        height, width = image.shape[:2]

        scale = min(
            MAX_DISPLAY_WIDTH / width,
            MAX_DISPLAY_HEIGHT / height,
            1.0,
        )

        if scale < 1:

            image = cv2.resize(
                image,
                (
                    int(width * scale),
                    int(height * scale),
                ),
                interpolation=cv2.INTER_AREA,
            )

        cv2.imshow("Dataset Visualization", image)

        while True:

            key = cv2.waitKey(0)

            if key in (13, 10, 32):
                break

            if key in (ord("q"), ord("Q")):
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()


def main():

    args = parse_arguments()

    print("=" * 55)
    print("Visualizing dataset")
    print("=" * 55)

    if args.image:

        image_path = find_image(args.image)

        if image_path is None:

            print(f"\nImage not found: {args.image}")
            sys.exit(1)

        image_paths = [image_path]

    else:

        image_paths = get_image_list(args.split)

        if not image_paths:

            print("\nNo images found.")
            sys.exit(1)

        random.shuffle(image_paths)

        image_paths = image_paths[: args.num]

    print(f"\nImages : {len(image_paths)}")
    print(f"Split  : {args.split}\n")

    visualize_images(image_paths)


if __name__ == "__main__":
    main()