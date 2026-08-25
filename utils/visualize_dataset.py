"""
visualize_dataset.py

Visualize wing-tag dataset annotations.

Two visualization modes are supported:

1. YOLO annotations
   Visualizes the generated YOLO dataset.

2. Raw annotations
   Visualizes the original annotations.csv directly on the raw images,
   including tag identifier, plate color, and text color.

Examples
--------
Random 5 YOLO images:
    python utils/visualize_dataset.py

Random 20 validation images:
    python utils/visualize_dataset.py --split val --num 20

Specific YOLO image:
    python utils/visualize_dataset.py --image IMG_123

Random 5 raw annotated images:
    python utils/visualize_dataset.py --raw

Random 50 raw annotated images:
    python utils/visualize_dataset.py --raw --num 50

Specific raw image:
    python utils/visualize_dataset.py --raw --image IMG_123.jpg
"""

import argparse
import csv
import random
import sys
from collections import defaultdict
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
        description="Visualize dataset annotations."
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
        help="YOLO dataset split to visualize.",
    )

    parser.add_argument(
        "--image",
        type=str,
        help="Specific image filename (with or without extension).",
    )

    parser.add_argument(
        "--raw",
        action="store_true",
        help="Visualize original annotations.csv on raw images.",
    )

    return parser.parse_args()


# ============================================================
# YOLO visualization
# ============================================================

def get_yolo_image_list(split):
    """Return a list of YOLO image paths."""

    image_paths = []

    splits = (
        config.DATASET_SPLITS.keys()
        if split == "all"
        else [split]
    )

    for current_split in splits:

        image_directory = (
            config.YOLO_IMAGE_DIRECTORY
            / current_split
        )

        for extension in config.SUPPORTED_IMAGE_EXTENSIONS:
            image_paths.extend(
                image_directory.glob(f"*{extension}")
            )

    return image_paths


def find_yolo_image(image_name):
    """Find a specific image across all YOLO dataset splits."""

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
            config.YOLO_IMAGE_DIRECTORY
            / split
        )

        for candidate in candidates:

            path = image_directory / candidate

            if path.exists():
                return path

    return None


def draw_yolo_annotations(image, label_path):
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

            draw_label(
                image,
                x1,
                y1,
                class_name,
            )

            object_count += 1

    return image, object_count


# ============================================================
# Raw annotation visualization
# ============================================================

def load_raw_annotations():
    """
    Load annotations.csv and group annotations by image filename.
    """

    annotations = defaultdict(list)

    with open(
        config.ANNOTATION_FILE,
        mode="r",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        reader = csv.DictReader(csv_file)

        for row in reader:

            image_name = row["image_file"].strip()

            if not image_name:
                continue

            annotations[image_name.lower()].append(row)

    return annotations


def get_raw_image_list(annotations):
    """
    Return raw images that have at least one annotation.
    """

    image_paths = []

    for image_name in annotations:

        path = (
            config.RAW_IMAGE_DIRECTORY
            / image_name
        )

        if path.exists():
            image_paths.append(path)
            continue

        #
        # Filename casing may differ on some systems.
        #

        for candidate in config.RAW_IMAGE_DIRECTORY.iterdir():

            if (
                candidate.is_file()
                and candidate.name.lower() == image_name
            ):
                image_paths.append(candidate)
                break

    return image_paths


def find_raw_image(image_name):
    """Find a specific image in the raw image directory."""

    image_path = Path(image_name)

    if image_path.suffix:

        candidates = [
            image_path.name
        ]

    else:

        candidates = [
            image_name + extension
            for extension in config.SUPPORTED_IMAGE_EXTENSIONS
        ]

    for candidate in candidates:

        path = (
            config.RAW_IMAGE_DIRECTORY
            / candidate
        )

        if path.exists():
            return path

    #
    # Case-insensitive fallback.
    #

    candidate_names = {
        candidate.lower()
        for candidate in candidates
    }

    for path in config.RAW_IMAGE_DIRECTORY.iterdir():

        if (
            path.is_file()
            and path.name.lower() in candidate_names
        ):
            return path

    return None


def draw_raw_annotations(image, rows):
    """
    Draw original annotations.csv bounding boxes and labels.

    Label format:
        ID | plate_color / text_color

    Example:
        K97 | yellow / blue
    """

    object_count = 0

    for row in rows:

        try:

            x1 = int(float(row["x1"]))
            y1 = int(float(row["y1"]))
            x2 = int(float(row["x2"]))
            y2 = int(float(row["y2"]))

        except (ValueError, TypeError, KeyError):
            continue

        tag_text = (
            row[
                config.CSV_PLATE_TEXT
            ]
            .strip()
            .upper()
        )

        plate_color = (
            row[
                config.CSV_PLATE_COLOR
            ]
            .strip()
            .lower()
        )

        text_color = (
            row[
                config.CSV_TEXT_COLOR
            ]
            .strip()
            .lower()
        )

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3,
        )

        label = (
            f"{tag_text} | "
            f"{plate_color} / {text_color}"
        )

        draw_label(
            image,
            x1,
            y1,
            label,
        )

        object_count += 1

    return image, object_count


# ============================================================
# Display helpers
# ============================================================

def draw_label(
    image,
    x,
    y,
    text,
):
    """Draw a readable label above a bounding box."""

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    (
        text_width,
        text_height,
    ), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness,
    )

    label_y = max(
        y,
        text_height + 10,
    )

    cv2.rectangle(
        image,
        (
            x,
            label_y - text_height - 8,
        ),
        (
            x + text_width + 8,
            label_y,
        ),
        (0, 255, 0),
        -1,
    )

    cv2.putText(
        image,
        text,
        (
            x + 4,
            label_y - 5,
        ),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
    )


def draw_image_info(
    image,
    index,
    total_images,
    image_path,
    object_count,
    mode,
    split=None,
):
    """Draw image information in the upper-left corner."""

    info = [
        f"Image   : {index}/{total_images}",
        f"Mode    : {mode}",
        f"File    : {image_path.name}",
        f"Objects : {object_count}",
    ]

    if split is not None:
        info.insert(
            2,
            f"Split   : {split}",
        )

    y = 30

    for line in info:

        cv2.putText(
            image,
            line,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            3,
        )

        cv2.putText(
            image,
            line,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            1,
        )

        y += 30


def resize_for_display(image):
    """Resize image to fit on screen."""

    height, width = image.shape[:2]

    scale = min(
        MAX_DISPLAY_WIDTH / width,
        MAX_DISPLAY_HEIGHT / height,
        1.0,
    )

    if scale >= 1:
        return image

    return cv2.resize(
        image,
        (
            int(width * scale),
            int(height * scale),
        ),
        interpolation=cv2.INTER_AREA,
    )


# ============================================================
# Visualization
# ============================================================

def visualize_yolo_images(image_paths):
    """Display YOLO annotated images one by one."""

    total_images = len(image_paths)

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):

        split = image_path.parent.name

        label_path = (
            config.YOLO_LABEL_DIRECTORY
            / split
            / f"{image_path.stem}.txt"
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            continue

        image, object_count = (
            draw_yolo_annotations(
                image,
                label_path,
            )
        )

        draw_image_info(
            image=image,
            index=index,
            total_images=total_images,
            image_path=image_path,
            object_count=object_count,
            mode="YOLO",
            split=split,
        )

        image = resize_for_display(
            image
        )

        if not show_image(image):
            return


def visualize_raw_images(
    image_paths,
    annotations,
):
    """Display original annotated images one by one."""

    total_images = len(image_paths)

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            continue

        rows = annotations.get(
            image_path.name.lower(),
            [],
        )

        image, object_count = (
            draw_raw_annotations(
                image,
                rows,
            )
        )

        draw_image_info(
            image=image,
            index=index,
            total_images=total_images,
            image_path=image_path,
            object_count=object_count,
            mode="RAW",
        )

        image = resize_for_display(
            image
        )

        if not show_image(image):
            return


def show_image(image):
    """
    Show one image.

    Enter / Space -> next image
    Q             -> quit
    """

    cv2.imshow(
        "Dataset Visualization",
        image,
    )

    while True:

        key = cv2.waitKey(0)

        if key in (
            13,
            10,
            32,
        ):
            return True

        if key in (
            ord("q"),
            ord("Q"),
        ):
            cv2.destroyAllWindows()
            return False


# ============================================================
# Main
# ============================================================

def main():

    args = parse_arguments()

    print("=" * 55)
    print("Visualizing dataset")
    print("=" * 55)

    #
    # Raw annotations.csv mode.
    #

    if args.raw:

        annotations = load_raw_annotations()

        if args.image:

            image_path = find_raw_image(
                args.image
            )

            if image_path is None:

                print(
                    f"\nRaw image not found: "
                    f"{args.image}"
                )

                sys.exit(1)

            image_paths = [
                image_path
            ]

        else:

            image_paths = (
                get_raw_image_list(
                    annotations
                )
            )

            if not image_paths:

                print(
                    "\nNo annotated raw images found."
                )

                sys.exit(1)

            random.shuffle(
                image_paths
            )

            image_paths = (
                image_paths[
                    :args.num
                ]
            )

        print("\nMode   : RAW annotations")
        print(f"Images : {len(image_paths)}")
        print()

        visualize_raw_images(
            image_paths,
            annotations,
        )

    #
    # Existing YOLO mode.
    #

    else:

        if args.image:

            image_path = find_yolo_image(
                args.image
            )

            if image_path is None:

                print(
                    f"\nImage not found: "
                    f"{args.image}"
                )

                sys.exit(1)

            image_paths = [
                image_path
            ]

        else:

            image_paths = (
                get_yolo_image_list(
                    args.split
                )
            )

            if not image_paths:

                print(
                    "\nNo images found."
                )

                sys.exit(1)

            random.shuffle(
                image_paths
            )

            image_paths = (
                image_paths[
                    :args.num
                ]
            )

        print("\nMode   : YOLO")
        print(f"Images : {len(image_paths)}")
        print(f"Split  : {args.split}")
        print()

        visualize_yolo_images(
            image_paths
        )

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()