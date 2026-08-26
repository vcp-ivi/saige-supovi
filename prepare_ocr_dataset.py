"""
prepare_ocr_dataset.py

Create PaddleOCR text-recognition datasets from the annotated wing tags.

The existing YOLO train/val/test split is reused. Only fully readable
identifiers are included.

Usage:
    python prepare_ocr_dataset.py
    python prepare_ocr_dataset.py --augment
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2

import config


OCR_DIRECTORY = config.OCR_DATA_DIRECTORY

OCR_IMAGE_DIRECTORY = (
    config.OCR_IMAGE_DIRECTORY
)

OCR_CHARACTER_DICT = (
    config.OCR_CHARACTER_DICT
)

OCR_METADATA_FILE = (
    config.OCR_METADATA_FILE
)

AUGMENTED_IMAGE_DIRECTORY = (
    OCR_IMAGE_DIRECTORY
    / "train_augmented"
)

AUGMENTED_LABEL_FILE = (
    OCR_DIRECTORY
    / "train_augmented.txt"
)

OCR_CHARACTERS = (
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
)

CROP_MARGIN_RATIO = 0.05

ROTATIONS = {
    "r90": cv2.ROTATE_90_CLOCKWISE,
    "r180": cv2.ROTATE_180,
    "r270": cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def parse_arguments():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Prepare the PaddleOCR wing-tag dataset."
    )

    parser.add_argument(
        "--augment",
        action="store_true",
        help=(
            "Create additional 90°, 180°, and 270° rotations "
            "of every training crop."
        ),
    )

    return parser.parse_args()


def load_annotations():
    """Load annotation rows."""

    with config.ANNOTATION_FILE.open(
        mode="r",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        return list(
            csv.DictReader(csv_file)
        )


def get_split_lookup():
    """
    Map each source image filename to its existing YOLO split.
    """

    split_lookup = {}

    for split in config.DATASET_SPLITS:

        split_directory = (
            config.YOLO_IMAGE_DIRECTORY
            / split
        )

        if not split_directory.exists():
            continue

        for image_path in split_directory.iterdir():

            if not image_path.is_file():
                continue

            image_key = (
                image_path.name
                .strip()
                .lower()
            )

            if (
                image_key in split_lookup
                and split_lookup[image_key] != split
            ):
                raise ValueError(
                    f"Image appears in multiple YOLO splits: "
                    f"{image_path.name}"
                )

            split_lookup[image_key] = split

    return split_lookup


def is_readable_text(text):
    """
    Return True only for complete alphanumeric annotations.

    Included:
        46
        K97
        A31

    Excluded:
        *
        **
        *4
        8*
        P3*
    """

    text = (
        text
        .strip()
        .upper()
    )

    if not text:
        return False

    if "*" in text:
        return False

    return all(
        character in OCR_CHARACTERS
        for character in text
    )


def crop_tag(
    image,
    row,
):
    """
    Crop one tag using its ground-truth bounding box.

    A small margin is added around the annotation.

    Returns:
        crop
        final crop coordinates (x1, y1, x2, y2)

    Returns None when the bounding box is invalid.
    """

    image_height, image_width = image.shape[:2]

    try:
        x1 = int(
            float(
                row[config.CSV_X1]
            )
        )

        y1 = int(
            float(
                row[config.CSV_Y1]
            )
        )

        x2 = int(
            float(
                row[config.CSV_X2]
            )
        )

        y2 = int(
            float(
                row[config.CSV_Y2]
            )
        )

    except (
        ValueError,
        TypeError,
    ):
        return None

    width = x2 - x1
    height = y2 - y1

    if width <= 0 or height <= 0:
        return None

    margin_x = int(
        width
        * CROP_MARGIN_RATIO
    )

    margin_y = int(
        height
        * CROP_MARGIN_RATIO
    )

    x1 = max(
        0,
        x1 - margin_x,
    )

    y1 = max(
        0,
        y1 - margin_y,
    )

    x2 = min(
        image_width,
        x2 + margin_x,
    )

    y2 = min(
        image_height,
        y2 + margin_y,
    )

    crop = image[
        y1:y2,
        x1:x2,
    ]

    if crop.size == 0:
        return None

    return (
        crop,
        (
            x1,
            y1,
            x2,
            y2,
        ),
    )


def create_output_directories():
    """
    Recreate the generated OCR dataset directory.

    Ask for confirmation before deleting an existing OCR dataset.
    """

    if OCR_DIRECTORY.exists():

        print()
        print(
            f"OCR dataset already exists:\n"
            f"{OCR_DIRECTORY}"
        )

        print()
        print(
            "Running this script will delete the existing OCR dataset "
            "and create a new one."
        )

        response = input(
            "\nContinue? [y/N]: "
        ).strip().lower()

        if response not in (
            "y",
            "yes",
        ):
            print(
                "\nDataset preparation cancelled."
            )
            return False

        shutil.rmtree(
            OCR_DIRECTORY
        )

    for split in config.DATASET_SPLITS:

        (
            OCR_IMAGE_DIRECTORY
            / split
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    (OCR_DIRECTORY / ".gitkeep").touch()

    return True


def write_label_files(labels):
    """Write PaddleOCR label files for each dataset split."""

    for split in config.DATASET_SPLITS:

        label_file = (
            OCR_DIRECTORY
            / f"{split}.txt"
        )

        with label_file.open(
            mode="w",
            encoding="utf-8",
        ) as file:

            for line in labels[split]:
                file.write(
                    line + "\n"
                )


def write_character_dictionary():
    """Write the PaddleOCR character dictionary."""

    with OCR_CHARACTER_DICT.open(
        mode="w",
        encoding="utf-8",
    ) as file:

        for character in OCR_CHARACTERS:
            file.write(
                character + "\n"
            )


def write_metadata(metadata_rows):
    """
    Write metadata linking every OCR crop to its source image.
    """

    fieldnames = [
        "crop_file",
        "source_image",
        "split",
        "text",
        "plate_color",
        "text_color",
        "annotation_x1",
        "annotation_y1",
        "annotation_x2",
        "annotation_y2",
        "crop_x1",
        "crop_y1",
        "crop_x2",
        "crop_y2",
    ]

    with OCR_METADATA_FILE.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            metadata_rows
        )


def create_base_dataset():
    """
    Create the standard OCR train/val/test dataset.

    Returns:
        counters containing the number of crops per split
        exclusion statistics
    """

    rows = load_annotations()

    split_lookup = get_split_lookup()

    if not create_output_directories():
        return None, None

    labels = {
        split: []
        for split in config.DATASET_SPLITS
    }

    counters = {
        split: 0
        for split in config.DATASET_SPLITS
    }

    metadata_rows = []

    excluded = {
        "unreadable": 0,
        "missing_split": 0,
        "missing_image": 0,
        "invalid_crop": 0,
    }

    for row in rows:

        image_name = (
            row[config.CSV_IMAGE_FILE]
            .strip()
        )

        tag_text = (
            row[config.CSV_PLATE_TEXT]
            .strip()
            .upper()
        )

        plate_color = (
            row[config.CSV_PLATE_COLOR]
            .strip()
            .lower()
        )

        text_color = (
            row[config.CSV_TEXT_COLOR]
            .strip()
            .lower()
        )

        if not is_readable_text(
            tag_text
        ):
            excluded["unreadable"] += 1
            continue

        split = split_lookup.get(
            image_name.lower()
        )

        if split is None:
            excluded["missing_split"] += 1
            continue

        image_path = (
            config.RAW_IMAGE_DIRECTORY
            / image_name
        )

        if not image_path.is_file():
            excluded["missing_image"] += 1
            continue

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            excluded["missing_image"] += 1
            continue

        crop_result = crop_tag(
            image,
            row,
        )

        if crop_result is None:
            excluded["invalid_crop"] += 1
            continue

        (
            crop,
            crop_coordinates,
        ) = crop_result

        (
            crop_x1,
            crop_y1,
            crop_x2,
            crop_y2,
        ) = crop_coordinates

        counters[split] += 1

        crop_name = (
            f"tag_{counters[split]:06d}.jpg"
        )

        crop_path = (
            OCR_IMAGE_DIRECTORY
            / split
            / crop_name
        )

        success = cv2.imwrite(
            str(crop_path),
            crop,
        )

        if not success:
            counters[split] -= 1
            excluded["invalid_crop"] += 1
            continue

        relative_path = (
            Path("images")
            / split
            / crop_name
        )

        relative_path_string = (
            relative_path.as_posix()
        )

        labels[split].append(
            f"{relative_path_string}\t{tag_text}"
        )

        metadata_rows.append(
            {
                "crop_file": relative_path_string,
                "source_image": image_name,
                "split": split,
                "text": tag_text,
                "plate_color": plate_color,
                "text_color": text_color,

                "annotation_x1": row[
                    config.CSV_X1
                ],
                "annotation_y1": row[
                    config.CSV_Y1
                ],
                "annotation_x2": row[
                    config.CSV_X2
                ],
                "annotation_y2": row[
                    config.CSV_Y2
                ],

                "crop_x1": crop_x1,
                "crop_y1": crop_y1,
                "crop_x2": crop_x2,
                "crop_y2": crop_y2,
            }
        )

    write_label_files(
        labels
    )

    write_character_dictionary()

    write_metadata(
        metadata_rows
    )

    return counters, excluded


def load_training_samples():
    """Load the generated OCR training samples."""

    train_label_file = (
        OCR_DIRECTORY
        / "train.txt"
    )

    samples = []

    with train_label_file.open(
        mode="r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            parts = line.split(
                "\t",
                1,
            )

            if len(parts) != 2:
                raise ValueError(
                    f"Invalid OCR label line: {line}"
                )

            image_path, label = parts

            samples.append(
                (
                    image_path,
                    label.strip(),
                )
            )

    return samples


def create_augmented_training_set():
    """
    Create rotated copies of all OCR training crops.

    Each original training crop is included together with:
        90°
        180°
        270°

    Validation and test images are not augmented.
    """

    samples = load_training_samples()

    AUGMENTED_IMAGE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    augmented_lines = []

    original_count = 0
    rotated_count = 0
    failed_count = 0

    for index, (
        relative_image_path,
        label,
    ) in enumerate(
        samples,
        start=1,
    ):

        source_path = (
            OCR_DIRECTORY
            / relative_image_path
        )

        image = cv2.imread(
            str(source_path)
        )

        if image is None:
            print(
                f"Could not read: "
                f"{source_path}"
            )

            failed_count += 1
            continue

        original_name = (
            f"tag_{index:06d}.jpg"
        )

        original_output_path = (
            AUGMENTED_IMAGE_DIRECTORY
            / original_name
        )

        if not cv2.imwrite(
            str(original_output_path),
            image,
        ):
            failed_count += 1
            continue

        original_relative_path = (
            Path("images")
            / "train_augmented"
            / original_name
        )

        augmented_lines.append(
            f"{original_relative_path.as_posix()}"
            f"\t{label}"
        )

        original_count += 1

        for (
            suffix,
            rotation_code,
        ) in ROTATIONS.items():

            rotated = cv2.rotate(
                image,
                rotation_code,
            )

            rotated_name = (
                f"tag_{index:06d}_{suffix}.jpg"
            )

            rotated_output_path = (
                AUGMENTED_IMAGE_DIRECTORY
                / rotated_name
            )

            if not cv2.imwrite(
                str(rotated_output_path),
                rotated,
            ):
                failed_count += 1
                continue

            rotated_relative_path = (
                Path("images")
                / "train_augmented"
                / rotated_name
            )

            augmented_lines.append(
                f"{rotated_relative_path.as_posix()}"
                f"\t{label}"
            )

            rotated_count += 1

    with AUGMENTED_LABEL_FILE.open(
        mode="w",
        encoding="utf-8",
    ) as file:

        for line in augmented_lines:
            file.write(
                line + "\n"
            )

    return {
        "original": original_count,
        "rotated": rotated_count,
        "failed": failed_count,
    }


def print_summary(
    counters,
    excluded,
    augmentation_statistics=None,
):
    """Print dataset-preparation statistics."""

    print()
    print("=" * 55)
    print("OCR Dataset Preparation")
    print("=" * 55)

    print()

    print(
        f"Train crops : "
        f"{counters['train']}"
    )

    print(
        f"Val crops   : "
        f"{counters['val']}"
    )

    print(
        f"Test crops  : "
        f"{counters['test']}"
    )

    print()

    print(
        f"Excluded unreadable/partial : "
        f"{excluded['unreadable']}"
    )

    print(
        f"Excluded missing split      : "
        f"{excluded['missing_split']}"
    )

    print(
        f"Excluded missing image      : "
        f"{excluded['missing_image']}"
    )

    print(
        f"Excluded invalid crop       : "
        f"{excluded['invalid_crop']}"
    )

    if augmentation_statistics is not None:

        print()
        print("-" * 55)
        print("Training augmentation")
        print("-" * 55)

        print(
            f"Original samples : "
            f"{augmentation_statistics['original']}"
        )

        print(
            f"Rotated samples  : "
            f"{augmentation_statistics['rotated']}"
        )

        print(
            f"Total samples    : "
            f"{augmentation_statistics['original'] + augmentation_statistics['rotated']}"
        )

        print(
            f"Failed images    : "
            f"{augmentation_statistics['failed']}"
        )

        print()

        print(
            f"Augmented labels:\n"
            f"{AUGMENTED_LABEL_FILE}"
        )

    print()

    print(
        f"OCR dataset created at:\n"
        f"{OCR_DIRECTORY}"
    )

    print()

    print(
        f"Crop metadata saved to:\n"
        f"{OCR_METADATA_FILE}"
    )


def main():
    """Prepare the PaddleOCR dataset."""

    args = parse_arguments()

    if not config.ANNOTATION_FILE.is_file():
        raise FileNotFoundError(
            f"Annotation file not found:\n"
            f"{config.ANNOTATION_FILE}"
        )

    if not config.RAW_IMAGE_DIRECTORY.is_dir():
        raise FileNotFoundError(
            f"Raw image directory not found:\n"
            f"{config.RAW_IMAGE_DIRECTORY}"
        )

    if not config.YOLO_IMAGE_DIRECTORY.is_dir():
        raise FileNotFoundError(
            f"YOLO dataset not found:\n"
            f"{config.YOLO_IMAGE_DIRECTORY}\n\n"
            "Prepare the YOLO dataset first."
        )

    counters, excluded = (
        create_base_dataset()
    )

    if counters is None:
        return

    augmentation_statistics = None

    if args.augment:
        augmentation_statistics = (
            create_augmented_training_set()
        )

    print_summary(
        counters=counters,
        excluded=excluded,
        augmentation_statistics=augmentation_statistics,
    )


if __name__ == "__main__":
    main()