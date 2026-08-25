"""
prepare_ocr_dataset.py

Create a PaddleOCR text-recognition dataset from wing-tag annotations.

The existing YOLO train/val/test image split is reused so that OCR
training and evaluation remain separated by source image.

Only fully readable tag annotations are included.
Annotations containing "*" are excluded.

Output:
    data/ocr/
        images/
            train/
            val/
            test/
        train.txt
        val.txt
        test.txt
        character_dict.txt
        metadata.csv

Usage:
    python prepare_ocr_dataset.py
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import cv2

import config


OCR_DIRECTORY = (
    config.PROJECT_ROOT
    / "data"
    / "ocr"
)

OCR_IMAGE_DIRECTORY = (
    OCR_DIRECTORY
    / "images"
)

OCR_CHARACTER_DICT = (
    OCR_DIRECTORY
    / "character_dict.txt"
)

OCR_METADATA_FILE = (
    OCR_DIRECTORY
    / "metadata.csv"
)

OCR_CHARACTERS = (
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
)

CROP_MARGIN_RATIO = 0.05


def load_annotations():
    """Load annotation rows."""

    with open(
        config.ANNOTATION_FILE,
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

    for split in (
        "train",
        "val",
        "test",
    ):

        split_directory = (
            config.YOLO_DATA_DIRECTORY
            / "images"
            / split
        )

        if not split_directory.exists():
            continue

        for image_path in split_directory.iterdir():

            if not image_path.is_file():
                continue

            split_lookup[
                image_path.name.lower()
            ] = split

    return split_lookup


def is_readable_text(
    text,
):
    """
    Return True only for complete alphanumeric annotations.

    Examples included:
        46
        K97
        A31

    Examples excluded:
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
            float(row["x1"])
        )

        y1 = int(
            float(row["y1"])
        )

        x2 = int(
            float(row["x2"])
        )

        y2 = int(
            float(row["y2"])
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


def write_character_dictionary():
    """
    Write PaddleOCR recognition character dictionary.
    """

    with open(
        OCR_CHARACTER_DICT,
        "w",
        encoding="utf-8",
    ) as file:

        for character in OCR_CHARACTERS:
            file.write(
                character + "\n"
            )


def write_metadata(
    metadata_rows,
):
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

    with open(
        OCR_METADATA_FILE,
        "w",
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


def main():
    """Prepare PaddleOCR dataset."""

    rows = load_annotations()

    split_lookup = get_split_lookup()

    #
    # Recreate OCR directory.
    #

    if OCR_DIRECTORY.exists():
        shutil.rmtree(
            OCR_DIRECTORY
        )

    for split in (
        "train",
        "val",
        "test",
    ):

        (
            OCR_IMAGE_DIRECTORY
            / split
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    labels = {
        "train": [],
        "val": [],
        "test": [],
    }

    counters = {
        "train": 0,
        "val": 0,
        "test": 0,
    }

    metadata_rows = []

    excluded_unreadable = 0
    excluded_missing_split = 0
    excluded_invalid_crop = 0
    excluded_missing_image = 0

    for row in rows:

        image_name = (
            row["image_file"]
            .strip()
        )

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

        #
        # Exclude *, **, *4, 8*, etc.
        #

        if not is_readable_text(
            tag_text
        ):
            excluded_unreadable += 1
            continue

        split = split_lookup.get(
            image_name.lower()
        )

        if split is None:
            excluded_missing_split += 1
            continue

        image_path = (
            config.RAW_IMAGE_DIRECTORY
            / image_name
        )

        if not image_path.is_file():
            excluded_missing_image += 1
            continue

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            excluded_missing_image += 1
            continue

        crop_result = crop_tag(
            image,
            row,
        )

        if crop_result is None:
            excluded_invalid_crop += 1
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
            excluded_invalid_crop += 1
            continue

        #
        # PaddleOCR expects:
        #
        # image_path<TAB>text
        #

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

        #
        # Store information needed to trace this crop
        # back to the original image and annotation.
        #

        metadata_rows.append(
            {
                "crop_file": relative_path_string,
                "source_image": image_name,
                "split": split,
                "text": tag_text,
                "plate_color": plate_color,
                "text_color": text_color,

                "annotation_x1": row["x1"],
                "annotation_y1": row["y1"],
                "annotation_x2": row["x2"],
                "annotation_y2": row["y2"],

                "crop_x1": crop_x1,
                "crop_y1": crop_y1,
                "crop_x2": crop_x2,
                "crop_y2": crop_y2,
            }
        )

    #
    # Write PaddleOCR label files.
    #

    for split in (
        "train",
        "val",
        "test",
    ):

        label_file = (
            OCR_DIRECTORY
            / f"{split}.txt"
        )

        with open(
            label_file,
            "w",
            encoding="utf-8",
        ) as file:

            for line in labels[split]:
                file.write(
                    line + "\n"
                )

    #
    # Character dictionary.
    #

    write_character_dictionary()

    #
    # Metadata.
    #

    write_metadata(
        metadata_rows
    )

    #
    # Summary.
    #

    print()
    print("=" * 50)
    print("OCR Dataset Preparation")
    print("=" * 50)

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
        f"{excluded_unreadable}"
    )

    print(
        f"Excluded missing split      : "
        f"{excluded_missing_split}"
    )

    print(
        f"Excluded missing image      : "
        f"{excluded_missing_image}"
    )

    print(
        f"Excluded invalid crop       : "
        f"{excluded_invalid_crop}"
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


if __name__ == "__main__":
    main()