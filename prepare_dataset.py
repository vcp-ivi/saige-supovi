"""
prepare_dataset.py

Convert the original wing-tag annotations into a YOLO dataset.

data/raw -> data/yolo
"""

import csv
import random
import shutil
from collections import defaultdict

from PIL import Image
import yaml

import config


def read_annotations() -> dict[str, list[tuple[float, float, float, float]]]:
    """
    Read the annotation CSV and group bounding boxes by image.

    Images with empty coordinates are treated as negative examples.
    """
    annotations = defaultdict(list)

    with config.ANNOTATION_FILE.open(
        mode="r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        for row_number, row in enumerate(reader, start=2):
            image_file = row[config.CSV_IMAGE_FILE].strip()

            if not image_file:
                raise ValueError(
                    f"Missing image filename in CSV row {row_number}."
                )

            # Ensure negative images are included even when they have no box.
            annotations[image_file]

            coordinate_values = [
                row[config.CSV_X1],
                row[config.CSV_Y1],
                row[config.CSV_X2],
                row[config.CSV_Y2],
            ]

            # Empty coordinates mean that this image contains no wing tag.
            if all(not value.strip() for value in coordinate_values):
                continue

            # A bounding box must contain all four coordinates.
            if any(not value.strip() for value in coordinate_values):
                raise ValueError(
                    f"Incomplete bounding box for '{image_file}' "
                    f"in CSV row {row_number}."
                )

            bounding_box = (
                float(row[config.CSV_X1]),
                float(row[config.CSV_Y1]),
                float(row[config.CSV_X2]),
                float(row[config.CSV_Y2]),
            )

            annotations[image_file].append(bounding_box)

    return dict(annotations)


def split_image_group(
    image_files: list[str],
    random_generator: random.Random,
) -> tuple[list[str], list[str], list[str]]:
    """
    Split one group of images into train, validation, and test subsets.
    """
    image_files = image_files.copy()
    random_generator.shuffle(image_files)

    number_of_images = len(image_files)

    splits = config.DATASET_SPLITS

    train_count = round(
        number_of_images * splits["train"]
    )

    validation_count = round(
        number_of_images * splits["val"]
    )

    train_images = image_files[:train_count]

    validation_images = image_files[
        train_count:train_count + validation_count
    ]

    test_images = image_files[
        train_count + validation_count:
    ]

    return train_images, validation_images, test_images


def split_dataset(
    annotations: dict[str, list[tuple[float, float, float, float]]],
) -> dict[str, list[str]]:
    """
    Split the dataset by image.

    Positive and negative images are split separately so that each subset
    contains approximately the same proportion of both.
    """
    positive_images = [
        image_file
        for image_file, bounding_boxes in annotations.items()
        if bounding_boxes
    ]

    negative_images = [
        image_file
        for image_file, bounding_boxes in annotations.items()
        if not bounding_boxes
    ]

    random_generator = random.Random(config.RANDOM_SEED)

    positive_split = split_image_group(
        positive_images,
        random_generator,
    )

    negative_split = split_image_group(
        negative_images,
        random_generator,
    )

    dataset_splits = {
        "train": positive_split[0] + negative_split[0],
        "val": positive_split[1] + negative_split[1],
        "test": positive_split[2] + negative_split[2],
    }

    for image_files in dataset_splits.values():
        random_generator.shuffle(image_files)

    return dataset_splits


def create_output_directories() -> None:
    """
    Delete the previous generated dataset and create a clean YOLO structure.
    """
    if config.YOLO_DATA_DIRECTORY.exists():
        shutil.rmtree(config.YOLO_DATA_DIRECTORY)

    for split_name in config.DATASET_SPLITS:
        image_directory = (
            config.YOLO_IMAGE_DIRECTORY / split_name
        )

        label_directory = (
            config.YOLO_LABEL_DIRECTORY / split_name
        )

        image_directory.mkdir(parents=True, exist_ok=True)
        label_directory.mkdir(parents=True, exist_ok=True)


def convert_to_yolo(
    bounding_box: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """
    Convert x1, y1, x2, y2 pixel coordinates into normalized YOLO format.
    """
    x1, y1, x2, y2 = bounding_box

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Invalid bounding box: {bounding_box}"
        )

    if (
        x1 < 0
        or y1 < 0
        or x2 > image_width
        or y2 > image_height
    ):
        raise ValueError(
            f"Bounding box {bounding_box} is outside an image "
            f"of size {image_width} x {image_height}."
        )

    box_width = x2 - x1
    box_height = y2 - y1

    x_center = x1 + box_width / 2
    y_center = y1 + box_height / 2

    return (
        x_center / image_width,
        y_center / image_height,
        box_width / image_width,
        box_height / image_height,
    )


def process_dataset(
    annotations: dict[str, list[tuple[float, float, float, float]]],
    dataset_splits: dict[str, list[str]],
) -> None:
    """
    Copy images and create one YOLO label file for each image.
    """
    for split_name, image_files in dataset_splits.items():
        print(
            f"Creating {split_name} subset: "
            f"{len(image_files)} images"
        )

        for image_file in image_files:
            source_image = config.RAW_IMAGE_DIRECTORY / image_file

            if not source_image.is_file():
                raise FileNotFoundError(
                    f"Image not found: {source_image}"
                )

            if source_image.suffix.lower() not in config.SUPPORTED_IMAGE_EXTENSIONS:
                raise ValueError(
                    f"Unsupported image format: {source_image}"
                )

            destination_image = (
                config.YOLO_IMAGE_DIRECTORY
                / split_name
                / source_image.name
            )

            destination_label = (
                config.YOLO_LABEL_DIRECTORY
                / split_name
                / f"{source_image.stem}.txt"
            )

            with Image.open(source_image) as image:
                image_width, image_height = image.size

            label_lines = []

            for bounding_box in annotations[image_file]:
                x_center, y_center, width, height = convert_to_yolo(
                    bounding_box,
                    image_width,
                    image_height,
                )

                label_lines.append(
                    f"0 "
                    f"{x_center:.6f} "
                    f"{y_center:.6f} "
                    f"{width:.6f} "
                    f"{height:.6f}"
                )

            # Negative images receive an empty label file.
            label_text = "\n".join(label_lines)

            if label_text:
                label_text += "\n"

            destination_label.write_text(
                label_text,
                encoding="utf-8",
            )

            shutil.copy2(
                source_image,
                destination_image,
            )


def create_data_yaml() -> None:
    """
    Create the dataset configuration file used by Ultralytics YOLO.
    """
    class_names = [
        class_name
        for _, class_name in sorted(config.OBJECT_CLASSES.items())
    ]

    dataset_config = {
        "path": str(config.YOLO_DATA_DIRECTORY.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": class_names,
    }

    with config.YOLO_DATASET_CONFIG.open(
        mode="w",
        encoding="utf-8",
    ) as yaml_file:
        yaml.safe_dump(
            dataset_config,
            yaml_file,
            sort_keys=False,
        )


def main() -> None:
    """
    Run the complete dataset-preparation process.
    """
    if not config.ANNOTATION_FILE.is_file():
        raise FileNotFoundError(
            f"Annotation file not found: {config.ANNOTATION_FILE}"
        )

    if not config.RAW_IMAGE_DIRECTORY.is_dir():
        raise FileNotFoundError(
            f"Image directory not found: "
            f"{config.RAW_IMAGE_DIRECTORY}"
        )

    print("Reading annotations...")
    annotations = read_annotations()

    print(f"Found {len(annotations)} unique images.")

    dataset_splits = split_dataset(annotations)

    create_output_directories()

    process_dataset(
        annotations,
        dataset_splits,
    )

    create_data_yaml()

    positive_images = sum(
        bool(bounding_boxes)
        for bounding_boxes in annotations.values()
    )

    negative_images = len(annotations) - positive_images

    print("\nDataset preparation completed.")
    print(f"Positive images: {positive_images}")
    print(f"Negative images: {negative_images}")
    print(f"Train images:    {len(dataset_splits['train'])}")
    print(f"Val images:      {len(dataset_splits['val'])}")
    print(f"Test images:     {len(dataset_splits['test'])}")
    print(f"Dataset saved to: {config.YOLO_DATA_DIRECTORY}")


if __name__ == "__main__":
    main()