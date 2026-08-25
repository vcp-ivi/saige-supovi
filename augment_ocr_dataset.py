"""
augment_ocr_dataset.py

Create rotated OCR training samples.

For every image listed in data/ocr/train.txt, create:
    original
    90-degree rotation
    180-degree rotation
    270-degree rotation

The original train.txt is NOT modified.

Output:
    data/ocr/train_augmented.txt
    data/ocr/images/train_augmented/

Usage:
    python augment_ocr_dataset.py
"""

from pathlib import Path
import shutil

import cv2

import config


OCR_DIRECTORY = (
    config.PROJECT_ROOT
    / "data"
    / "ocr"
)

TRAIN_LABEL_FILE = (
    OCR_DIRECTORY
    / "train.txt"
)

AUGMENTED_LABEL_FILE = (
    OCR_DIRECTORY
    / "train_augmented.txt"
)

AUGMENTED_IMAGE_DIRECTORY = (
    OCR_DIRECTORY
    / "images"
    / "train_augmented"
)


ROTATIONS = {
    "r90": cv2.ROTATE_90_CLOCKWISE,
    "r180": cv2.ROTATE_180,
    "r270": cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def load_training_samples():
    """Load image paths and labels from train.txt."""

    samples = []

    with TRAIN_LABEL_FILE.open(
        "r",
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
                continue

            image_path, label = parts

            samples.append(
                (
                    image_path,
                    label.strip(),
                )
            )

    return samples


def main():
    """Create rotated OCR training dataset."""

    samples = load_training_samples()

    if not samples:
        print("No training samples found.")
        return

    #
    # Recreate augmented image directory.
    #

    if AUGMENTED_IMAGE_DIRECTORY.exists():
        shutil.rmtree(
            AUGMENTED_IMAGE_DIRECTORY
        )

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

        #
        # Save a copy of the original image.
        #

        original_name = (
            f"tag_{index:06d}.jpg"
        )

        original_output_path = (
            AUGMENTED_IMAGE_DIRECTORY
            / original_name
        )

        cv2.imwrite(
            str(original_output_path),
            image,
        )

        original_relative_path = (
            Path("images")
            / "train_augmented"
            / original_name
        )

        augmented_lines.append(
            f"{original_relative_path.as_posix()}\t{label}"
        )

        original_count += 1

        #
        # Rotated copies.
        #

        for suffix, rotation_code in ROTATIONS.items():

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

            cv2.imwrite(
                str(rotated_output_path),
                rotated,
            )

            rotated_relative_path = (
                Path("images")
                / "train_augmented"
                / rotated_name
            )

            augmented_lines.append(
                f"{rotated_relative_path.as_posix()}\t{label}"
            )

            rotated_count += 1

    #
    # Write augmented label file.
    #

    with AUGMENTED_LABEL_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        for line in augmented_lines:

            file.write(
                line + "\n"
            )

    print()
    print("=" * 55)
    print("OCR Training Augmentation")
    print("=" * 55)

    print(
        f"Original samples : {original_count}"
    )

    print(
        f"Rotated samples  : {rotated_count}"
    )

    print(
        f"Total samples    : "
        f"{original_count + rotated_count}"
    )

    print(
        f"Failed images    : {failed_count}"
    )

    print()

    print(
        f"Augmented images:\n"
        f"{AUGMENTED_IMAGE_DIRECTORY}"
    )

    print()

    print(
        f"Training labels:\n"
        f"{AUGMENTED_LABEL_FILE}"
    )


if __name__ == "__main__":
    main()