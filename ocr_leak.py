"""
check_ocr_leak.py

Check whether OCR train/val/test splits contain crops originating
from the same source image.
"""

import csv
from collections import defaultdict
from pathlib import Path

import config


OCR_DIRECTORY = (
    config.PROJECT_ROOT
    / "data"
    / "ocr"
)

METADATA_FILE = (
    OCR_DIRECTORY
    / "metadata.csv"
)


def load_metadata():
    rows = []

    with METADATA_FILE.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        rows.extend(reader)

    return rows


def main():
    rows = load_metadata()

    source_splits = defaultdict(set)

    for row in rows:
        source_image = (
            row["source_image"]
            .strip()
            .lower()
        )

        split = (
            row["split"]
            .strip()
            .lower()
        )

        if source_image and split:
            source_splits[source_image].add(split)

    leaks = {
        source_image: splits
        for source_image, splits in source_splits.items()
        if len(splits) > 1
    }

    print("=" * 60)
    print("OCR SPLIT LEAK CHECK")
    print("=" * 60)

    print(
        f"Unique source images: "
        f"{len(source_splits)}"
    )

    print(
        f"Source images appearing in multiple splits: "
        f"{len(leaks)}"
    )

    if leaks:
        print()
        print("LEAKS FOUND:")
        print("-" * 60)

        for source_image, splits in sorted(leaks.items()):
            print(
                f"{source_image}: "
                f"{', '.join(sorted(splits))}"
            )

    else:
        print()
        print("No source-image leakage detected.")


if __name__ == "__main__":
    main()