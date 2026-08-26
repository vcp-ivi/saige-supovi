"""
dataset_statistics.py

Print summary statistics for the raw wing-tag annotation dataset.

Usage:
    python tools/dataset_statistics.py
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import config

def natural_sort_key(text: str):
    """
    Sort strings naturally.

    Example:
        1, 2, 10, 11, AA, AB
    """

    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    ]


def load_annotations():
    """
    Load all annotation rows.
    """

    with open(
        config.ANNOTATION_FILE,
        newline="",
        encoding="utf-8",
    ) as csv_file:

        reader = csv.DictReader(csv_file)

        return list(reader)


def print_header(title: str):
    print()
    print("-" * 50)
    print(title)
    print("-" * 50)


def print_fallback_colors(rows):
    """
    Print the most common text color for each plate color.

    Rows with missing or 'none' colors are ignored.
    """

    plate_to_text_colors = defaultdict(Counter)

    for row in rows:

        plate_color = row[config.CSV_PLATE_COLOR].strip()
        text_color = row[config.CSV_TEXT_COLOR].strip()

        if not plate_color or not text_color:
            continue

        if plate_color == "none" or text_color == "none":
            continue

        plate_to_text_colors[plate_color][text_color] += 1

    fallback_colors = {}

    print_header("Fallback color statistics")

    for plate_color in sorted(plate_to_text_colors):

        text_counts = plate_to_text_colors[plate_color]

        fallback_text_color, fallback_count = text_counts.most_common(1)[0]

        total_count = sum(text_counts.values())

        percentage = (
            fallback_count / total_count * 100
        )

        fallback_colors[plate_color] = fallback_text_color

        print(
            f"{plate_color:<10} -> "
            f"{fallback_text_color:<10} "
            f"{fallback_count}/{total_count} "
            f"({percentage:.1f}%)"
        )

    print_header("Fallback colors")

    print("fallback_colors = {")

    for plate_color in sorted(fallback_colors):
        print(
            f'    "{plate_color}": '
            f'"{fallback_colors[plate_color]}",'
        )

    print("}")


def main():

    rows = load_annotations()

    total_annotations = len(rows)

    color_counter = Counter()
    text_counter = Counter()

    color_text_counter = defaultdict(Counter)

    for row in rows:

        plate_color = row[config.CSV_PLATE_COLOR].strip()
        text_color = row[config.CSV_TEXT_COLOR].strip()
        plate_text = row[config.CSV_PLATE_TEXT].strip()

        color_key = (plate_color, text_color)

        color_counter[color_key] += 1
        text_counter[plate_text] += 1
        color_text_counter[color_key][plate_text] += 1

    print("=" * 50)
    print("Annotation Analysis")
    print("=" * 50)

    print()
    print(f"Total annotated plates: {total_annotations}")

    #
    # Color combinations
    #

    print_header("Plate/Text color combinations")

    longest_color_name = max(
        len(f"{p} / {t}")
        for p, t in color_counter
    )

    for (plate_color, text_color), count in sorted(
        color_counter.items(),
        key=lambda x: (-x[1], x[0]),
    ):

        label = f"{plate_color} / {text_color}"

        print(
            f"{label:<{longest_color_name}} : {count}"
        )

    #
    # Unique texts
    #

    print_header("Unique tag texts")

    print(f"Unique texts: {len(text_counter)}")
    print()

    longest_text = max(
        len(text)
        for text in text_counter
    )

    for text in sorted(
        text_counter,
        key=natural_sort_key,
    ):

        print(
            f"{text:<{longest_text}} : {text_counter[text]}"
        )

    #
    # Colors + text
    #

    print_header("Plate/Text color + text")

    for (plate_color, text_color), total_count in sorted(
        color_counter.items(),
        key=lambda x: (-x[1], x[0]),
    ):

        label = f"{plate_color} / {text_color}"

        print()
        print(f"{label} : {total_count}")
        print("-" * (len(label) + len(str(total_count)) + 3))

        text_counts = color_text_counter[(plate_color, text_color)]

        longest_text = max(
            len(text)
            for text in text_counts
        )

        for text in sorted(
            text_counts,
            key=natural_sort_key,
        ):

            print(
                f"{text:<{longest_text}} : {text_counts[text]}"
            )

    print_fallback_colors(rows)


if __name__ == "__main__":
    main()