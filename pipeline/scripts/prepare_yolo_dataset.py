"""Create reproducible train/validation/test splits in YOLO format.
     Regenerate with:
    .\.venv\Scripts\python.exe .\scripts\prepare_yolo_dataset.py --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print(
        "Pillow is required. Install dependencies with: "
        "python -m pip install -r requirements.txt",
        file=sys.stderr,
    )
    raise SystemExit(2)


BOX_FIELDS = ("x1", "y1", "x2", "y2")
SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Split the bird-tag dataset at image level and convert bounding "
            "boxes to the single YOLO class 'tag'."
        )
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=project_root / "data" / "annotations.csv",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=project_root / "data" / "images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data" / "yolo",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory.",
    )
    return parser.parse_args()


def allocate_counts(total: int, ratios: tuple[float, float, float]) -> list[int]:
    """Allocate integer counts with the largest-remainder method."""
    exact = [total * ratio for ratio in ratios]
    counts = [int(value) for value in exact]
    remainder = total - sum(counts)
    order = sorted(
        range(len(ratios)),
        key=lambda index: (exact[index] - counts[index], -index),
        reverse=True,
    )
    for index in order[:remainder]:
        counts[index] += 1
    return counts


def split_group(
    image_names: list[str],
    ratios: tuple[float, float, float],
    rng: random.Random,
) -> dict[str, list[str]]:
    shuffled = sorted(image_names)
    rng.shuffle(shuffled)
    counts = allocate_counts(len(shuffled), ratios)

    result: dict[str, list[str]] = {}
    start = 0
    for split_name, count in zip(SPLIT_NAMES, counts):
        result[split_name] = shuffled[start : start + count]
        start += count
    return result


def read_annotations(
    annotations_path: Path,
) -> tuple[dict[str, list[Box]], set[str]]:
    boxes_by_image: dict[str, list[Box]] = defaultdict(list)
    csv_images: set[str] = set()

    with annotations_path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        missing_columns = {"image_file", *BOX_FIELDS} - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing required CSV columns: {missing}")

        for row_number, row in enumerate(reader, start=2):
            image_file = (row.get("image_file") or "").strip()
            if not image_file:
                raise ValueError(f"CSV row {row_number}: image_file is empty")
            csv_images.add(image_file)

            values = [(row.get(field) or "").strip() for field in BOX_FIELDS]
            if not any(values):
                continue
            if not all(values):
                raise ValueError(
                    f"CSV row {row_number}: bounding box is partially populated"
                )

            try:
                x1, y1, x2, y2 = (float(value) for value in values)
            except ValueError as exc:
                raise ValueError(
                    f"CSV row {row_number}: coordinates must be numeric"
                ) from exc

            if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
                raise ValueError(
                    f"CSV row {row_number}: invalid bounding box "
                    f"({x1}, {y1}, {x2}, {y2})"
                )
            boxes_by_image[image_file].append(Box(x1, y1, x2, y2))

    return boxes_by_image, csv_images


def to_yolo_line(box: Box, image_width: int, image_height: int) -> str:
    if box.x2 > image_width or box.y2 > image_height:
        raise ValueError(
            f"box ({box.x1}, {box.y1}, {box.x2}, {box.y2}) exceeds "
            f"image size {image_width}x{image_height}"
        )

    center_x = ((box.x1 + box.x2) / 2) / image_width
    center_y = ((box.y1 + box.y2) / 2) / image_height
    width = (box.x2 - box.x1) / image_width
    height = (box.y2 - box.y1) / image_height
    return f"0 {center_x:.8f} {center_y:.8f} {width:.8f} {height:.8f}"


def write_data_yaml(output_dir: Path) -> None:
    content = (
        f"path: {output_dir.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "\n"
        "names:\n"
        "  0: tag\n"
    )
    (output_dir / "data.yaml").write_text(content, encoding="utf-8")


def main() -> int:
    args = parse_args()
    annotations_path = args.annotations.resolve()
    images_dir = args.images_dir.resolve()
    output_dir = args.output_dir.resolve()
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)

    if any(ratio <= 0 for ratio in ratios):
        print("All split ratios must be greater than zero.", file=sys.stderr)
        return 2
    if abs(sum(ratios) - 1.0) > 1e-9:
        print("Split ratios must sum to 1.0.", file=sys.stderr)
        return 2
    if not annotations_path.is_file():
        print(f"Annotations file not found: {annotations_path}", file=sys.stderr)
        return 2
    if not images_dir.is_dir():
        print(f"Images directory not found: {images_dir}", file=sys.stderr)
        return 2
    if output_dir.exists():
        if not args.overwrite:
            print(
                f"Output directory already exists: {output_dir}\n"
                "Use --overwrite to replace it.",
                file=sys.stderr,
            )
            return 2
        shutil.rmtree(output_dir)

    try:
        boxes_by_image, csv_images = read_annotations(annotations_path)
    except ValueError as exc:
        print(f"Annotation error: {exc}", file=sys.stderr)
        return 1

    image_paths = {
        path.name: path
        for path in images_dir.iterdir()
        if path.is_file()
        and path.suffix.lower()
        in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    }
    missing_images = sorted(csv_images - image_paths.keys())
    unannotated_images = sorted(image_paths.keys() - csv_images)
    if missing_images:
        print(
            "CSV references missing images:\n  " + "\n  ".join(missing_images),
            file=sys.stderr,
        )
        return 1
    if unannotated_images:
        print(
            "Images without CSV rows:\n  " + "\n  ".join(unannotated_images),
            file=sys.stderr,
        )
        return 1

    positive_images = sorted(boxes_by_image)
    negative_images = sorted(csv_images - boxes_by_image.keys())
    rng = random.Random(args.seed)
    positive_splits = split_group(positive_images, ratios, rng)
    negative_splits = split_group(negative_images, ratios, rng)
    splits = {
        split_name: sorted(
            positive_splits[split_name] + negative_splits[split_name]
        )
        for split_name in SPLIT_NAMES
    }

    split_sets = [set(splits[name]) for name in SPLIT_NAMES]
    if any(split_sets[i] & split_sets[j] for i in range(3) for j in range(i + 1, 3)):
        print("Internal error: split overlap detected.", file=sys.stderr)
        return 1
    if set().union(*split_sets) != csv_images:
        print("Internal error: split does not cover every image.", file=sys.stderr)
        return 1
    if any(not positive_splits[name] for name in SPLIT_NAMES):
        print(
            "Each split must contain at least one positive image.",
            file=sys.stderr,
        )
        return 1

    manifest: dict[str, object] = {
        "seed": args.seed,
        "ratios": dict(zip(SPLIT_NAMES, ratios)),
        "class_names": ["tag"],
        "splits": {},
    }

    try:
        for split_name in SPLIT_NAMES:
            images_output = output_dir / "images" / split_name
            labels_output = output_dir / "labels" / split_name
            images_output.mkdir(parents=True, exist_ok=True)
            labels_output.mkdir(parents=True, exist_ok=True)

            box_count = 0
            for image_name in splits[split_name]:
                source_path = image_paths[image_name]
                destination_path = images_output / image_name
                shutil.copy2(source_path, destination_path)

                with Image.open(source_path) as image:
                    image_width, image_height = image.size

                boxes = boxes_by_image.get(image_name, [])
                lines = [
                    to_yolo_line(box, image_width, image_height)
                    for box in boxes
                ]
                box_count += len(lines)
                label_path = labels_output / f"{source_path.stem}.txt"
                label_path.write_text(
                    "\n".join(lines) + ("\n" if lines else ""),
                    encoding="ascii",
                )

            manifest["splits"][split_name] = {
                "images": len(splits[split_name]),
                "positive_images": len(positive_splits[split_name]),
                "negative_images": len(negative_splits[split_name]),
                "boxes": box_count,
                "image_files": splits[split_name],
            }
    except (OSError, ValueError) as exc:
        print(f"Dataset conversion failed: {exc}", file=sys.stderr)
        return 1

    write_data_yaml(output_dir)
    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"YOLO dataset created: {output_dir}")
    for split_name in SPLIT_NAMES:
        details = manifest["splits"][split_name]
        print(
            f"  {split_name:5} "
            f"images={details['images']:3} "
            f"positive={details['positive_images']:2} "
            f"negative={details['negative_images']:2} "
            f"boxes={details['boxes']:2}"
        )
    print(f"  Config:   {output_dir / 'data.yaml'}")
    print(f"  Manifest: {output_dir / 'split_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
