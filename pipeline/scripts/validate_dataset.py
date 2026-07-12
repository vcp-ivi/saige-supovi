"""Validate bird-tag annotations and render bounding-box previews."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError:
    print(
        "Pillow is required. Install dependencies with: "
        "python -m pip install -r requirements.txt",
        file=sys.stderr,
    )
    raise SystemExit(2)


BOX_FIELDS = ("x1", "y1", "x2", "y2")


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
            "Validate CSV annotations. Every complete bounding box is treated "
            "as the single class 'tag'; text and color columns are ignored."
        )
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=project_root / "data" / "annotations.csv",
        help="Path to annotations.csv.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=project_root / "data" / "images",
        help="Directory containing source images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for previews and validation_report.json.",
    )
    parser.add_argument(
        "--yolo-dir",
        type=Path,
        default=None,
        help=(
            "Validate an exported YOLO dataset instead of the source CSV. "
            "Expected layout: images/{train,val,test} and "
            "labels/{train,val,test}."
        ),
    )
    parser.add_argument(
        "--max-previews",
        type=int,
        default=50,
        help="Maximum positive images to render. Use 0 to disable previews.",
    )
    return parser.parse_args()


def add_issue(
    issues: list[dict[str, object]],
    *,
    severity: str,
    message: str,
    row: int | None = None,
    image_file: str | None = None,
) -> None:
    issue: dict[str, object] = {"severity": severity, "message": message}
    if row is not None:
        issue["row"] = row
    if image_file is not None:
        issue["image_file"] = image_file
    issues.append(issue)


def parse_box(
    row: dict[str, str],
    row_number: int,
    issues: list[dict[str, object]],
) -> Box | None:
    values = [(row.get(field) or "").strip() for field in BOX_FIELDS]
    image_file = (row.get("image_file") or "").strip()

    if not any(values):
        return None
    if not all(values):
        add_issue(
            issues,
            severity="error",
            message="Bounding box is only partially populated.",
            row=row_number,
            image_file=image_file,
        )
        return None

    try:
        x1, y1, x2, y2 = (float(value) for value in values)
    except ValueError:
        add_issue(
            issues,
            severity="error",
            message="Bounding-box coordinates must be numeric.",
            row=row_number,
            image_file=image_file,
        )
        return None

    if x1 < 0 or y1 < 0:
        add_issue(
            issues,
            severity="error",
            message="Bounding-box coordinates cannot be negative.",
            row=row_number,
            image_file=image_file,
        )
        return None
    if x2 <= x1 or y2 <= y1:
        add_issue(
            issues,
            severity="error",
            message="Bounding box must satisfy x2 > x1 and y2 > y1.",
            row=row_number,
            image_file=image_file,
        )
        return None

    return Box(x1=x1, y1=y1, x2=x2, y2=y2)


def render_preview(
    image_path: Path,
    boxes: list[tuple[int, Box]],
    output_path: Path,
) -> None:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    line_width = max(2, round(min(image.size) / 300))

    for _, box in boxes:
        draw.rectangle(
            (box.x1, box.y1, box.x2, box.y2),
            outline=(255, 40, 40),
            width=line_width,
        )
        label_y = max(0, box.y1 - 14)
        draw.text((box.x1 + 2, label_y), "tag", fill=(255, 40, 40))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=90)


def validate_yolo_dataset(args: argparse.Namespace, output_dir: Path) -> int:
    yolo_dir = args.yolo_dir.resolve()
    issues: list[dict[str, object]] = []
    split_summaries: dict[str, dict[str, int]] = {}
    seen_image_names: dict[str, str] = {}
    previews_written = 0
    total_boxes = 0
    total_positive = 0
    total_negative = 0
    total_images = 0

    if not yolo_dir.is_dir():
        print(f"YOLO dataset directory not found: {yolo_dir}", file=sys.stderr)
        return 2
    if not (yolo_dir / "data.yaml").is_file():
        add_issue(
            issues,
            severity="error",
            message="YOLO dataset is missing data.yaml.",
        )

    for split_name in ("train", "val", "test"):
        images_dir = yolo_dir / "images" / split_name
        labels_dir = yolo_dir / "labels" / split_name
        if not images_dir.is_dir():
            add_issue(
                issues,
                severity="error",
                message=f"Missing images/{split_name} directory.",
            )
            continue
        if not labels_dir.is_dir():
            add_issue(
                issues,
                severity="error",
                message=f"Missing labels/{split_name} directory.",
            )
            continue

        image_paths = {
            path.stem: path
            for path in images_dir.iterdir()
            if path.is_file()
            and path.suffix.lower()
            in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        }
        label_paths = {
            path.stem: path
            for path in labels_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".txt"
        }

        for stem in sorted(image_paths.keys() - label_paths.keys()):
            add_issue(
                issues,
                severity="error",
                message=f"Image has no label file in split '{split_name}'.",
                image_file=image_paths[stem].name,
            )
        for stem in sorted(label_paths.keys() - image_paths.keys()):
            add_issue(
                issues,
                severity="error",
                message=f"Label has no image in split '{split_name}'.",
                image_file=label_paths[stem].name,
            )

        split_boxes = 0
        split_positive = 0
        split_negative = 0
        for stem, image_path in sorted(image_paths.items()):
            previous_split = seen_image_names.get(image_path.name)
            if previous_split is not None:
                add_issue(
                    issues,
                    severity="error",
                    message=(
                        f"Image occurs in both '{previous_split}' and "
                        f"'{split_name}' splits."
                    ),
                    image_file=image_path.name,
                )
            else:
                seen_image_names[image_path.name] = split_name

            try:
                with Image.open(image_path) as image:
                    image.verify()
                with Image.open(image_path) as image:
                    image_width, image_height = image.size
            except (OSError, ValueError) as exc:
                add_issue(
                    issues,
                    severity="error",
                    message=f"Image cannot be opened: {exc}",
                    image_file=image_path.name,
                )
                continue

            pixel_boxes: list[tuple[int, Box]] = []
            label_path = label_paths.get(stem)
            if label_path is not None:
                for line_number, raw_line in enumerate(
                    label_path.read_text(encoding="utf-8").splitlines(),
                    start=1,
                ):
                    line = raw_line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        add_issue(
                            issues,
                            severity="error",
                            message=(
                                f"YOLO label line {line_number} must contain "
                                "exactly 5 values."
                            ),
                            image_file=label_path.name,
                        )
                        continue
                    try:
                        class_id = int(parts[0])
                        center_x, center_y, width, height = (
                            float(value) for value in parts[1:]
                        )
                    except ValueError:
                        add_issue(
                            issues,
                            severity="error",
                            message=f"YOLO label line {line_number} is not numeric.",
                            image_file=label_path.name,
                        )
                        continue
                    if class_id != 0:
                        add_issue(
                            issues,
                            severity="error",
                            message=(
                                f"YOLO label line {line_number} uses class "
                                f"{class_id}; only class 0 ('tag') is valid."
                            ),
                            image_file=label_path.name,
                        )
                        continue
                    if not (
                        0 <= center_x <= 1
                        and 0 <= center_y <= 1
                        and 0 < width <= 1
                        and 0 < height <= 1
                    ):
                        add_issue(
                            issues,
                            severity="error",
                            message=(
                                f"YOLO label line {line_number} has coordinates "
                                "outside normalized bounds."
                            ),
                            image_file=label_path.name,
                        )
                        continue

                    x1 = (center_x - width / 2) * image_width
                    y1 = (center_y - height / 2) * image_height
                    x2 = (center_x + width / 2) * image_width
                    y2 = (center_y + height / 2) * image_height
                    tolerance = 1e-4
                    if (
                        x1 < -tolerance
                        or y1 < -tolerance
                        or x2 > image_width + tolerance
                        or y2 > image_height + tolerance
                    ):
                        add_issue(
                            issues,
                            severity="error",
                            message=(
                                f"YOLO label line {line_number} converts to a "
                                "box outside image bounds."
                            ),
                            image_file=label_path.name,
                        )
                        continue
                    pixel_boxes.append(
                        (
                            line_number,
                            Box(
                                max(0, x1),
                                max(0, y1),
                                min(image_width, x2),
                                min(image_height, y2),
                            ),
                        )
                    )

            if pixel_boxes:
                split_positive += 1
                split_boxes += len(pixel_boxes)
                if previews_written < args.max_previews:
                    render_preview(
                        image_path,
                        pixel_boxes,
                        output_dir / "previews" / split_name / image_path.name,
                    )
                    previews_written += 1
            else:
                split_negative += 1

        split_summaries[split_name] = {
            "images": len(image_paths),
            "positive_images": split_positive,
            "negative_images": split_negative,
            "bounding_boxes": split_boxes,
        }
        total_images += len(image_paths)
        total_positive += split_positive
        total_negative += split_negative
        total_boxes += split_boxes

    severity_counts = Counter(issue["severity"] for issue in issues)
    report = {
        "format": "yolo",
        "class_name": "tag",
        "dataset_directory": str(yolo_dir),
        "summary": {
            "image_files": total_images,
            "positive_images": total_positive,
            "negative_images": total_negative,
            "bounding_boxes": total_boxes,
            "previews_written": previews_written,
            "errors": severity_counts["error"],
            "warnings": severity_counts["warning"],
        },
        "splits": split_summaries,
        "issues": issues,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary = report["summary"]
    print("YOLO dataset validation complete")
    for split_name, details in split_summaries.items():
        print(
            f"  {split_name:5} "
            f"images={details['images']:3} "
            f"positive={details['positive_images']:2} "
            f"negative={details['negative_images']:2} "
            f"boxes={details['bounding_boxes']:2}"
        )
    print(f"  Total images: {summary['image_files']}")
    print(f"  Tag boxes:    {summary['bounding_boxes']}")
    print(f"  Previews:     {summary['previews_written']}")
    print(f"  Errors:       {summary['errors']}")
    print(f"  Warnings:     {summary['warnings']}")
    print(f"  Report:       {report_path}")
    return 1 if severity_counts["error"] else 0


def validate_csv_dataset(args: argparse.Namespace, output_dir: Path) -> int:
    annotations_path = args.annotations.resolve()
    images_dir = args.images_dir.resolve()
    issues: list[dict[str, object]] = []
    boxes_by_image: dict[str, list[tuple[int, Box]]] = defaultdict(list)
    csv_image_names: set[str] = set()
    row_count = 0
    negative_rows = 0

    if args.max_previews < 0:
        print("--max-previews cannot be negative.", file=sys.stderr)
        return 2
    if not annotations_path.is_file():
        print(f"Annotations file not found: {annotations_path}", file=sys.stderr)
        return 2
    if not images_dir.is_dir():
        print(f"Images directory not found: {images_dir}", file=sys.stderr)
        return 2

    with annotations_path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        missing_columns = {"image_file", *BOX_FIELDS} - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            print(f"Missing required CSV columns: {missing}", file=sys.stderr)
            return 2

        for row_number, row in enumerate(reader, start=2):
            row_count += 1
            image_file = (row.get("image_file") or "").strip()
            if not image_file:
                add_issue(
                    issues,
                    severity="error",
                    message="image_file is empty.",
                    row=row_number,
                )
                continue

            csv_image_names.add(image_file)
            box = parse_box(row, row_number, issues)
            if box is None:
                if not any((row.get(field) or "").strip() for field in BOX_FIELDS):
                    negative_rows += 1
                continue
            boxes_by_image[image_file].append((row_number, box))

    image_paths = {
        path.name: path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    }

    for image_file in sorted(csv_image_names - image_paths.keys()):
        add_issue(
            issues,
            severity="error",
            message="Image referenced by CSV does not exist.",
            image_file=image_file,
        )
    for image_file in sorted(image_paths.keys() - csv_image_names):
        add_issue(
            issues,
            severity="warning",
            message="Image exists but has no CSV row.",
            image_file=image_file,
        )

    image_sizes: dict[str, tuple[int, int]] = {}
    for image_file, image_path in sorted(image_paths.items()):
        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                image_sizes[image_file] = image.size
        except (OSError, ValueError) as exc:
            add_issue(
                issues,
                severity="error",
                message=f"Image cannot be opened: {exc}",
                image_file=image_file,
            )

    for image_file, boxes in sorted(boxes_by_image.items()):
        size = image_sizes.get(image_file)
        if size is None:
            continue
        width, height = size
        for row_number, box in boxes:
            if box.x2 > width or box.y2 > height:
                add_issue(
                    issues,
                    severity="error",
                    message=(
                        f"Bounding box exceeds image bounds "
                        f"({width}x{height})."
                    ),
                    row=row_number,
                    image_file=image_file,
                )

    previews_written = 0
    if args.max_previews:
        previews_dir = output_dir / "previews"
        for image_file, boxes in sorted(boxes_by_image.items()):
            if previews_written >= args.max_previews:
                break
            image_path = image_paths.get(image_file)
            if image_path is None or image_file not in image_sizes:
                continue
            render_preview(image_path, boxes, previews_dir / image_file)
            previews_written += 1

    severity_counts = Counter(issue["severity"] for issue in issues)
    report = {
        "class_name": "tag",
        "annotations_file": str(annotations_path),
        "images_directory": str(images_dir),
        "summary": {
            "csv_rows": row_count,
            "image_files": len(image_paths),
            "positive_images": len(boxes_by_image),
            "negative_rows": negative_rows,
            "bounding_boxes": sum(len(boxes) for boxes in boxes_by_image.values()),
            "previews_written": previews_written,
            "errors": severity_counts["error"],
            "warnings": severity_counts["warning"],
        },
        "issues": issues,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary = report["summary"]
    print("Dataset validation complete")
    print(f"  Images:          {summary['image_files']}")
    print(f"  Positive images: {summary['positive_images']}")
    print(f"  Negative rows:   {summary['negative_rows']}")
    print(f"  Tag boxes:       {summary['bounding_boxes']}")
    print(f"  Previews:        {summary['previews_written']}")
    print(f"  Errors:          {summary['errors']}")
    print(f"  Warnings:        {summary['warnings']}")
    print(f"  Report:          {report_path}")

    return 1 if severity_counts["error"] else 0


def main() -> int:
    args = parse_args()
    if args.max_previews < 0:
        print("--max-previews cannot be negative.", file=sys.stderr)
        return 2

    project_root = Path(__file__).resolve().parents[1]
    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
    elif args.yolo_dir is not None:
        output_dir = project_root / "outputs" / "yolo_validation"
    else:
        output_dir = project_root / "outputs" / "dataset_validation"

    if args.yolo_dir is not None:
        return validate_yolo_dataset(args, output_dir)
    return validate_csv_dataset(args, output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
