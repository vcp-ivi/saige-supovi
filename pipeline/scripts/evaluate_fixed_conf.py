"""Evaluate YOLO predictions at fixed confidence thresholds."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / "outputs" / "ultralytics_config"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / "matplotlib_config"))

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, default=Path("data/yolo/images/test"))
    parser.add_argument("--labels-dir", type=Path, default=Path("data/yolo/labels/test"))
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default=0)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--conf", type=float, nargs="+", default=[0.001, 0.01, 0.05, 0.1, 0.25])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/fixed_conf_test_metrics.csv"),
        help="CSV file to write threshold-based metrics.",
    )
    return parser.parse_args()


def yolo_box_to_xyxy(line: str, image_width: int, image_height: int) -> tuple[float, float, float, float]:
    _, cx, cy, width, height = line.split()
    cx = float(cx) * image_width
    cy = float(cy) * image_height
    width = float(width) * image_width
    height = float(height) * image_height
    return (
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
    )


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if intersection == 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return intersection / (area_a + area_b - intersection)


def load_labels(label_path: Path, image_path: Path) -> list[tuple[float, float, float, float]]:
    with Image.open(image_path) as image:
        image_width, image_height = image.size
    if not label_path.exists():
        return []
    return [
        yolo_box_to_xyxy(line, image_width, image_height)
        for line in label_path.read_text(encoding="ascii").splitlines()
        if line.strip()
    ]


def match_counts(
    predictions: list[tuple[float, tuple[float, float, float, float]]],
    targets: list[tuple[float, float, float, float]],
    threshold: float,
    iou_threshold: float,
) -> tuple[int, int, int]:
    filtered = [(conf, box) for conf, box in predictions if conf >= threshold]
    filtered.sort(key=lambda item: item[0], reverse=True)
    matched_targets: set[int] = set()
    true_positive = 0
    false_positive = 0

    for _, pred_box in filtered:
        best_iou = 0.0
        best_index = None
        for index, target_box in enumerate(targets):
            if index in matched_targets:
                continue
            iou = box_iou(pred_box, target_box)
            if iou > best_iou:
                best_iou = iou
                best_index = index
        if best_index is not None and best_iou >= iou_threshold:
            matched_targets.add(best_index)
            true_positive += 1
        else:
            false_positive += 1

    false_negative = len(targets) - true_positive
    return true_positive, false_positive, false_negative


def main() -> int:
    args = parse_args()
    image_paths = sorted(
        path for path in args.images_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    )
    labels_by_image = {
        image_path.name: load_labels(args.labels_dir / f"{image_path.stem}.txt", image_path)
        for image_path in image_paths
    }

    model = YOLO(str(args.model))
    results = model.predict(
        [str(path) for path in image_paths],
        imgsz=args.imgsz,
        conf=min(args.conf),
        device=args.device,
        verbose=False,
    )

    predictions_by_image: dict[str, list[tuple[float, tuple[float, float, float, float]]]] = {}
    for image_path, result in zip(image_paths, results):
        predictions = []
        if result.boxes is not None:
            for xyxy, conf in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist()):
                predictions.append((float(conf), tuple(float(value) for value in xyxy)))
        predictions_by_image[image_path.name] = predictions

    total_targets = sum(len(labels) for labels in labels_by_image.values())
    print(f"images={len(image_paths)} targets={total_targets} iou={args.iou}")
    rows = []
    for threshold in args.conf:
        tp = fp = fn = 0
        for image_name, targets in labels_by_image.items():
            image_tp, image_fp, image_fn = match_counts(
                predictions_by_image[image_name],
                targets,
                threshold,
                args.iou,
            )
            tp += image_tp
            fp += image_fp
            fn += image_fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        rows.append(
            {
                "confidence": threshold,
                "iou": args.iou,
                "images": len(image_paths),
                "targets": total_targets,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": precision,
                "recall": recall,
            }
        )
        print(
            f"conf={threshold:g} tp={tp} fp={fp} fn={fn} "
            f"precision={precision:.3f} recall={recall:.3f}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
