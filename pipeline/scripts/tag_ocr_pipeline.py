r"""Detect tag plates, crop them, run OCR, and estimate plate/text colors.

Example:
    .\.venv311\Scripts\python.exe scripts\tag_ocr_pipeline.py ^
        --source data\yolo\images\test ^
        --output outputs\tag_ocr_pipeline ^
        --conf 0.05
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / "outputs" / "ultralytics_config"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / "matplotlib_config"))

DEFAULT_MODEL = (
    PROJECT_ROOT
    / "outputs"
    / "grid_search_big"
    / "yolo26s_i1024_m0p5_s0p05_t0p02_e0p0_hs0p3_hv0p4_f0p0"
    / "weights"
    / "best.pt"
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

COLOR_PALETTE = {
    "black": (20, 20, 20),
    "white": (235, 235, 235),
    "gray": (128, 128, 128),
    "red": (200, 40, 40),
    "orange": (230, 120, 35),
    "yellow": (220, 200, 45),
    "green": (50, 155, 75),
    "blue": (45, 95, 200),
    "purple": (130, 65, 170),
    "pink": (220, 95, 160),
    "brown": (120, 75, 45),
}


@dataclass
class ColorEstimate:
    name: str
    hex: str
    rgb: tuple[int, int, int]
    coverage: float


@dataclass
class TagResult:
    image: str
    detection_id: int
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    crop_path: str
    ocr_text: str
    ocr_confidence: float
    plate_color: str
    plate_hex: str
    plate_rgb: str
    plate_coverage: float
    text_color: str
    text_hex: str
    text_rgb: str
    text_coverage: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO tag detection -> crop -> OCR -> OpenCV plate/text color inference."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--source", type=Path, default=Path("data/yolo/images/test"))
    parser.add_argument("--output", type=Path, default=Path("outputs/tag_ocr_pipeline"))
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crop-padding", type=float, default=0.08)
    parser.add_argument("--ocr-languages", default="en")
    parser.add_argument("--ocr-gpu", action="store_true", help="Use GPU for EasyOCR.")
    parser.add_argument("--no-ocr", action="store_true", help="Skip OCR and only save crops/colors.")
    parser.add_argument("--save-debug-masks", action="store_true")
    return parser.parse_args()


def image_paths(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(path for path in source.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def clean_text(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9 -]+", "", text).strip()


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def bgr_to_rgb_tuple(bgr: Iterable[float]) -> tuple[int, int, int]:
    b, g, r = [int(round(float(value))) for value in bgr]
    return (r, g, b)


def color_name(rgb: tuple[int, int, int]) -> str:
    sample = np.uint8([[rgb]])
    sample_lab = cv2.cvtColor(sample, cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
    best_name = "unknown"
    best_distance = float("inf")
    for name, palette_rgb in COLOR_PALETTE.items():
        palette_lab = cv2.cvtColor(np.uint8([[palette_rgb]]), cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
        distance = float(np.linalg.norm(sample_lab - palette_lab))
        if distance < best_distance:
            best_name = name
            best_distance = distance
    return best_name


def estimate_dominant_plate_color(crop: np.ndarray) -> tuple[ColorEstimate, np.ndarray]:
    h, w = crop.shape[:2]
    margin_x = max(1, int(w * 0.04))
    margin_y = max(1, int(h * 0.04))
    inner = crop[margin_y : h - margin_y or h, margin_x : w - margin_x or w]
    pixels = inner.reshape((-1, 3)).astype(np.float32)
    if len(pixels) < 4:
        rgb = bgr_to_rgb_tuple(np.mean(pixels, axis=0))
        return ColorEstimate(color_name(rgb), rgb_to_hex(rgb), rgb, 1.0), np.full((h, w), 255, np.uint8)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    k = min(4, len(pixels))
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k)
    dominant_index = int(np.argmax(counts))
    dominant_bgr = centers[dominant_index]
    rgb = bgr_to_rgb_tuple(dominant_bgr)

    diff = np.linalg.norm(crop.astype(np.float32) - dominant_bgr.reshape(1, 1, 3), axis=2)
    plate_mask = (diff < max(35.0, float(np.percentile(diff, 55)))).astype(np.uint8) * 255
    coverage = float(np.count_nonzero(plate_mask) / plate_mask.size)
    return ColorEstimate(color_name(rgb), rgb_to_hex(rgb), rgb, coverage), plate_mask


def estimate_text_color(crop: np.ndarray, plate_rgb: tuple[int, int, int]) -> tuple[ColorEstimate, np.ndarray]:
    h, w = crop.shape[:2]
    plate_bgr = np.array([plate_rgb[2], plate_rgb[1], plate_rgb[0]], dtype=np.float32)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    plate_lab = cv2.cvtColor(np.uint8([[[plate_rgb[0], plate_rgb[1], plate_rgb[2]]]]), cv2.COLOR_RGB2LAB)[
        0, 0
    ].astype(np.float32)
    color_distance = np.linalg.norm(lab - plate_lab.reshape(1, 1, 3), axis=2)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    distance_threshold = max(28.0, float(np.percentile(color_distance, 82)))
    mask = ((color_distance >= distance_threshold) | (edges > 0)).astype(np.uint8) * 255

    border = max(1, int(min(h, w) * 0.04))
    mask[:border, :] = 0
    mask[-border:, :] = 0
    mask[:, :border] = 0
    mask[:, -border:] = 0
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    pixels = crop[mask > 0]
    if len(pixels) == 0:
        diff = np.linalg.norm(crop.astype(np.float32) - plate_bgr.reshape(1, 1, 3), axis=2)
        threshold = float(np.percentile(diff, 90))
        pixels = crop[diff >= threshold]
        mask = (diff >= threshold).astype(np.uint8) * 255

    if len(pixels) == 0:
        rgb = (0, 0, 0)
        return ColorEstimate("unknown", rgb_to_hex(rgb), rgb, 0.0), mask

    rgb = bgr_to_rgb_tuple(np.median(pixels, axis=0))
    coverage = float(np.count_nonzero(mask) / mask.size)
    return ColorEstimate(color_name(rgb), rgb_to_hex(rgb), rgb, coverage), mask


def padded_box(
    xyxy: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy
    width = x2 - x1
    height = y2 - y1
    pad_x = width * padding_ratio
    pad_y = height * padding_ratio
    return (
        max(0, int(round(x1 - pad_x))),
        max(0, int(round(y1 - pad_y))),
        min(image_width, int(round(x2 + pad_x))),
        min(image_height, int(round(y2 + pad_y))),
    )


def build_ocr_reader(languages: str, use_gpu: bool):
    import easyocr

    return easyocr.Reader([lang.strip() for lang in languages.split(",") if lang.strip()], gpu=use_gpu)


def run_ocr(reader, crop: np.ndarray) -> tuple[str, float]:
    if reader is None:
        return "", 0.0
    scale = 3 if max(crop.shape[:2]) < 220 else 2
    enlarged = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    enhanced = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    results = reader.readtext(
        enhanced,
        detail=1,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789- ",
        paragraph=False,
    )
    if not results:
        return "", 0.0
    pieces = []
    confidences = []
    for _, text, confidence in sorted(results, key=lambda item: item[0][0][0]):
        text = clean_text(text)
        if text:
            pieces.append(text)
            confidences.append(float(confidence))
    if not pieces:
        return "", 0.0
    return " ".join(pieces), float(np.mean(confidences))


def save_csv(path: Path, rows: list[TagResult]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> int:
    args = parse_args()
    if not args.model.is_file():
        print(f"Model not found: {args.model}")
        return 2
    if not args.source.exists():
        print(f"Source not found: {args.source}")
        return 2

    images = image_paths(args.source)
    args.output.mkdir(parents=True, exist_ok=True)
    crops_dir = args.output / "crops"
    annotated_dir = args.output / "annotated"
    debug_dir = args.output / "debug_masks"
    crops_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)
    if args.save_debug_masks:
        debug_dir.mkdir(parents=True, exist_ok=True)

    reader = None if args.no_ocr else build_ocr_reader(args.ocr_languages, args.ocr_gpu)
    model = YOLO(str(args.model))
    results: list[TagResult] = []

    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not read image: {image_path}")
            continue
        image_height, image_width = image.shape[:2]
        prediction = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )[0]
        annotated = image.copy()
        boxes = [] if prediction.boxes is None else prediction.boxes
        for detection_id, box in enumerate(boxes):
            xyxy = tuple(float(value) for value in box.xyxy[0].cpu().tolist())
            confidence = float(box.conf[0].cpu().item())
            x1, y1, x2, y2 = padded_box(xyxy, image_width, image_height, args.crop_padding)
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            stem = f"{image_path.stem}_tag{detection_id:02d}"
            crop_path = crops_dir / f"{stem}.jpg"
            cv2.imwrite(str(crop_path), crop)

            plate_estimate, plate_mask = estimate_dominant_plate_color(crop)
            text_estimate, text_mask = estimate_text_color(crop, plate_estimate.rgb)
            ocr_text, ocr_confidence = run_ocr(reader, crop)

            if args.save_debug_masks:
                cv2.imwrite(str(debug_dir / f"{stem}_plate_mask.png"), plate_mask)
                cv2.imwrite(str(debug_dir / f"{stem}_text_mask.png"), text_mask)

            label = f"{ocr_text or 'tag'} {confidence:.2f} plate:{plate_estimate.name} text:{text_estimate.name}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (30, 210, 70), 2)
            cv2.putText(
                annotated,
                label[:80],
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (30, 210, 70),
                2,
                cv2.LINE_AA,
            )

            results.append(
                TagResult(
                    image=image_path.name,
                    detection_id=detection_id,
                    confidence=confidence,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    crop_path=str(crop_path),
                    ocr_text=ocr_text,
                    ocr_confidence=ocr_confidence,
                    plate_color=plate_estimate.name,
                    plate_hex=plate_estimate.hex,
                    plate_rgb=str(plate_estimate.rgb),
                    plate_coverage=plate_estimate.coverage,
                    text_color=text_estimate.name,
                    text_hex=text_estimate.hex,
                    text_rgb=str(text_estimate.rgb),
                    text_coverage=text_estimate.coverage,
                )
            )

        cv2.imwrite(str(annotated_dir / image_path.name), annotated)

    save_csv(args.output / "tag_ocr_results.csv", results)
    (args.output / "tag_ocr_results.json").write_text(
        json.dumps([asdict(row) for row in results], indent=2),
        encoding="utf-8",
    )
    print(f"Processed images: {len(images)}")
    print(f"Detected tags: {len(results)}")
    print(f"Results written to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
