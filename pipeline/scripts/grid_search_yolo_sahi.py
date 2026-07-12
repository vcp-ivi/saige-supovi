"""Run YOLO training grid search and compare normal vs SAHI inference.

Example:
    .\.venv311\Scripts\python.exe .\scripts\grid_search_yolo_sahi.py

The default grid is intentionally small:
    models: yolo26n.pt, yolo26s.pt
    mosaic: 0.0, 0.5

Each run is saved under outputs/grid_search/<parameter-name>/.
"""

from __future__ import annotations

import argparse
import csv
import gc
import itertools
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / "outputs" / "ultralytics_config"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / "matplotlib_config"))
os.environ.setdefault("WANDB_MODE", "disabled")

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, predict as sahi_predict
from ultralytics import YOLO

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Experiment:
    model: str
    imgsz: int
    mosaic: float
    scale: float
    translate: float
    erasing: float
    hsv_s: float
    hsv_v: float
    fliplr: float


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO grid-search runs and evaluate normal YOLO and SAHI inference."
    )
    parser.add_argument("--data", type=Path, default=Path("data/yolo/data.yaml"))
    parser.add_argument("--test-images", type=Path, default=Path("data/yolo/images/test"))
    parser.add_argument("--test-labels", type=Path, default=Path("data/yolo/labels/test"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/grid_search"))
    parser.add_argument("--models", default="yolo26n.pt,yolo26s.pt")
    parser.add_argument("--imgsz", default="1024")
    parser.add_argument("--mosaic", default="0.0,0.5")
    parser.add_argument("--scale", default="0.0")
    parser.add_argument("--translate", default="0.0")
    parser.add_argument("--erasing", default="0.4")
    parser.add_argument("--hsv-s", default="0.3")
    parser.add_argument("--hsv-v", default="0.4")
    parser.add_argument("--fliplr", default="0.5")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--batch",
        default="1",
        help="Training batch size. Use -1 for AutoBatch, but fixed 1 is safer on 4 GB GPUs.",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--conf", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.1, 0.25])
    parser.add_argument("--visual-conf", type=float, default=0.02)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--sahi-slice-size", type=int, default=640)
    parser.add_argument("--sahi-overlap", type=float, default=0.2)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing experiment folders before rerunning.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip experiment folders that already contain weights/best.pt.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N generated experiments.",
    )
    parser.add_argument(
        "--no-visuals",
        action="store_true",
        help="Skip saving YOLO and SAHI prediction visualizations.",
    )
    return parser.parse_args()


def normalize_batch(value: str) -> int | float:
    if "." in value:
        return float(value)
    return int(value)


def safe_name_part(value: object) -> str:
    text = str(value).replace(".pt", "")
    text = text.replace(".", "p").replace("-", "m")
    return re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")


def experiment_name(experiment: Experiment) -> str:
    return (
        f"{safe_name_part(experiment.model)}"
        f"_i{experiment.imgsz}"
        f"_m{safe_name_part(experiment.mosaic)}"
        f"_s{safe_name_part(experiment.scale)}"
        f"_t{safe_name_part(experiment.translate)}"
        f"_e{safe_name_part(experiment.erasing)}"
        f"_hs{safe_name_part(experiment.hsv_s)}"
        f"_hv{safe_name_part(experiment.hsv_v)}"
        f"_f{safe_name_part(experiment.fliplr)}"
    )


def image_paths(images_dir: Path) -> list[Path]:
    return sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


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


def metric_rows(
    *,
    predictions_by_image: dict[str, list[tuple[float, tuple[float, float, float, float]]]],
    labels_by_image: dict[str, list[tuple[float, float, float, float]]],
    confidence_thresholds: list[float],
    iou_threshold: float,
    inference_type: str,
) -> list[dict[str, object]]:
    total_targets = sum(len(labels) for labels in labels_by_image.values())
    rows: list[dict[str, object]] = []
    for threshold in confidence_thresholds:
        tp = fp = fn = 0
        for image_name, targets in labels_by_image.items():
            image_tp, image_fp, image_fn = match_counts(
                predictions_by_image.get(image_name, []),
                targets,
                threshold,
                iou_threshold,
            )
            tp += image_tp
            fp += image_fp
            fn += image_fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        rows.append(
            {
                "inference": inference_type,
                "confidence": threshold,
                "iou": iou_threshold,
                "images": len(labels_by_image),
                "targets": total_targets,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": precision,
                "recall": recall,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def clear_cuda_cache() -> None:
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def labels_for_images(images: list[Path], labels_dir: Path) -> dict[str, list[tuple[float, float, float, float]]]:
    return {
        image_path.name: load_labels(labels_dir / f"{image_path.stem}.txt", image_path)
        for image_path in images
    }


def yolo_predictions(
    model: YOLO,
    images: list[Path],
    *,
    imgsz: int,
    min_conf: float,
    device: str,
) -> dict[str, list[tuple[float, tuple[float, float, float, float]]]]:
    results = model.predict(
        [str(path) for path in images],
        imgsz=imgsz,
        conf=min_conf,
        device=device,
        verbose=False,
    )
    predictions: dict[str, list[tuple[float, tuple[float, float, float, float]]]] = {}
    for image_path, result in zip(images, results):
        image_predictions = []
        if result.boxes is not None:
            for xyxy, conf in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist()):
                image_predictions.append((float(conf), tuple(float(value) for value in xyxy)))
        predictions[image_path.name] = image_predictions
    return predictions


def sahi_predictions(
    model_path: Path,
    images: list[Path],
    *,
    min_conf: float,
    device: str,
    slice_size: int,
    overlap: float,
) -> dict[str, list[tuple[float, tuple[float, float, float, float]]]]:
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(model_path),
        confidence_threshold=min_conf,
        device=f"cuda:{device}" if device.isdigit() else device,
    )
    predictions: dict[str, list[tuple[float, tuple[float, float, float, float]]]] = {}
    for image_path in images:
        result = get_sliced_prediction(
            str(image_path),
            detection_model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
            verbose=0,
        )
        image_predictions = []
        for prediction in result.object_prediction_list:
            box = prediction.bbox
            image_predictions.append(
                (
                    float(prediction.score.value),
                    (float(box.minx), float(box.miny), float(box.maxx), float(box.maxy)),
                )
            )
        predictions[image_path.name] = image_predictions
    del detection_model
    clear_cuda_cache()
    return predictions


def summarize_training_results(results_csv: Path) -> dict[str, object]:
    if not results_csv.is_file():
        return {}
    with results_csv.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        return {}
    best = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    final = rows[-1]
    return {
        "best_epoch": best["epoch"],
        "best_val_precision": best["metrics/precision(B)"],
        "best_val_recall": best["metrics/recall(B)"],
        "best_val_map50": best["metrics/mAP50(B)"],
        "best_val_map50_95": best["metrics/mAP50-95(B)"],
        "best_train_box_loss": best["train/box_loss"],
        "best_train_cls_loss": best["train/cls_loss"],
        "best_train_dfl_loss": best["train/dfl_loss"],
        "best_val_box_loss": best["val/box_loss"],
        "best_val_cls_loss": best["val/cls_loss"],
        "best_val_dfl_loss": best["val/dfl_loss"],
        "final_epoch": final["epoch"],
        "final_val_map50_95": final["metrics/mAP50-95(B)"],
    }


def save_visuals(
    model_path: Path,
    source: Path,
    run_dir: Path,
    *,
    imgsz: int,
    device: str,
    confidence: float,
    slice_size: int,
    overlap: float,
) -> None:
    yolo_visual_dir = run_dir / "inference_yolo_visuals"
    model = YOLO(str(model_path))
    model.predict(
        source=str(source),
        imgsz=imgsz,
        conf=confidence,
        device=device,
        project=str(yolo_visual_dir),
        name="",
        save=True,
        exist_ok=True,
        verbose=False,
    )
    del model
    clear_cuda_cache()

    sahi_visual_dir = run_dir / "inference_sahi_visuals"
    sahi_predict(
        model_type="ultralytics",
        model_path=str(model_path),
        model_device=f"cuda:{device}" if device.isdigit() else device,
        model_confidence_threshold=confidence,
        source=str(source),
        project=str(sahi_visual_dir),
        name="",
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
    )


def run_experiment(
    experiment: Experiment,
    *,
    args: argparse.Namespace,
    batch: int | float,
) -> dict[str, object]:
    run_name = experiment_name(experiment)
    run_dir = args.output_dir / run_name

    if args.overwrite and run_dir.exists():
        shutil.rmtree(run_dir)
    if args.skip_existing and (run_dir / "weights" / "best.pt").is_file():
        print(f"Skipping existing run: {run_name}")
        summary = summarize_training_results(run_dir / "results.csv")
        return {"run": run_name, **summary, "status": "skipped"}

    print(f"Starting run: {run_name}")
    model = YOLO(experiment.model)
    train_args = {
        "data": str(args.data),
        "device": args.device,
        "project": str(args.output_dir),
        "name": run_name,
        "exist_ok": args.overwrite,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch": batch,
        "workers": args.workers,
        "imgsz": experiment.imgsz,
        "mosaic": experiment.mosaic,
        "close_mosaic": 0,
        "hsv_s": experiment.hsv_s,
        "hsv_v": experiment.hsv_v,
        "fliplr": experiment.fliplr,
        "scale": experiment.scale,
        "translate": experiment.translate,
        "erasing": experiment.erasing,
        "plots": True,
    }
    model.train(**train_args)
    del model
    clear_cuda_cache()

    best_model = run_dir / "weights" / "best.pt"
    if not best_model.is_file():
        raise FileNotFoundError(f"Training did not produce best.pt: {best_model}")

    images = image_paths(args.test_images)
    labels_by_image = labels_for_images(images, args.test_labels)
    min_conf = min(args.conf)

    trained_model = YOLO(str(best_model))
    yolo_preds = yolo_predictions(
        trained_model,
        images,
        imgsz=experiment.imgsz,
        min_conf=min_conf,
        device=args.device,
    )
    del trained_model
    clear_cuda_cache()
    yolo_rows = metric_rows(
        predictions_by_image=yolo_preds,
        labels_by_image=labels_by_image,
        confidence_thresholds=args.conf,
        iou_threshold=args.iou,
        inference_type="yolo",
    )

    sahi_preds = sahi_predictions(
        best_model,
        images,
        min_conf=min_conf,
        device=args.device,
        slice_size=args.sahi_slice_size,
        overlap=args.sahi_overlap,
    )
    clear_cuda_cache()
    sahi_rows = metric_rows(
        predictions_by_image=sahi_preds,
        labels_by_image=labels_by_image,
        confidence_thresholds=args.conf,
        iou_threshold=args.iou,
        inference_type="sahi",
    )

    fixed_rows = yolo_rows + sahi_rows
    write_csv(run_dir / "fixed_conf_test_metrics.csv", fixed_rows)

    if not args.no_visuals:
        save_visuals(
            best_model,
            args.test_images,
            run_dir,
            imgsz=experiment.imgsz,
            device=args.device,
            confidence=args.visual_conf,
            slice_size=args.sahi_slice_size,
            overlap=args.sahi_overlap,
        )
        clear_cuda_cache()

    metadata = {
        "experiment": experiment.__dict__,
        "train_args": train_args,
        "confidence_thresholds": args.conf,
        "visual_confidence": args.visual_conf,
        "iou_threshold": args.iou,
        "sahi_slice_size": args.sahi_slice_size,
        "sahi_overlap": args.sahi_overlap,
    }
    (run_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    summary = summarize_training_results(run_dir / "results.csv")
    visual_row = next(
        row for row in fixed_rows
        if row["inference"] == "yolo" and row["confidence"] == args.visual_conf
    )
    sahi_visual_row = next(
        row for row in fixed_rows
        if row["inference"] == "sahi" and row["confidence"] == args.visual_conf
    )
    return {
        "run": run_name,
        "status": "completed",
        "model": experiment.model,
        "imgsz": experiment.imgsz,
        "mosaic": experiment.mosaic,
        "scale": experiment.scale,
        "translate": experiment.translate,
        "erasing": experiment.erasing,
        "hsv_s": experiment.hsv_s,
        "hsv_v": experiment.hsv_v,
        "fliplr": experiment.fliplr,
        **summary,
        "visual_conf": args.visual_conf,
        "yolo_precision_at_visual_conf": visual_row["precision"],
        "yolo_recall_at_visual_conf": visual_row["recall"],
        "yolo_tp_at_visual_conf": visual_row["true_positives"],
        "yolo_fp_at_visual_conf": visual_row["false_positives"],
        "yolo_fn_at_visual_conf": visual_row["false_negatives"],
        "sahi_precision_at_visual_conf": sahi_visual_row["precision"],
        "sahi_recall_at_visual_conf": sahi_visual_row["recall"],
        "sahi_tp_at_visual_conf": sahi_visual_row["true_positives"],
        "sahi_fp_at_visual_conf": sahi_visual_row["false_positives"],
        "sahi_fn_at_visual_conf": sahi_visual_row["false_negatives"],
    }


def main() -> int:
    args = parse_args()
    args.data = args.data.resolve()
    args.test_images = args.test_images.resolve()
    args.test_labels = args.test_labels.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    models = parse_str_list(args.models)
    imgsz_values = parse_int_list(args.imgsz)
    mosaic_values = parse_float_list(args.mosaic)
    scale_values = parse_float_list(args.scale)
    translate_values = parse_float_list(args.translate)
    erasing_values = parse_float_list(args.erasing)
    hsv_s_values = parse_float_list(args.hsv_s)
    hsv_v_values = parse_float_list(args.hsv_v)
    fliplr_values = parse_float_list(args.fliplr)
    batch = normalize_batch(args.batch)

    experiments = [
        Experiment(*values)
        for values in itertools.product(
            models,
            imgsz_values,
            mosaic_values,
            scale_values,
            translate_values,
            erasing_values,
            hsv_s_values,
            hsv_v_values,
            fliplr_values,
        )
    ]
    if args.limit is not None:
        experiments = experiments[: args.limit]

    summary_rows = []
    summary_path = args.output_dir / "summary.csv"
    for experiment in experiments:
        try:
            summary_rows.append(run_experiment(experiment, args=args, batch=batch))
            write_csv(summary_path, summary_rows)
        except Exception as exc:
            run_name = experiment_name(experiment)
            error_row = {
                "run": run_name,
                "status": "failed",
                "model": experiment.model,
                "imgsz": experiment.imgsz,
                "mosaic": experiment.mosaic,
                "scale": experiment.scale,
                "translate": experiment.translate,
                "erasing": experiment.erasing,
                "hsv_s": experiment.hsv_s,
                "hsv_v": experiment.hsv_v,
                "fliplr": experiment.fliplr,
                "error": repr(exc),
            }
            summary_rows.append(error_row)
            write_csv(summary_path, summary_rows)
            print(f"Run failed: {run_name}: {exc}")
            raise

    print(f"Grid search summary written to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
