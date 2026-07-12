r"""Run sliced inference with SAHI on trained YOLO weights.

Example:
    .\.venv311\Scripts\python.exe .\scripts\sahi_predict.py `
        --model outputs\yolo_results\tag_yolo26s_baseline\weights\best.pt `
        --source data\yolo\images\test `
        --conf 0.01
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / "outputs" / "ultralytics_config"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / "matplotlib_config"))

from sahi.predict import predict


def confidence_output_suffix(confidence: float) -> str:
    suffix = f"{confidence:.6f}".rstrip("0").rstrip(".")
    return suffix.replace(".", "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO + SAHI sliced inference on images or a directory."
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained YOLO weights, usually weights/best.pt.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/yolo/images/test"),
        help="Image file or directory to run inference on.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Directory for SAHI prediction outputs. Defaults to "
            "outputs/sahi_predictions_conf<value>, for example conf001."
        ),
    )
    parser.add_argument("--device", default="cuda:0", help="Use cuda:0 or cpu.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--slice-size", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.model.is_file():
        print(f"Model file not found: {args.model}")
        return 2
    if not args.source.exists():
        print(f"Source not found: {args.source}")
        return 2
    output_dir = args.output
    if output_dir is None:
        output_dir = Path("outputs") / f"sahi_predictions_conf{confidence_output_suffix(args.conf)}"

    predict(
        model_type="ultralytics",
        model_path=str(args.model),
        model_device=args.device,
        model_confidence_threshold=args.conf,
        source=str(args.source),
        project=str(output_dir),
        name="",
        slice_height=args.slice_size,
        slice_width=args.slice_size,
        overlap_height_ratio=args.overlap,
        overlap_width_ratio=args.overlap,
    )
    print(f"SAHI predictions written under: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
