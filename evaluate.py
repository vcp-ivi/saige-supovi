"""
evaluate.py

Evaluate YOLO detection, color classification, and OCR against
data/raw/annotations.csv.

The dataset split is selected with --split.

The evaluated stages depend on config.py:

    ENABLE_COLOR_CLASSIFICATION = False
    ENABLE_OCR = False

Examples
--------
Evaluate the validation split:

    python evaluate.py --split val

Evaluate the test split:

    python evaluate.py --split test

YOLO only:

    ENABLE_COLOR_CLASSIFICATION = False
    ENABLE_OCR = False

YOLO + colors:

    ENABLE_COLOR_CLASSIFICATION = True
    ENABLE_OCR = False

Full pipeline:

    ENABLE_COLOR_CLASSIFICATION = True
    ENABLE_OCR = True
"""

import argparse
import csv
from collections import defaultdict

import config

from predict import (
    load_model,
    run_inference,
    extract_detections,
    crop_detections,
    classify_tag_colors,
    read_tag_texts,
)

from utils.image_utils import get_images


IOU_THRESHOLD = 0.50


def parse_arguments():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Evaluate the wing-tag recognition pipeline."
    )

    parser.add_argument(
        "--split",
        required=True,
        choices=config.DATASET_SPLITS.keys(),
        help="Dataset split to evaluate: train, val, or test.",
    )

    return parser.parse_args()


def load_annotations():
    """
    Load annotations and group them by image filename.
    """

    annotations = defaultdict(list)

    with config.ANNOTATION_FILE.open(
        mode="r",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        reader = csv.DictReader(
            csv_file
        )

        for row in reader:

            image_name = (
                row[config.CSV_IMAGE_FILE]
                .strip()
                .lower()
            )

            plate_color = (
                row[config.CSV_PLATE_COLOR]
                .strip()
                .lower()
            )

            text_color = (
                row[config.CSV_TEXT_COLOR]
                .strip()
                .lower()
            )

            plate_text = (
                row[config.CSV_PLATE_TEXT]
                .strip()
                .upper()
            )

            # Images marked none / none contain no annotated tag.
            if (
                plate_color == "none"
                and text_color == "none"
            ):
                continue

            try:
                annotation = {
                    "x1": float(
                        row[config.CSV_X1]
                    ),
                    "y1": float(
                        row[config.CSV_Y1]
                    ),
                    "x2": float(
                        row[config.CSV_X2]
                    ),
                    "y2": float(
                        row[config.CSV_Y2]
                    ),
                    "plate_color": plate_color,
                    "text_color": text_color,
                    "text": plate_text,
                }

            except (
                ValueError,
                TypeError,
            ):
                continue

            annotations[
                image_name
            ].append(
                annotation
            )

    return annotations


def calculate_iou(
    box_a,
    box_b,
):
    """
    Calculate intersection over union between two bounding boxes.
    """

    x1 = max(
        box_a["x1"],
        box_b["x1"],
    )

    y1 = max(
        box_a["y1"],
        box_b["y1"],
    )

    x2 = min(
        box_a["x2"],
        box_b["x2"],
    )

    y2 = min(
        box_a["y2"],
        box_b["y2"],
    )

    intersection = (
        max(
            0,
            x2 - x1,
        )
        * max(
            0,
            y2 - y1,
        )
    )

    area_a = (
        (box_a["x2"] - box_a["x1"])
        * (box_a["y2"] - box_a["y1"])
    )

    area_b = (
        (box_b["x2"] - box_b["x1"])
        * (box_b["y2"] - box_b["y1"])
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return (
        intersection
        / union
    )


def match_detections(
    ground_truth,
    detections,
):
    """
    Match predictions to ground-truth annotations using highest IoU.

    Each ground-truth tag and prediction can be matched at most once.
    """

    candidates = []

    for gt_index, gt in enumerate(
        ground_truth
    ):

        for pred_index, pred in enumerate(
            detections
        ):

            iou = calculate_iou(
                gt,
                pred,
            )

            if iou >= IOU_THRESHOLD:

                candidates.append(
                    (
                        iou,
                        gt_index,
                        pred_index,
                    )
                )

    candidates.sort(
        reverse=True
    )

    matched_gt = set()
    matched_pred = set()

    matches = []

    for (
        iou,
        gt_index,
        pred_index,
    ) in candidates:

        if gt_index in matched_gt:
            continue

        if pred_index in matched_pred:
            continue

        matches.append(
            (
                gt_index,
                pred_index,
                iou,
            )
        )

        matched_gt.add(
            gt_index
        )

        matched_pred.add(
            pred_index
        )

    missed = [
        index
        for index in range(
            len(ground_truth)
        )
        if index not in matched_gt
    ]

    false_positives = [
        index
        for index in range(
            len(detections)
        )
        if index not in matched_pred
    ]

    return (
        matches,
        missed,
        false_positives,
    )


def classify_text_annotation(
    text,
):
    """
    Classify ground-truth OCR text.

    Returns:
        readable
            All characters are known.

        partial
            Some characters are unknown.
            Examples: *4, 8*, P3*

        unreadable
            All characters are unknown.
            Examples: *, **

        missing
            Empty annotation.
    """

    text = (
        text
        .strip()
        .upper()
    )

    if not text:
        return "missing"

    if set(text) == {"*"}:
        return "unreadable"

    if "*" in text:
        return "partial"

    return "readable"


def partial_text_matches(
    ground_truth,
    prediction,
):
    """
    Check whether OCR output agrees with all known characters
    in a partially readable ground-truth annotation.

    Examples:
        *4  vs A4  -> True
        *4  vs A7  -> False
        8*  vs 87  -> True
        P3* vs P30 -> True
    """

    ground_truth = (
        ground_truth
        .strip()
        .upper()
    )

    prediction = (
        prediction
        .strip()
        .upper()
    )

    if len(
        ground_truth
    ) != len(
        prediction
    ):
        return False

    for (
        gt_character,
        pred_character,
    ) in zip(
        ground_truth,
        prediction,
    ):

        if gt_character == "*":
            continue

        if gt_character != pred_character:
            return False

    return True


def main():
    """Run dataset evaluation."""

    args = parse_arguments()

    image_directory = (
        config.YOLO_IMAGE_DIRECTORY
        / args.split
    )

    if not image_directory.is_dir():
        raise FileNotFoundError(
            f"YOLO {args.split} split not found:\n"
            f"{image_directory}\n\n"
            "Run prepare_yolo_dataset.py first."
        )

    image_paths = get_images(
        image_directory
    )

    if not image_paths:
        raise ValueError(
            f"No supported images found in:\n"
            f"{image_directory}"
        )

    output_file = (
        config.OUTPUT_DIRECTORY
        / f"evaluation_{args.split}.csv"
    )

    print("=" * 60)
    print("Evaluation")
    print("=" * 60)

    print(
        f"Split  : "
        f"{args.split}"
    )

    print(
        f"Images : "
        f"{len(image_paths)}"
    )

    print()

    annotations = load_annotations()

    model = load_model()

    results = []

    # Detection counters
    gt_total = 0
    detected_total = 0
    missed_total = 0
    false_positive_total = 0

    # Color counters
    plate_correct = 0
    text_color_correct = 0
    pair_correct = 0
    color_total = 0

    # OCR counters
    ocr_correct = 0
    ocr_total = 0

    partial_correct = 0
    partial_total = 0

    unreadable_total = 0

    # Evaluate images
    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):

        print(
            f"[{index}/{len(image_paths)}] "
            f"{image_path.name}"
        )

        ground_truth = annotations.get(
            image_path.name.lower(),
            [],
        )

        gt_total += len(
            ground_truth
        )

        (
            image,
            prediction_results,
            _,
        ) = run_inference(
            model,
            image_path,
        )

        detections = extract_detections(
            prediction_results
        )

        # Crops are only needed for colors or OCR.
        if (
            config.ENABLE_COLOR_CLASSIFICATION
            or config.ENABLE_OCR
        ):
            crop_detections(
                image,
                detections,
            )

        if config.ENABLE_COLOR_CLASSIFICATION:
            classify_tag_colors(
                detections
            )

        if config.ENABLE_OCR:
            read_tag_texts(
                detections
            )

        (
            matches,
            missed,
            false_positives,
        ) = match_detections(
            ground_truth,
            detections,
        )

        detected_total += len(
            matches
        )

        missed_total += len(
            missed
        )

        false_positive_total += len(
            false_positives
        )

        # ----------------------------------------------------
        # Matched tags
        # ----------------------------------------------------

        for (
            gt_index,
            pred_index,
            iou,
        ) in matches:

            gt = ground_truth[
                gt_index
            ]

            pred = detections[
                pred_index
            ]

            pred_plate = pred.get(
                "plate_color",
                "",
            )

            pred_text_color = pred.get(
                "text_color",
                "",
            )

            pred_text = (
                pred.get(
                    "text",
                    "",
                )
                .strip()
                .upper()
            )

            errors = []

            # Color evaluation
            plate_ok = ""
            text_color_ok = ""
            pair_ok = ""

            if config.ENABLE_COLOR_CLASSIFICATION:

                color_total += 1

                plate_ok = (
                    pred_plate
                    == gt["plate_color"]
                )

                text_color_ok = (
                    pred_text_color
                    == gt["text_color"]
                )

                pair_ok = (
                    plate_ok
                    and text_color_ok
                )

                plate_correct += int(
                    plate_ok
                )

                text_color_correct += int(
                    text_color_ok
                )

                pair_correct += int(
                    pair_ok
                )

                if not pair_ok:
                    errors.append(
                        "color_error"
                    )

            # OCR evaluation
            text_type = classify_text_annotation(
                gt["text"]
            )

            ocr_result = ""

            if config.ENABLE_OCR:

                if text_type == "readable":

                    ocr_total += 1

                    ocr_ok = (
                        pred_text
                        == gt["text"]
                    )

                    ocr_correct += int(
                        ocr_ok
                    )

                    ocr_result = (
                        "correct"
                        if ocr_ok
                        else "incorrect"
                    )

                    if not ocr_ok:
                        errors.append(
                            "ocr_error"
                        )

                elif text_type == "partial":

                    partial_total += 1

                    partial_ok = partial_text_matches(
                        gt["text"],
                        pred_text,
                    )

                    partial_correct += int(
                        partial_ok
                    )

                    ocr_result = (
                        "compatible"
                        if partial_ok
                        else "incompatible"
                    )

                    if not partial_ok:
                        errors.append(
                            "partial_ocr_error"
                        )

                elif text_type == "unreadable":

                    unreadable_total += 1

                    ocr_result = (
                        "not_evaluated"
                    )

            results.append(
                {
                    "image": image_path.name,

                    "status": (
                        "correct"
                        if not errors
                        else "|".join(
                            errors
                        )
                    ),

                    "iou": round(
                        iou,
                        3,
                    ),

                    "gt_plate": gt[
                        "plate_color"
                    ],

                    "pred_plate": (
                        pred_plate
                    ),

                    "plate_correct": (
                        plate_ok
                    ),

                    "gt_text_color": gt[
                        "text_color"
                    ],

                    "pred_text_color": (
                        pred_text_color
                    ),

                    "text_color_correct": (
                        text_color_ok
                    ),

                    "gt_text": gt[
                        "text"
                    ],

                    "gt_text_type": (
                        text_type
                    ),

                    "pred_text": (
                        pred_text
                    ),

                    "ocr_result": (
                        ocr_result
                    ),

                    "yolo_confidence": round(
                        pred[
                            "confidence"
                        ],
                        3,
                    ),

                    "color_confidence": (
                        round(
                            pred.get(
                                "color_confidence",
                                0.0,
                            ),
                            3,
                        )
                        if config.ENABLE_COLOR_CLASSIFICATION
                        else ""
                    ),

                    "ocr_confidence": (
                        round(
                            pred.get(
                                "text_confidence",
                                0.0,
                            ),
                            3,
                        )
                        if config.ENABLE_OCR
                        else ""
                    ),
                }
            )

        # ----------------------------------------------------
        # Missed tags
        # ----------------------------------------------------

        for gt_index in missed:

            gt = ground_truth[
                gt_index
            ]

            results.append(
                {
                    "image": image_path.name,
                    "status": "missed_tag",
                    "iou": "",

                    "gt_plate": gt[
                        "plate_color"
                    ],
                    "pred_plate": "",
                    "plate_correct": "",

                    "gt_text_color": gt[
                        "text_color"
                    ],
                    "pred_text_color": "",
                    "text_color_correct": "",

                    "gt_text": gt[
                        "text"
                    ],

                    "gt_text_type": (
                        classify_text_annotation(
                            gt["text"]
                        )
                    ),

                    "pred_text": "",
                    "ocr_result": "",

                    "yolo_confidence": "",
                    "color_confidence": "",
                    "ocr_confidence": "",
                }
            )

        # ----------------------------------------------------
        # False-positive detections
        # ----------------------------------------------------

        for pred_index in false_positives:

            pred = detections[
                pred_index
            ]

            results.append(
                {
                    "image": image_path.name,
                    "status": "false_positive",
                    "iou": "",

                    "gt_plate": "",
                    "pred_plate": pred.get(
                        "plate_color",
                        "",
                    ),
                    "plate_correct": "",

                    "gt_text_color": "",
                    "pred_text_color": pred.get(
                        "text_color",
                        "",
                    ),
                    "text_color_correct": "",

                    "gt_text": "",
                    "gt_text_type": "",

                    "pred_text": pred.get(
                        "text",
                        "",
                    ),

                    "ocr_result": "",

                    "yolo_confidence": round(
                        pred[
                            "confidence"
                        ],
                        3,
                    ),

                    "color_confidence": (
                        round(
                            pred.get(
                                "color_confidence",
                                0.0,
                            ),
                            3,
                        )
                        if config.ENABLE_COLOR_CLASSIFICATION
                        else ""
                    ),

                    "ocr_confidence": (
                        round(
                            pred.get(
                                "text_confidence",
                                0.0,
                            ),
                            3,
                        )
                        if config.ENABLE_OCR
                        else ""
                    ),
                }
            )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if results:

        with output_file.open(
            mode="w",
            newline="",
            encoding="utf-8",
        ) as csv_file:

            writer = csv.DictWriter(
                csv_file,
                fieldnames=results[
                    0
                ].keys(),
            )

            writer.writeheader()

            writer.writerows(
                results
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 60)

    print(
        f"Evaluation Summary — "
        f"{args.split}"
    )

    print("=" * 60)

    # Detection
    print()
    print("Tag detection")
    print("-" * 60)

    print(
        f"Ground-truth tags : "
        f"{gt_total}"
    )

    print(
        f"Detected correctly: "
        f"{detected_total}"
    )

    print(
        f"Missed tags       : "
        f"{missed_total}"
    )

    print(
        f"False positives   : "
        f"{false_positive_total}"
    )

    precision_denominator = (
        detected_total
        + false_positive_total
    )

    precision = (
        detected_total
        / precision_denominator
        if precision_denominator
        else 0
    )

    recall = (
        detected_total
        / gt_total
        if gt_total
        else 0
    )

    print(
        f"Precision         : "
        f"{precision * 100:.1f}%"
    )

    print(
        f"Recall            : "
        f"{recall * 100:.1f}%"
    )

    # Colors
    if config.ENABLE_COLOR_CLASSIFICATION:

        print()
        print("Colors")
        print("-" * 60)

        if color_total:

            print(
                f"Plate color       : "
                f"{plate_correct}/{color_total} "
                f"({plate_correct / color_total * 100:.1f}%)"
            )

            print(
                f"Text color        : "
                f"{text_color_correct}/{color_total} "
                f"({text_color_correct / color_total * 100:.1f}%)"
            )

            print(
                f"Full color pair   : "
                f"{pair_correct}/{color_total} "
                f"({pair_correct / color_total * 100:.1f}%)"
            )

        else:

            print(
                "No matched tags available."
            )

    # OCR
    if config.ENABLE_OCR:

        print()
        print("OCR")
        print("-" * 60)

        if ocr_total:

            print(
                f"Exact text        : "
                f"{ocr_correct}/{ocr_total} "
                f"({ocr_correct / ocr_total * 100:.1f}%)"
            )

        else:

            print(
                "Exact text        : N/A"
            )

        if partial_total:

            print(
                f"Partial compatible: "
                f"{partial_correct}/{partial_total} "
                f"({partial_correct / partial_total * 100:.1f}%)"
            )

        else:

            print(
                "Partial compatible: N/A"
            )

        print(
            f"Unreadable GT     : "
            f"{unreadable_total}"
        )

    print()

    print(
        f"Results saved to:\n"
        f"{output_file}"
    )


if __name__ == "__main__":
    main()