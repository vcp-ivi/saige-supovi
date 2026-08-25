"""
evaluate_ocr.py

Evaluate the fine-tuned PaddleOCR recognizer on the cleaned OCR test set.

Outputs:
    outputs/ocr_test_results.csv

Usage:
    python evaluate_ocr.py
"""

from pathlib import Path
import csv
import sys

PROJECT_ROOT = Path(__file__).resolve().parent

PADDLEOCR_ROOT = (
    PROJECT_ROOT.parent
    / "PaddleOCR"
)

sys.path.insert(
    0,
    str(PADDLEOCR_ROOT),
)

import cv2
import paddle
import yaml

from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process


# ============================================================
# Paths
# ============================================================

OCR_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "ocr"
)

TEST_LABEL_FILE = (
    OCR_DIRECTORY
    / "test.txt"
)

PADDLE_CONFIG_FILE = (
    PADDLEOCR_ROOT
    / "configs"
    / "rec"
    / "PP-OCRv5"
    / "wing_tags_rec.yml"
)

MODEL_PATH = (
    PADDLEOCR_ROOT
    / "output"
    / "wing_tags_rec"
    / "best_accuracy.pdparams"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "ocr_test_results.csv"
)


# ============================================================
# Load test set
# ============================================================

def load_test_samples():

    samples = []

    with open(
        TEST_LABEL_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            image_path, ground_truth = (
                line.split("\t", 1)
            )

            samples.append(
                (
                    OCR_DIRECTORY / image_path,
                    ground_truth.strip().upper(),
                )
            )

    return samples


# ============================================================
# Load model
# ============================================================

def load_model():

    with open(
        PADDLE_CONFIG_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        config = yaml.safe_load(file)

    post_process_class = build_post_process(
        config["PostProcess"],
        config["Global"],
    )

    config["Architecture"]["Head"][
        "out_channels_list"
    ] = {
        "CTCLabelDecode": (
            len(post_process_class.character)
        ),
        "NRTRLabelDecode": (
            len(post_process_class.character)
            + 3
        ),
    }

    model = build_model(
        config["Architecture"]
    )

    state_dict = paddle.load(
        str(MODEL_PATH)
    )

    model.set_state_dict(
        state_dict
    )

    model.eval()

    return (
        model,
        post_process_class,
        config,
    )


# ============================================================
# Preprocessing
# ============================================================

def build_eval_operators(config):

    transforms = (
        config["Eval"]
        ["dataset"]
        ["transforms"]
    )

    operators = []

    for transform_config in transforms:

        if "MultiLabelEncode" in transform_config:
            continue

        if "KeepKeys" in transform_config:
            continue

        operators.append(
            transform_config
        )

    operators.append(
        {
            "KeepKeys": {
                "keep_keys": [
                    "image",
                    "valid_ratio",
                ]
            }
        }
    )

    return create_operators(
        operators,
        config["Global"],
    )


# ============================================================
# Prediction
# ============================================================

def predict_image(
    image_path,
    model,
    post_process_class,
    operators,
):

    with open(
        image_path,
        "rb",
    ) as file:

        image_bytes = file.read()

    data = {
        "image": image_bytes,
    }

    batch = transform(
        data,
        operators,
    )

    if batch is None:
        return "", 0.0

    image = batch[0]

    image = (
        paddle.to_tensor(image)
        .unsqueeze(0)
    )

    with paddle.no_grad():

        predictions = model(
            image
        )

    result = post_process_class(
        predictions
    )

    if not result:
        return "", 0.0

    text, confidence = result[0]

    return (
        text.strip().upper(),
        float(confidence),
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 55)
    print("Fine-tuned PaddleOCR Test Evaluation")
    print("=" * 55)

    samples = load_test_samples()

    print(
        f"\nTest samples: {len(samples)}"
    )

    (
        model,
        post_process_class,
        config,
    ) = load_model()

    operators = build_eval_operators(
        config
    )

    results = []

    correct = 0

    for index, (
        image_path,
        ground_truth,
    ) in enumerate(
        samples,
        start=1,
    ):

        prediction, confidence = (
            predict_image(
                image_path,
                model,
                post_process_class,
                operators,
            )
        )

        is_correct = (
            prediction == ground_truth
        )

        if is_correct:
            correct += 1

        results.append(
            {
                "image": image_path.name,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "confidence": confidence,
                "correct": is_correct,
            }
        )

        print(
            f"{index:3d}/{len(samples)}  "
            f"GT={ground_truth:<4}  "
            f"PRED={prediction:<4}  "
            f"CONF={confidence:.3f}  "
            f"{'OK' if is_correct else 'WRONG'}"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "image",
                "ground_truth",
                "prediction",
                "confidence",
                "correct",
            ],
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    accuracy = (
        correct / len(samples)
        if samples
        else 0.0
    )

    print()
    print("=" * 55)
    print("Results")
    print("=" * 55)

    print(
        f"Correct : {correct}/{len(samples)}"
    )

    print(
        f"Accuracy: {accuracy * 100:.1f}%"
    )

    print()

    print(
        f"Saved to:\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()