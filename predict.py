"""
predict.py

Run YOLO inference on a single image or all images in a folder.

Examples
--------
Predict on all images in folder:
     python predict.py --folder data/yolo/images/train/

Specific image:
    python predict.py --image IMG_123
"""

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

import config
from utils.image_utils import get_images
from utils.color_utils import detect_tag_colors
from utils.ocr_utils import read_tag_text


def parse_arguments():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run YOLO inference."
    )

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--image",
        help="Run inference on a single image.",
    )

    group.add_argument(
        "--folder",
        help="Run inference on all images in a folder.",
    )

    return parser.parse_args()


def load_model():
    """Load the trained YOLO model."""

    model_path = (
        config.MODEL_DIRECTORY
        / config.BEST_MODEL_NAME
    )

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found:\n{model_path}\n\n"
            "Run train.py first."
        )

    return YOLO(model_path)


def run_inference(model, image_path):
    """Run YOLO inference on an image."""

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"Could not load image:\n{image_path}"
        )

    start_time = time.perf_counter()

    prediction_results = model(
        image,
        conf=config.CONFIDENCE_THRESHOLD,
        verbose=False,
    )[0]

    inference_time = (
        time.perf_counter() - start_time
    ) * 1000

    return image, prediction_results, inference_time


def extract_detections(prediction_results):
    """Extract detection information from YOLO prediction results."""

    detections = []

    for box in prediction_results.boxes:

        x1, y1, x2, y2 = (
            box.xyxy[0]
            .cpu()
            .numpy()
            .astype(int)
        )

        class_id = int(box.cls[0])

        detections.append(
            {
                "class_name": config.OBJECT_CLASSES.get(
                    class_id,
                    str(class_id),
                ),
                "confidence": float(box.conf[0]),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )

    return detections


def draw_predictions(image, detections):
    """Draw predicted bounding boxes."""

    print("\nDetections")
    print("-" * 60)

    for index, detection in enumerate(
        detections,
        start=1,
    ):

        x1 = detection["x1"]
        y1 = detection["y1"]
        x2 = detection["x2"]
        y2 = detection["y2"]

        confidence = detection["confidence"]
        class_name = detection["class_name"]

        plate_color = detection.get("plate_color", "unknown")
        text_color = detection.get("text_color", "unknown")

        text = detection.get("text", "")

        label_parts = [
            class_name,
            f"{confidence:.2f}",
            f"{plate_color}/{text_color}",
        ]

        if text:
            label_parts.append(text)

        label = " ".join(label_parts)

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            config.BOUNDING_BOX_COLOR,
            config.LINE_THICKNESS,
        )

        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2,
        )

        cv2.rectangle(
            image,
            (x1, y1 - text_height - 8),
            (x1 + text_width + 6, y1),
            config.LABEL_BACKGROUND_COLOR,
            -1,
        )

        cv2.putText(
            image,
            label,
            (x1 + 3, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            config.LABEL_TEXT_COLOR,
            2,
        )

        print(f"{index})")
        print(
            f"    Class          : {class_name}"
        )
        print(
            f"    Confidence     : {confidence:.3f}"
        )
        print(
            f"    Bounding box   : ({x1}, {y1}) -> ({x2}, {y2})"
        )
        if "plate_color" in detection:
            print(
                f"    Plate color    : {detection['plate_color']}"
            )

        if "text_color" in detection:
            print(
                f"    Text color     : {detection['text_color']}"
            )

        if "color_confidence" in detection:
            print(
                f"    Color conf.    : "
                f"{detection['color_confidence']:.3f}"
            )

        if "text" in detection:
            print(
                f"    OCR text       : {detection['text']}"
            )

        if "text_confidence" in detection:
            print(
                f"    OCR conf.      : "
                f"{detection['text_confidence']:.3f}"
            )

    if len(detections) == 0:
        print("No objects detected.")

    return image


def crop_detections(image, detections):
    """Crop detected objects from the image."""

    image_height, image_width = image.shape[:2]

    for detection in detections:

        x1 = max(0, detection["x1"])
        y1 = max(0, detection["y1"])
        x2 = min(image_width, detection["x2"])
        y2 = min(image_height, detection["y2"])

        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        detection["crop"] = crop.copy()


def classify_tag_colors(detections):
    """Classify plate and text colors for each detected tag."""

    for detection in detections:

        crop = detection.get("crop")

        if crop is None or crop.size == 0:
            detection.update(
                {
                    "plate_color": "unknown",
                    "text_color": "unknown",
                    "color_confidence": 0.0,
                }
            )
            continue

        color_result = detect_tag_colors(crop)
        detection.update(color_result)



def read_tag_texts(detections):
    """Read text from each detected wing tag."""

    for detection in detections:

        crop = detection.get("crop")

        if crop is None or crop.size == 0:
            continue

        ocr_result = read_tag_text(crop)

        detection["text"] = ocr_result["text"]
        detection["text_confidence"] = ocr_result["confidence"]


def save_prediction(image, image_path):
    """Save the prediction image."""

    config.PREDICTION_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        config.PREDICTION_DIRECTORY
        / f"{image_path.stem}_prediction.jpg"
    )

    cv2.imwrite(
        str(output_path),
        image,
    )

    return output_path


def main():
    """Run YOLO inference."""

    args = parse_arguments()

    if args.image:
        image_paths = [Path(args.image)]
    else:
        image_paths = get_images(args.folder)

    model = load_model()

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):

        print("=" * 60)
        print(
            f"Image {index} / {len(image_paths)}"
        )
        print("=" * 60)

        print("Image")
        print(f"    {image_path}")

        image, prediction_results, inference_time = (
            run_inference(
                model,
                image_path,
            )
        )

        detections = extract_detections(
            prediction_results
        )

        crop_detections(
            image,
            detections,
        )

        classify_tag_colors(
            detections,
        )

        read_tag_texts(
            detections,
        )

        image = draw_predictions(
            image,
            detections,
        )

        output_path = save_prediction(
            image,
            image_path,
        )

        cv2.imshow(
            f"Prediction ({index}/{len(image_paths)})",
            image,
        )

        print("-" * 60)
        print(
            f"Inference time : {inference_time:.1f} ms"
        )
        print(
            f"Prediction saved to:\n{output_path}"
        )

        while True:

            key = cv2.waitKey(0)

            if key in (13, 32):
                break

            if key in (
                ord("q"),
                ord("Q"),
            ):
                cv2.destroyAllWindows()
                return

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()