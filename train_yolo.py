"""
train_yolo.py

Train a YOLO model on the prepared wing-tag dataset.
"""

import shutil

from ultralytics import YOLO

import config


def train_model():
    """Train a YOLO model and return the training results."""

    model = YOLO(
        config.YOLO_MODEL_NAME
    )

    training_results = model.train(
        data=str(
            config.YOLO_DATASET_CONFIG
        ),
        imgsz=config.YOLO_IMAGE_SIZE,
        epochs=config.YOLO_EPOCHS,
        batch=config.YOLO_BATCH_SIZE,
        patience=config.YOLO_PATIENCE,
        workers=config.YOLO_WORKERS,
        device=config.YOLO_DEVICE,
        project=str(
            config.YOLO_RUN_DIRECTORY
        ),
        mosaic=config.YOLO_MOSAIC,
        scale=config.YOLO_SCALE,
        translate=config.YOLO_TRANSLATE,
        fliplr=config.YOLO_FLIP_LEFT_RIGHT,
        hsv_h=config.YOLO_HSV_H,
        hsv_s=config.YOLO_HSV_S,
        hsv_v=config.YOLO_HSV_V,
    )

    return training_results


def save_best_model(
    training_results,
):
    """
    Copy the best trained model into the project's YOLO model directory.
    """

    best_model = (
        training_results.save_dir
        / "weights"
        / config.YOLO_BEST_MODEL_NAME
    )

    if not best_model.exists():
        raise FileNotFoundError(
            f"Could not find trained model:\n"
            f"{best_model}"
        )

    config.YOLO_MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination = (
        config.YOLO_MODEL_DIRECTORY
        / config.YOLO_BEST_MODEL_NAME
    )

    shutil.copy2(
        best_model,
        destination,
    )

    return destination


def main():
    """Train a YOLO model and save the best weights."""

    if not config.YOLO_DATASET_CONFIG.is_file():
        raise FileNotFoundError(
            f"Dataset configuration not found:\n"
            f"{config.YOLO_DATASET_CONFIG}\n\n"
            "Run prepare_yolo_dataset.py first."
        )

    print("=" * 60)
    print("Training YOLO")
    print("=" * 60)

    print(
        f"Model       : "
        f"{config.YOLO_MODEL_NAME}"
    )

    print(
        f"Dataset     : "
        f"{config.YOLO_DATASET_CONFIG}"
    )

    print(
        f"Image size  : "
        f"{config.YOLO_IMAGE_SIZE}"
    )

    print(
        f"Epochs      : "
        f"{config.YOLO_EPOCHS}"
    )

    print(
        f"Batch size  : "
        f"{config.YOLO_BATCH_SIZE}"
    )

    print(
        f"Device      : "
        f"{config.YOLO_DEVICE}"
    )

    print()

    print("-" * 60)
    print("Training started...")
    print("-" * 60)

    training_results = (
        train_model()
    )

    destination = save_best_model(
        training_results
    )

    print()
    print("-" * 60)
    print("Training completed.")
    print("-" * 60)

    print(
        f"Best model saved to:\n"
        f"{destination}"
    )


if __name__ == "__main__":
    main()