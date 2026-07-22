"""
train.py

Train a YOLO model on the prepared dataset.
"""

import shutil

from ultralytics import YOLO

import config


def train_model():
    """Train a YOLO model and return the training results."""

    model = YOLO(config.MODEL_NAME)

    training_results = model.train(
        data=str(config.YOLO_DATASET_CONFIG),
        imgsz=config.IMAGE_SIZE,
        epochs=config.EPOCHS,
        batch=config.BATCH_SIZE,
        patience=config.PATIENCE,
        workers=config.WORKERS,
        device=config.DEVICE,
        project=str(config.RUN_DIRECTORY),
        mosaic=config.MOSAIC,
        scale=config.SCALE,
        translate=config.TRANSLATE,
        fliplr=config.FLIP_LEFT_RIGHT,
        hsv_h=config.HSV_H,
        hsv_s=config.HSV_S,
        hsv_v=config.HSV_V,
    )

    return training_results


def save_best_model(training_results):
    """Copy the best trained model into the project's model directory."""

    best_model = (
        training_results.save_dir
        / "weights"
        / config.BEST_MODEL_NAME
    )

    if not best_model.exists():
        raise FileNotFoundError(
            f"Could not find trained model:\n{best_model}"
        )

    config.MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination = (
        config.MODEL_DIRECTORY
        / config.BEST_MODEL_NAME
    )

    shutil.copy2(best_model, destination)

    return destination


def main():
    """Train a YOLO model and save the best weights."""

    if not config.YOLO_DATASET_CONFIG.is_file():
        raise FileNotFoundError(
            f"Dataset configuration not found:\n"
            f"{config.YOLO_DATASET_CONFIG}"
        )

    print("=" * 60)
    print("Training YOLO")
    print("=" * 60)

    print(f"Model       : {config.MODEL_NAME}")
    print(f"Dataset     : {config.YOLO_DATASET_CONFIG}")
    print(f"Image size  : {config.IMAGE_SIZE}")
    print(f"Epochs      : {config.EPOCHS}")
    print(f"Batch size  : {config.BATCH_SIZE}")
    print(f"Device      : {config.DEVICE}")
    print()

    print("-" * 60)
    print("Training started...")
    print("-" * 60)

    training_results = train_model()

    destination = save_best_model(training_results)

    print()
    print("-" * 60)
    print("Training completed.")
    print("-" * 60)
    print(f"Best model saved to:\n{destination}")


if __name__ == "__main__":
    main()