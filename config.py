"""
config.py

Central configuration file containing user-configurable project parameters.
"""

from pathlib import Path


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIRECTORY = PROJECT_ROOT / "data"

RAW_DATA_DIRECTORY = DATA_DIRECTORY / "raw"
RAW_IMAGE_DIRECTORY = RAW_DATA_DIRECTORY / "images"
ANNOTATION_FILE = RAW_DATA_DIRECTORY / "annotations.csv"


# =============================================================================
# YOLO DATASET PATHS
# =============================================================================

YOLO_DATA_DIRECTORY = DATA_DIRECTORY / "yolo"

YOLO_IMAGE_DIRECTORY = YOLO_DATA_DIRECTORY / "images"
YOLO_LABEL_DIRECTORY = YOLO_DATA_DIRECTORY / "labels"

YOLO_DATASET_CONFIG = YOLO_DATA_DIRECTORY / "data.yaml"


# =============================================================================
# OCR DATASET PATHS
# =============================================================================

OCR_DATA_DIRECTORY = DATA_DIRECTORY / "ocr"

OCR_IMAGE_DIRECTORY = OCR_DATA_DIRECTORY / "images"

OCR_CHARACTER_DICT = (
    OCR_DATA_DIRECTORY
    / "character_dict.txt"
)

OCR_METADATA_FILE = (
    OCR_DATA_DIRECTORY
    / "metadata.csv"
)


# =============================================================================
# MODEL PATHS
# =============================================================================

YOLO_MODEL_DIRECTORY = (
    PROJECT_ROOT
    / "models"
    / "yolo"
)

OCR_MODEL_DIRECTORY = (
    PROJECT_ROOT
    / "models"
    / "paddleocr"
)


# =============================================================================
# OUTPUT PATHS
# =============================================================================

OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs"

PREDICTION_DIRECTORY = (
    OUTPUT_DIRECTORY
    / "predictions"
)

YOLO_RUN_DIRECTORY = (
    PROJECT_ROOT
    / "runs"
)


# =============================================================================
# DATASET SETTINGS
# =============================================================================
#
# Values must sum to 1.0.
# =============================================================================

DATASET_SPLITS = {
    "train": 0.80,
    "val": 0.10,
    "test": 0.10,
}

assert (
    abs(sum(DATASET_SPLITS.values()) - 1.0) < 1e-9
), "Dataset splits must sum to 1.0."

RANDOM_SEED = 42


# =============================================================================
# YOLO MODEL SETTINGS
# =============================================================================
#
# YOLO_MODEL_NAME can be changed to another Ultralytics model:
#
# yolo11n.pt
# yolo11s.pt
# yolo11m.pt
# yolo11l.pt
# yolo11x.pt
#
# YOLO_DEVICE = 0      -> first GPU
# YOLO_DEVICE = 1      -> second GPU
# YOLO_DEVICE = "cpu"  -> CPU only
# =============================================================================

YOLO_MODEL_NAME = "yolo11s.pt"

YOLO_BEST_MODEL_NAME = "best.pt"

YOLO_DEVICE = 0


# =============================================================================
# YOLO TRAINING SETTINGS
# =============================================================================

YOLO_IMAGE_SIZE = 1024

YOLO_EPOCHS = 100

YOLO_BATCH_SIZE = 16

YOLO_PATIENCE = 25

YOLO_WORKERS = 4


# =============================================================================
# YOLO DATA AUGMENTATION
# =============================================================================

YOLO_MOSAIC = 0.5

YOLO_SCALE = 0.10

YOLO_TRANSLATE = 0.10

YOLO_FLIP_LEFT_RIGHT = 0.0

YOLO_HSV_H = 0.015

YOLO_HSV_S = 0.7

YOLO_HSV_V = 0.4


# =============================================================================
# YOLO CLASS DEFINITIONS
# =============================================================================

YOLO_OBJECT_CLASSES = {
    0: "wing_tag",
}


# =============================================================================
# OCR MODEL SETTINGS
# =============================================================================

OCR_MODEL_NAME = "PP-OCRv5_server_rec"

OCR_MIN_CONFIDENCE = 0.40


# =============================================================================
# CSV COLUMN NAMES
# =============================================================================

CSV_IMAGE_FILE = "image_file"

CSV_PLATE_TEXT = "plate_text"

CSV_PLATE_COLOR = "plate_color"

CSV_TEXT_COLOR = "text_color"

CSV_X1 = "x1"

CSV_Y1 = "y1"

CSV_X2 = "x2"

CSV_Y2 = "y2"


# =============================================================================
# IMAGE FILE TYPES
# =============================================================================

SUPPORTED_IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
)


# =============================================================================
# INFERENCE SETTINGS
# =============================================================================

YOLO_CONFIDENCE_THRESHOLD = 0.25

ENABLE_COLOR_CLASSIFICATION = True

ENABLE_OCR = True


# =============================================================================
# VISUALIZATION SETTINGS
# =============================================================================

LINE_THICKNESS = 2

BOUNDING_BOX_COLOR = (0, 255, 0)

LABEL_TEXT_COLOR = (0, 0, 0)

LABEL_BACKGROUND_COLOR = (0, 255, 0)


# =============================================================================
# TAG COLOR COMBINATIONS
# =============================================================================

VALID_TAG_TYPES = [
    {
        "plate_color": "red",
        "text_color": "yellow",
    },
    {
        "plate_color": "yellow",
        "text_color": "blue",
    },
    {
        "plate_color": "yellow",
        "text_color": "black",
    },
    {
        "plate_color": "gray",
        "text_color": "black",
    },
    {
        "plate_color": "cyan",
        "text_color": "yellow",
    },
    {
        "plate_color": "cyan",
        "text_color": "blue",
    },
    {
        "plate_color": "cyan",
        "text_color": "white",
    },
    {
        "plate_color": "white",
        "text_color": "black",
    },
    {
        "plate_color": "black",
        "text_color": "white",
    },
    {
        "plate_color": "magenta",
        "text_color": "black",
    },
]

PLATE_COLORS = sorted(
    {
        tag_type["plate_color"]
        for tag_type in VALID_TAG_TYPES
    }
)

TEXT_COLORS = sorted(
    {
        tag_type["text_color"]
        for tag_type in VALID_TAG_TYPES
    }
)