"""
config.py

Central configuration file containing all user-configurable parameters for the project.
"""

from pathlib import Path


# =============================================================================
# PROJECT PATHS
# =============================================================================
#
# All project paths are created relative to this file.
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIRECTORY = PROJECT_ROOT / "data"

RAW_DATA_DIRECTORY = DATA_DIRECTORY / "raw"
RAW_IMAGE_DIRECTORY = RAW_DATA_DIRECTORY / "images"
ANNOTATION_FILE = RAW_DATA_DIRECTORY / "annotations.csv"

YOLO_DATA_DIRECTORY = DATA_DIRECTORY / "yolo"

YOLO_IMAGE_DIRECTORY = YOLO_DATA_DIRECTORY / "images"
YOLO_LABEL_DIRECTORY = YOLO_DATA_DIRECTORY / "labels"

YOLO_DATASET_CONFIG = YOLO_DATA_DIRECTORY / "data.yaml"

MODEL_DIRECTORY = PROJECT_ROOT / "models" / "yolo"

OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs"

PREDICTION_DIRECTORY = OUTPUT_DIRECTORY / "predictions"

RUN_DIRECTORY = PROJECT_ROOT / "runs"

PADDLEOCR_MODEL_DIRECTORY = (
    PROJECT_ROOT
    / "models"
    / "paddleocr"
)

PADDLEOCR_MODEL_PATH = (
    PADDLEOCR_MODEL_DIRECTORY
)

PADDLEOCR_CHARACTER_DICT = (
    PADDLEOCR_MODEL_DIRECTORY
    / "character_dict.txt"
)


# =============================================================================
# DATASET SETTINGS
# =============================================================================
#
# These values control how the original dataset is split into training,
# validation and testing datasets.
#
# The values should always sum to 1.0.
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
# MODEL_NAME can be changed to any Ultralytics model, such as
# yolo11n.pt
# yolo11s.pt
# yolo11m.pt
# yolo11l.pt
# yolo11x.pt
# ...
#
# DEVICE = 0        -> first GPU
# DEVICE = 1        -> second GPU
# DEVICE = "cpu"    -> CPU only
#
# Later this can easily be changed to YOLO26 or another model.
# =============================================================================

MODEL_NAME = "yolo11s.pt"

DEVICE = 0

BEST_MODEL_NAME = "best.pt"


# =============================================================================
# TRAINING SETTINGS
# =============================================================================

IMAGE_SIZE = 1024

EPOCHS = 100

BATCH_SIZE = 16

PATIENCE = 25

WORKERS = 4


# =============================================================================
# DATA AUGMENTATION
# =============================================================================
#
# Commonly changed augmentation parameters are exposed here. Add more if needed.
# =============================================================================

MOSAIC = 0.5

SCALE = 0.10

TRANSLATE = 0.10

FLIP_LEFT_RIGHT = 0.0

HSV_H = 0.015

HSV_S = 0.7

HSV_V = 0.4


# =============================================================================
# CLASS DEFINITIONS
# =============================================================================
#
# YOLO requires every object class to have a numerical ID.
# =============================================================================

OBJECT_CLASSES = {
    0: "wing_tag",
}


# =============================================================================
# CSV COLUMN NAMES
# =============================================================================
#
# The annotation CSV is expected to contain the following columns.
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
#
# Parameters controlling model inference and visualization.
# =============================================================================

CONFIDENCE_THRESHOLD = 0.25

LINE_THICKNESS = 2

BOUNDING_BOX_COLOR = (0, 255, 0)

LABEL_TEXT_COLOR = (0, 0, 0)

LABEL_BACKGROUND_COLOR = (0, 255, 0)

# Prediction pipeline
ENABLE_COLOR_CLASSIFICATION = True  # yes/no to color classification
ENABLE_OCR = True  # yes/no to text reading


# =============================================================================
# TAG COLOR COMBINATIONS
# =============================================================================

# Valid plate and text color combinations currently present in the dataset.
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