"""
Utilities for reading text from wing tags using EasyOCR.
"""

from typing import Any

import cv2
import easyocr
import numpy as np

# Create the OCR model only once.
_READER = easyocr.Reader(
    ["en"],
    gpu=False,
)


def read_tag_text(
        crop: np.ndarray,
) -> dict[str, Any]:
    """
    Read the text printed on a wing tag.

    Args:
        crop:
            BGR image containing one detected wing tag.

    Returns:
        Dictionary containing:
            text
            confidence
    """

    if crop is None or crop.size == 0:
        return _unknown_result()

    processed_crop = _preprocess_crop(
        crop,
    )

    results = _READER.readtext(
        processed_crop,
        detail=1,
        paragraph=False,
    )

    if len(results) == 0:
        return _unknown_result()

    # EasyOCR returns the detections sorted by confidence.
    _, text, confidence = max(
        results,
        key=lambda item: item[2],
    )

    text = text.strip()

    if not text:
        return _unknown_result()

    return {
        "text": text,
        "confidence": float(confidence),
    }


def _preprocess_crop(
        crop: np.ndarray,
) -> np.ndarray:
    """
    Apply lightweight preprocessing before OCR.
    """

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY,
    )

    # Upscale small crops.
    gray = cv2.resize(
        gray,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC,
    )

    # Reduce JPEG artifacts.
    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
    )

    return gray


def _unknown_result() -> dict[str, Any]:
    """
    Return an empty OCR result.
    """
    return {
        "text": "",
        "confidence": 0.0,
    }
