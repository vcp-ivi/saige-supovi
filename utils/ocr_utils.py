"""
Utilities for reading text from wing tags using EasyOCR.
"""

from typing import Any
from collections import defaultdict

import cv2
import easyocr
import numpy as np


# Allowed characters printed on wing tags.
OCR_ALLOWLIST = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Reject OCR results below this confidence.
OCR_MIN_SCORE = 0.40

# Print all OCR candidates for debugging.
DEBUG_OCR = False


# Create the OCR model only once.
_READER = easyocr.Reader(
    ["en"],
    gpu=True,
)


def read_tag_text(
    crop: np.ndarray,
) -> dict[str, Any]:
    """
    Read the text printed on a wing tag.

    OCR is attempted at four orientations:
        0 degrees
        90 degrees
        180 degrees
        270 degrees

    The valid candidate with the highest EasyOCR confidence
    is returned.

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

    processed_variants = _preprocess_crop(
        crop,
    )

    candidates = []

    for variant_name, processed_crop in processed_variants.items():

        for rotation in (
                0,
                90,
                180,
                270,
        ):

            rotated_crop = _rotate_image(
                processed_crop,
                rotation,
            )

            rotation_candidates = _read_single_variant(
                rotated_crop,
            )

            for candidate in rotation_candidates:
                candidate["rotation"] = rotation
                candidate["variant"] = variant_name

                candidates.append(
                    candidate,
                )

    if DEBUG_OCR:
        _print_ocr_candidates(
            candidates,
        )

    if not candidates:
        return _unknown_result()

    best_candidate = _select_best_candidate(
        candidates,
    )

    if best_candidate is None:
        return _unknown_result()

    if best_candidate["confidence"] < OCR_MIN_SCORE:
        return _unknown_result()

    return {
        "text": best_candidate["text"],
        "confidence": float(
            best_candidate["confidence"],
        ),
    }


def _select_best_candidate(
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """
    Select the best OCR result using both confidence and
    agreement across preprocessing variants.
    """

    if not candidates:
        return None

    grouped = defaultdict(list)

    for candidate in candidates:
        key = (
            candidate["text"],
            candidate["rotation"],
        )

        grouped[key].append(
            candidate["confidence"],
        )

    scored_candidates = []

    for (text, rotation), confidences in grouped.items():

        best_confidence = max(confidences)
        agreement_count = len(confidences)

        #
        # Reward agreement between preprocessing variants.
        #
        agreement_bonus = 0.15 * (
            agreement_count - 1
        )

        #
        # Very short OCR results are common false positives
        # when a vertically oriented tag is interpreted as
        # one isolated character.
        #
        if len(text) == 1:
            length_penalty = 0.20
        else:
            length_penalty = 0.0

        score = (
            best_confidence
            + agreement_bonus
            - length_penalty
        )

        scored_candidates.append(
            {
                "text": text,
                "rotation": rotation,
                "confidence": best_confidence,
                "agreement": agreement_count,
                "score": score,
            }
        )

    return max(
        scored_candidates,
        key=lambda candidate: candidate["score"],
    )


def _read_single_variant(
    image: np.ndarray,
) -> list[dict[str, Any]]:
    """
    Run EasyOCR on one image orientation.
    """

    results = _READER.readtext(
        image,
        detail=1,
        paragraph=False,
        allowlist=OCR_ALLOWLIST,
    )

    candidates = []

    for _, text, confidence in results:

        normalized_text = _normalize_ocr_text(
            text,
        )

        if not normalized_text:
            continue

        candidates.append(
            {
                "text": normalized_text,
                "confidence": float(confidence),
            }
        )

    return candidates


def _normalize_ocr_text(
    text: str,
) -> str:
    """
    Normalize OCR output.

    Text is converted to uppercase and all characters
    outside the configured wing-tag alphabet are removed.
    """

    text = text.strip().upper()

    normalized_text = "".join(
        character
        for character in text
        if character in OCR_ALLOWLIST
    )

    return normalized_text


def _rotate_image(
    image: np.ndarray,
    rotation: int,
) -> np.ndarray:
    """
    Rotate an image by 0, 90, 180 or 270 degrees.
    """

    if rotation == 0:
        return image

    if rotation == 90:
        return cv2.rotate(
            image,
            cv2.ROTATE_90_CLOCKWISE,
        )

    if rotation == 180:
        return cv2.rotate(
            image,
            cv2.ROTATE_180,
        )

    if rotation == 270:
        return cv2.rotate(
            image,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
        )

    raise ValueError(
        f"Unsupported rotation: {rotation}"
    )


def _preprocess_crop(
    crop: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Create several preprocessing variants for OCR.
    """

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY,
    )

    #
    # Upscale first.
    #
    gray = cv2.resize(
        gray,
        None,
        fx=6,
        fy=6,
        interpolation=cv2.INTER_CUBIC,
    )

    variants = {
        "gray": gray,
    }

    #
    # CLAHE contrast enhancement.
    #
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(4, 4),
    )

    enhanced = clahe.apply(
        gray,
    )

    variants["clahe"] = enhanced

    #
    # Otsu threshold.
    #
    _, otsu = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    variants["otsu"] = otsu

    #
    # Adaptive threshold.
    #
    adaptive = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5,
    )

    variants["adaptive"] = adaptive

    dark_text = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        21,
        7,
    )

    variants["dark_text"] = dark_text

    variants["dark_text_inv"] = cv2.bitwise_not(
        dark_text,
    )

    return variants


def _print_ocr_candidates(
    candidates: list[dict[str, Any]],
) -> None:
    """
    Print all OCR candidates sorted by confidence.
    """

    print()
    print("OCR candidates")
    print("-" * 40)

    if not candidates:
        print("No OCR candidates.")
        return

    for candidate in sorted(
        candidates,
        key=lambda item: item["confidence"],
        reverse=True,
    ):
        print(
            f'{candidate["variant"]:<10} '
            f'{candidate["rotation"]:>3}°  '
            f'{candidate["text"]:<8} '
            f'{candidate["confidence"]:.3f}'
        )


def _unknown_result() -> dict[str, Any]:
    """
    Return an empty OCR result.
    """

    return {
        "text": "",
        "confidence": 0.0,
    }