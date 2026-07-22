"""
color_utils.py

Utilities for identifying wing-tag plate and text colors.

The module classifies the plate and text colors of a cropped wing tag.

YOLO crop
      │
      ▼
Crop away image borders
      │
      ▼
Convert to HSV
      │
      ▼
Determine plate color
      │
      ▼
Find largest connected component
      │
      ▼
Reconstruct plate (convex hull)
      │
      ▼
Determine text color
      │
      ▼
Compute confidence
      │
      ▼
Return result
"""

from typing import Any

import cv2
import numpy as np

from config import PLATE_COLORS, VALID_TAG_TYPES


DEBUG_COLOR_DETECTION = False


# OpenCV HSV ranges:
# Hue:        0–179
# Saturation: 0–255
# Value:      0–255
HSV_COLOR_RANGES = {
    "red": [
        ((0, 80, 50), (12, 255, 255)),
        ((165, 80, 50), (179, 255, 255)),
    ],
    "yellow": [
        ((13, 70, 70), (42, 255, 255)),
    ],
    "blue": [
        ((80, 60, 40), (135, 255, 255)),
    ],
    "black": [
        ((0, 0, 0), (179, 70, 50)),
    ],
    "white": [
        ((0, 0, 140), (179, 80, 255)),
    ],
}


def detect_tag_colors(crop: np.ndarray) -> dict[str, Any]:
    """
    Identify the plate and text colors of a cropped wing tag.
    """

    if crop is None or crop.size == 0:
        return _unknown_result()

    working_crop, _ = _crop_center_region(crop)

    if working_crop.size == 0:
        return _unknown_result()

    hsv_crop = cv2.cvtColor(
        working_crop,
        cv2.COLOR_BGR2HSV,
    )

    #
    # -------- Plate --------
    #

    plate_result = _find_plate_color(hsv_crop)

    if plate_result is None:
        empty_mask = np.zeros(
            hsv_crop.shape[:2],
            dtype=np.uint8,
        )

        _show_debug_images(
            crop=working_crop,
            plate_mask=empty_mask,
            reconstructed_plate=empty_mask,
            text_mask=empty_mask,
            plate_color="unknown",
            text_color="unknown",
            plate_counts={},
            text_counts={},
            plate_score=0.0,
            text_score=0.0,
            color_confidence=0.0,
        )

        return _unknown_result()

    plate_color = plate_result["color"]
    plate_mask = plate_result["mask"]
    plate_score = plate_result["score"]
    plate_counts = plate_result["counts"]

    plate_component = _find_largest_component(
        plate_mask,
    )

    reconstructed_plate = _reconstruct_plate_mask(
        plate_component,
        reference_mask=plate_mask,
    )

    #
    # -------- Text --------
    #

    text_result = _find_text_color(
        hsv_crop=hsv_crop,
        plate_color=plate_color,
        plate_mask=reconstructed_plate,
    )

    if text_result is None:
        text_color = "unknown"
        text_score = 0.0
        text_counts = {}

        text_mask = np.zeros_like(
            plate_mask,
        )
    else:
        text_color = text_result["color"]
        text_score = text_result["score"]
        text_counts = text_result["counts"]
        text_mask = text_result["mask"]

    #
    # -------- Confidence --------
    #

    color_confidence = plate_score * text_score

    #
    # -------- Debug --------
    #

    _show_debug_images(
        crop=working_crop,
        plate_mask=plate_mask,
        reconstructed_plate=reconstructed_plate,
        text_mask=text_mask,
        plate_color=plate_color,
        text_color=text_color,
        plate_counts=plate_counts,
        text_counts=text_counts,
        plate_score=plate_score,
        text_score=text_score,
        color_confidence=color_confidence,
    )

    #
    # -------- Return --------
    #

    if text_result is None:
        return _unknown_result()

    return {
        "plate_color": plate_color,
        "text_color": text_color,
        "color_confidence": float(color_confidence),
    }


def _find_plate_color(
    hsv_crop: np.ndarray,
) -> dict[str, Any] | None:
    """
    Determine which valid plate color occupies the largest part of the crop.
    """

    color_masks = {}
    color_counts = {}

    for color_name in PLATE_COLORS:
        mask = _create_color_mask(
            hsv_crop,
            color_name,
        )

        color_masks[color_name] = mask
        color_counts[color_name] = cv2.countNonZero(
            mask,
        )

    if not color_counts:
        return None

    plate_color = max(
        color_counts,
        key=color_counts.get,
    )

    winning_count = color_counts[plate_color]

    if winning_count == 0:
        return None

    plate_score = _calculate_pixel_fraction(
        pixel_count=winning_count,
        image_shape=hsv_crop.shape[:2],
    )

    return {
        "color": plate_color,
        "mask": color_masks[plate_color],
        "score": plate_score,
        "counts": color_counts,
    }


def _find_text_color(
    hsv_crop: np.ndarray,
    plate_color: str,
    plate_mask: np.ndarray,
) -> dict[str, Any] | None:
    """
    Determine the most likely text color inside the isolated plate.
    """

    valid_text_colors = _valid_text_colors_for_plate(
        plate_color,
    )

    if not valid_text_colors:
        return None

    if plate_mask is None or cv2.countNonZero(plate_mask) == 0:
        return None

    color_masks = {}
    color_counts = {}

    for color_name in valid_text_colors:
        full_color_mask = _create_color_mask(
            hsv_crop,
            color_name,
        )

        mask_inside_plate = cv2.bitwise_and(
            full_color_mask,
            plate_mask,
        )

        color_masks[color_name] = mask_inside_plate
        color_counts[color_name] = cv2.countNonZero(
            mask_inside_plate,
        )

    if not color_counts:
        return None

    text_color = max(
        color_counts,
        key=color_counts.get,
    )

    winning_count = color_counts[text_color]

    if winning_count == 0:
        return None

    plate_pixel_count = cv2.countNonZero(
        plate_mask,
    )

    text_score = _calculate_fraction(
        numerator=winning_count,
        denominator=plate_pixel_count,
    )

    return {
        "color": text_color,
        "mask": color_masks[text_color],
        "score": text_score,
        "counts": color_counts,
    }


def _create_color_mask(
    hsv_image: np.ndarray,
    color_name: str,
) -> np.ndarray:
    """
    Create a binary mask for one configured color.
    """

    if color_name not in HSV_COLOR_RANGES:
        raise ValueError(
            f"Unsupported color: {color_name}"
        )

    mask = np.zeros(
        hsv_image.shape[:2],
        dtype=np.uint8,
    )

    for lower, upper in HSV_COLOR_RANGES[color_name]:
        lower_array = np.array(
            lower,
            dtype=np.uint8,
        )

        upper_array = np.array(
            upper,
            dtype=np.uint8,
        )

        range_mask = cv2.inRange(
            hsv_image,
            lower_array,
            upper_array,
        )

        mask = cv2.bitwise_or(
            mask,
            range_mask,
        )

    return mask


def _find_largest_component(
    mask: np.ndarray,
) -> np.ndarray | None:
    """
    Return a mask containing the largest connected component.
    """

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None

    largest_contour = max(
        contours,
        key=cv2.contourArea,
    )

    if cv2.contourArea(largest_contour) <= 0:
        return None

    component_mask = np.zeros_like(
        mask,
    )

    cv2.drawContours(
        component_mask,
        [largest_contour],
        contourIdx=-1,
        color=255,
        thickness=cv2.FILLED,
    )

    return component_mask


def _reconstruct_plate_mask(
    component_mask: np.ndarray | None,
    reference_mask: np.ndarray,
) -> np.ndarray:
    """
    Fill gaps in the plate region using its convex hull.

    Return an empty mask when no plate component was identified.
    """

    if component_mask is None:
        return np.zeros_like(
            reference_mask,
        )

    contours, _ = cv2.findContours(
        component_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return np.zeros_like(
            reference_mask,
        )

    largest_contour = max(
        contours,
        key=cv2.contourArea,
    )

    hull = cv2.convexHull(
        largest_contour,
    )

    reconstructed = np.zeros_like(
        reference_mask,
    )

    cv2.drawContours(
        reconstructed,
        [hull],
        contourIdx=-1,
        color=255,
        thickness=cv2.FILLED,
    )

    return reconstructed


def _calculate_pixel_fraction(
    pixel_count: int,
    image_shape: tuple[int, int],
) -> float:
    """
    Calculate the fraction of image pixels represented by a count.
    """

    total_pixels = image_shape[0] * image_shape[1]

    return _calculate_fraction(
        numerator=pixel_count,
        denominator=total_pixels,
    )


def _calculate_fraction(
    numerator: int,
    denominator: int,
) -> float:
    """
    Safely calculate and constrain a fraction to the range 0–1.
    """

    if denominator <= 0:
        return 0.0

    score = numerator / denominator

    return float(
        np.clip(
            score,
            0.0,
            1.0,
        )
    )


def _valid_text_colors_for_plate(
    plate_color: str,
) -> list[str]:
    """
    Return valid text colors for one plate color.
    """

    return list(
        dict.fromkeys(
            tag_type["text_color"]
            for tag_type in VALID_TAG_TYPES
            if tag_type["plate_color"] == plate_color
        )
    )


def _unknown_result() -> dict[str, Any]:
    """
    Return an unknown color result.
    """

    return {
        "plate_color": "unknown",
        "text_color": "unknown",
        "color_confidence": 0.0,
    }


def _crop_center_region(
    image: np.ndarray,
    margin_ratio: float = 0.15,
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Return the central part of the crop.

    The outer border often contains feathers or background,
    while the wing tag is usually near the center.
    """

    if not 0.0 <= margin_ratio < 0.5:
        raise ValueError(
            "margin_ratio must be between 0.0 and 0.5."
        )

    height, width = image.shape[:2]

    margin_x = int(width * margin_ratio)
    margin_y = int(height * margin_ratio)

    cropped = image[
        margin_y:height - margin_y,
        margin_x:width - margin_x,
    ]

    return cropped, (margin_x, margin_y)


def _show_debug_images(
    crop: np.ndarray,
    plate_mask: np.ndarray,
    reconstructed_plate: np.ndarray,
    text_mask: np.ndarray,
    plate_color: str,
    text_color: str,
    plate_counts: dict[str, int],
    text_counts: dict[str, int],
    plate_score: float,
    text_score: float,
    color_confidence: float,
) -> None:
    """
    Display intermediate color-detection results.
    """

    if not DEBUG_COLOR_DETECTION:
        return

    hsv_crop = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2HSV,
    )

    _show_debug_windows(
        crop=crop,
        hsv_crop=hsv_crop,
        plate_mask=plate_mask,
        reconstructed_plate=reconstructed_plate,
        text_mask=text_mask,
        plate_color=plate_color,
        text_color=text_color,
    )

    _print_debug_information(
        plate_color=plate_color,
        text_color=text_color,
        plate_counts=plate_counts,
        text_counts=text_counts,
        plate_score=plate_score,
        text_score=text_score,
        color_confidence=color_confidence,
    )

    print(
        "\nClick on the crop to inspect HSV values."
    )
    print(
        "Press any key to continue..."
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _show_debug_windows(
    crop: np.ndarray,
    hsv_crop: np.ndarray,
    plate_mask: np.ndarray,
    reconstructed_plate: np.ndarray,
    text_mask: np.ndarray,
    plate_color: str,
    text_color: str,
) -> None:
    """
    Open the color-detection debug windows.
    """

    crop_window_name = (
        f"Crop ({plate_color}/{text_color})"
    )

    cv2.imshow(
        crop_window_name,
        crop,
    )

    cv2.setMouseCallback(
        crop_window_name,
        _inspect_clicked_pixel,
        {
            "hsv": hsv_crop,
            "bgr": crop,
        },
    )

    cv2.imshow(
        f"Plate mask ({plate_color})",
        plate_mask,
    )

    cv2.imshow(
        f"Reconstructed plate ({plate_color})",
        reconstructed_plate,
    )

    cv2.imshow(
        f"Text mask ({text_color})",
        text_mask,
    )


def _inspect_clicked_pixel(
    event: int,
    x: int,
    y: int,
    flags: int,
    parameters: dict[str, np.ndarray],
) -> None:
    """
    Print average HSV and BGR values around a clicked pixel.
    """

    del flags

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    hsv_image = parameters["hsv"]
    bgr_image = parameters["bgr"]

    radius = 2

    x1 = max(0, x - radius)
    x2 = min(
        hsv_image.shape[1],
        x + radius + 1,
    )

    y1 = max(0, y - radius)
    y2 = min(
        hsv_image.shape[0],
        y + radius + 1,
    )

    hsv_patch = hsv_image[
        y1:y2,
        x1:x2,
    ]

    bgr_patch = bgr_image[
        y1:y2,
        x1:x2,
    ]

    if hsv_patch.size == 0 or bgr_patch.size == 0:
        return

    hsv_mean = hsv_patch.mean(
        axis=(0, 1),
    )

    bgr_mean = bgr_patch.mean(
        axis=(0, 1),
    )

    print(
        f"\nClicked ({x}, {y})"
    )
    print(
        f"Patch: {x1}:{x2}, {y1}:{y2}"
    )
    print(
        "Average HSV = "
        f"({hsv_mean[0]:.1f}, "
        f"{hsv_mean[1]:.1f}, "
        f"{hsv_mean[2]:.1f})"
    )
    print(
        "Average BGR = "
        f"({bgr_mean[0]:.1f}, "
        f"{bgr_mean[1]:.1f}, "
        f"{bgr_mean[2]:.1f})"
    )
    print(
        "H range: "
        f"{hsv_patch[:, :, 0].min()} - "
        f"{hsv_patch[:, :, 0].max()}"
    )
    print(
        "S range: "
        f"{hsv_patch[:, :, 1].min()} - "
        f"{hsv_patch[:, :, 1].max()}"
    )
    print(
        "V range: "
        f"{hsv_patch[:, :, 2].min()} - "
        f"{hsv_patch[:, :, 2].max()}"
    )


def _print_debug_information(
    plate_color: str,
    text_color: str,
    plate_counts: dict[str, int],
    text_counts: dict[str, int],
    plate_score: float,
    text_score: float,
    color_confidence: float,
) -> None:
    """
    Print intermediate color-detection values.
    """

    _print_color_counts(
        heading="Plate color counts",
        color_counts=plate_counts,
    )

    _print_color_counts(
        heading="Text color counts",
        color_counts=text_counts,
    )

    print("\nSelected result")
    print("-" * 30)
    print(
        f"Plate color     : {plate_color}"
    )
    print(
        f"Text color      : {text_color}"
    )
    print(
        f"Plate score     : {plate_score:.3f}"
    )
    print(
        f"Text score      : {text_score:.3f}"
    )
    print(
        f"Color confidence: {color_confidence:.3f}"
    )


def _print_color_counts(
    heading: str,
    color_counts: dict[str, int],
) -> None:
    """
    Print candidate colors and their corresponding pixel counts.
    """

    print(f"\n{heading}")
    print("-" * 30)

    if not color_counts:
        print("No color candidates.")
        return

    for color_name, pixel_count in sorted(
        color_counts.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(
            f"{color_name:>8}: "
            f"{pixel_count} pixels"
        )