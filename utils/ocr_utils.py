"""
ocr_utils.py

OCR utilities using the fine-tuned PaddleOCR wing-tag model.
"""

import cv2

import ssl
import certifi

_original_create_default_context = ssl.create_default_context


def _create_default_context_with_certifi(
    purpose=ssl.Purpose.SERVER_AUTH,
    *,
    cafile=None,
    capath=None,
    cadata=None,
):
    if (
        cafile is None
        and capath is None
        and cadata is None
    ):
        cafile = certifi.where()

    return _original_create_default_context(
        purpose=purpose,
        cafile=cafile,
        capath=capath,
        cadata=cadata,
    )


ssl.create_default_context = _create_default_context_with_certifi

from paddleocr import TextRecognition

import config


DEBUG_OCR = False


_READER = TextRecognition(
    model_name=config.OCR_MODEL_NAME,
    model_dir=str(
        config.OCR_MODEL_DIRECTORY
    ),
)


def _rotate_image(image, angle):
    """Rotate image by a multiple of 90 degrees."""

    if angle == 0:
        return image

    if angle == 90:
        return cv2.rotate(
            image,
            cv2.ROTATE_90_CLOCKWISE,
        )

    if angle == 180:
        return cv2.rotate(
            image,
            cv2.ROTATE_180,
        )

    if angle == 270:
        return cv2.rotate(
            image,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
        )

    raise ValueError(
        f"Unsupported rotation: {angle}"
    )


def _read_single_orientation(image):
    """Run PaddleOCR recognition on one orientation."""

    results = _READER.predict(
        input=image,
        batch_size=1,
    )

    for result in results:

        result_dict = result.json

        text = (
            result_dict["res"]
            .get(
                "rec_text",
                "",
            )
            .strip()
            .upper()
        )

        confidence = float(
            result_dict["res"]
            .get(
                "rec_score",
                0.0,
            )
        )

        return text, confidence

    return "", 0.0


def read_tag_text(crop):
    """
    Recognize text on a wing-tag crop.

    The crop is evaluated at four rotations and the prediction with
    the highest confidence is returned.
    """

    candidates = []

    for angle in (
        0,
        90,
        180,
        270,
    ):

        rotated_crop = _rotate_image(
            crop,
            angle,
        )

        text, confidence = (
            _read_single_orientation(
                rotated_crop
            )
        )

        if DEBUG_OCR:

            print(
                f"OCR {angle:3d}°: "
                f"{text!r} "
                f"{confidence:.3f}"
            )

        if text:

            candidates.append(
                {
                    "text": text,
                    "confidence": confidence,
                    "angle": angle,
                }
            )

    if not candidates:

        return {
            "text": "",
            "confidence": 0.0,
        }

    best = max(
        candidates,
        key=lambda candidate: (
            candidate["confidence"]
        ),
    )

    if (
        best["confidence"]
        < config.OCR_MIN_CONFIDENCE
    ):

        return {
            "text": "",
            "confidence": (
                best["confidence"]
            ),
        }

    return {
        "text": best["text"],
        "confidence": (
            best["confidence"]
        ),
    }