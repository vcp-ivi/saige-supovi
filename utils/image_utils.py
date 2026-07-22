"""
image_utils.py

Utility functions for working with image files.
"""

from pathlib import Path

import config


def get_images(image_directory):
    """Return all supported images in a directory."""

    image_directory = Path(image_directory)

    image_paths = []

    for extension in config.SUPPORTED_IMAGE_EXTENSIONS:
        image_paths.extend(
            image_directory.glob(f"*{extension}")
        )

    return sorted(image_paths)