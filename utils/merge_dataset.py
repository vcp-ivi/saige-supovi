"""
merge_dataset.py

Merge a newly received image folder and its annotations into the
project's existing raw dataset.
"""

import argparse
import csv
import shutil
import tempfile
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

RAW_DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw"
RAW_IMAGE_DIRECTORY = RAW_DATA_DIRECTORY / "images"
RAW_ANNOTATION_FILE = RAW_DATA_DIRECTORY / "annotations.csv"

ANNOTATION_FILE_NAME = "annotations.csv"

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
}

SKIPPED_IMAGE_EXTENSIONS = {
    ".bmp",
    ".xls",
    ".xlsx",
}


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Merge new images and annotations into the existing "
            "raw dataset."
        ),
    )

    parser.add_argument(
        "new_data_directory",
        type=Path,
        help=(
            "Directory containing the new images and annotations.csv."
        ),
    )

    return parser.parse_args()


def merge_raw_data(
    new_data_directory: Path,
) -> None:
    """
    Merge a folder of new images and annotations into the raw dataset.
    """

    new_data_directory = new_data_directory.resolve()

    _validate_directories(
        new_data_directory,
    )

    incoming_annotation_file = (
        new_data_directory / ANNOTATION_FILE_NAME
    )

    if not incoming_annotation_file.is_file():
        raise FileNotFoundError(
            "Incoming annotation file not found:\n"
            f"{incoming_annotation_file}"
        )

    incoming_images, skipped_images = _inspect_incoming_files(
        new_data_directory,
        incoming_annotation_file,
    )

    existing_fieldnames, existing_rows = _read_annotations(
        RAW_ANNOTATION_FILE,
        allow_missing=True,
    )

    incoming_fieldnames, incoming_rows = _read_annotations(
        incoming_annotation_file,
        allow_missing=False,
    )

    _validate_annotation_columns(
        existing_fieldnames=existing_fieldnames,
        incoming_fieldnames=incoming_fieldnames,
    )

    _validate_annotation_images(
        incoming_rows=incoming_rows,
        incoming_images=incoming_images,
    )

    rename_map = _build_rename_map(
        incoming_images,
    )

    updated_rows = _update_annotation_filenames(
        incoming_rows=incoming_rows,
        rename_map=rename_map,
    )

    _merge_files_transactionally(
        incoming_images=incoming_images,
        rename_map=rename_map,
        existing_rows=existing_rows,
        incoming_rows=updated_rows,
        fieldnames=incoming_fieldnames,
    )

    _print_summary(
        incoming_images=incoming_images,
        skipped_images=skipped_images,
        rename_map=rename_map,
        added_annotation_count=len(updated_rows),
    )


def _validate_directories(
    new_data_directory: Path,
) -> None:
    """
    Validate the source and destination directories.
    """

    if not new_data_directory.is_dir():
        raise NotADirectoryError(
            "New data directory not found:\n"
            f"{new_data_directory}"
        )

    if new_data_directory == RAW_DATA_DIRECTORY.resolve():
        raise ValueError(
            "The incoming folder cannot be the existing raw-data folder."
        )

    RAW_IMAGE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )


def _inspect_incoming_files(
    new_data_directory: Path,
    incoming_annotation_file: Path,
) -> tuple[list[Path], list[Path]]:
    """
    Identify supported images, skipped BMP images and unsupported files.

    Unsupported file types stop the process before any data is changed.
    """

    incoming_images = []
    skipped_images = []
    unsupported_files = []

    for path in new_data_directory.iterdir():
        if not path.is_file():
            continue

        if path == incoming_annotation_file:
            continue

        extension = path.suffix.lower()

        if extension in SUPPORTED_IMAGE_EXTENSIONS:
            incoming_images.append(path)

        elif extension in SKIPPED_IMAGE_EXTENSIONS:
            skipped_images.append(path)

        else:
            unsupported_files.append(path)

    if unsupported_files:
        detected_extensions = sorted(
            {
                path.suffix.lower() or "<no extension>"
                for path in unsupported_files
            }
        )

        extension_text = ", ".join(
            detected_extensions
        )

        file_text = "\n".join(
            f"  - {path.name}"
            for path in sorted(unsupported_files)
        )

        raise ValueError(
            "Merge stopped before any files were changed.\n\n"
            f"Note: {extension_text} file type detected.\n\n"
            "Unsupported files:\n"
            f"{file_text}"
        )

    if not incoming_images:
        raise ValueError(
            "No supported .jpg images were found in:\n"
            f"{new_data_directory}"
        )

    return (
        sorted(
            incoming_images,
            key=lambda path: path.name.lower(),
        ),
        sorted(
            skipped_images,
            key=lambda path: path.name.lower(),
        ),
    )


def _read_annotations(
    annotation_file: Path,
    allow_missing: bool,
) -> tuple[list[str], list[dict[str, str]]]:
    """
    Read an annotation CSV file.
    """

    if not annotation_file.is_file():
        if allow_missing:
            return [], []

        raise FileNotFoundError(
            "Annotation file not found:\n"
            f"{annotation_file}"
        )

    with annotation_file.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(
                "Annotation file has no header:\n"
                f"{annotation_file}"
            )

        fieldnames = [
            fieldname.strip()
            for fieldname in reader.fieldnames
        ]

        rows = []

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            normalized_row = {
                key.strip(): value
                for key, value in row.items()
                if key is not None
            }

            if None in row:
                raise ValueError(
                    "Malformed annotation row "
                    f"{row_number} in:\n{annotation_file}"
                )

            rows.append(
                normalized_row
            )

    if "image_file" not in fieldnames:
        raise ValueError(
            "Required column 'image_file' was not found in:\n"
            f"{annotation_file}"
        )

    return fieldnames, rows


def _validate_annotation_columns(
    existing_fieldnames: list[str],
    incoming_fieldnames: list[str],
) -> None:
    """
    Ensure the existing and incoming CSV files use the same columns.
    """

    if not existing_fieldnames:
        return

    if existing_fieldnames != incoming_fieldnames:
        raise ValueError(
            "The incoming annotations do not have the same columns "
            "as the existing annotations.\n\n"
            f"Existing columns:\n{existing_fieldnames}\n\n"
            f"Incoming columns:\n{incoming_fieldnames}"
        )


def _validate_annotation_images(
    incoming_rows: list[dict[str, str]],
    incoming_images: list[Path],
) -> None:
    """
    Ensure every image referenced by the incoming CSV is available.

    References to skipped BMP files are ignored because those images and
    their annotation rows will not be imported.
    """

    available_images = {
        path.name.lower(): path.name
        for path in incoming_images
    }

    missing_images = set()

    for row in incoming_rows:
        image_name = row["image_file"].strip()
        extension = Path(image_name).suffix.lower()

        if extension in SKIPPED_IMAGE_EXTENSIONS:
            continue

        if image_name.lower() not in available_images:
            missing_images.add(
                image_name
            )

    if missing_images:
        missing_text = "\n".join(
            f"  - {name}"
            for name in sorted(missing_images)
        )

        raise ValueError(
            "Some images referenced in annotations.csv were not found "
            "in the incoming folder.\n\n"
            f"{missing_text}"
        )


def _build_rename_map(
    incoming_images: list[Path],
) -> dict[str, str]:
    """
    Determine the destination name of every incoming image.

    Existing names are compared case-insensitively to match Windows file
    system behavior.
    """

    occupied_names = {
        path.name.lower()
        for path in RAW_IMAGE_DIRECTORY.iterdir()
        if path.is_file()
    }

    rename_map = {}

    for source_path in incoming_images:
        destination_name = _find_available_filename(
            original_name=source_path.name,
            occupied_names=occupied_names,
        )

        rename_map[source_path.name] = destination_name
        occupied_names.add(
            destination_name.lower()
        )

    return rename_map


def _find_available_filename(
    original_name: str,
    occupied_names: set[str],
) -> str:
    """
    Return the original name or a numbered alternative.

    Example:
        123.jpg
        123-1.jpg
        123-2.jpg
    """

    if original_name.lower() not in occupied_names:
        return original_name

    original_path = Path(original_name)
    stem = original_path.stem
    extension = original_path.suffix

    counter = 1

    while True:
        candidate = f"{stem}-{counter}{extension}"

        if candidate.lower() not in occupied_names:
            return candidate

        counter += 1


def _update_annotation_filenames(
    incoming_rows: list[dict[str, str]],
    rename_map: dict[str, str],
) -> list[dict[str, str]]:
    """
    Update image filenames in incoming annotation rows.

    Rows that refer to BMP files are intentionally omitted.
    """

    case_insensitive_rename_map = {
        source_name.lower(): destination_name
        for source_name, destination_name in rename_map.items()
    }

    updated_rows = []

    for row in incoming_rows:
        source_name = row["image_file"].strip()
        extension = Path(source_name).suffix.lower()

        if extension in SKIPPED_IMAGE_EXTENSIONS:
            continue

        destination_name = case_insensitive_rename_map.get(
            source_name.lower()
        )

        if destination_name is None:
            raise ValueError(
                "No destination filename was prepared for annotated "
                f"image: {source_name}"
            )

        updated_row = row.copy()
        updated_row["image_file"] = destination_name

        updated_rows.append(
            updated_row
        )

    return updated_rows


def _merge_files_transactionally(
    incoming_images: list[Path],
    rename_map: dict[str, str],
    existing_rows: list[dict[str, str]],
    incoming_rows: list[dict[str, str]],
    fieldnames: list[str],
) -> None:
    """
    Prepare all files first, then apply the merge.

    Temporary files are used so that a CSV-writing failure does not leave
    a partially written annotation file.
    """

    RAW_DATA_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory(
        dir=RAW_DATA_DIRECTORY,
        prefix=".merge_",
    ) as temporary_directory:
        temporary_directory = Path(
            temporary_directory
        )

        staged_image_directory = (
            temporary_directory / "images"
        )

        staged_image_directory.mkdir()

        for source_path in incoming_images:
            destination_name = rename_map[
                source_path.name
            ]

            shutil.copy2(
                source_path,
                staged_image_directory / destination_name,
            )

        temporary_annotation_file = (
            temporary_directory / "annotations.csv"
        )

        _write_annotations(
            annotation_file=temporary_annotation_file,
            fieldnames=fieldnames,
            rows=existing_rows + incoming_rows,
        )

        copied_destinations = []

        try:
            for staged_path in staged_image_directory.iterdir():
                destination_path = (
                    RAW_IMAGE_DIRECTORY / staged_path.name
                )

                shutil.move(
                    str(staged_path),
                    str(destination_path),
                )

                copied_destinations.append(
                    destination_path
                )

            shutil.move(
                str(temporary_annotation_file),
                str(RAW_ANNOTATION_FILE),
            )

        except Exception:
            for destination_path in copied_destinations:
                destination_path.unlink(
                    missing_ok=True,
                )

            raise


def _write_annotations(
    annotation_file: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    """
    Write annotation rows to a CSV file.
    """

    with annotation_file.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def _print_summary(
    incoming_images: list[Path],
    skipped_images: list[Path],
    rename_map: dict[str, str],
    added_annotation_count: int,
) -> None:
    """
    Print a summary of the completed merge.
    """

    renamed_images = {
        source_name: destination_name
        for source_name, destination_name in rename_map.items()
        if source_name != destination_name
    }

    print("\nRaw-data merge completed.")
    print("-" * 40)
    print(
        f"JPG images copied       : {len(incoming_images)}"
    )
    print(
        f"Annotation rows added   : {added_annotation_count}"
    )
    print(
        f"Files skipped           : {len(skipped_images)}"
    )
    print(
        f"Duplicate names changed : {len(renamed_images)}"
    )

    if skipped_images:
        print("\nSkipped files:")

        for path in skipped_images:
            print(
                f"  - {path.name}"
            )

    if renamed_images:
        print("\nRenamed images:")

        for source_name, destination_name in renamed_images.items():
            print(
                f"  - {source_name} -> {destination_name}"
            )

    print(
        f"\nImages saved to:\n{RAW_IMAGE_DIRECTORY}"
    )
    print(
        f"\nAnnotations updated:\n{RAW_ANNOTATION_FILE}"
    )


def main() -> None:
    """
    Run the raw-data merge utility.
    """

    arguments = parse_arguments()

    try:
        merge_raw_data(
            arguments.new_data_directory,
        )

    except (FileNotFoundError, NotADirectoryError, ValueError) as error:
        print(f"\nERROR\n{'-' * 40}\n{error}")
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()