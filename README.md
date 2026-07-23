# Wing Tag Detection and Recognition Pipeline

A computer vision pipeline for detecting griffon vulture wing tags, identifying their colors, and reading the alphanumeric code printed on each tag.

The project combines:

- **YOLO** for wing tag detection
- **OpenCV** for wing tag color classification
- **EasyOCR** for optical character recognition (OCR)

---

# Pipeline Overview

```
Input Image
      │
      ▼
YOLO
Wing Tag Detection
      │
      ▼
Crop Each Detected Tag
      │
      ├──────────────► Color Detection (OpenCV)
      │
      └──────────────► OCR (EasyOCR)
      │
      ▼
Final Prediction
```

For every detected wing tag, the pipeline returns:

- Bounding box
- Detection confidence
- Plate color
- Text color
- Color confidence
- Recognized text
- OCR confidence

---

# Project Structure

```
project/
│
├── config.py
├── prepare_dataset.py
├── train.py
├── predict.py
├── visualize_dataset.py
├── merge_dataset.py
├── dataset_statistics.py
│
├── color_utils.py
├── ocr_utils.py
├── image_utils.py
│
├── data/
│   ├── raw/
│   └── yolo/
│
├── outputs/
├── prediction/
└── runs/
```

---

# Script Overview

| Script | Purpose |
|---------------------------|---------------------------|
| **merge_dataset.py** | Merges a new raw dataset into the existing raw dataset, automatically renaming duplicate images and updating annotations. |
| **dataset_statistics.py** | Displays summary statistics of the raw annotation dataset. |
| **prepare_dataset.py** | Converts the raw dataset into the YOLO format, creates the train/validation/test split and generates `data.yaml`. |
| **visualize_dataset.py** | Displays annotated images from the YOLO dataset for inspection and quality control. |
| **train.py** | Trains a YOLO detector using the YOLO dataset and copies the best model to the output directory. |
| **predict.py** | Runs the complete inference pipeline on a single image or an entire directory. |
| **color_utils.py** | Determines wing tag plate and text colors using classical computer vision techniques. |
| **ocr_utils.py** | Reads the wing tag identifier using EasyOCR. |
| **image_utils.py** | Helper functions for loading and enumerating image files. |
| **config.py** | Central configuration file containing paths, model parameters and project settings. |

---

# Raw Dataset

The repository stores the original images and annotation file inside

```
data/raw/
```

The raw dataset consists of:

- original images
- `annotations.csv`

### Dataset Merging

Merge a new annotated dataset into the existing raw dataset:

```bash
python merge_dataset.py path/to/new_dataset
```

The input directory must contain the images together with an `annotations.csv` file.

The script automatically:

- copies supported images into `data/raw/images`
- appends annotations to `data/raw/annotations.csv`
- renames duplicate image filenames and updates the corresponding annotations

---

### Dataset Statistics

Display summary statistics for the raw dataset:

```bash
python dataset_statistics.py
```

The script reports:

- plate/text color combinations and their frequencies
- unique wing tag identifiers and their frequencies
- identifier frequencies grouped by plate/text color combination

---

### Dataset Preparation

Convert the raw dataset into the YOLO dataset:

```bash
python prepare_dataset.py
```

This script automatically:

- creates the train/validation/test split
- converts annotations to the YOLO format
- generates label files
- creates the YOLO `data.yaml` configuration file

---

# YOLO Dataset

After running `prepare_dataset.py`, the generated YOLO dataset is stored in

```
data/yolo/
```

### Dataset Visualization

Inspect random 5 images:

```bash
python visualize_dataset.py
```

Inspect a specific image:

```bash
python visualize_dataset.py --image IMG_0123.jpg
```

Inspect multiple random images:

```bash
python visualize_dataset.py --num 20
```

---

# Training

Train the detector:

```bash
python train.py
```

The best-performing model is automatically copied to

```
outputs/models/best.pt
```

The training configuration (model, image size, epochs, batch size, etc.) is defined in `config.py`.

---

# Prediction

Run inference on a single image:

```bash
python predict.py --image path/to/image.jpg
```

Run inference on an entire directory:

```bash
python predict.py --directory path/to/images
```

Annotated images are saved to

```
prediction/
```

---

# Prediction Pipeline

The prediction pipeline consists of four consecutive stages.

## 1. Wing Tag Detection

The input image is processed by a YOLO object detector trained to identify griffon vulture wing tags.

For every detected tag, the detector returns:

- bounding box
- confidence score
- class label

Each detected tag is cropped and processed independently by the remaining stages of the pipeline.

---

## 2. Color Detection (`color_utils.py`)

The cropped wing tag is analyzed using classical computer vision techniques.

The processing steps are:

1. Crop the central region of the detection to reduce background.
2. Convert the image from RGB to HSV.
3. Threshold the image using predefined HSV color ranges.
4. Determine the dominant wing tag plate color.
5. Extract the largest connected component.
6. Reconstruct the plate using its convex hull.
7. Search only for text colors that are valid for the detected plate color.
8. Compute a color confidence score based on the proportion of supporting pixels.

This module is entirely based on OpenCV image processing.

---

## 3. Optical Character Recognition (`ocr_utils.py`)

The cropped wing tag is preprocessed before OCR.

The preprocessing consists of:

- conversion to grayscale
- image upscaling
- Gaussian blur

The processed image is passed to EasyOCR, which returns all detected text candidates.

The candidate with the highest confidence is selected as the final wing tag identifier.

---

## 4. Result Visualization (`predict.py`)

Finally, all results are combined into a single prediction.

For every detected wing tag, the output image contains:

- bounding box
- detection confidence
- plate color
- text color
- recognized identifier
- OCR confidence

The annotated image is written to the `prediction/` directory.

---

# Output Format

Each detected wing tag contains the following information.

| Field | Description |
|-----------------------|-----------------------|
| `bbox` | Bounding box coordinates |
| `confidence` | Detection confidence |
| `plate_color` | Detected wing tag plate color |
| `text_color` | Detected text color |
| `color_confidence` | Confidence of the color classification |
| `text` | Recognized wing tag identifier |
| `text_confidence` | OCR confidence |

---

# Configuration

Most project settings are located in

```
config.py
```

These include:

- dataset paths
- train/validation/test split
- model configuration
- image size
- batch size
- confidence threshold
- output directories
- supported image extensions

---

# License

This repository was developed as part of an internal research project at the Institute for Artificial Intelligence Research and Development of Serbia (IVI).