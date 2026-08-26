# Wing Tag Detection and Recognition Pipeline

Computer vision pipeline for detecting and recognizing wing tags on griffon vultures.

The pipeline combines:

- **YOLO** for wing-tag detection
- **OpenCV** for plate and text color classification
- **PaddleOCR** for wing-tag identifier recognition

For each detected wing tag, the pipeline can return:

- bounding box
- detection confidence
- plate color
- text color
- color confidence
- recognized identifier
- OCR confidence

---

## Pipeline

```text
Input image
    │
    ▼
YOLO
Wing-tag detection
    │
    ▼
Detected tag crop
    │
    ├──► Color classification (OpenCV)
    │
    └──► Identifier recognition (PaddleOCR)
    │
    ▼
Final prediction
```

Color classification and OCR can be enabled or disabled independently in `config.py`.

---

## Project Structure

```text
saige-supovi/
│
├── config.py
├── prepare_yolo_dataset.py
├── prepare_ocr_dataset.py
├── train_yolo.py
├── predict.py
├── evaluate.py
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   └── images/
│   ├── yolo/
│   │   ├── images/
│   │   └── labels/
│   └── ocr/
│       └── images/
│
├── models/
│   ├── yolo/
│   │   └── best.pt
│   └── paddleocr/
│       ├── inference.json
│       ├── inference.pdiparams
│       └── inference.yml
│
├── outputs/
├── runs/
│
├── utils/
│   ├── color_utils.py
│   ├── image_utils.py
│   └── ocr_utils.py
│
└── tools/
    ├── dataset_statistics.py
    ├── merge_dataset.py
    └── visualize_dataset.py
```

The raw dataset is not included in the repository.

The trained YOLO detector and exported PaddleOCR recognition model used for inference are included under `models/`.

---

## Installation

Create and activate a Python virtual environment, then install PaddlePaddle separately according to the hardware being used.

PaddlePaddle provides separate CPU and GPU installations. The development environment for this project used:

```text
paddlepaddle-gpu==3.2.0
```

After installing the appropriate PaddlePaddle version, install the remaining project dependencies:

```bash
pip install -r requirements.txt
```

---

## Raw Dataset

The expected raw dataset structure is:

```text
data/raw/
├── annotations.csv
└── images/
```

`annotations.csv` contains the image filename, wing-tag bounding box, plate color, text color, and tag identifier.

The raw images and annotations are intentionally excluded from Git.

---

## Preparing the YOLO Dataset

Convert the raw dataset into YOLO format:

```bash
python prepare_yolo_dataset.py
```

The script:

- creates train, validation, and test splits
- keeps positive and negative images distributed across the splits
- converts bounding boxes to YOLO format
- creates YOLO label files
- generates `data/yolo/data.yaml`

The generated dataset is stored under:

```text
data/yolo/
```

---

## Training YOLO

Train the wing-tag detector:

```bash
python train_yolo.py
```

Training settings such as the YOLO model, image size, epochs, batch size, device, and augmentation parameters are defined in `config.py`.

Training runs are stored under:

```text
runs/
```

The best model is copied to:

```text
models/yolo/best.pt
```

A trained model is already included in the repository, so training is not required for inference.

---

## OCR Dataset Preparation

The OCR dataset is generated from the original annotations while reusing the existing YOLO train/validation/test split:

```bash
python prepare_ocr_dataset.py
```

Only fully readable identifiers are included. Annotations containing `*` are excluded from OCR training data.

Optional rotation augmentation can be generated with:

```bash
python prepare_ocr_dataset.py --augment
```

This adds 90°, 180°, and 270° rotations of each training crop.

The generated OCR dataset is stored under:

```text
data/ocr/
```

### OCR Model Training

The OCR model itself is **not trained in this repository**.

OCR fine-tuning was performed separately using the [PaddleOCR repository](https://github.com/PaddlePaddle/PaddleOCR).

This repository contains:

- the script used to prepare the OCR dataset
- the exported fine-tuned inference model under `models/paddleocr/`
- the inference code used by the final recognition pipeline

The PaddleOCR training code and training procedure remain external to this repository.

---

## Prediction

Run the complete pipeline on one image:

```bash
python predict.py --image path/to/image.jpg
```

Run it on all supported images in a folder:

```bash
python predict.py --folder path/to/images
```

Predicted images are saved under:

```text
outputs/predictions/
```

The pipeline stages are controlled in `config.py`:

```python
ENABLE_COLOR_CLASSIFICATION = True
ENABLE_OCR = True
```

Setting either option to `False` disables that stage.

---

## Evaluation

Evaluate one of the generated dataset splits:

```bash
python evaluate.py --split val
```

or:

```bash
python evaluate.py --split test
```

Supported values are:

```text
train
val
test
```

Detection predictions are matched to ground-truth tags using Intersection over Union (IoU).

Depending on the enabled pipeline stages, the evaluation reports:

- detection precision and recall
- plate-color accuracy
- text-color accuracy
- full color-pair accuracy
- exact OCR accuracy
- compatibility with partially readable OCR annotations

Detailed results are saved as:

```text
outputs/evaluation_train.csv
outputs/evaluation_val.csv
outputs/evaluation_test.csv
```

---

## Utility Tools

### Dataset Statistics

Display statistics for the raw annotation dataset:

```bash
python tools/dataset_statistics.py
```

### Dataset Visualization

Visualize random YOLO samples:

```bash
python tools/visualize_dataset.py
```

Visualize a specific split:

```bash
python tools/visualize_dataset.py --split val --num 20
```

Visualize raw annotations:

```bash
python tools/visualize_dataset.py --raw
```

### Merge Additional Data

Merge another annotated image collection into the existing raw dataset:

```bash
python tools/merge_dataset.py path/to/new_dataset
```

The input directory must contain its images and an `annotations.csv` file.

---

## Configuration

Project paths and configurable parameters are defined in:

```text
config.py
```

The main configuration groups include:

- dataset paths and splits
- YOLO model and training parameters
- YOLO augmentation
- OCR model settings
- inference thresholds
- optional prediction stages
- visualization settings
- valid wing-tag color combinations

---

## Models

The repository contains the models required for inference:

```text
models/yolo/best.pt
models/paddleocr/
```

The YOLO model detects wing tags.

The PaddleOCR model is a fine-tuned text-recognition model used to read the alphanumeric identifier from each detected tag.

OCR inference is evaluated at 0°, 90°, 180°, and 270° rotations, and the highest-confidence result is selected.

---

## Project Context

This repository was developed as part of a research project at the Institute for Artificial Intelligence Research and Development of Serbia.