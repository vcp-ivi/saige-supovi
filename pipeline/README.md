# Bird Tag Detection Pipeline

YOLO-based pipeline for detecting tagged bird plates in photographs, evaluating
YOLO vs SAHI inference, and running a downstream crop/OCR/color-analysis step.

## Files To Version

The intended push payload is:

- `scripts/*.py`
- `requirements.txt`
- `README.md`
- `outputs/grid_search_big/summary.csv`
- `outputs/grid_search_big/yolo26s_i1024_m0p5_s0p05_t0p02_e0p0_hs0p3_hv0p4_f0p0/weights/best.pt`

Do not commit local virtual environments, generated training runs, dataset
images, caches, or unrelated checkpoint folders.

## Selected Model

The selected detector checkpoint is:

```text
outputs/grid_search_big/yolo26s_i1024_m0p5_s0p05_t0p02_e0p0_hs0p3_hv0p4_f0p0/weights/best.pt
```

Training configuration for that run:

- Base model: `yolo26s.pt`
- Image size: `1024`
- Mosaic: `0.5`
- Scale: `0.05`
- Translate: `0.02`
- Erasing: `0.0`
- HSV saturation: `0.3`
- HSV value: `0.4`
- Horizontal flip: `0.0`
- Visual confidence used for comparison: `0.02`
- IoU threshold: `0.5`

The full grid-search table is stored in
`outputs/grid_search_big/summary.csv`.

## Setup

```powershell
python -m venv .venv311
.\.venv311\Scripts\python.exe -m pip install --upgrade pip
.\.venv311\Scripts\python.exe -m pip install -r requirements.txt
```

## Main Scripts

Prepare the YOLO dataset from annotated images:

```powershell
.\.venv311\Scripts\python.exe .\scripts\prepare_yolo_dataset.py --overwrite
```

Validate source annotations and generated YOLO labels:

```powershell
.\.venv311\Scripts\python.exe .\scripts\validate_dataset.py
```

Run grid search:

```powershell
.\.venv311\Scripts\python.exe .\scripts\grid_search_yolo_sahi.py --output-dir outputs\grid_search_big
```

Run tag detection, crop extraction, optional OCR, and color estimation:

```powershell
.\.venv311\Scripts\python.exe .\scripts\tag_ocr_pipeline.py --source data\yolo\images\test --output outputs\tag_ocr_pipeline --conf 0.05
```

## Notes

- `yolo26s.pt` and `yolo26n.pt` are base pretrained weights and do not need to
  be committed if they can be downloaded again.
- The selected `best.pt` is project output and should be committed only if the
  repository is expected to carry model weights. Prefer Git LFS if the remote
  enforces strict binary limits.
