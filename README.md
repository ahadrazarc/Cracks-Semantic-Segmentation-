# Cracks Semantic Segmentation

Semantic Segmentation using Ultralytics `yolo26x-sem.pt`

## Demo

https://github.com/user-attachments/assets/ea7966e9-fe6f-42d4-b71e-a7d19bb1468b

## Overview

- **Task:** Semantic segmentation (pixel-wise crack detection)
- **Model:** `yolo26x-sem.pt` (Ultralytics YOLO26, semantic segmentation variant)
- **Framework:** Ultralytics
- **Input dataset format:** Roboflow-style export (`_classes.csv` + `*_mask.png` per split)
- **Training image size:** 1024
- **Epochs:** 100

## Pipeline

### 1. Dataset Conversion

Raw dataset (`/content/crcsegformer-2`) is converted from a Roboflow-style export into the YOLO segmentation directory layout expected by Ultralytics:

```
dataset_yolo/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── masks/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

The conversion script:
- Reads `_classes.csv` to build the class index → name mapping.
- Splits `train` / `valid` / `test` folders into separate `images/` and `masks/` directories (`valid` is renamed to `val`).
- Strips the `_mask.png` suffix from mask filenames so they line up with their source images.
- Generates a `data.yaml` pointing to the image/mask directories and listing class names.

### 2. Training

```python
from ultralytics import YOLO

model = YOLO("yolo26x-sem.pt")
results = model.train(
    data="/content/dataset_yolo/data.yaml",
    epochs=100,
    imgsz=1024,
)
```

### 3. Inference

Run the trained model on a video (or image) and pull out the per-frame semantic mask:

```python
from ultralytics import YOLO

model = YOLO("/content/runs/semantic/train/weights/best.pt")
results = model("path/to/video_or_image", save=True)

for result in results:
    semantic_mask = result.semantic_mask.data
```

## Requirements

```bash
pip install ultralytics
```

## Notes

- Designed to run in Google Colab (paths reference `/content/...`); update paths if running locally.
- Trained weights are saved to `runs/semantic/train/weights/best.pt`.
- `result.semantic_mask.data` contains the raw pixel-wise class mask for each inference result — use this for downstream analysis (e.g. crack area, severity estimation, overlay rendering).

## Project Structure

```
.
├── cracks_semantic_segmentation.ipynb   # Main notebook: dataset prep, training, inference
└── README.md
```
