
YOLO Tree Crown Mapping Handover



This repository contains the cleaned workflow used for the Master's thesis:

**Mapping Individual Tree Crowns in Leipzig from RGBI Aerial Orthophotos Using YOLO26 Instance Segmentation**

By Maxim Mahnkopf September 2026

The goal of this repository is to provide a reproducible handover workflow for preparing tree-crown reference data, training and evaluating a YOLO26 segmentation model, running the final post-processing workflow, and generating a Leipzig-wide individual tree-crown map.

## Workflow

The main workflow is organized as numbered scripts:

01_prepare_data.py
    ↓
02_train_model.py
    ↓
03_evaluate_model.py
    ↓
04_predict_evaluation_tiles.py
    ↓
05_evaluate_predictions.py
    ↓
06_predict_leipzig_map.py


01 — Prepare data

Creates the YOLO segmentation dataset from RGBI image tiles and georeferenced tree-crown polygons.

Main steps:

match image tiles and reference polygons
convert polygons to YOLO segmentation labels
create an area-stratified 70/15/15 train/validation/test split
add five background-only training tiles
prepare four-band RGBI image tiles
create the YOLO data.yaml
validate the generated dataset
02 — Train model

Trains the final YOLO26x instance-segmentation model using four-channel RGBI imagery.

The four input bands are:

Red
Green
Blue
Near-infrared

Training settings are stored in:

configs/training.yaml
03 — Evaluate model

Evaluates the trained model with the standard Ultralytics validation procedure.

Reported segmentation metrics include:

Precision
Recall
F1
mAP50
mAP50-95
04 — Predict evaluation areas

Runs the final large-area prediction workflow on the georeferenced validation and test areas.

Final inference settings:

window size             1000 px
window overlap           500 px
phase offsets              0 / 250 px
prediction image size    2048 px
confidence threshold       0.16
edge-contact filter       15 px
polygon IoU-NMS            0.50
retained-area fraction     0.70

Predictions from both sliding-window phases are merged before global polygon post-processing.

05 — Evaluate final predictions

Compares the final post-processed crown polygons with the georeferenced reference crowns.

Prediction and reference polygons are greedily matched one-to-one using:

IoU >= 0.50

Reported metrics:

Precision
Recall
F1
Mean matched IoU
Crown-count ratio
Crown-area ratio

Results are summarized for:

complete validation and test sets
Leipzig evaluation area (Leipzig + Auwald)
individual study areas
urban/non-forest and forest test contexts
06 — Predict Leipzig-wide crown map

Generates the final Leipzig-wide individual tree-crown map.

Leipzig is processed district by district. For each district the script:

reads the administrative district boundary
creates a buffered prediction extent
identifies the required RGBI orthophotos
builds a temporary four-band mosaic
runs the same sliding-window prediction workflow used in script 04
applies polygon post-processing
assigns crowns to districts based on their centroid
writes full and district-clipped crown polygons

The ten Leipzig districts are:

Mitte
Nordost
Ost
Südost
Süd
Südwest
West
Alt-West
Nordwest
Nord

The district outputs are finally merged into one Leipzig-wide GeoPackage.

Repository structure
yolo_tree_crown_project/
├── configs/
│   └── training.yaml
├── data/
│   ├── dops/
│   ├── tree_crown_reference_data/
│   ├── yolo_dataset/
│   ├── source_full_tiles_crop_val_test/
│   └── leipzig_boundaries/
├── scripts/
│   ├── 01_prepare_data.py
│   ├── 02_train_model.py
│   ├── 03_evaluate_model.py
│   ├── 04_predict_evaluation_tiles.py
│   ├── 05_evaluate_predictions.py
│   └── 06_predict_leipzig_map.py
├── src/
│   ├── data_preparation/
│   ├── training/
│   ├── inference/
│   ├── postprocessing/
│   ├── evaluation/
│   └── mapping/
├── pyproject.toml
├── uv.lock
└── README.md
Installation

The project uses uv.

Create the environment with:

uv sync

Run workflow scripts with:

uv run python scripts/<script_name>.py

For example:

uv run python scripts/01_prepare_data.py
External data

Large input and generated datasets are not stored in Git.

Reference data

Place the reference crown dataset under:

data/tree_crown_reference_data/

Expected study areas:

Auwald
Chemnitz
Hohes_Holz
Kalebsberg
Leipzig
Rostock
Tharandter_Wald
Leipzig RGBI orthophotos

Place the Leipzig DOP imagery under:

data/dops/

Script 06 searches this directory recursively for GeoTIFFs and zipped GeoTIFFs.

Expected imagery:

4-band RGBI
20 cm spatial resolution
8-bit
EPSG:25833
Leipzig district boundaries

The Leipzig Stadtbezirk shapefile is stored under:

data/leipzig_boundaries/

Current source:

sbz.shp

All associated shapefile files should remain together.

Model weights

Model weights are intentionally not stored in Git.

The workflow currently expects the final selected model under:

runs/X_yolo26x_tuned_exact/weights/best.pt

The main model used for the thesis was:

X_yolo26x_tuned_exact
Notes

This repository contains the cleaned operational workflow.

Experimental hyperparameter-search scripts, intermediate development scripts, large imagery, generated datasets, model weights, and run outputs are intentionally excluded.

Scripts 04 and 06 use the same sliding-window inference and polygon post-processing implementation so that evaluation and Leipzig-wide mapping follow the same prediction workflow.