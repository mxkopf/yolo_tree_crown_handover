# Generate the final Leipzig-wide individual tree-crown map.
#
# Leipzig is processed district by district. Each district is predicted
# with surrounding imagery so crowns near administrative boundaries have
# sufficient image context.

from pathlib import Path

from mapping.leipzig_map import predict_leipzig_map


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

WEIGHTS_PATH = (
    PROJECT_ROOT
    / "runs"
    / "X_yolo26x_tuned_exact"
    / "weights"
    / "best.pt"
)

# One Leipzig administrative-boundary vector dataset.
# It may contain either the ten districts directly or smaller units with
# an attribute containing their district name.
DISTRICT_BOUNDARIES = (
    PROJECT_ROOT
    / "data"
    / "leipzig_boundaries"
    / "leipzig_districts.gpkg"
)

# Original georeferenced RGBI DOPs.
# TIFFs may also be stored inside ZIP archives.
DOP_DIR = (
    PROJECT_ROOT
    / "data"
    / "dops"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "runs"
    / "X_yolo26x_tuned_exact"
    / "leipzig_map"
)


# District processing
DISTRICT_BUFFER_M = 250
CROP_EXPAND_M = 200

# Sliding-window inference
WINDOW_SIZE = 1000
STRIDE = 500
PHASE_OFFSETS = [0, 250]

PREDICT_IMGSZ = 2048
EDGE_FILTER_PX = 15

# Final prediction settings
CONFIDENCE = 0.16
POLYGON_IOU_NMS = 0.50
RETAINED_AREA_FRACTION = 0.70

DEVICE = 0


def main() -> None:
    predict_leipzig_map(
        weights_path=WEIGHTS_PATH,
        boundary_file=DISTRICT_BOUNDARIES,
        dop_dir=DOP_DIR,
        output_dir=OUTPUT_DIR,
        district_buffer_m=DISTRICT_BUFFER_M,
        crop_expand_m=CROP_EXPAND_M,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        phase_offsets=PHASE_OFFSETS,
        predict_imgsz=PREDICT_IMGSZ,
        edge_filter_px=EDGE_FILTER_PX,
        confidence=CONFIDENCE,
        iou_nms_threshold=POLYGON_IOU_NMS,
        retained_area_fraction=RETAINED_AREA_FRACTION,
        device=DEVICE,
    )


if __name__ == "__main__":
    main()