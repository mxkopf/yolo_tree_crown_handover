# Generate post-processed predictions for the validation and test areas.
#
# Large source crops are processed using two-phase sliding-window inference,
# followed by confidence filtering and geometric post-processing.

from pathlib import Path

from evaluation.prediction_tiles import predict_evaluation_tiles


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

CROP_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "source_full_tiles_crop_val_test"
    / "source_full_tiles_crop_manifest.csv"
)

ASSIGNMENT_CSV = (
    PROJECT_ROOT
    / "data"
    / "source_full_tiles_crop_val_test"
    / "validation_test_crop_assignment.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "runs"
    / "X_yolo26x_tuned_exact"
    / "evaluation_predictions"
)


# Sliding-window inference
WINDOW_SIZE = 1000
STRIDE = 500
PHASE_OFFSETS = [0, 250]

PREDICT_IMGSZ = 2048
EDGE_FILTER_PX = 15

# Prediction filtering
CONFIDENCE = 0.16

# Polygon post-processing
POLYGON_IOU_NMS = 0.50
RETAINED_AREA_FRACTION = 0.70

DEVICE = 0


def main() -> None:
    predict_evaluation_tiles(
        project_root=PROJECT_ROOT,
        weights_path=WEIGHTS_PATH,
        crop_manifest=CROP_MANIFEST,
        assignment_csv=ASSIGNMENT_CSV,
        output_dir=OUTPUT_DIR,
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