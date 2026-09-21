# Evaluate the final post-processed validation and test predictions.

from pathlib import Path

from evaluation.prediction_metrics import (
    evaluate_predictions,
)


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTION_DIR = (
    PROJECT_ROOT
    / "runs"
    / "X_yolo26x_tuned_exact"
    / "evaluation_predictions"
    / "per_tile"
)

ASSIGNMENT_CSV = (
    PROJECT_ROOT
    / "data"
    / "source_full_tiles_crop_val_test"
    / "validation_test_crop_assignment.csv"
)

LABEL_BASE_DIR = (
    PROJECT_ROOT
    / "data"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "runs"
    / "X_yolo26x_tuned_exact"
    / "prediction_evaluation"
)


# A prediction and reference crown are matched when IoU >= 0.50.
MATCH_IOU = 0.50

# Tiny polygon fragments created by clipping are ignored.
MIN_PREDICTION_AREA_M2 = 1.0


def main() -> None:
    evaluate_predictions(
        project_root=PROJECT_ROOT,
        prediction_dir=PREDICTION_DIR,
        assignment_csv=ASSIGNMENT_CSV,
        label_base_dir=LABEL_BASE_DIR,
        output_dir=OUTPUT_DIR,
        match_iou=MATCH_IOU,
        min_prediction_area_m2=MIN_PREDICTION_AREA_M2,
    )


if __name__ == "__main__":
    main()