# Evaluate the final trained YOLO26x-seg model.
#
# Standard Ultralytics segmentation metrics are calculated for the
# validation and test datasets and written to a CSV summary.

from pathlib import Path

from evaluation.evaluator import (
    evaluate_model,
    print_summary,
    save_summary,
)


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

DATA_YAML = PROJECT_ROOT / "data" / "yolo_dataset" / "data.yaml"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "runs"
    / "X_yolo26x_tuned_exact"
    / "evaluation"
)

SPLITS = ["val", "test"]

IMGSZ = 1024
DEVICE = 0
PLOTS = True


def main() -> None:
    results = evaluate_model(
        weights_path=WEIGHTS_PATH,
        data_yaml=DATA_YAML,
        splits=SPLITS,
        imgsz=IMGSZ,
        device=DEVICE,
        output_dir=OUTPUT_DIR,
    )

    save_summary(
        results,
        OUTPUT_DIR / "evaluation_summary.csv",
    )

    print_summary(results)


if __name__ == "__main__":
    main()