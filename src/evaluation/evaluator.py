import csv
from pathlib import Path

from ultralytics import YOLO


def compute_f1(
    precision: float,
    recall: float,
) -> float:
    """Calculate F1 from precision and recall."""

    if precision + recall == 0:
        return 0.0

    return (
        2 * precision * recall
        / (precision + recall)
    )

def evaluate_model(
    *,
    weights_path: Path,
    data_yaml: Path,
    splits: list[str],
    imgsz: int,
    device,
    output_dir: Path,
) -> list[dict]:
    """Evaluate a trained segmentation model."""

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {weights_path}"
        )

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found: {data_yaml}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = YOLO(str(weights_path))

    results = []

    for split in splits:
        print(f"Evaluating {split} split...")
        
        metrics = model.val(
            data=str(data_yaml),
            split=split,
            imgsz=imgsz,
            device=device,
            plots=False,
            project=str(output_dir),
            name=split,
            exist_ok=True,
        )

        precision = float(metrics.seg.mp)
        recall = float(metrics.seg.mr)

        results.append(
            {
                "split": split,
                "precision": precision,
                "recall": recall,
                "f1": compute_f1(
                    precision,
                    recall,
                ),
                "map50": float(metrics.seg.map50),
                "map50_95": float(metrics.seg.map),
            }
        )

    return results


def save_summary(
    results: list[dict],
    output_path: Path,
) -> None:
    """Write evaluation results to CSV."""

    fieldnames = [
        "split",
        "precision",
        "recall",
        "f1",
        "map50",
        "map50_95",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)


def print_summary(
    results: list[dict],
) -> None:
    """Print the evaluation results."""

    print()
    print("YOLO model evaluation")
    print("-" * 66)

    for result in results:
        print(
            f"{result['split']:<5} "
            f"P={result['precision']:.3f}  "
            f"R={result['recall']:.3f}  "
            f"F1={result['f1']:.3f}  "
            f"mAP50={result['map50']:.3f}  "
            f"mAP50-95={result['map50_95']:.3f}"
        )

    print()