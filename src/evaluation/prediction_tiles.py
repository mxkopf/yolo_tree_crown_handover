import csv
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box
from ultralytics import YOLO

from inference.sliding_window import predict_multiphase


def read_csv(path: Path) -> list[dict]:
    """Read a CSV file into a list of dictionaries."""

    if not path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def empty_predictions(crs) -> gpd.GeoDataFrame:
    """Create an empty prediction GeoDataFrame."""

    return gpd.GeoDataFrame(
        columns=[
            "geometry",
            "confidence",
            "class_id",
            "area_m2",
        ],
        geometry="geometry",
        crs=crs,
    )


def clip_predictions_to_tile(
    predictions: gpd.GeoDataFrame,
    tile: dict,
) -> gpd.GeoDataFrame:
    """Clip crop predictions to one reference tile extent."""

    tile_extent = box(
        float(tile["tile_left"]),
        float(tile["tile_bottom"]),
        float(tile["tile_right"]),
        float(tile["tile_top"]),
    )

    if predictions.empty:
        return empty_predictions(
            predictions.crs
        )

    possible = predictions[
        predictions.geometry.intersects(tile_extent)
    ].copy()

    if possible.empty:
        return empty_predictions(
            predictions.crs
        )

    possible["geometry"] = (
        possible.geometry
        .intersection(tile_extent)
    )

    possible = possible[
        ~possible.geometry.is_empty
    ].copy()

    return possible


def save_csv(
    rows: list[dict],
    output_path: Path,
) -> None:
    """Write a list of dictionaries to CSV."""

    if not rows:
        return

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def predict_evaluation_tiles(
    *,
    project_root: Path,
    weights_path: Path,
    crop_manifest: Path,
    assignment_csv: Path,
    output_dir: Path,
    window_size: int,
    stride: int,
    phase_offsets: list[int],
    predict_imgsz: int,
    edge_filter_px: float,
    confidence: float,
    iou_nms_threshold: float,
    retained_area_fraction: float,
    device,
) -> None:
    """
    Predict the validation/test source crops and clip predictions
    back to the original reference tile extents.
    """

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {weights_path}"
        )

    crops = read_csv(crop_manifest)
    assignments = read_csv(assignment_csv)

    model = YOLO(str(weights_path))

    crop_summary = []
    tile_summary = []

    for crop_index, crop in enumerate(
        crops,
        start=1,
    ):
        crop_id = str(crop["crop_id"])

        crop_assignments = [
            row
            for row in assignments
            if str(row["crop_id"]) == crop_id
        ]

        if not crop_assignments:
            continue

        crop_path = Path(
            crop["crop_tif"]
        )

        if not crop_path.is_absolute():
            crop_path = (
                project_root
                / crop_path
            ).resolve()

        print()
        print(
            f"Predicting crop "
            f"{crop_index}/{len(crops)}: "
            f"{crop['crop_name']}"
        )

        predictions, summary = predict_multiphase(
            model=model,
            image_path=crop_path,
            window_size=window_size,
            stride=stride,
            phase_offsets=phase_offsets,
            predict_imgsz=predict_imgsz,
            confidence=confidence,
            edge_filter_px=edge_filter_px,
            iou_nms_threshold=iou_nms_threshold,
            retained_area_fraction=retained_area_fraction,
            device=device,
        )

        crop_output = (
            output_dir
            / "crop_predictions"
            / f"{crop['crop_name']}.geojson"
        )

        crop_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        predictions.to_file(
            crop_output,
            driver="GeoJSON",
        )

        crop_summary.append(
            {
                "crop_id": crop_id,
                "crop_name": crop["crop_name"],
                **summary,
            }
        )

        for tile in crop_assignments:
            clipped = clip_predictions_to_tile(
                predictions,
                tile,
            )

            tile_output = (
                output_dir
                / "per_tile"
                / tile["split"]
                / tile["area"]
                / f"{tile['val_stem']}.geojson"
            )

            tile_output.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            clipped.to_file(
                tile_output,
                driver="GeoJSON",
            )

            tile_summary.append(
                {
                    "split": tile["split"],
                    "area": tile["area"],
                    "tile": tile["val_stem"],
                    "crop_id": crop_id,
                    "predictions": len(clipped),
                    "prediction_file": str(tile_output),
                }
            )

    save_csv(
        crop_summary,
        output_dir / "crop_summary.csv",
    )

    save_csv(
        tile_summary,
        output_dir / "tile_summary.csv",
    )

    print()
    print("Evaluation predictions completed")
    print("-" * 58)
    print(f"Source crops: {len(crop_summary)}")
    print(f"Reference tiles: {len(tile_summary)}")
    print(f"Output: {output_dir}")