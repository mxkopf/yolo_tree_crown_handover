import math
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from postprocessing.polygons import (
    clean_all_parts,
    polygon_iou,
)


LEIPZIG_EVALUATION_AREAS = [
    "Leipzig",
    "Auwald",
]

URBAN_NONFOREST_AREAS = [
    "Leipzig",
    "Chemnitz",
    "Rostock",
]

FOREST_AREAS = [
    "Auwald",
    "Hohes_Holz",
    "Kalebsberg",
    "Tharandter_Wald",
]

SUMMARY_METRICS = [
    "precision",
    "recall",
    "f1",
    "mean_matched_iou",
    "count_ratio",
    "area_ratio",
]


# ---------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------

def read_assignment(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Assignment CSV not found: {path}"
        )

    frame = pd.read_csv(path)

    required = {
        "split",
        "area",
        "val_stem",
        "tile_left",
        "tile_bottom",
        "tile_right",
        "tile_top",
        "source_crs",
    }

    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            "Assignment CSV is missing columns: "
            + ", ".join(sorted(missing))
        )

    return frame


def resolve_label_path(
    row,
    *,
    project_root: Path,
    label_base_dir: Path,
) -> Path:
    # Prefer the exact georeferenced label path stored in the assignment CSV.
    if "label_file" in row:
        value = row["label_file"]

        if pd.notna(value) and str(value).strip():
            path = Path(str(value))

            if not path.is_absolute():
                path = (
                    project_root
                    / path
                ).resolve()

            if path.exists():
                return path

    # Fallback used by the original evaluation workflow.
    fallback = (
        label_base_dir
        / f"{row['split']}_labels_areastratified_georef"
        / f"{row['val_stem']}.geojson"
    )

    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        "Reference label not found for "
        f"{row['split']} / {row['area']} / {row['val_stem']}"
    )


def prediction_path(
    prediction_dir: Path,
    row,
) -> Path:
    return (
        prediction_dir
        / str(row["split"])
        / str(row["area"])
        / f"{row['val_stem']}.geojson"
    )


def read_polygons(
    path: Path,
    *,
    crs,
    min_area_m2: float = 0.0,
) -> list:
    if not path.exists():
        raise FileNotFoundError(
            f"Polygon file not found: {path}"
        )

    # Handle GeoJSON files containing no features.
    text = path.read_text(
        encoding="utf-8",
        errors="ignore",
    )[:1000]

    if '"features": []' in text or '"features":[]' in text:
        return []

    frame = gpd.read_file(path)

    if frame.empty:
        return []

    frame = frame.set_crs(
        crs,
        allow_override=True,
    )

    polygons = []

    for geometry in frame.geometry:
        geometry = clean_all_parts(
            geometry
        )

        if geometry is None:
            continue

        if geometry.area < min_area_m2:
            continue

        polygons.append(geometry)

    return polygons


# ---------------------------------------------------------------------
# Polygon matching
# ---------------------------------------------------------------------

def greedy_iou_match(
    predictions: list,
    references: list,
    threshold: float,
) -> dict:
    # Collect every possible prediction-reference match above the IoU threshold.
    candidate_pairs = []

    for prediction_index, prediction in enumerate(
        predictions
    ):
        for reference_index, reference in enumerate(
            references
        ):
            if not prediction.intersects(reference):
                continue

            iou = polygon_iou(
                prediction,
                reference,
            )

            if iou >= threshold:
                candidate_pairs.append(
                    (
                        iou,
                        prediction_index,
                        reference_index,
                    )
                )

    # Highest-IoU matches are assigned first.
    candidate_pairs.sort(
        key=lambda pair: pair[0],
        reverse=True,
    )

    matched_predictions = set()
    matched_references = set()
    matched_ious = []

    for (
        iou,
        prediction_index,
        reference_index,
    ) in candidate_pairs:

        if prediction_index in matched_predictions:
            continue

        if reference_index in matched_references:
            continue

        matched_predictions.add(
            prediction_index
        )

        matched_references.add(
            reference_index
        )

        matched_ious.append(
            float(iou)
        )

    true_positive = len(
        matched_predictions
    )

    false_positive = (
        len(predictions)
        - true_positive
    )

    false_negative = (
        len(references)
        - true_positive
    )

    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "iou_sum": sum(matched_ious),
    }


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def safe_ratio(
    numerator: float,
    denominator: float,
) -> float:
    if denominator == 0:
        if numerator == 0:
            return 1.0

        return math.inf

    return numerator / denominator


def calculate_metrics(
    *,
    tp: int,
    fp: int,
    fn: int,
    pred_count: int,
    ref_count: int,
    pred_area_m2: float,
    ref_area_m2: float,
    iou_sum: float,
) -> dict:

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    mean_matched_iou = (
        iou_sum / tp
        if tp > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_iou": mean_matched_iou,
        "count_ratio": safe_ratio(
            pred_count,
            ref_count,
        ),
        "area_ratio": safe_ratio(
            pred_area_m2,
            ref_area_m2,
        ),
    }


# ---------------------------------------------------------------------
# Tile evaluation
# ---------------------------------------------------------------------

def evaluate_tile(
    row,
    *,
    project_root: Path,
    prediction_dir: Path,
    label_base_dir: Path,
    match_iou: float,
    min_prediction_area_m2: float,
) -> dict:

    crs = str(
        row["source_crs"]
    )

    tile_extent = box(
        float(row["tile_left"]),
        float(row["tile_bottom"]),
        float(row["tile_right"]),
        float(row["tile_top"]),
    )

    label_path = resolve_label_path(
        row,
        project_root=project_root,
        label_base_dir=label_base_dir,
    )

    pred_path = prediction_path(
        prediction_dir,
        row,
    )

    references = read_polygons(
        label_path,
        crs=crs,
    )

    predictions = read_polygons(
        pred_path,
        crs=crs,
        min_area_m2=min_prediction_area_m2,
    )

    # Clip both datasets to exactly the same evaluation tile.
    clipped_references = []

    for geometry in references:
        if not geometry.intersects(
            tile_extent
        ):
            continue

        clipped = clean_all_parts(
            geometry.intersection(
                tile_extent
            )
        )

        if (
            clipped is not None
            and clipped.area > 0
        ):
            clipped_references.append(
                clipped
            )

    clipped_predictions = []

    for geometry in predictions:
        if not geometry.intersects(
            tile_extent
        ):
            continue

        clipped = clean_all_parts(
            geometry.intersection(
                tile_extent
            )
        )

        if (
            clipped is not None
            and clipped.area
            >= min_prediction_area_m2
        ):
            clipped_predictions.append(
                clipped
            )

    match = greedy_iou_match(
        clipped_predictions,
        clipped_references,
        threshold=match_iou,
    )

    pred_count = len(
        clipped_predictions
    )

    ref_count = len(
        clipped_references
    )

    pred_area_m2 = sum(
        geometry.area
        for geometry in clipped_predictions
    )

    ref_area_m2 = sum(
        geometry.area
        for geometry in clipped_references
    )

    metrics = calculate_metrics(
        tp=match["tp"],
        fp=match["fp"],
        fn=match["fn"],
        pred_count=pred_count,
        ref_count=ref_count,
        pred_area_m2=pred_area_m2,
        ref_area_m2=ref_area_m2,
        iou_sum=match["iou_sum"],
    )

    return {
        "split": str(
            row["split"]
        ),
        "area": str(
            row["area"]
        ),
        "tile": str(
            row["val_stem"]
        ),
        "tp": match["tp"],
        "fp": match["fp"],
        "fn": match["fn"],
        "pred_count": pred_count,
        "ref_count": ref_count,
        "pred_area_m2": pred_area_m2,
        "ref_area_m2": ref_area_m2,
        "iou_sum": match["iou_sum"],
        **metrics,
    }


# ---------------------------------------------------------------------
# Pooled summaries
# ---------------------------------------------------------------------

def aggregate(
    frame: pd.DataFrame,
) -> dict:

    tp = int(
        frame["tp"].sum()
    )

    fp = int(
        frame["fp"].sum()
    )

    fn = int(
        frame["fn"].sum()
    )

    pred_count = int(
        frame["pred_count"].sum()
    )

    ref_count = int(
        frame["ref_count"].sum()
    )

    pred_area_m2 = float(
        frame["pred_area_m2"].sum()
    )

    ref_area_m2 = float(
        frame["ref_area_m2"].sum()
    )

    iou_sum = float(
        frame["iou_sum"].sum()
    )

    return {
        "n_tiles": len(frame),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "pred_count": pred_count,
        "ref_count": ref_count,
        "pred_area_m2": pred_area_m2,
        "ref_area_m2": ref_area_m2,
        **calculate_metrics(
            tp=tp,
            fp=fp,
            fn=fn,
            pred_count=pred_count,
            ref_count=ref_count,
            pred_area_m2=pred_area_m2,
            ref_area_m2=ref_area_m2,
            iou_sum=iou_sum,
        ),
    }


def summary_by_study_area(
    tile_metrics: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for (
        split,
        area,
    ), group in tile_metrics.groupby(
        ["split", "area"],
        sort=True,
    ):
        rows.append(
            {
                "split": split,
                "area": area,
                **aggregate(group),
            }
        )

    return pd.DataFrame(
        rows
    )


def complete_and_leipzig_summary(
    tile_metrics: pd.DataFrame,
) -> pd.DataFrame:

    complete_validation = aggregate(
        tile_metrics[
            tile_metrics["split"] == "val"
        ]
    )

    complete_test = aggregate(
        tile_metrics[
            tile_metrics["split"] == "test"
        ]
    )

    leipzig = tile_metrics[
        tile_metrics["area"].isin(
            LEIPZIG_EVALUATION_AREAS
        )
    ]

    leipzig_validation = aggregate(
        leipzig[
            leipzig["split"] == "val"
        ]
    )

    leipzig_test = aggregate(
        leipzig[
            leipzig["split"] == "test"
        ]
    )

    rows = []

    for metric in SUMMARY_METRICS:
        rows.append(
            {
                "metric": metric,
                "complete_validation":
                    complete_validation[metric],
                "complete_test":
                    complete_test[metric],
                "leipzig_validation":
                    leipzig_validation[metric],
                "leipzig_test":
                    leipzig_test[metric],
            }
        )

    return pd.DataFrame(
        rows
    )


def forest_nonforest_summary(
    tile_metrics: pd.DataFrame,
) -> pd.DataFrame:

    test = tile_metrics[
        tile_metrics["split"] == "test"
    ]

    urban_nonforest = aggregate(
        test[
            test["area"].isin(
                URBAN_NONFOREST_AREAS
            )
        ]
    )

    forest = aggregate(
        test[
            test["area"].isin(
                FOREST_AREAS
            )
        ]
    )

    rows = []

    for metric in SUMMARY_METRICS:
        rows.append(
            {
                "metric": metric,
                "urban_nonforest_test":
                    urban_nonforest[metric],
                "forest_test":
                    forest[metric],
            }
        )

    return pd.DataFrame(
        rows
    )


# ---------------------------------------------------------------------
# Complete evaluation
# ---------------------------------------------------------------------

def evaluate_predictions(
    *,
    project_root: Path,
    prediction_dir: Path,
    assignment_csv: Path,
    label_base_dir: Path,
    output_dir: Path,
    match_iou: float,
    min_prediction_area_m2: float,
) -> None:

    assignment = read_assignment(
        assignment_csv
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for index, row in assignment.iterrows():
        result = evaluate_tile(
            row,
            project_root=project_root,
            prediction_dir=prediction_dir,
            label_base_dir=label_base_dir,
            match_iou=match_iou,
            min_prediction_area_m2=min_prediction_area_m2,
        )

        rows.append(
            result
        )

        print(
            f"Evaluated "
            f"{index + 1}/{len(assignment)}: "
            f"{result['split']} / "
            f"{result['area']} / "
            f"{result['tile']}"
        )

    tile_metrics = pd.DataFrame(
        rows
    )

    area_summary = summary_by_study_area(
        tile_metrics
    )

    main_summary = complete_and_leipzig_summary(
        tile_metrics
    )

    context_summary = forest_nonforest_summary(
        tile_metrics
    )

    tile_metrics.to_csv(
        output_dir / "tile_metrics.csv",
        index=False,
    )

    area_summary.to_csv(
        output_dir / "summary_by_study_area.csv",
        index=False,
    )

    main_summary.to_csv(
        output_dir / "summary_complete_and_leipzig.csv",
        index=False,
    )

    context_summary.to_csv(
        output_dir / "summary_forest_nonforest.csv",
        index=False,
    )

    print()
    print("Complete set and Leipzig evaluation area")
    print("-" * 70)
    print(
        main_summary.to_string(
            index=False
        )
    )

    print()
    print("Urban / non-forest vs forest")
    print("-" * 70)
    print(
        context_summary.to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved evaluation results to: "
        f"{output_dir}"
    )