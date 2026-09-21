import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import Window
from shapely.geometry import Polygon

from postprocessing.polygons import (
    clean_single_polygon,
    polygon_iou_nms,
    subtract_overlaps,
)


EDGE_TOLERANCE_PX = 2.0
MAX_DETECTIONS = 1000


def phase_window_starts(
    length: int,
    stride: int,
    offset: int,
) -> list[int]:
    """Create window starts for one prediction phase."""

    if offset >= length:
        return []

    return list(
        range(offset, length, stride)
    )


def read_rgbi_window(
    source,
    row_offset: int,
    column_offset: int,
    window_size: int,
) -> tuple[np.ndarray, int, int]:
    """Read one RGBI image window as H x W x 4 uint8."""

    width = min(
        window_size,
        source.width - column_offset,
    )

    height = min(
        window_size,
        source.height - row_offset,
    )

    window = Window(
        col_off=column_offset,
        row_off=row_offset,
        width=width,
        height=height,
    )

    red = source.read(1, window=window)
    green = source.read(2, window=window)
    blue = source.read(3, window=window)
    nir = source.read(4, window=window)

    image = np.dstack([
        red,
        green,
        blue,
        nir,
    ]).astype(np.uint8)

    # Windows at the outer raster border can be smaller.
    if height < window_size or width < window_size:
        padded = np.zeros(
            (window_size, window_size, 4),
            dtype=np.uint8,
        )

        padded[:height, :width, :] = image
        image = padded

    return image, height, width


def mask_to_map_polygon(
    mask_xy,
    *,
    column_offset: int,
    row_offset: int,
    width: int,
    height: int,
    transform,
):
    """Convert one local YOLO mask into map coordinates."""

    if mask_xy is None or len(mask_xy) < 3:
        return None

    coordinates = np.asarray(
        mask_xy,
        dtype=float,
    ).copy()

    coordinates[:, 0] = np.clip(
        coordinates[:, 0],
        0,
        width,
    )

    coordinates[:, 1] = np.clip(
        coordinates[:, 1],
        0,
        height,
    )

    map_coordinates = [
        transform * (
            column_offset + x,
            row_offset + y,
        )
        for x, y in coordinates
    ]

    return clean_single_polygon(
        Polygon(map_coordinates)
    )


def records_from_result(
    result,
    *,
    column_offset: int,
    row_offset: int,
    width: int,
    height: int,
    transform,
    phase: str,
) -> list[dict]:
    """Convert YOLO predictions into georeferenced polygon records."""

    if result.masks is None or result.boxes is None:
        return []

    confidences = (
        result.boxes.conf
        .detach()
        .cpu()
        .numpy()
    )

    classes = (
        result.boxes.cls
        .detach()
        .cpu()
        .numpy()
        .astype(int)
    )

    records = []

    for prediction_id, mask_xy in enumerate(
        result.masks.xy
    ):
        polygon = mask_to_map_polygon(
            mask_xy,
            column_offset=column_offset,
            row_offset=row_offset,
            width=width,
            height=height,
            transform=transform,
        )

        if polygon is None:
            continue

        records.append(
            {
                "geometry": polygon,
                "confidence": float(
                    confidences[prediction_id]
                ),
                "class_id": int(
                    classes[prediction_id]
                ),
                "area_m2": float(polygon.area),
                "phase": phase,
                "window_row": row_offset,
                "window_col": column_offset,
                "window_width": width,
                "window_height": height,
            }
        )

    return records


def edge_contact_lengths(
    geometry,
    *,
    transform,
    column_offset: int,
    row_offset: int,
    width: int,
    height: int,
    raster_width: int,
    raster_height: int,
) -> dict[str, float]:
    """Measure polygon boundary contact with internal window edges."""

    contacts = {
        "left": 0.0,
        "right": 0.0,
        "top": 0.0,
        "bottom": 0.0,
    }

    if geometry.geom_type != "Polygon":
        return contacts

    inverse_transform = ~transform
    coordinates = list(geometry.exterior.coords)

    for first, second in zip(
        coordinates[:-1],
        coordinates[1:],
    ):
        col1, row1 = inverse_transform * first
        col2, row2 = inverse_transform * second

        x1 = col1 - column_offset
        y1 = row1 - row_offset
        x2 = col2 - column_offset
        y2 = row2 - row_offset

        segment_length = math.hypot(
            x2 - x1,
            y2 - y1,
        )

        if column_offset > 0:
            if max(abs(x1), abs(x2)) <= EDGE_TOLERANCE_PX:
                contacts["left"] += segment_length

        if column_offset + width < raster_width:
            if max(
                abs(x1 - width),
                abs(x2 - width),
            ) <= EDGE_TOLERANCE_PX:
                contacts["right"] += segment_length

        if row_offset > 0:
            if max(abs(y1), abs(y2)) <= EDGE_TOLERANCE_PX:
                contacts["top"] += segment_length

        if row_offset + height < raster_height:
            if max(
                abs(y1 - height),
                abs(y2 - height),
            ) <= EDGE_TOLERANCE_PX:
                contacts["bottom"] += segment_length

    return contacts


def touches_internal_edge(
    record: dict,
    *,
    transform,
    raster_width: int,
    raster_height: int,
    minimum_contact_px: float,
) -> bool:
    """Return True for a prediction strongly truncated by a window edge."""

    contacts = edge_contact_lengths(
        record["geometry"],
        transform=transform,
        column_offset=record["window_col"],
        row_offset=record["window_row"],
        width=record["window_width"],
        height=record["window_height"],
        raster_width=raster_width,
        raster_height=raster_height,
    )

    return max(contacts.values()) >= minimum_contact_px


def predict_multiphase(
    *,
    model,
    image_path: Path,
    window_size: int,
    stride: int,
    phase_offsets: list[int],
    predict_imgsz: int,
    confidence: float,
    edge_filter_px: float,
    iou_nms_threshold: float,
    retained_area_fraction: float,
    device,
):
    """Run the final multi-phase sliding-window prediction workflow."""

    survivors = []
    raw_count = 0
    edge_deleted_count = 0

    with rasterio.open(image_path) as source:
        if source.count < 4:
            raise ValueError(
                f"Expected four RGBI bands: {image_path}"
            )

        for offset in phase_offsets:
            phase_name = f"phase{offset}"

            row_starts = phase_window_starts(
                source.height,
                stride,
                offset,
            )

            column_starts = phase_window_starts(
                source.width,
                stride,
                offset,
            )

            for row_offset in row_starts:
                for column_offset in column_starts:
                    image, height, width = read_rgbi_window(
                        source,
                        row_offset,
                        column_offset,
                        window_size,
                    )

                    results = model.predict(
                        source=image,
                        imgsz=predict_imgsz,
                        conf=confidence,
                        task="segment",
                        device=device,
                        retina_masks=True,
                        max_det=MAX_DETECTIONS,
                        save=False,
                        verbose=False,
                    )

                    if not results:
                        continue

                    records = records_from_result(
                        results[0],
                        column_offset=column_offset,
                        row_offset=row_offset,
                        width=width,
                        height=height,
                        transform=source.transform,
                        phase=phase_name,
                    )

                    raw_count += len(records)

                    for record in records:
                        if touches_internal_edge(
                            record,
                            transform=source.transform,
                            raster_width=source.width,
                            raster_height=source.height,
                            minimum_contact_px=edge_filter_px,
                        ):
                            edge_deleted_count += 1
                            continue

                        survivors.append(record)

        crs = source.crs

    after_nms = polygon_iou_nms(
        survivors,
        threshold=iou_nms_threshold,
    )

    final_records = subtract_overlaps(
        after_nms,
        min_retained_fraction=retained_area_fraction,
    )

    final = gpd.GeoDataFrame(
        final_records,
        geometry="geometry",
        crs=crs,
    )

    summary = {
        "raw_predictions": raw_count,
        "edge_deleted": edge_deleted_count,
        "before_global_postprocessing": len(survivors),
        "after_iou_nms": len(after_nms),
        "final_predictions": len(final_records),
    }

    return final, summary