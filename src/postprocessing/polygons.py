import math

from shapely.geometry import MultiPolygon
from shapely.ops import unary_union
from shapely.validation import make_valid


SPATIAL_INDEX_CELL_M = 10.0


def polygonal_parts(geometry):
    """Return polygon parts from a Shapely geometry."""

    if geometry is None or geometry.is_empty:
        return []

    if geometry.geom_type == "Polygon":
        return [geometry]

    if geometry.geom_type == "MultiPolygon":
        return [
            polygon
            for polygon in geometry.geoms
            if not polygon.is_empty and polygon.area > 0
        ]

    if geometry.geom_type == "GeometryCollection":
        parts = []

        for member in geometry.geoms:
            parts.extend(polygonal_parts(member))

        return parts

    return []


def clean_single_polygon(geometry):
    """Return one valid polygon, keeping the largest part if necessary."""

    if geometry is None:
        return None

    if not geometry.is_valid:
        geometry = make_valid(geometry)

    parts = polygonal_parts(geometry)

    if not parts:
        return None

    polygon = max(parts, key=lambda part: part.area)

    if polygon.is_empty or polygon.area <= 0:
        return None

    return polygon


def clean_all_parts(geometry):
    """
    Return a valid Polygon or MultiPolygon.

    All remaining polygon parts are kept together as one prediction.
    This implements the final no-split rule.
    """

    if geometry is None:
        return None

    if not geometry.is_valid:
        geometry = make_valid(geometry)

    parts = polygonal_parts(geometry)

    if not parts:
        return None

    geometry = unary_union(parts)

    if not geometry.is_valid:
        geometry = make_valid(geometry)

    parts = polygonal_parts(geometry)

    if not parts:
        return None

    if len(parts) == 1:
        return parts[0]

    return MultiPolygon(parts)


def polygon_iou(first, second) -> float:
    """Calculate polygon intersection over union."""

    intersection = first.intersection(second).area

    if intersection <= 0:
        return 0.0

    union = first.union(second).area

    if union <= 0:
        return 0.0

    return float(intersection / union)


def bounds_intersect(first, second) -> bool:
    """Check whether two polygon bounding boxes intersect."""

    minx1, miny1, maxx1, maxy1 = first
    minx2, miny2, maxx2, maxy2 = second

    return not (
        maxx1 < minx2
        or maxx2 < minx1
        or maxy1 < miny2
        or maxy2 < miny1
    )


class SpatialIndex:
    """Simple spatial index for efficient polygon comparisons."""

    def __init__(self, cell_size: float):
        self.cell_size = float(cell_size)
        self.records = []
        self.cells = {}

    def cell_keys(self, bounds):
        minx, miny, maxx, maxy = bounds

        x_start = math.floor(minx / self.cell_size)
        x_end = math.floor(maxx / self.cell_size)
        y_start = math.floor(miny / self.cell_size)
        y_end = math.floor(maxy / self.cell_size)

        for x in range(x_start, x_end + 1):
            for y in range(y_start, y_end + 1):
                yield x, y

    def add(self, record):
        index = len(self.records)
        self.records.append(record)

        for key in self.cell_keys(record["geometry"].bounds):
            self.cells.setdefault(key, []).append(index)

    def query(self, geometry, class_id):
        indices = set()

        for key in self.cell_keys(geometry.bounds):
            indices.update(self.cells.get(key, []))

        return [
            self.records[index]
            for index in indices
            if self.records[index]["class_id"] == class_id
            and bounds_intersect(
                geometry.bounds,
                self.records[index]["geometry"].bounds,
            )
        ]


def polygon_iou_nms(
    records: list[dict],
    threshold: float,
) -> list[dict]:
    """
    Remove lower-confidence duplicate polygons using polygon IoU.
    """

    index = SpatialIndex(SPATIAL_INDEX_CELL_M)
    kept = []

    records = sorted(
        records,
        key=lambda record: record["confidence"],
        reverse=True,
    )

    for candidate in records:
        neighbours = index.query(
            candidate["geometry"],
            candidate["class_id"],
        )

        duplicate = any(
            polygon_iou(
                candidate["geometry"],
                existing["geometry"],
            ) >= threshold
            for existing in neighbours
        )

        if not duplicate:
            kept.append(candidate)
            index.add(candidate)

    return kept


def subtract_overlaps(
    records: list[dict],
    min_retained_fraction: float,
) -> list[dict]:
    """
    Subtract areas already occupied by higher-confidence predictions.

    A polygon is retained only if the required fraction of its original
    area remains after subtraction.
    """

    index = SpatialIndex(SPATIAL_INDEX_CELL_M)
    kept = []

    records = sorted(
        records,
        key=lambda record: record["confidence"],
        reverse=True,
    )

    for candidate in records:
        original = candidate["geometry"]
        original_area = float(original.area)

        if original_area <= 0:
            continue

        earlier = index.query(
            original,
            candidate["class_id"],
        )

        if earlier:
            occupied = unary_union([
                record["geometry"]
                for record in earlier
            ])

            cleaned = original.difference(occupied)

        else:
            cleaned = original

        # Final no-split rule: keep all remaining parts together.
        cleaned = clean_all_parts(cleaned)

        if cleaned is None:
            continue

        retained_fraction = (
            float(cleaned.area)
            / original_area
        )

        if retained_fraction < min_retained_fraction:
            continue

        output = dict(candidate)
        output["geometry"] = cleaned
        output["area_m2"] = float(cleaned.area)
        output["retained_fraction"] = retained_fraction

        kept.append(output)
        index.add(output)

    return kept