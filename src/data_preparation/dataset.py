from dataclasses import dataclass
import math
import random
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import tifffile
from rasterio.transform import rowcol


@dataclass(frozen=True)
class Tile:
    area: str
    name: str
    image_path: Path
    label_path: Path | None


def find_tifs(folder: Path) -> list[Path]:
    """Return all TIFF files in a folder."""
    return sorted([
        *folder.glob("*.tif"),
        *folder.glob("*.tiff"),
    ])

def collect_reference_tiles(
    reference_dir: Path,
    study_areas: list[str],
) -> tuple[dict[str, list[Tile]], list[Tile]]:
    """
    Collect positive reference tiles and background candidates.

    Positive tiles are read from each area's tiles/ folder. Their corresponding
    crown labels must be stored as label_<tile_name>.geojson in labels/.

    Background candidates are read from pool_tiles/.
    """
    if not reference_dir.is_dir():
        raise FileNotFoundError(
            f"Reference data directory not found:\n{reference_dir}"
        )

    positive_tiles = {}
    background_tiles = []

    for area in study_areas:
        area_dir = reference_dir / area

        if not area_dir.is_dir():
            raise FileNotFoundError(
                f"Study area not found: {area_dir}"
            )

        tile_dir = area_dir / "tiles"
        label_dir = area_dir / "labels"
        pool_dir = area_dir / "pool_tiles"

        if not tile_dir.is_dir():
            raise FileNotFoundError(f"Missing folder: {tile_dir}")

        if not label_dir.is_dir():
            raise FileNotFoundError(f"Missing folder: {label_dir}")

        if not pool_dir.is_dir():
            raise FileNotFoundError(f"Missing folder: {pool_dir}")

        area_tiles = []

        for image_path in find_tifs(tile_dir):
            label_path = label_dir / f"label_{image_path.stem}.geojson"

            if not label_path.exists():
                raise FileNotFoundError(
                    f"Missing label for {image_path.name}:\n"
                    f"{label_path}"
                )

            area_tiles.append(
                Tile(
                    area=area,
                    name=f"{area}__{image_path.stem}",
                    image_path=image_path,
                    label_path=label_path,
                )
            )

        positive_tiles[area] = area_tiles

        for image_path in find_tifs(pool_dir):
            background_tiles.append(
                Tile(
                    area=area,
                    name=f"{area}__{image_path.stem}",
                    image_path=image_path,
                    label_path=None,
                )
            )

    return positive_tiles, background_tiles


def allocate_by_area(
    area_counts: dict[str, int],
    ratio: float,
    target: int,
) -> dict[str, int]:
    """
    Distribute a global split target across study areas.

    Each area first receives the whole-number part of its proportional share.
    Remaining tiles are assigned to the areas with the largest decimal
    remainders until the global target is reached.
    """
    exact = {
        area: count * ratio
        for area, count in area_counts.items()
    }

    allocation = {
        area: math.floor(value)
        for area, value in exact.items()
    }

    remaining = target - sum(allocation.values())

    ranked_areas = sorted(
        area_counts,
        key=lambda area: (
            -(exact[area] - allocation[area]),
            area,
        ),
    )

    for area in ranked_areas[:remaining]:
        allocation[area] += 1

    return allocation


def create_area_stratified_split(
    tiles_by_area: dict[str, list[Tile]],
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[dict[str, list[Tile]], dict[str, dict[str, int]]]:
    """Create reproducible area-stratified train, validation and test splits."""

    if not math.isclose(
        train_ratio + val_ratio + test_ratio,
        1.0,
    ):
        raise ValueError(
            "TRAIN_RATIO, VAL_RATIO and TEST_RATIO must sum to 1."
        )

    area_counts = {
        area: len(tiles)
        for area, tiles in sorted(tiles_by_area.items())
    }

    total_tiles = sum(area_counts.values())

    train_target = round(total_tiles * train_ratio)
    val_target = round(total_tiles * val_ratio)

    # Test receives the remaining tiles. This avoids rounding the three
    # global targets independently and accidentally changing the total.
    test_target = total_tiles - train_target - val_target

    train_allocation = allocate_by_area(
        area_counts,
        train_ratio,
        train_target,
    )

    val_allocation = allocate_by_area(
        area_counts,
        val_ratio,
        val_target,
    )

    test_allocation = {
        area: (
            area_counts[area]
            - train_allocation[area]
            - val_allocation[area]
        )
        for area in area_counts
    }

    if sum(test_allocation.values()) != test_target:
        raise RuntimeError("Could not create the requested test split.")

    if any(count < 0 for count in test_allocation.values()):
        raise RuntimeError("Invalid split allocation.")

    splits = {
        "train": [],
        "val": [],
        "test": [],
    }

    rng = random.Random(seed)

    for area in sorted(tiles_by_area):
        tiles = sorted(
            tiles_by_area[area],
            key=lambda tile: tile.name,
        )

        rng.shuffle(tiles)

        n_train = train_allocation[area]
        n_val = val_allocation[area]

        splits["train"].extend(
            tiles[:n_train]
        )

        splits["val"].extend(
            tiles[n_train:n_train + n_val]
        )

        splits["test"].extend(
            tiles[n_train + n_val:]
        )

    area_allocation = {
        area: {
            "train": train_allocation[area],
            "val": val_allocation[area],
            "test": test_allocation[area],
            "total": area_counts[area],
        }
        for area in area_counts
    }

    return splits, area_allocation


def select_background_tiles(
    candidates: list[Tile],
    *,
    count: int,
    seed: int,
) -> list[Tile]:
    """Select training background tiles reproducibly."""

    if len(candidates) < count:
        raise ValueError(
            f"Requested {count} background tiles, "
            f"but only {len(candidates)} are available."
        )

    candidates = sorted(
        candidates,
        key=lambda tile: tile.name,
    )

    rng = random.Random(seed)
    selected = rng.sample(candidates, count)

    return sorted(
        selected,
        key=lambda tile: tile.name,
    )


def read_rgbi_tile(
    image_path: Path,
) -> tuple[np.ndarray, object, object, int, int]:
    """Read a four-band RGBI tile."""

    with rasterio.open(image_path) as src:
        if src.count != 4:
            raise ValueError(
                f"{image_path.name}: expected 4 bands, "
                f"found {src.count}."
            )

        image = src.read()

        transform = src.transform
        crs = src.crs
        width = src.width
        height = src.height

    # Rasterio returns channels x height x width.
    # The training TIFFs are stored as height x width x channels.
    image = np.moveaxis(image, 0, -1)

    return image, transform, crs, width, height


def geometry_to_yolo(
    geometry,
    transform,
    width: int,
    height: int,
) -> list[str]:
    """Convert a polygon geometry to YOLO segmentation format."""

    if geometry.geom_type == "Polygon":
        polygons = [geometry]

    elif geometry.geom_type == "MultiPolygon":
        polygons = list(geometry.geoms)

    else:
        raise ValueError(
            f"Unsupported geometry type: {geometry.geom_type}"
        )

    yolo_lines = []

    for polygon in polygons:
        coordinates = list(polygon.exterior.coords)

        if len(coordinates) < 4:
            continue

        # The last polygon coordinate repeats the first one.
        coordinates = coordinates[:-1]

        x_coords = [
            coordinate[0]
            for coordinate in coordinates
        ]

        y_coords = [
            coordinate[1]
            for coordinate in coordinates
        ]

        rows, cols = rowcol(
            transform,
            x_coords,
            y_coords,
        )

        values = []

        for row, col in zip(rows, cols):
            x = min(max(col / width, 0.0), 1.0)
            y = min(max(row / height, 0.0), 1.0)

            values.extend([x, y])

        if len(values) < 6:
            continue

        coordinates_text = " ".join(
            f"{value:.6f}"
            for value in values
        )

        yolo_lines.append(
            f"0 {coordinates_text}"
        )

    return yolo_lines


def convert_label(
    label_path: Path,
    *,
    image_crs,
    transform,
    width: int,
    height: int,
) -> list[str]:
    """Read one GeoJSON label file and create YOLO segmentation labels."""

    labels = gpd.read_file(label_path)

    if labels.empty:
        raise ValueError(
            f"No crown polygons found in {label_path}"
        )

    if image_crs is None:
        raise ValueError(
            f"Image CRS missing for label {label_path.name}"
        )

    if labels.crs is None:
        raise ValueError(
            f"Label CRS missing: {label_path}"
        )

    if labels.crs != image_crs:
        labels = labels.to_crs(image_crs)

    yolo_lines = []

    for geometry in labels.geometry:
        if geometry is None or geometry.is_empty:
            continue

        yolo_lines.extend(
            geometry_to_yolo(
                geometry,
                transform,
                width,
                height,
            )
        )

    if not yolo_lines:
        raise ValueError(
            f"No usable crown polygons found in {label_path}"
        )

    return yolo_lines


def write_positive_tile(
    tile: Tile,
    image_output: Path,
    label_output: Path,
) -> int:
    """Write one positive RGBI tile and its YOLO label."""

    image, transform, crs, width, height = read_rgbi_tile(
        tile.image_path
    )

    if tile.label_path is None:
        raise ValueError(
            f"Positive tile has no label: {tile.name}"
        )

    yolo_lines = convert_label(
        tile.label_path,
        image_crs=crs,
        transform=transform,
        width=width,
        height=height,
    )

    # The training image does not need georeferencing.
    tifffile.imwrite(
        image_output,
        image,
    )

    label_output.write_text(
        "\n".join(yolo_lines) + "\n",
        encoding="utf-8",
    )

    return len(yolo_lines)


def write_background_tile(
    tile: Tile,
    image_output: Path,
    label_output: Path,
) -> None:
    """Write one background tile with an empty YOLO label."""

    image, _, _, _, _ = read_rgbi_tile(
        tile.image_path
    )

    tifffile.imwrite(
        image_output,
        image,
    )

    label_output.write_text(
        "",
        encoding="utf-8",
    )


def write_data_yaml(output_dir: Path) -> None:
    """Write the Ultralytics dataset configuration."""

    content = (
        f"path: {output_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "channels: 4\n"
        "names:\n"
        "  0: tree crown\n"
    )

    (output_dir / "data.yaml").write_text(
        content,
        encoding="utf-8",
    )


def write_dataset(
    splits: dict[str, list[Tile]],
    output_dir: Path,
) -> dict[str, dict[str, int]]:
    """Write the prepared YOLO dataset."""

    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists:\n{output_dir}\n"
            "Remove it before rebuilding the dataset."
        )

    summary = {}

    for split in ("train", "val", "test"):
        image_dir = output_dir / "images" / split
        label_dir = output_dir / "labels" / split

        image_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        label_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        positive_count = 0
        background_count = 0
        crown_count = 0

        tiles = sorted(
            splits[split],
            key=lambda tile: tile.name,
        )

        for tile in tiles:
            image_output = image_dir / f"{tile.name}.tif"
            label_output = label_dir / f"{tile.name}.txt"

            if tile.label_path is None:
                write_background_tile(
                    tile,
                    image_output,
                    label_output,
                )

                background_count += 1

            else:
                crown_count += write_positive_tile(
                    tile,
                    image_output,
                    label_output,
                )

                positive_count += 1

        summary[split] = {
            "images": len(tiles),
            "positive": positive_count,
            "background": background_count,
            "crowns": crown_count,
        }

    write_data_yaml(output_dir)

    return summary


def validate_dataset(
    output_dir: Path,
    *,
    expected_background_tiles: int,
) -> None:
    """Check image/label pairing, split leakage and background placement."""

    seen_tiles = set()

    for split in ("train", "val", "test"):
        image_dir = output_dir / "images" / split
        label_dir = output_dir / "labels" / split

        image_names = {
            path.stem
            for path in find_tifs(image_dir)
        }

        label_paths = sorted(
            label_dir.glob("*.txt")
        )

        label_names = {
            path.stem
            for path in label_paths
        }

        if image_names != label_names:
            raise RuntimeError(
                f"Image/label mismatch in {split} split."
            )

        overlap = seen_tiles.intersection(
            image_names
        )

        if overlap:
            raise RuntimeError(
                f"Tiles occur in more than one split: "
                f"{sorted(overlap)[:5]}"
            )

        seen_tiles.update(
            image_names
        )

        background_count = sum(
            not path.read_text(
                encoding="utf-8"
            ).strip()
            for path in label_paths
        )

        expected = (
            expected_background_tiles
            if split == "train"
            else 0
        )

        if background_count != expected:
            raise RuntimeError(
                f"{split} contains {background_count} "
                f"background tiles; expected {expected}."
            )


def print_summary(
    output_dir: Path,
    area_allocation: dict[str, dict[str, int]],
    summary: dict[str, dict[str, int]],
    background_tiles: list[Tile],
) -> None:
    """Print a short dataset summary."""

    print()
    print("YOLO dataset created")
    print("-" * 58)

    print(
        f"{'Area':<18}"
        f"{'Train':>8}"
        f"{'Val':>8}"
        f"{'Test':>8}"
        f"{'Total':>8}"
    )

    for area, counts in area_allocation.items():
        print(
            f"{area:<18}"
            f"{counts['train']:>8}"
            f"{counts['val']:>8}"
            f"{counts['test']:>8}"
            f"{counts['total']:>8}"
        )

    print()

    for split in ("train", "val", "test"):
        values = summary[split]

        print(
            f"{split:<5} "
            f"images={values['images']:<4} "
            f"positive={values['positive']:<4} "
            f"background={values['background']:<2} "
            f"crowns={values['crowns']}"
        )

    print()
    print("Background tiles used:")

    for tile in background_tiles:
        print(f"  {tile.name}")

    print()
    print(f"Output: {output_dir}")