import math
import re
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterio.windows import Window
from shapely.geometry import box
from shapely.ops import unary_union
from ultralytics import YOLO

from inference.sliding_window import predict_multiphase
from postprocessing.polygons import clean_all_parts


TARGET_CRS = "EPSG:25833"
RESOLUTION_M = 0.20
MOSAIC_CHUNK_PX = 2000
MIN_AREA_M2 = 1.0


DISTRICTS = {
    "mitte": "Mitte",
    "nordost": "Nordost",
    "ost": "Ost",
    "suedost": "Südost",
    "sued": "Süd",
    "suedwest": "Südwest",
    "west": "West",
    "alt_west": "Alt-West",
    "nordwest": "Nordwest",
    "nord": "Nord",
}


# ---------------------------------------------------------------------
# District boundaries
# ---------------------------------------------------------------------

def normalize_name(value: str) -> str:
    """Normalize district names for robust matching."""

    text = str(value).lower()

    replacements = {
        "ä": "ae",
        "ö": "oe",
        "ü": "ue",
        "ß": "ss",
    }

    for source, target in replacements.items():
        text = text.replace(
            source,
            target,
        )

    return re.sub(
        r"[^a-z0-9]",
        "",
        text,
    )


def find_district_column(
    frame: gpd.GeoDataFrame,
) -> str:
    """
    Find a text column containing the ten Leipzig district names.

    This also works when the input contains smaller administrative units
    with a district-name attribute, because they are dissolved afterwards.
    """

    required = {
        normalize_name(name)
        for name in DISTRICTS.values()
    }

    for column in frame.columns:
        if column == frame.geometry.name:
            continue

        values = {
            normalize_name(value)
            for value in frame[column].dropna()
        }

        if required.issubset(values):
            return column

    raise ValueError(
        "Could not identify a column containing all ten Leipzig district names."
    )


def load_districts(
    boundary_file: Path,
) -> gpd.GeoDataFrame:
    """Read and dissolve Leipzig administrative boundaries by district."""

    if not boundary_file.exists():
        raise FileNotFoundError(
            f"District boundary file not found: {boundary_file}"
        )

    frame = gpd.read_file(
        boundary_file
    ).to_crs(
        TARGET_CRS
    )

    district_column = find_district_column(
        frame
    )

    lookup = {
        normalize_name(name): slug
        for slug, name in DISTRICTS.items()
    }

    frame["district"] = (
        frame[district_column]
        .astype(str)
        .map(normalize_name)
        .map(lookup)
    )

    frame = frame[
        frame["district"].notna()
    ].copy()

    districts = (
        frame[["district", "geometry"]]
        .dissolve(by="district")
        .reset_index()
    )

    missing = (
        set(DISTRICTS)
        - set(districts["district"])
    )

    if missing:
        raise ValueError(
            "Missing districts: "
            + ", ".join(sorted(missing))
        )

    return districts


# ---------------------------------------------------------------------
# Orthophoto discovery
# ---------------------------------------------------------------------

def zip_raster_paths(
    zip_path: Path,
) -> list[str]:
    """Return GDAL /vsizip paths for TIFFs inside one ZIP archive."""

    paths = []

    with zipfile.ZipFile(
        zip_path
    ) as archive:
        for member in archive.namelist():
            if not member.lower().endswith(
                (".tif", ".tiff")
            ):
                continue

            paths.append(
                f"/vsizip/{zip_path.resolve()}/{member}"
            )

    return paths


def discover_dops(
    dop_dir: Path,
) -> list[dict]:
    """
    Discover RGBI GeoTIFFs stored directly or inside ZIP archives.

    Raster bounds are recorded once so districts can later select only
    orthophotos intersecting their prediction crop.
    """

    if not dop_dir.exists():
        raise FileNotFoundError(
            f"DOP directory not found: {dop_dir}"
        )

    raster_paths = [
        str(path.resolve())
        for path in sorted(
            list(dop_dir.rglob("*.tif"))
            + list(dop_dir.rglob("*.tiff"))
        )
    ]

    for zip_path in sorted(
        dop_dir.rglob("*.zip")
    ):
        raster_paths.extend(
            zip_raster_paths(
                zip_path
            )
        )

    if not raster_paths:
        raise FileNotFoundError(
            f"No TIFFs or zipped TIFFs found in: {dop_dir}"
        )

    sources = []

    print(
        f"Scanning {len(raster_paths)} orthophotos..."
    )

    for index, raster_path in enumerate(
        raster_paths,
        start=1,
    ):
        with rasterio.open(
            raster_path
        ) as source:

            if source.count < 4:
                continue

            if str(source.crs) != TARGET_CRS:
                continue

            bounds = source.bounds

            sources.append(
                {
                    "path": raster_path,
                    "bounds": box(
                        bounds.left,
                        bounds.bottom,
                        bounds.right,
                        bounds.top,
                    ),
                }
            )

        if (
            index % 50 == 0
            or index == len(raster_paths)
        ):
            print(
                f"  scanned {index}/{len(raster_paths)}"
            )

    if not sources:
        raise ValueError(
            "No four-band RGBI orthophotos in EPSG:25833 were found."
        )

    print(
        f"Usable RGBI orthophotos: {len(sources)}"
    )

    return sources


# ---------------------------------------------------------------------
# District mosaic
# ---------------------------------------------------------------------

def align_bounds(
    bounds,
    *,
    expand_m: float,
) -> tuple[float, float, float, float]:
    """Expand and align crop bounds to the 20 cm raster grid."""

    minx, miny, maxx, maxy = bounds

    minx = (
        math.floor(
            (minx - expand_m)
            / RESOLUTION_M
        )
        * RESOLUTION_M
    )

    miny = (
        math.floor(
            (miny - expand_m)
            / RESOLUTION_M
        )
        * RESOLUTION_M
    )

    maxx = (
        math.ceil(
            (maxx + expand_m)
            / RESOLUTION_M
        )
        * RESOLUTION_M
    )

    maxy = (
        math.ceil(
            (maxy + expand_m)
            / RESOLUTION_M
        )
        * RESOLUTION_M
    )

    return (
        float(minx),
        float(miny),
        float(maxx),
        float(maxy),
    )


def select_sources(
    sources: list[dict],
    crop_bounds,
) -> list[dict]:
    """Select orthophotos intersecting one district crop."""

    crop_geometry = box(
        *crop_bounds
    )

    return [
        source
        for source in sources
        if source["bounds"].intersects(
            crop_geometry
        )
    ]


def ensure_shape(
    array: np.ndarray,
    expected_shape: tuple[int, int, int],
) -> np.ndarray:
    """Ensure a raster merge has the requested output shape."""

    if hasattr(
        array,
        "filled",
    ):
        array = array.filled(0)

    array = np.asarray(
        array
    )

    if array.shape == expected_shape:
        return array

    fixed = np.zeros(
        expected_shape,
        dtype=array.dtype,
    )

    channels = min(
        array.shape[0],
        expected_shape[0],
    )

    height = min(
        array.shape[1],
        expected_shape[1],
    )

    width = min(
        array.shape[2],
        expected_shape[2],
    )

    fixed[
        :channels,
        :height,
        :width,
    ] = array[
        :channels,
        :height,
        :width,
    ]

    return fixed


def build_district_mosaic(
    *,
    crop_path: Path,
    crop_bounds,
    sources: list[dict],
) -> None:
    """Build one temporary four-band RGBI mosaic for a district."""

    selected = select_sources(
        sources,
        crop_bounds,
    )

    if not selected:
        raise ValueError(
            f"No orthophotos intersect crop: {crop_bounds}"
        )

    xmin, ymin, xmax, ymax = (
        crop_bounds
    )

    width = int(
        round(
            (xmax - xmin)
            / RESOLUTION_M
        )
    )

    height = int(
        round(
            (ymax - ymin)
            / RESOLUTION_M
        )
    )

    transform = from_origin(
        xmin,
        ymax,
        RESOLUTION_M,
        RESOLUTION_M,
    )

    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 4,
        "dtype": "uint8",
        "crs": TARGET_CRS,
        "transform": transform,
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "BIGTIFF": "IF_SAFER",
        "nodata": 0,
    }

    crop_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    datasets = [
        rasterio.open(
            source["path"]
        )
        for source in selected
    ]

    try:
        with rasterio.open(
            crop_path,
            "w",
            **profile,
        ) as destination:

            chunks_x = math.ceil(
                width
                / MOSAIC_CHUNK_PX
            )

            chunks_y = math.ceil(
                height
                / MOSAIC_CHUNK_PX
            )

            for chunk_y in range(
                chunks_y
            ):
                for chunk_x in range(
                    chunks_x
                ):
                    col_start = (
                        chunk_x
                        * MOSAIC_CHUNK_PX
                    )

                    row_start = (
                        chunk_y
                        * MOSAIC_CHUNK_PX
                    )

                    col_end = min(
                        col_start
                        + MOSAIC_CHUNK_PX,
                        width,
                    )

                    row_end = min(
                        row_start
                        + MOSAIC_CHUNK_PX,
                        height,
                    )

                    chunk_width = (
                        col_end
                        - col_start
                    )

                    chunk_height = (
                        row_end
                        - row_start
                    )

                    chunk_bounds = (
                        xmin
                        + col_start
                        * RESOLUTION_M,

                        ymax
                        - row_end
                        * RESOLUTION_M,

                        xmin
                        + col_end
                        * RESOLUTION_M,

                        ymax
                        - row_start
                        * RESOLUTION_M,
                    )

                    chunk_geometry = box(
                        *chunk_bounds
                    )

                    active = [
                        dataset
                        for dataset, source
                        in zip(
                            datasets,
                            selected,
                        )
                        if source[
                            "bounds"
                        ].intersects(
                            chunk_geometry
                        )
                    ]

                    if active:
                        array, _ = merge(
                            active,
                            bounds=chunk_bounds,
                            res=(
                                RESOLUTION_M,
                                RESOLUTION_M,
                            ),
                            indexes=[
                                1,
                                2,
                                3,
                                4,
                            ],
                            nodata=0,
                        )

                        array = ensure_shape(
                            array,
                            (
                                4,
                                chunk_height,
                                chunk_width,
                            ),
                        )

                        if array.dtype != np.uint8:
                            array = (
                                np.clip(
                                    array,
                                    0,
                                    255,
                                )
                                .astype(
                                    np.uint8
                                )
                            )

                    else:
                        array = np.zeros(
                            (
                                4,
                                chunk_height,
                                chunk_width,
                            ),
                            dtype=np.uint8,
                        )

                    destination.write(
                        array,
                        window=Window(
                            col_start,
                            row_start,
                            chunk_width,
                            chunk_height,
                        ),
                    )

    finally:
        for dataset in datasets:
            dataset.close()


# ---------------------------------------------------------------------
# District outputs
# ---------------------------------------------------------------------

def assign_to_district(
    predictions: gpd.GeoDataFrame,
    *,
    district: str,
    district_geometry,
) -> tuple[
    gpd.GeoDataFrame,
    gpd.GeoDataFrame,
]:
    """
    Assign predictions by centroid.

    Full polygons are kept for the main crown dataset.
    A second version is clipped to the district boundary.
    """

    full_rows = []
    clipped_rows = []

    for _, row in predictions.iterrows():
        geometry = clean_all_parts(
            row.geometry
        )

        if (
            geometry is None
            or geometry.area
            < MIN_AREA_M2
        ):
            continue

        if not district_geometry.covers(
            geometry.centroid
        ):
            continue

        attributes = (
            row.drop(
                labels="geometry",
                errors="ignore",
            )
            .to_dict()
        )

        attributes["district"] = (
            district
        )

        attributes["area_m2"] = float(
            geometry.area
        )

        full_rows.append(
            {
                **attributes,
                "geometry": geometry,
            }
        )

        clipped = clean_all_parts(
            geometry.intersection(
                district_geometry
            )
        )

        if (
            clipped is not None
            and clipped.area
            >= MIN_AREA_M2
        ):
            clipped_attributes = dict(
                attributes
            )

            clipped_attributes[
                "area_m2"
            ] = float(
                clipped.area
            )

            clipped_rows.append(
                {
                    **clipped_attributes,
                    "geometry": clipped,
                }
            )

    full = gpd.GeoDataFrame(
        full_rows,
        geometry="geometry",
        crs=TARGET_CRS,
    )

    clipped = gpd.GeoDataFrame(
        clipped_rows,
        geometry="geometry",
        crs=TARGET_CRS,
    )

    return full, clipped


# ---------------------------------------------------------------------
# Complete Leipzig workflow
# ---------------------------------------------------------------------

def predict_leipzig_map(
    *,
    weights_path: Path,
    boundary_file: Path,
    dop_dir: Path,
    output_dir: Path,
    district_buffer_m: float,
    crop_expand_m: float,
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
    """Predict the Leipzig crown map district by district."""

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {weights_path}"
        )

    districts = load_districts(
        boundary_file
    )

    sources = discover_dops(
        dop_dir
    )

    model = YOLO(
        str(weights_path)
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    district_outputs = []
    summary_rows = []

    for district_number, (
        district_slug,
        district_name,
    ) in enumerate(
        DISTRICTS.items(),
        start=1,
    ):
        print()
        print("=" * 70)
        print(
            f"{district_number}/{len(DISTRICTS)} "
            f"{district_name}"
        )
        print("=" * 70)

        district_row = districts[
            districts["district"]
            == district_slug
        ].iloc[0]

        district_geometry = clean_all_parts(
            district_row.geometry
        )

        buffer_geometry = (
            district_geometry.buffer(
                district_buffer_m
            )
        )

        crop_bounds = align_bounds(
            buffer_geometry.bounds,
            expand_m=crop_expand_m,
        )

        district_dir = (
            output_dir
            / district_slug
        )

        work_dir = (
            district_dir
            / "_work"
        )

        crop_path = (
            work_dir
            / f"{district_slug}_rgbi_crop.tif"
        )

        print(
            "Building district RGBI mosaic..."
        )

        build_district_mosaic(
            crop_path=crop_path,
            crop_bounds=crop_bounds,
            sources=sources,
        )

        print(
            "Running sliding-window prediction..."
        )

        predictions, prediction_summary = (
            predict_multiphase(
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
        )

        full, clipped = assign_to_district(
            predictions,
            district=district_slug,
            district_geometry=district_geometry,
        )

        district_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        full_path = (
            district_dir
            / "crowns.gpkg"
        )

        clipped_path = (
            district_dir
            / "crowns_clipped.gpkg"
        )

        full.to_file(
            full_path,
            driver="GPKG",
        )

        clipped.to_file(
            clipped_path,
            driver="GPKG",
        )

        district_outputs.append(
            full
        )

        summary_rows.append(
            {
                "district": district_slug,
                "district_name": district_name,
                "crowns": len(full),
                "crown_area_m2": float(
                    full.geometry.area.sum()
                )
                if len(full)
                else 0.0,
                "output_file": str(
                    full_path
                ),
                **prediction_summary,
            }
        )

        print(
            f"Final crowns: {len(full)}"
        )

    # Merge the district-assigned crowns into the final Leipzig dataset.
    leipzig = gpd.GeoDataFrame(
        pd.concat(
            district_outputs,
            ignore_index=True,
        ),
        geometry="geometry",
        crs=TARGET_CRS,
    )

    leipzig.insert(
        0,
        "prediction_id",
        range(
            1,
            len(leipzig) + 1,
        ),
    )

    final_path = (
        output_dir
        / "leipzig_tree_crowns.gpkg"
    )

    leipzig.to_file(
        final_path,
        driver="GPKG",
    )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        output_dir
        / "district_summary.csv",
        index=False,
    )

    print()
    print("=" * 70)
    print("LEIPZIG MAP COMPLETE")
    print("=" * 70)
    print(
        f"Crowns: {len(leipzig)}"
    )
    print(
        f"Final map: {final_path}"
    )