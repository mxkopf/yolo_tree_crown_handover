
# Prepare an area-stratified YOLO segmentation dataset.

# The reference data are organised by study area. Positive image tiles are split
# into train, validation and test sets while preserving representation from each
# area. The requested split sizes are first calculated for the complete dataset
# and then distributed proportionally across the study areas.

# Background tiles are sampled separately and added to the training set only.

# Expected input structure:
#     data/tree_crown_reference_data/
#         Auwald/
#             tiles/
#             labels/
#             pool_tiles/
#         Chemnitz/
#             ...
#         ...

# Output:
#     data/yolo_dataset/
#         images/train/
#         images/val/
#         images/test/
#         labels/train/
#         labels/val/
#         labels/test/
#         data.yaml


from pathlib import Path

from data_preparation.dataset import (
    create_area_stratified_split,
    collect_reference_tiles,
    print_summary,
    select_background_tiles,
    validate_dataset,
    write_dataset,
)


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42
N_BACKGROUND_TILES = 5

STUDY_AREAS = [
    "Auwald",
    "Chemnitz",
    "Hohes_Holz",
    "Kalebsberg",
    "Leipzig",
    "Rostock",
    "Tharandter_Wald",
]


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REFERENCE_DATA_DIR = PROJECT_ROOT / "data" / "tree_crown_reference_data"
OUTPUT_DIR = PROJECT_ROOT / "data" / "yolo_dataset"


def main() -> None:
    positive_tiles, background_candidates = collect_reference_tiles(
        REFERENCE_DATA_DIR,
        STUDY_AREAS,
    )

    splits, area_allocation = create_area_stratified_split(
        positive_tiles,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=SEED,
    )

    background_tiles = select_background_tiles(
        background_candidates,
        count=N_BACKGROUND_TILES,
        seed=SEED,
    )

    splits["train"].extend(background_tiles)

    summary = write_dataset(
        splits,
        OUTPUT_DIR,
    )

    validate_dataset(
        OUTPUT_DIR,
        expected_background_tiles=N_BACKGROUND_TILES,
    )

    print_summary(
        OUTPUT_DIR,
        area_allocation,
        summary,
        background_tiles,
    )


if __name__ == "__main__":
    main()