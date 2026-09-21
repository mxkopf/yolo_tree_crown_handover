# Train the final YOLO26x-seg tree-crown model.

# The training configuration is stored in:
#     configs/training.yaml

# Use --dry-run to inspect the resolved configuration without starting
# model training.
# uv run python scripts/02_train_model.py
# or
# uv run python scripts/02_train_model.py --dry-run

import argparse
from pathlib import Path

from training.trainer import (
     load_training_config,
     print_training_config,
     resolve_training_paths,
     train_model,
 )


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "configs" / "training.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the training configuration without starting training.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_training_config(CONFIG_PATH)

    config = resolve_training_paths(
        config,
        PROJECT_ROOT,
    )

    print_training_config(config)

    if args.dry_run:
        print("Dry run only. Training was not started.")
        return

    train_model(config)


if __name__ == "__main__":
    main()