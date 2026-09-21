from pathlib import Path

import yaml
from ultralytics import YOLO


def load_training_config(config_path: Path) -> dict:
    """Load the YOLO training configuration."""

    if not config_path.exists():
        raise FileNotFoundError(
            f"Training config not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            f"Invalid training config: {config_path}"
        )

    return config


def resolve_training_paths(
    config: dict,
    project_root: Path,
) -> dict:
    """Resolve repository-relative paths in the training configuration."""

    config = config.copy()

    data_path = Path(config["data"])

    if not data_path.is_absolute():
        data_path = project_root / data_path

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found: {data_path}"
        )

    config["data"] = str(data_path.resolve())

    project_path = Path(config["project"])

    if not project_path.is_absolute():
        project_path = project_root / project_path

    config["project"] = str(project_path.resolve())

    return config


def print_training_config(config: dict) -> None:
    """Print the resolved training configuration."""

    print()
    print("YOLO training configuration")
    print("-" * 58)

    for key, value in config.items():
        print(f"{key:<20} {value}")

    print()


def train_model(config: dict) -> None:
    """Train the YOLO segmentation model."""

    config = config.copy()

    model_name = config.pop("model")

    model = YOLO(model_name)

    model.train(**config)