from pathlib import Path


def get_project_root() -> Path:
    """Return the absolute path to the project root (parent of studies/)."""
    return Path(__file__).resolve().parent.parent.parent.parent


def get_config_dir() -> str:
    """Return the absolute path to the Hydra config directory."""
    return str(get_project_root() / "configs")
