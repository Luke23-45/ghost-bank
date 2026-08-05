"""Configuration loader (OmegaConf-backed, mirrors the reference framework)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

_CACHE: Optional[DictConfig] = None
_CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"     # studies/analysis/configs
_DEFAULT_CONFIG = _CONFIGS_DIR / "base.yaml"


def _detect_project_root() -> Path:
    """src/common/config.py -> studies/analysis/src/common -> repo root."""
    return Path(__file__).resolve().parents[4]


def get_config(
    config_path: Optional[Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
    *,
    use_cache: bool = True,
) -> DictConfig:
    """Load, resolve, and cache the analysis configuration."""
    global _CACHE

    if use_cache and _CACHE is not None and overrides is None:
        return _CACHE

    cfg_path = config_path or _DEFAULT_CONFIG
    if not cfg_path.exists():
        cfg = OmegaConf.create({
            "paths": {"project_root": None, "data_root": None, "output_root": "studies/analysis/outputs"},
        })
    else:
        cfg = OmegaConf.load(cfg_path)

    assert isinstance(cfg, DictConfig)

    project_root = _detect_project_root()
    OmegaConf.update(cfg, "paths.project_root", str(project_root.resolve()))

    if cfg.paths.data_root is None:
        data_root = project_root / "experiment_output"
        OmegaConf.update(cfg, "paths.data_root", str(data_root.resolve()))

    if overrides:
        override_cfg = OmegaConf.from_dotlist([f"{k}={v}" for k, v in overrides.items()])
        cfg = OmegaConf.merge(cfg, override_cfg)

    if overrides is None:
        _CACHE = cfg

    logger.debug("Configuration loaded: %s", cfg)
    return cfg


def get_data_root(cfg: Optional[DictConfig] = None) -> Path:
    if cfg is None:
        cfg = get_config()
    root = Path(cfg.paths.data_root)
    if not root.is_absolute():
        root = Path(cfg.paths.project_root) / root
    return root.resolve()


def get_output_root(cfg: Optional[DictConfig] = None) -> Path:
    if cfg is None:
        cfg = get_config()
    out_dir = Path(cfg.paths.project_root) / cfg.paths.output_root
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
