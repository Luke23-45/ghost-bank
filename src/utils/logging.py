from __future__ import annotations

import logging
import os
import sys
import warnings


_LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
    "none": logging.CRITICAL + 10,
}


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def setup_logging(level: str = "info") -> None:
    """Globally configure all logging for the project.

    Parameters
    ----------
    level : str, optional
        One of ``"debug"``, ``"info"``, ``"warning"``, ``"error"``,
        ``"critical"``, ``"none"``.

        - ``"info"`` (default) — balanced output (epoch bars, PL banners).
        - ``"warning"`` — suppress per-epoch output, PL banners, and
          informational messages; keep warnings and errors.
        - ``"error"`` — suppress everything except errors.
        - ``"none"`` — suppress all Python + PL output.
        - ``"debug"`` — full verbose output.

    The environment variable ``GHOST_BANK_LOG_LEVEL`` takes precedence
    over the *level* argument so operators can silence output at
    deployment time without touching config files.
    """
    env_level = os.environ.get("GHOST_BANK_LOG_LEVEL")
    if env_level is not None and env_level in _LOG_LEVELS:
        level = env_level.lower()

    py_level = _LOG_LEVELS.get(level, logging.INFO)

    _quiet = py_level > logging.INFO
    _silent = py_level >= _LOG_LEVELS["error"]
    _dead = py_level >= _LOG_LEVELS["none"]

    # --- Reset any prior global disable (in case level was "none" before) ----
    logging.disable(logging.NOTSET)

    # --- Python-level loggers ------------------------------------------------
    root = logging.getLogger("src")
    for h in root.handlers:
        h.setLevel(py_level)
    root.setLevel(py_level)
    root.propagate = False

    if _dead:
        logging.disable(logging.CRITICAL + 10)

    # --- PyTorch Lightning / Lightning ---------------------------------------
    for ns in ("lightning", "pytorch_lightning"):
        pl_logger = logging.getLogger(ns)
        pl_logger.setLevel(logging.WARNING if _quiet else py_level)
        pl_logger.propagate = False

    # --- Suppress common noisy warnings --------------------------------------
    if _quiet:
        warnings.filterwarnings(
            "ignore",
            message="This DataLoader will create",
        )
        warnings.filterwarnings(
            "ignore",
            message=".*You defined a `validation_step`.*",
        )
        warnings.filterwarnings(
            "ignore",
            message=".*Found \\d+ module\\(s\\) in eval mode.*",
        )
        warnings.filterwarnings(
            "ignore",
            message=".*Consider increasing the capacity of your GPU.*",
        )
