"""Loguru setup — structured logs with rotation."""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parents[2]


def setup_logger(level: str = "INFO") -> None:
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level=level, colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> — {message}")
    logger.add(log_dir / "quant_pea.log", level="DEBUG",
               rotation="10 MB", retention="30 days", compression="gz",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} — {message}")


setup_logger()
