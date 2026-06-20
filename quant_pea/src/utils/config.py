"""Central configuration loader with environment variable support."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


@lru_cache(maxsize=1)
def get_config() -> dict:
    cfg_path = ROOT / "config" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Inject env vars where needed
    fred_key = os.getenv("FRED_API_KEY", "")
    if fred_key:
        cfg["data_sources"]["fred"]["api_key"] = fred_key
    return cfg


def cfg() -> dict:
    return get_config()
