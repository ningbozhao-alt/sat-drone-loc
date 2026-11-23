"""运行参数。瓦片地址可用环境变量 SATDRONE_TILE_URL 覆盖。"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TILE_CACHE = DATA / "tiles" / "online"
GEOTIFF_CACHE = DATA / "tiles" / "geotiff"
JOBS = DATA / "jobs"
OFFLINE_DIR = DATA / "offline"
SETTINGS_PATH = DATA / "settings.json"

DEFAULT_TILE_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)

MAX_MOSAIC_SIDE = 8192
MIN_GSD_M = 0.45
MAX_GSD_M = 2.2
TARGET_GSD_M = 1.0
MIN_INLIERS = 30
MAX_REPROJ_PX = 4.0
MIN_NCC = 0.35
MAX_QUERY_SIDE = 1600
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


def tile_url_template() -> str:
    return os.environ.get("SATDRONE_TILE_URL", DEFAULT_TILE_URL).strip()
