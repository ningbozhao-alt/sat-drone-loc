"""在线缓存、XYZ 目录、MBTiles 与 GeoTIFF 的统一瓦片读取。"""

from __future__ import annotations

import json
import sqlite3
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.config import (
    GEOTIFF_CACHE,
    OFFLINE_DIR,
    SETTINGS_PATH,
    TILE_CACHE,
    tile_url_template,
)

_HEADERS = {"User-Agent": "SatDroneLoc/1.0 (local visual localization)"}
_BLANK = None


def ensure_dirs() -> None:
    TILE_CACHE.mkdir(parents=True, exist_ok=True)
    GEOTIFF_CACHE.mkdir(parents=True, exist_ok=True)
    OFFLINE_DIR.mkdir(parents=True, exist_ok=True)


def load_settings() -> dict:
    ensure_dirs()
    if not SETTINGS_PATH.is_file():
        return {"tile_url": tile_url_template(), "offline": None}
    data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    data.setdefault("tile_url", tile_url_template())
    data.setdefault("offline", None)
    return data


def save_settings(data: dict) -> dict:
    ensure_dirs()
    current = load_settings()
    if "tile_url" in data and data["tile_url"]:
        url = str(data["tile_url"]).strip()
        if not all(token in url for token in ("{z}", "{x}", "{y}")):
            raise ValueError("瓦片 URL 需要包含 {z}、{x}、{y}")
        current["tile_url"] = url
    if "offline" in data:
        current["offline"] = data["offline"]
    SETTINGS_PATH.write_text(
        json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return current


def configure_offline(kind: str, path: Path) -> dict:
    if kind not in {"xyz", "mbtiles", "geotiff"}:
        raise ValueError("离线类型无效")
    resolved = path.expanduser().resolve()
    if kind == "xyz":
        if not resolved.is_dir():
            raise ValueError("XYZ 目录不存在")
    elif not resolved.is_file():
        raise ValueError("离线文件不存在")
    return save_settings({"offline": {"kind": kind, "path": str(resolved)}})


def tile_png(z: int, x: int, y: int, mode: str) -> bytes:
    """返回 PNG 字节。mode 为 online、cache 或 offline。"""
    if mode not in {"online", "cache", "offline"}:
        raise ValueError("底图来源无效")
    if mode == "offline":
        return _offline_png(z, x, y)
    cached = _cache_path(z, x, y)
    if cached.is_file() and cached.stat().st_size > 0:
        return cached.read_bytes()
    if mode == "cache":
        raise FileNotFoundError("该瓦片不在本地缓存中")
    png = _download_png(z, x, y)
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(png)
    return png


def tile_bgr(z: int, x: int, y: int, mode: str) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(tile_png(z, x, y, mode), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"瓦片 {z}/{x}/{y} 无法解码")
    if image.shape[0] != 256 or image.shape[1] != 256:
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    return image


def read_mbtiles(path: Path, z: int, x: int, y: int) -> bytes | None:
    uri = f"file:{path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        tms_y = (1 << z) - 1 - y
        for row_y in (tms_y, y):
            row = conn.execute(
                "SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?",
                (z, x, row_y),
            ).fetchone()
            if row and row[0]:
                return _to_png(row[0])
        return None
    finally:
        conn.close()


def _cache_path(z: int, x: int, y: int) -> Path:
    return TILE_CACHE / str(z) / str(x) / f"{y}.png"


def _download_png(z: int, x: int, y: int) -> bytes:
    template = load_settings()["tile_url"]
    url = (
        template.replace("{z}", str(z))
        .replace("{x}", str(x))
        .replace("{y}", str(y))
    )
    request = urllib.request.Request(url, headers=_HEADERS)
    last_error: Exception | None = None
    for _ in range(2):
        try:
            with urllib.request.urlopen(request, timeout=25) as response:
                return _to_png(response.read())
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
    raise RuntimeError(f"下载瓦片失败 {z}/{x}/{y}：{last_error}")


def _offline_png(z: int, x: int, y: int) -> bytes:
    offline = load_settings().get("offline")
    if not offline:
        raise FileNotFoundError("尚未导入离线底图")
    kind = offline["kind"]
    path = Path(offline["path"])
    if kind == "xyz":
        return _xyz_png(path, z, x, y)
    if kind == "mbtiles":
        data = read_mbtiles(path, z, x, y)
        if data is None:
            return _blank_png()
        return data
    if kind == "geotiff":
        return _geotiff_png(path, z, x, y)
    raise ValueError("离线类型无效")


def _xyz_png(root: Path, z: int, x: int, y: int) -> bytes:
    folder = root / str(z) / str(x)
    for name in (f"{y}.png", f"{y}.jpg", f"{y}.jpeg", f"{y}.webp"):
        file = folder / name
        if file.is_file():
            return _to_png(file.read_bytes())
    return _blank_png()


def _geotiff_png(path: Path, z: int, x: int, y: int) -> bytes:
    cached = GEOTIFF_CACHE / str(z) / str(x) / f"{y}.png"
    if cached.is_file() and cached.stat().st_size > 0:
        return cached.read_bytes()
    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.warp import transform_bounds
        from rasterio.windows import from_bounds
    except ImportError as exc:
        raise RuntimeError("读取 GeoTIFF 需要安装 rasterio") from exc
    import mercantile

    west, south, east, north = mercantile.bounds(x, y, z)
    with rasterio.open(path) as src:
        left, bottom, right, top = transform_bounds(
            "EPSG:4326", src.crs, west, south, east, north, densify_pts=21
        )
        window = from_bounds(left, bottom, right, top, transform=src.transform)
        window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        if window.width < 1 or window.height < 1:
            return _blank_png()
        data = src.read(
            indexes=_rgb_indexes(src.count),
            window=window,
            out_shape=(3, 256, 256),
            resampling=Resampling.bilinear,
            boundless=False,
            fill_value=0,
        )
    image = np.transpose(data, (1, 2, 0))
    if image.dtype != np.uint8:
        image = _scale_to_uint8(image)
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("GeoTIFF 瓦片编码失败")
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(encoded.tobytes())
    return encoded.tobytes()


def mosaic_from_geotiff(
    path: Path, west: float, south: float, east: float, north: float, zoom: int
) -> tuple[np.ndarray, float, float, float]:
    """把任务区从 GeoTIFF 重采样成镶嵌图，返回 BGR、origin_x、origin_y、m_per_px。"""
    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.warp import reproject, transform_bounds
    except ImportError as exc:
        raise RuntimeError("读取 GeoTIFF 需要安装 rasterio") from exc

    from app.geo import mercator_m_per_px, tile_span

    x0, y0, x1, y1, width, height = tile_span(west, south, east, north, zoom)
    import mercantile

    north_west = mercantile.xy_bounds(x0, y0, zoom)
    south_east = mercantile.xy_bounds(x1, y1, zoom)
    origin_x = float(north_west.left)
    origin_y = float(north_west.top)
    dst_bounds = (origin_x, float(south_east.bottom), float(south_east.right), origin_y)
    m_per_px = mercator_m_per_px(zoom)
    with rasterio.open(path) as src:
        left, bottom, right, top = transform_bounds(
            "EPSG:3857", src.crs, *dst_bounds, densify_pts=21
        )
        dst = np.zeros((3, height, width), dtype=np.float32)
        src_transform = src.transform
        reproject(
            source=src.read(indexes=_rgb_indexes(src.count)).astype(np.float32),
            destination=dst,
            src_transform=src_transform,
            src_crs=src.crs,
            dst_transform=rasterio.transform.from_bounds(
                dst_bounds[0], dst_bounds[1], dst_bounds[2], dst_bounds[3], width, height
            ),
            dst_crs="EPSG:3857",
            resampling=Resampling.bilinear,
        )
    image = np.transpose(dst, (1, 2, 0))
    if image.max(initial=0) <= 1.0:
        image = image * 255.0
    image = np.clip(image, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), origin_x, origin_y, m_per_px


def _rgb_indexes(count: int) -> list[int]:
    if count == 1:
        return [1, 1, 1]
    return [1, 2, 3]


def _scale_to_uint8(image: np.ndarray) -> np.ndarray:
    if image.max(initial=0) <= 1.0:
        image = image * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def _to_png(raw: bytes) -> bytes:
    with Image.open(__import__("io").BytesIO(raw)) as image:
        converted = image.convert("RGB")
        out = __import__("io").BytesIO()
        converted.save(out, format="PNG")
        return out.getvalue()


def _blank_png() -> bytes:
    global _BLANK
    if _BLANK is None:
        canvas = np.zeros((256, 256, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", canvas)
        if not ok:
            raise RuntimeError("空白瓦片编码失败")
        _BLANK = encoded.tobytes()
    return _BLANK
