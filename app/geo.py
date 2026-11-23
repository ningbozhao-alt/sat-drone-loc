"""XYZ 瓦片与 Web Mercator 像素互转。"""

from __future__ import annotations

import math

import mercantile

from app.config import MAX_GSD_M, MAX_MOSAIC_SIDE, MIN_GSD_M, TARGET_GSD_M

EARTH_CIRCUMFERENCE = 2 * math.pi * 6378137.0
TILE_SIZE = 256


def gsd_m(lat_deg: float, zoom: int) -> float:
    """该纬度、该 zoom 下的地面分辨率，单位米/像素。"""
    return (
        abs(math.cos(math.radians(lat_deg)))
        * EARTH_CIRCUMFERENCE
        / (TILE_SIZE * (2**zoom))
    )


def mercator_m_per_px(zoom: int) -> float:
    """EPSG:3857 下该 zoom 的米/像素，不随纬度变化。"""
    return EARTH_CIRCUMFERENCE / (TILE_SIZE * (2**zoom))


def tile_span(
    west: float, south: float, east: float, north: float, zoom: int
) -> tuple[int, int, int, int, int, int]:
    """用任务区四角所在的瓦片求范围，避免把整幅地图的瓦片枚举出来。"""
    northwest = mercantile.tile(west, north, zoom)
    southeast = mercantile.tile(min(east, 179.999999), max(south, -85.051128), zoom)
    x0, x1 = northwest.x, southeast.x
    y0, y1 = northwest.y, southeast.y
    if x1 < x0:
        raise ValueError("任务区跨过了日期变更线，请分成两块")
    if y1 < y0:
        y0, y1 = y1, y0
    width = (x1 - x0 + 1) * TILE_SIZE
    height = (y1 - y0 + 1) * TILE_SIZE
    return x0, y0, x1, y1, width, height


def select_zoom(
    west: float,
    south: float,
    east: float,
    north: float,
    max_side: int = MAX_MOSAIC_SIDE,
) -> int:
    """选取地面分辨率约 1 米、镶嵌图长边不超过 max_side 的 zoom。"""
    if not (west < east and south < north):
        raise ValueError("任务区范围无效，需要 west < east 且 south < north")
    if west < -180 or east > 180 or south < -85 or north > 85:
        raise ValueError("任务区经纬度超出可投影范围")
    lat = (south + north) / 2.0
    fitting: list[tuple[float, int]] = []
    for zoom in range(19, 9, -1):
        _x0, _y0, _x1, _y1, width, height = tile_span(west, south, east, north, zoom)
        if width > max_side or height > max_side or width < 64 or height < 64:
            continue
        gsd = gsd_m(lat, zoom)
        if MIN_GSD_M <= gsd <= MAX_GSD_M:
            fitting.append((abs(gsd - TARGET_GSD_M), zoom))
    if not fitting:
        raise ValueError(
            "任务区过大或过小。请画大约 5 公里见方的范围，"
            "使地面分辨率落在 0.5–2 米，且镶嵌图长边不超过 8192 像素"
        )
    fitting.sort()
    return fitting[0][1]


def mosaic_origin(tile_x: int, tile_y: int, zoom: int) -> tuple[float, float]:
    bounds = mercantile.xy_bounds(tile_x, tile_y, zoom)
    return float(bounds.left), float(bounds.top)


def pixel_to_lonlat(
    px: float, py: float, origin_x: float, origin_y: float, m_per_px: float
) -> tuple[float, float]:
    mx = origin_x + px * m_per_px
    my = origin_y - py * m_per_px
    lng, lat = mercantile.lnglat(mx, my)
    return float(lng), float(lat)


def lonlat_to_pixel(
    lon: float, lat: float, origin_x: float, origin_y: float, m_per_px: float
) -> tuple[float, float]:
    mx, my = mercantile.xy(lon, lat)
    px = (mx - origin_x) / m_per_px
    py = (origin_y - my) / m_per_px
    return float(px), float(py)
