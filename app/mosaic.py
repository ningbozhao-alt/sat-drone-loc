"""按任务区拼接带地理参考的卫星镶嵌图。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from app.config import MAX_MOSAIC_SIDE
from app.geo import gsd_m, mercator_m_per_px, mosaic_origin, select_zoom, tile_span


def build_mosaic(fetch_tile, west: float, south: float, east: float, north: float) -> dict:
    """fetch_tile(z, x, y) 返回 256×256 BGR。超出尺寸限制时抛出 ValueError。"""
    zoom = select_zoom(west, south, east, north, MAX_MOSAIC_SIDE)
    x0, y0, x1, y1, width, height = tile_span(west, south, east, north, zoom)
    coords = [(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]
    tiles: dict[tuple[int, int], np.ndarray] = {}
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(fetch_tile, zoom, x, y): (x, y) for x, y in coords}
        for future in as_completed(futures):
            x, y = futures[future]
            try:
                tiles[(x, y)] = future.result()
            except Exception as exc:  # 单张失败要汇总，避免线程里的异常被吞掉
                errors.append(f"{zoom}/{x}/{y}: {exc}")
    if errors:
        preview = "；".join(errors[:3])
        raise RuntimeError(f"有 {len(errors)} 张瓦片读取失败。{preview}")
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for (x, y), image in tiles.items():
        row = (y - y0) * 256
        col = (x - x0) * 256
        canvas[row : row + 256, col : col + 256] = image
    origin_x, origin_y = mosaic_origin(x0, y0, zoom)
    lat = (south + north) / 2.0
    return {
        "image": canvas,
        "zoom": zoom,
        "width": width,
        "height": height,
        "origin_x": origin_x,
        "origin_y": origin_y,
        "m_per_px": mercator_m_per_px(zoom),
        "gsd_m": gsd_m(lat, zoom),
        "tile_count": len(coords),
        "west": west,
        "south": south,
        "east": east,
        "north": north,
    }
