"""用镶嵌图自己的裁块做端到端自检。"""

from __future__ import annotations

import math

import cv2
import numpy as np

from app.localize import localize


def run_self_check(mosaic_bgr: np.ndarray) -> dict:
    height, width = mosaic_bgr.shape[:2]
    side = min(800, height // 2, width // 2)
    if side < 128:
        raise ValueError("镶嵌图太小，无法自检")
    x0 = width // 2 - side // 2
    y0 = height // 2 - side // 2
    crop = mosaic_bgr[y0 : y0 + side, x0 : x0 + side].copy()
    expected_x = x0 + (side - 1) / 2.0
    expected_y = y0 + (side - 1) / 2.0
    straight = localize(mosaic_bgr, crop)
    rotated = localize(mosaic_bgr, cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE))
    foreign = localize(mosaic_bgr, _decoy(side))
    straight_error = _error(straight, expected_x, expected_y)
    rotated_error = _error(rotated, expected_x, expected_y)
    passed = (
        straight.get("ok")
        and rotated.get("ok")
        and straight_error is not None
        and rotated_error is not None
        and straight_error < 3.0
        and rotated_error < 3.0
        and not foreign.get("ok")
    )
    return {
        "ok": bool(passed),
        "straight_error_px": straight_error,
        "rotated_error_px": rotated_error,
        "unrelated_rejected": not foreign.get("ok"),
        "unrelated_reason": foreign.get("reason") or "",
        "straight_reason": straight.get("reason") or "",
        "rotated_reason": rotated.get("reason") or "",
    }


def _error(result: dict, expected_x: float, expected_y: float) -> float | None:
    if not result.get("ok") or not result.get("center_px"):
        return None
    dx = result["center_px"][0] - expected_x
    dy = result["center_px"][1] - expected_y
    return float(math.hypot(dx, dy))


def _decoy(side: int) -> np.ndarray:
    rng = np.random.default_rng(2026)
    image = np.full((side, side, 3), 180, np.uint8)
    for index in range(12):
        cv2.circle(
            image,
            (int(side * (index + 1) / 13), side // 2),
            max(8, side // 20),
            (20, 20, 20),
            -1,
        )
    return image
