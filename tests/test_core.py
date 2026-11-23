import math
import tempfile
import unittest
from pathlib import Path

import cv2
import mercantile
import numpy as np

from app.geo import gsd_m, mosaic_origin, pixel_to_lonlat, select_zoom, tile_span
from app.localize import _apply, _orient_image, localize
from app.mosaic import build_mosaic
from app.tiles import read_mbtiles


def _scene(seed: int, size: int = 1500) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = np.full((size, size, 3), 36, np.uint8)
    for _ in range(140):
        x, y = rng.integers(0, size - 100, size=2)
        w, h = rng.integers(16, 96, size=2)
        color = rng.integers(15, 255, size=3).tolist()
        cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, -1)
    for _ in range(50):
        center = rng.integers(20, size - 20, size=2)
        color = rng.integers(0, 255, size=3).tolist()
        cv2.circle(image, (int(center[0]), int(center[1])), int(rng.integers(6, 28)), color, -1)
    for _ in range(40):
        p1 = tuple(int(v) for v in rng.integers(0, size, size=2))
        p2 = tuple(int(v) for v in rng.integers(0, size, size=2))
        color = rng.integers(0, 255, size=3).tolist()
        cv2.line(image, p1, p2, color, int(rng.integers(1, 4)))
    return image


class GeoTests(unittest.TestCase):
    def test_select_zoom_for_five_kilometres(self):
        west, south, east, north = 116.30, 39.85, 116.36, 39.90
        zoom = select_zoom(west, south, east, north)
        gsd = gsd_m((south + north) / 2.0, zoom)
        _x0, _y0, _x1, _y1, width, height = tile_span(west, south, east, north, zoom)
        self.assertGreaterEqual(gsd, 0.45)
        self.assertLessEqual(gsd, 2.2)
        self.assertLessEqual(max(width, height), 8192)

    def test_huge_area_is_rejected(self):
        with self.assertRaises(ValueError):
            select_zoom(70, 15, 140, 55)

    def test_pixel_origin_is_tile_northwest(self):
        west, south, east, north = 116.30, 39.85, 116.36, 39.90

        def fetch(z, x, y):
            image = np.zeros((256, 256, 3), np.uint8)
            image[:] = (40, 80, 120)
            return image

        built = build_mosaic(fetch, west, south, east, north)
        lon, lat = pixel_to_lonlat(0, 0, built["origin_x"], built["origin_y"], built["m_per_px"])
        x0, y0, _x1, _y1, width, height = tile_span(west, south, east, north, built["zoom"])
        bounds = mercantile.bounds(x0, y0, built["zoom"])
        self.assertAlmostEqual(lon, bounds.west, places=6)
        self.assertAlmostEqual(lat, bounds.north, places=6)
        self.assertEqual(built["image"].shape[1], width)
        self.assertEqual(built["image"].shape[0], height)
        origin_x, origin_y = mosaic_origin(x0, y0, built["zoom"])
        self.assertAlmostEqual(origin_x, built["origin_x"])
        self.assertAlmostEqual(origin_y, built["origin_y"])


class TileTests(unittest.TestCase):
    def test_mbtiles_reads_tms_row(self):
        image = np.zeros((256, 256, 3), np.uint8)
        image[:] = (10, 20, 30)
        ok, encoded = cv2.imencode(".png", image)
        self.assertTrue(ok)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "demo.mbtiles"
            import sqlite3

            conn = sqlite3.connect(path)
            try:
                conn.execute(
                    "CREATE TABLE tiles (zoom_level INTEGER, tile_column INTEGER, tile_row INTEGER, tile_data BLOB)"
                )
                conn.execute(
                    "INSERT INTO tiles VALUES (?,?,?,?)",
                    (2, 1, (1 << 2) - 1 - 3, encoded.tobytes()),
                )
                conn.commit()
            finally:
                conn.close()
            raw = read_mbtiles(path, 2, 1, 3)
        self.assertIsNotNone(raw)
        decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(tuple(decoded[0, 0]), (10, 20, 30))


class LocalizeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mosaic = _scene(7, 1500)

    def test_orientation_matches_opencv_rotate(self):
        image = np.zeros((30, 50, 3), np.uint8)
        image[4, 7] = (255, 0, 0)
        for angle in (90, 180, 270):
            rotated, matrix = _orient_image(image, angle)
            x, y = _apply(matrix, np.array([[7.0, 4.0]]))[0]
            pixel = rotated[int(round(y)), int(round(x))]
            self.assertEqual(int(pixel[0]), 255, msg=f"angle {angle}")

    def test_center_crop_returns_within_three_pixels(self):
        side = 800
        x0 = self.mosaic.shape[1] // 2 - side // 2
        y0 = self.mosaic.shape[0] // 2 - side // 2
        crop = self.mosaic[y0 : y0 + side, x0 : x0 + side].copy()
        result = localize(self.mosaic, crop)
        self.assertTrue(result["ok"], result["reason"])
        expected_x = x0 + (side - 1) / 2.0
        expected_y = y0 + (side - 1) / 2.0
        error = math.hypot(result["center_px"][0] - expected_x, result["center_px"][1] - expected_y)
        self.assertLess(error, 3.0)

    def test_rotated_crop_returns_within_three_pixels(self):
        side = 800
        x0 = self.mosaic.shape[1] // 2 - side // 2
        y0 = self.mosaic.shape[0] // 2 - side // 2
        crop = self.mosaic[y0 : y0 + side, x0 : x0 + side].copy()
        rotated = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        result = localize(self.mosaic, rotated)
        self.assertTrue(result["ok"], result["reason"])
        expected_x = x0 + (side - 1) / 2.0
        expected_y = y0 + (side - 1) / 2.0
        error = math.hypot(result["center_px"][0] - expected_x, result["center_px"][1] - expected_y)
        self.assertLess(error, 3.0)

    def test_unrelated_image_fails(self):
        foreign = _scene(99, 800)
        result = localize(self.mosaic, foreign)
        self.assertFalse(result["ok"])


if __name__ == "__main__":
    unittest.main()
