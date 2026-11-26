import unittest

from app.geocode import places_from_photon


class GeocodeParseTests(unittest.TestCase):
    def test_reads_name_and_coordinates(self):
        payload = {
            "features": [
                {
                    "geometry": {"coordinates": [121.55, 29.87]},
                    "properties": {"name": "宁波市", "state": "浙江省", "country": "中国"},
                },
                {"geometry": {"coordinates": []}, "properties": {"name": "缺坐标"}},
            ]
        }
        places = places_from_photon(payload)
        self.assertEqual(len(places), 1)
        self.assertEqual(places[0]["name"], "宁波市")
        self.assertEqual(places[0]["label"], "宁波市 · 浙江省 · 中国")
        self.assertAlmostEqual(places[0]["lon"], 121.55)
        self.assertAlmostEqual(places[0]["lat"], 29.87)


if __name__ == "__main__":
    unittest.main()
