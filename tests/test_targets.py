import tempfile
import unittest
import uuid
from pathlib import Path

from app.targets import clear_targets, delete_target, list_targets, upsert_target


class TargetStoreTests(unittest.TestCase):
    def test_upsert_lists_newest_first_and_updates_label(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "targets.json"
            first = _item("东门")
            second = _item("桥")
            upsert_target(first, path)
            upsert_target(second, path)
            self.assertEqual([row["name"] for row in list_targets(path)], ["桥", "东门"])
            second["note"] = "北侧"
            upsert_target(second, path)
            rows = list_targets(path)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["note"], "北侧")

    def test_delete_and_clear(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "targets.json"
            item = _item("塔")
            upsert_target(item, path)
            self.assertTrue(delete_target(item["id"], path))
            self.assertFalse(delete_target(item["id"], path))
            upsert_target(item, path)
            clear_targets(path)
            self.assertEqual(list_targets(path), [])


def _item(name: str) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "name": name,
        "note": "",
        "lon": 116.39,
        "lat": 39.91,
        "kind": "target",
    }


if __name__ == "__main__":
    unittest.main()
