"""已定位目标的名称、备注和经纬度。"""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path

from app.config import TARGETS_PATH

_LOCK = threading.Lock()


def list_targets(path: Path | None = None) -> list[dict]:
    with _LOCK:
        return _read(path or TARGETS_PATH)


def upsert_target(item: dict, path: Path | None = None) -> dict:
    stored = _normalize(item)
    file = path or TARGETS_PATH
    with _LOCK:
        rows = _read(file)
        for index, row in enumerate(rows):
            if row["id"] == stored["id"]:
                rows[index] = stored
                _write(file, rows)
                return stored
        rows.insert(0, stored)
        _write(file, rows)
        return stored


def delete_target(target_id: str, path: Path | None = None) -> bool:
    file = path or TARGETS_PATH
    with _LOCK:
        rows = _read(file)
        kept = [row for row in rows if row["id"] != target_id]
        if len(kept) == len(rows):
            return False
        _write(file, kept)
        return True


def clear_targets(path: Path | None = None) -> None:
    file = path or TARGETS_PATH
    with _LOCK:
        _write(file, [])


def _normalize(item: dict) -> dict:
    try:
        target_id = str(uuid.UUID(str(item["id"])))
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError("目标编号无效") from exc
    try:
        lon = float(item["lon"])
        lat = float(item["lat"])
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError("经纬度无效") from exc
    if not (-180 <= lon <= 180 and -85 <= lat <= 85):
        raise ValueError("经纬度超出范围")
    name = str(item.get("name") or "").strip()
    note = str(item.get("note") or "").strip()
    if not name:
        raise ValueError("名称不能为空")
    if len(name) > 40 or len(note) > 200:
        raise ValueError("名称或备注过长")
    kind = item.get("kind") if item.get("kind") in {"center", "target"} else "target"
    return {
        "id": target_id,
        "name": name,
        "note": note,
        "lon": lon,
        "lat": lat,
        "kind": kind,
    }


def _read(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []
    return data


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
