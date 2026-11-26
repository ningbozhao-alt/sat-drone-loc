"""在线地名检索。结果用于把态势地图飞到该位置。"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

_HEADERS = {"User-Agent": "SatDroneLoc/1.0 (local visual localization)"}
_PHOTON = "https://photon.komoot.io/api/"


def search_places(query: str, limit: int = 6) -> list[dict]:
    text = " ".join(query.split())
    if not text:
        raise ValueError("请输入地名")
    if len(text) > 80:
        raise ValueError("地名过长")
    count = min(max(limit, 1), 8)
    url = _PHOTON + "?" + urllib.parse.urlencode({"q": text, "limit": count})
    request = urllib.request.Request(url, headers=_HEADERS)
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"地名检索失败：{exc}") from exc
    return places_from_photon(payload)


def places_from_photon(payload: dict) -> list[dict]:
    places = []
    for feature in payload.get("features") or []:
        geometry = feature.get("geometry") or {}
        coordinates = geometry.get("coordinates") or []
        if len(coordinates) < 2:
            continue
        lon, lat = float(coordinates[0]), float(coordinates[1])
        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
            continue
        props = feature.get("properties") or {}
        name = str(props.get("name") or "").strip()
        if not name:
            continue
        detail = [
            str(props[key]).strip()
            for key in ("city", "state", "country")
            if props.get(key) and str(props[key]).strip() != name
        ]
        places.append(
            {
                "name": name,
                "label": " · ".join([name, *detail]),
                "lon": lon,
                "lat": lat,
            }
        )
    return places
