"""本地 Web：态势地图、任务区镶嵌、上传定位。"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.check import run_self_check
from app.config import MAX_UPLOAD_BYTES, OFFLINE_DIR, ROOT
from app.geo import gsd_m, pixel_to_lonlat, select_zoom
from app.jobs import create_job, job_dir, load_meta, load_mosaic, save_meta
from app.localize import localize, map_query_pixel
from app.mosaic import build_mosaic
from app.targets import clear_targets, delete_target, list_targets, upsert_target
from app.tiles import (
    configure_offline,
    ensure_dirs,
    load_settings,
    mosaic_from_geotiff,
    save_settings,
    tile_bgr,
    tile_png,
)

STATIC = ROOT / "static"


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    ensure_dirs()
    (ROOT / "data" / "jobs").mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="SatDroneLoc", lifespan=_lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class AoiBody(BaseModel):
    west: float
    south: float
    east: float
    north: float
    source: str = "online"


class PointBody(BaseModel):
    job_id: str
    x: float
    y: float


class JobBody(BaseModel):
    job_id: str


class SettingsBody(BaseModel):
    tile_url: str = Field(default="")


class TargetBody(BaseModel):
    id: str
    name: str
    note: str = ""
    lon: float
    lat: float
    kind: str = "target"


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/targets")
def get_targets():
    return list_targets()


@app.post("/api/targets")
def save_target(body: TargetBody):
    try:
        return upsert_target(body.model_dump())
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/targets/clear")
def clear_target_list():
    clear_targets()
    return {"ok": True}


@app.delete("/api/targets/{target_id}")
def remove_target(target_id: str):
    delete_target(target_id)
    return {"ok": True}


@app.get("/api/settings")
def get_settings():
    settings = load_settings()
    offline = settings.get("offline")
    return {
        "tile_url": settings.get("tile_url"),
        "offline": offline,
    }


@app.post("/api/settings")
def update_settings(body: SettingsBody):
    try:
        saved = save_settings({"tile_url": body.tile_url} if body.tile_url else {})
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"tile_url": saved["tile_url"]}


@app.get("/tiles/{z}/{x}/{y}.png")
def tiles(z: int, x: int, y: int, mode: str = "online"):
    try:
        payload = tile_png(z, x, y, mode)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from exc
    return Response(content=payload, media_type="image/png")


@app.post("/api/aoi")
async def create_aoi(body: AoiBody):
    try:
        meta = await asyncio.to_thread(
            _build_aoi, body.west, body.south, body.east, body.north, body.source
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from exc
    return meta


@app.post("/api/localize")
async def localize_photo(job_id: str = Form(...), file: UploadFile = File(...)):
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(400, "图片超过 50 MB")
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "无法读取图片，请上传 jpg、png 或 tif")
    try:
        mosaic = load_mosaic(job_id)
        meta = load_meta(job_id)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(404, str(exc)) from exc
    result = await asyncio.to_thread(localize, mosaic, image)
    payload = _place(result, meta)
    if result["ok"]:
        meta["transform"] = result["transform"]
        save_meta(job_id, meta)
    else:
        meta.pop("transform", None)
        save_meta(job_id, meta)
    return payload


@app.post("/api/point")
def point(body: PointBody):
    try:
        meta = load_meta(body.job_id)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(404, str(exc)) from exc
    transform = meta.get("transform")
    if not transform:
        raise HTTPException(400, "请先完成一次成功的定位")
    px, py = map_query_pixel(transform, body.x, body.y)
    lon, lat = pixel_to_lonlat(px, py, meta["origin_x"], meta["origin_y"], meta["m_per_px"])
    return {"ok": True, "lon": lon, "lat": lat, "pixel": {"x": px, "y": py}}


@app.post("/api/self-check")
async def self_check(body: JobBody):
    try:
        mosaic = load_mosaic(body.job_id)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(404, str(exc)) from exc
    try:
        return await asyncio.to_thread(run_self_check, mosaic)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/jobs/{job_id}/mosaic.png")
def mosaic_image(job_id: str):
    try:
        path = job_dir(job_id) / "mosaic.png"
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(404, str(exc)) from exc
    if not path.is_file():
        raise HTTPException(404, "镶嵌图不存在")
    return FileResponse(path, media_type="image/png")


@app.post("/api/offline")
async def offline(file: UploadFile | None = File(None), xyz_dir: str = Form("")):
    ensure_dirs()
    if xyz_dir.strip():
        try:
            saved = configure_offline("xyz", Path(xyz_dir.strip()))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return saved["offline"]
    if file is None or not file.filename:
        raise HTTPException(400, "请上传 MBTiles 或 GeoTIFF，或填写本机 XYZ 目录")
    suffix = Path(file.filename).suffix.lower()
    if suffix == ".mbtiles":
        kind = "mbtiles"
    elif suffix in {".tif", ".tiff"}:
        kind = "geotiff"
    else:
        raise HTTPException(400, "只接受 .mbtiles、.tif、.tiff")
    dest = OFFLINE_DIR / f"basemap{suffix}"
    written = 0
    with dest.open("wb") as handle:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            handle.write(chunk)
    if written == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "上传文件是空的")
    saved = configure_offline(kind, dest)
    return saved["offline"]


def _build_aoi(west: float, south: float, east: float, north: float, source: str) -> dict:
    if source not in {"online", "cache", "offline"}:
        raise ValueError("底图来源无效")
    offline = load_settings().get("offline")
    if source == "offline" and offline and offline.get("kind") == "geotiff":
        zoom = select_zoom(west, south, east, north)
        image, origin_x, origin_y, m_per_px = mosaic_from_geotiff(
            Path(offline["path"]), west, south, east, north, zoom
        )
        built = {
            "image": image,
            "zoom": zoom,
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "origin_x": origin_x,
            "origin_y": origin_y,
            "m_per_px": m_per_px,
            "gsd_m": gsd_m((south + north) / 2.0, zoom),
            "tile_count": 0,
            "west": west,
            "south": south,
            "east": east,
            "north": north,
        }
    else:
        built = build_mosaic(
            lambda z, x, y: tile_bgr(z, x, y, source),
            west,
            south,
            east,
            north,
        )
    image = built.pop("image")
    job_id = create_job(image, built)
    return {
        "job_id": job_id,
        "zoom": built["zoom"],
        "width": built["width"],
        "height": built["height"],
        "gsd_m": round(built["gsd_m"], 3),
        "tile_count": built["tile_count"],
    }


def _place(result: dict, meta: dict) -> dict:
    payload = {
        "ok": result["ok"],
        "reason": result["reason"],
        "inliers": result["inliers"],
        "reproj_error_px": result["reproj_error_px"],
        "confidence": result["confidence"],
        "ncc": result["ncc"],
        "center": None,
        "footprint": None,
    }
    if not result["ok"]:
        return payload
    payload["center"] = _lonlat(result["center_px"], meta)
    payload["footprint"] = [_lonlat(point, meta) for point in result["corners_px"]]
    return payload


def _lonlat(point, meta: dict) -> dict:
    lon, lat = pixel_to_lonlat(
        point[0], point[1], meta["origin_x"], meta["origin_y"], meta["m_per_px"]
    )
    return {"lon": lon, "lat": lat}
