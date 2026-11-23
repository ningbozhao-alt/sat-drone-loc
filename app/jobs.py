"""一次任务区搜索的镶嵌图与配准结果。"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

import cv2
import numpy as np

from app.config import JOBS

_JOB_ID = re.compile(r"^[0-9a-f]{32}$")


def ensure_dirs() -> None:
    JOBS.mkdir(parents=True, exist_ok=True)


def new_id() -> str:
    return uuid.uuid4().hex


def job_dir(job_id: str) -> Path:
    if not _JOB_ID.fullmatch(job_id or ""):
        raise ValueError("任务编号无效")
    path = JOBS / job_id
    if not path.is_dir():
        raise FileNotFoundError("找不到该任务，请先建立搜索底图")
    return path


def create_job(mosaic_bgr: np.ndarray, meta: dict) -> str:
    ensure_dirs()
    job_id = new_id()
    path = JOBS / job_id
    path.mkdir(parents=True, exist_ok=False)
    if not cv2.imwrite(str(path / "mosaic.png"), mosaic_bgr):
        raise RuntimeError("镶嵌图写入失败")
    (path / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return job_id


def load_meta(job_id: str) -> dict:
    path = job_dir(job_id) / "meta.json"
    return json.loads(path.read_text(encoding="utf-8"))


def save_meta(job_id: str, meta: dict) -> None:
    path = job_dir(job_id) / "meta.json"
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_mosaic(job_id: str) -> np.ndarray:
    image = cv2.imread(str(job_dir(job_id) / "mosaic.png"), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError("镶嵌图已丢失，请重新建立搜索底图")
    return image
