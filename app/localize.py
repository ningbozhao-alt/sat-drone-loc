"""金字塔粗匹配，再用 SIFT 单应把像素反算到镶嵌图。"""

from __future__ import annotations

import math

import cv2
import numpy as np

from app.config import MAX_QUERY_SIDE, MAX_REPROJ_PX, MIN_INLIERS, MIN_NCC

_SCALES = (1.0, 0.85, 1.15, 0.7, 1.4, 0.5, 2.0, 0.35, 3.0)
_SEARCH_SIDE = 720


def localize(mosaic_bgr: np.ndarray, query_bgr: np.ndarray) -> dict:
    """返回镶嵌图像素坐标、置信度和可保存的变换。失败时 ok 为 False。"""
    if mosaic_bgr is None or query_bgr is None or mosaic_bgr.size == 0 or query_bgr.size == 0:
        return _fail("图片为空")
    working, pre_scale = _limit_query(query_bgr)
    coarse = _coarse_match(mosaic_bgr, working)
    if coarse is None:
        return _fail("未找到可靠的粗匹配位置")
    refined = _refine(mosaic_bgr, working, coarse)
    homography = _sift_homography(refined["scaled"], refined["window"])
    if homography is None:
        return _fail("内点不足", ncc=refined["score"])
    matrix, inliers, median, good_count = homography
    if inliers < MIN_INLIERS:
        return _fail("内点不足", ncc=refined["score"], inliers=inliers, reproj=median)
    if median > MAX_REPROJ_PX:
        return _fail(
            "重投影误差过大",
            ncc=refined["score"],
            inliers=inliers,
            reproj=median,
        )
    transform = {
        "pre_scale": pre_scale,
        "orient": refined["orient"].tolist(),
        "scale_xy": refined["scale_xy"],
        "H": matrix.tolist(),
        "window_origin": [refined["x0"], refined["y0"]],
        "query_size": [int(query_bgr.shape[1]), int(query_bgr.shape[0])],
    }
    height, width = query_bgr.shape[:2]
    center = _map_points(transform, np.array([[(width - 1) / 2.0, (height - 1) / 2.0]]))[0]
    corners = _map_points(
        transform,
        np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float64,
        ),
    )
    area = _polygon_area(corners)
    expected = float(refined["scaled"].shape[0] * refined["scaled"].shape[1])
    if expected <= 0 or not (0.35 * expected <= area <= 3.0 * expected):
        return _fail("配准范围异常", ncc=refined["score"], inliers=inliers, reproj=median)
    mh, mw = mosaic_bgr.shape[:2]
    if not (0 <= center[0] < mw and 0 <= center[1] < mh):
        return _fail("配准中心落在任务区之外", ncc=refined["score"], inliers=inliers, reproj=median)
    ratio = inliers / max(good_count, 1)
    reproj_term = max(0.0, 1.0 - median / MAX_REPROJ_PX)
    ncc_term = max(0.0, min(1.0, (refined["score"] - MIN_NCC) / (1.0 - MIN_NCC)))
    confidence = 0.4 * ratio + 0.3 * reproj_term + 0.3 * ncc_term
    return {
        "ok": True,
        "reason": "",
        "center_px": [float(center[0]), float(center[1])],
        "corners_px": corners.tolist(),
        "inliers": inliers,
        "reproj_error_px": median,
        "confidence": float(confidence),
        "ncc": float(refined["score"]),
        "transform": transform,
    }


def map_query_pixel(transform: dict, x: float, y: float) -> tuple[float, float]:
    point = _map_points(transform, np.array([[x, y]], dtype=np.float64))[0]
    return float(point[0]), float(point[1])


def _fail(reason: str, ncc: float = 0.0, inliers: int = 0, reproj: float = 0.0) -> dict:
    return {
        "ok": False,
        "reason": reason,
        "center_px": None,
        "corners_px": None,
        "inliers": inliers,
        "reproj_error_px": reproj,
        "confidence": 0.0,
        "ncc": float(ncc),
        "transform": None,
    }


def _limit_query(query: np.ndarray) -> tuple[np.ndarray, float]:
    height, width = query.shape[:2]
    long_side = max(height, width)
    if long_side <= MAX_QUERY_SIDE:
        return query, [1.0, 1.0]
    scale = MAX_QUERY_SIDE / long_side
    resized = cv2.resize(
        query,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, [resized.shape[1] / width, resized.shape[0] / height]


def _coarse_match(mosaic_bgr: np.ndarray, query_bgr: np.ndarray) -> dict | None:
    mosaic_gray = _gray(mosaic_bgr)
    small, factors = _downsample(mosaic_gray, _SEARCH_SIDE)
    best = _search_angles(small, query_bgr, _SCALES[:1], (0, 90, 180, 270), factors)
    if best is None or best["score"] < 0.9:
        wider = _search_angles(small, query_bgr, _SCALES, (0, 90, 180, 270), factors)
        if wider is not None and (best is None or wider["score"] > best["score"]):
            best = wider
    if best is None or best["score"] < 0.2:
        return None
    return best


def _refine(mosaic_bgr: np.ndarray, query_bgr: np.ndarray, coarse: dict) -> dict:
    if coarse["score"] < 0.97:
        mosaic_gray = _gray(mosaic_bgr)
        small, factors = _downsample(mosaic_gray, _SEARCH_SIDE)
        angles = [((coarse["angle"] + delta) % 360) for delta in (-15, -10, -5, 5, 10, 15)]
        tuned = _search_angles(small, query_bgr, (coarse["scale"],), angles, factors)
        if tuned is not None and tuned["score"] > coarse["score"]:
            coarse = tuned
    oriented, orient = _orient_image(query_bgr, coarse["angle"])
    oh, ow = oriented.shape[:2]
    scale = coarse["scale"]
    sw = max(16, int(round(ow * scale)))
    sh = max(16, int(round(oh * scale)))
    scaled = cv2.resize(
        oriented,
        (sw, sh),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    x0, y0, x1, y1 = _window(mosaic_bgr.shape, coarse)
    return {
        "scaled": scaled,
        "window": mosaic_bgr[y0:y1, x0:x1],
        "orient": orient,
        "scale_xy": [sw / ow, sh / oh],
        "x0": x0,
        "y0": y0,
        "score": coarse["score"],
        "angle": coarse["angle"],
        "scale": scale,
    }


def _search_angles(mosaic_gray, query_bgr, scales, angles, factors) -> dict | None:
    factor_x, factor_y = factors
    best = None
    for scale in scales:
        for angle in angles:
            oriented, _matrix = _orient_image(_gray(query_bgr), angle)
            oh, ow = oriented.shape[:2]
            tw = int(round(ow * scale * factor_x))
            th = int(round(oh * scale * factor_y))
            if th < 24 or tw < 24:
                continue
            if th >= mosaic_gray.shape[0] or tw >= mosaic_gray.shape[1]:
                continue
            template = cv2.resize(oriented, (tw, th), interpolation=cv2.INTER_AREA)
            result = cv2.matchTemplate(mosaic_gray, template, cv2.TM_CCOEFF_NORMED)
            _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(result)
            if best is None or max_val > best["score"]:
                best = {
                    "score": float(max_val),
                    "angle": float(angle),
                    "scale": float(scale),
                    "x": max_loc[0] / factor_x,
                    "y": max_loc[1] / factor_y,
                    "w": tw / factor_x,
                    "h": th / factor_y,
                }
            if best["score"] >= 0.985:
                return best
    return best


def _window(shape, coarse: dict) -> tuple[int, int, int, int]:
    height, width = shape[:2]
    pad_x = coarse["w"] * 0.2
    pad_y = coarse["h"] * 0.2
    x0 = max(0, int(math.floor(coarse["x"] - pad_x)))
    y0 = max(0, int(math.floor(coarse["y"] - pad_y)))
    x1 = min(width, int(math.ceil(coarse["x"] + coarse["w"] + pad_x)))
    y1 = min(height, int(math.ceil(coarse["y"] + coarse["h"] + pad_y)))
    if x1 - x0 < 32:
        x1 = min(width, x0 + 32)
    if y1 - y0 < 32:
        y1 = min(height, y0 + 32)
    return x0, y0, x1, y1


def _sift_homography(src_bgr: np.ndarray, dst_bgr: np.ndarray):
    if src_bgr.size == 0 or dst_bgr.size == 0:
        return None
    if src_bgr.shape[0] >= dst_bgr.shape[0] or src_bgr.shape[1] >= dst_bgr.shape[1]:
        # 模板必须落在窗口内部，边界贴齐时留出 1 像素余量没有意义，SIFT 仍可在等尺寸上工作
        pass
    sift = cv2.SIFT_create(nfeatures=4000)
    keypoints_src, desc_src = sift.detectAndCompute(_gray(src_bgr), None)
    keypoints_dst, desc_dst = sift.detectAndCompute(_gray(dst_bgr), None)
    if desc_src is None or desc_dst is None or len(keypoints_src) < 8 or len(keypoints_dst) < 8:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(desc_src, desc_dst, k=2)
    good = []
    for pair in pairs:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < 0.75 * second.distance:
            good.append(first)
    if len(good) < MIN_INLIERS:
        return None
    src_pts = np.float32([keypoints_src[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_dst[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    cv2.setRNGSeed(0)
    matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0, maxIters=4000)
    if matrix is None or mask is None:
        return None
    inlier_mask = mask.ravel().astype(bool)
    inliers = int(inlier_mask.sum())
    projected = cv2.perspectiveTransform(src_pts, matrix)
    error = np.linalg.norm((projected - dst_pts).reshape(-1, 2), axis=1)
    median = float(np.median(error[inlier_mask])) if inliers else 1e9
    return matrix, inliers, median, len(good)


def _map_points(transform: dict, points: np.ndarray) -> np.ndarray:
    oriented = _apply(np.array(transform["orient"], dtype=np.float64), points * transform["pre_scale"])
    scale_x, scale_y = transform["scale_xy"]
    scaled = np.column_stack([oriented[:, 0] * scale_x, oriented[:, 1] * scale_y])
    projected = cv2.perspectiveTransform(scaled.reshape(-1, 1, 2).astype(np.float32), np.array(transform["H"], dtype=np.float64))
    projected = projected.reshape(-1, 2)
    origin = np.array(transform["window_origin"], dtype=np.float64)
    return projected + origin


def _orient_image(image: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    matrix = _orientation_matrix(width, height, angle_deg)
    quarter = int(round(angle_deg)) % 360
    if abs((angle_deg % 360) - quarter) < 1e-6 and quarter % 90 == 0:
        if quarter == 0:
            return image.copy(), matrix
        code = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }[quarter]
        return cv2.rotate(image, code), matrix
    corners = _apply(matrix, np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float64))
    min_xy = np.floor(corners.min(axis=0)).astype(int)
    max_xy = np.ceil(corners.max(axis=0)).astype(int)
    shift = np.eye(3)
    shift[0, 2] = -min_xy[0]
    shift[1, 2] = -min_xy[1]
    matrix = shift @ matrix
    out_w = max(1, int(max_xy[0] - min_xy[0] + 1))
    out_h = max(1, int(max_xy[1] - min_xy[1] + 1))
    warped = cv2.warpAffine(
        image,
        matrix[:2],
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return warped, matrix


def _orientation_matrix(width: int, height: int, angle_deg: float) -> np.ndarray:
    quarter = int(round(angle_deg)) % 360
    if abs((angle_deg % 360) - quarter) < 1e-6 and quarter % 90 == 0:
        if quarter == 0:
            return np.eye(3, dtype=np.float64)
        if quarter == 90:
            return np.array([[0, -1, height - 1], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        if quarter == 180:
            return np.array([[-1, 0, width - 1], [0, -1, height - 1], [0, 0, 1]], dtype=np.float64)
        return np.array([[0, 1, 0], [-1, 0, width - 1], [0, 0, 1]], dtype=np.float64)
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    affine = cv2.getRotationMatrix2D(center, -float(angle_deg), 1.0)
    matrix = np.eye(3, dtype=np.float64)
    matrix[:2] = affine
    return matrix


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate([points.astype(np.float64), np.ones((len(points), 1))], axis=1)
    transformed = (matrix @ homogeneous.T).T
    return transformed[:, :2] / transformed[:, 2:3]


def _polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _downsample(image: np.ndarray, max_side: int) -> tuple[np.ndarray, tuple[float, float]]:
    height, width = image.shape[:2]
    factor = min(1.0, max_side / max(height, width))
    if factor >= 0.999:
        return image, (1.0, 1.0)
    resized = cv2.resize(
        image,
        (max(1, int(round(width * factor))), max(1, int(round(height * factor)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, (resized.shape[1] / float(width), resized.shape[0] / float(height))
