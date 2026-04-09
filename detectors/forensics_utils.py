from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class QualityMetrics:
    blur_score: float
    blockiness: float
    noise_level: float
    dynamic_range: float
    quality_risk: float


def clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def normalize(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return clamp01((value - low) / (high - low))


def decode_bgr_image(image_bytes: bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    if nparr.size == 0:
        return None
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def decode_gray_image(image_bytes: bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    if nparr.size == 0:
        return None
    return cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)


def ensure_gray(image):
    if image is None:
        return None
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def central_crop(image, ratio: float = 0.6):
    if image is None or image.size == 0:
        return image
    h, w = image.shape[:2]
    crop_h = max(int(h * ratio), 1)
    crop_w = max(int(w * ratio), 1)
    y1 = max((h - crop_h) // 2, 0)
    x1 = max((w - crop_w) // 2, 0)
    return image[y1 : y1 + crop_h, x1 : x1 + crop_w]


def detect_largest_face(gray_image, face_cascade) -> Optional[Tuple[int, int, int, int]]:
    if gray_image is None or face_cascade.empty():
        return None
    faces = face_cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(48, 48),
    )
    if len(faces) == 0:
        return None
    return max(faces, key=lambda face: int(face[2]) * int(face[3]))


def crop_face_region(image, face, pad_x_ratio: float = 0.18, pad_y_ratio: float = 0.22):
    if image is None or image.size == 0:
        return image
    h, w = image.shape[:2]
    if face is None:
        return central_crop(image, ratio=0.65)

    x, y, fw, fh = face
    pad_x = int(fw * pad_x_ratio)
    pad_y = int(fh * pad_y_ratio)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + fw + pad_x)
    y2 = min(h, y + fh + pad_y)
    return image[y1:y2, x1:x2]


def estimate_quality_metrics(image) -> QualityMetrics:
    gray = ensure_gray(image)
    if gray is None or gray.size == 0:
        return QualityMetrics(
            blur_score=0.0,
            blockiness=1.0,
            noise_level=1.0,
            dynamic_range=0.0,
            quality_risk=1.0,
        )

    gray_f = gray.astype(np.float32)
    lap_var = float(cv2.Laplacian(gray_f, cv2.CV_32F).var())
    blur_score = normalize(lap_var, 40.0, 260.0)

    diff_h = np.abs(np.diff(gray_f, axis=1))
    diff_v = np.abs(np.diff(gray_f, axis=0))
    block_cols = list(range(7, gray_f.shape[1] - 1, 8))
    block_rows = list(range(7, gray_f.shape[0] - 1, 8))
    block_h = float(np.mean(diff_h[:, block_cols])) if block_cols else 0.0
    block_v = float(np.mean(diff_v[block_rows, :])) if block_rows else 0.0
    smooth_h = float(np.mean(diff_h)) + 1e-5
    smooth_v = float(np.mean(diff_v)) + 1e-5
    blockiness = normalize(0.5 * (block_h / smooth_h + block_v / smooth_v), 1.0, 2.2)

    blurred = cv2.GaussianBlur(gray_f, (3, 3), 0)
    noise_residual = gray_f - blurred
    noise_level = normalize(float(np.std(noise_residual)), 2.0, 18.0)

    p5, p95 = np.percentile(gray_f, [5, 95])
    dynamic_range = normalize(float((p95 - p5) / 255.0), 0.18, 0.75)

    quality_risk = clamp01(
        0.35 * blockiness
        + 0.25 * (1.0 - blur_score)
        + 0.20 * noise_level
        + 0.20 * (1.0 - dynamic_range)
    )

    return QualityMetrics(
        blur_score=blur_score,
        blockiness=blockiness,
        noise_level=noise_level,
        dynamic_range=dynamic_range,
        quality_risk=quality_risk,
    )


def build_skin_mask(roi):
    if roi is None or roi.size == 0:
        return None
    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)
    mask = cv2.medianBlur(mask, 5)
    return (mask > 0).astype(np.uint8)


def compute_chroma_noise_inconsistency(roi) -> float:
    if roi is None or roi.size == 0:
        return 0.0

    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]

    h, w = cr.shape
    gh, gw = max(h // 4, 1), max(w // 4, 1)
    local_vars = []
    for yi in range(0, h, gh):
        for xi in range(0, w, gw):
            cr_patch = cr[yi : yi + gh, xi : xi + gw]
            cb_patch = cb[yi : yi + gh, xi : xi + gw]
            if cr_patch.size < 16:
                continue
            local_vars.append(float(np.var(cr_patch) + np.var(cb_patch)))

    if len(local_vars) < 2:
        return 0.0

    local_vars = np.array(local_vars, dtype=np.float32)
    dispersion = float(np.std(local_vars) / (np.mean(local_vars) + 1e-5))
    return normalize(dispersion, 0.08, 0.90)


def compute_spectral_profile(gray_image) -> Dict[str, float]:
    if gray_image is None or gray_image.size == 0:
        return {
            "raw_score": 0.0,
            "score": 0.0,
            "global_high_ratio": 0.0,
            "patch_inconsistency": 0.0,
            "patch_peakiness": 0.0,
        }

    resized = cv2.resize(gray_image, (256, 256), interpolation=cv2.INTER_AREA)
    resized = resized.astype(np.float32) / 255.0
    window = np.outer(np.hanning(256), np.hanning(256)).astype(np.float32)

    spectrum = np.fft.fftshift(np.fft.fft2(resized * window))
    magnitude = np.log1p(np.abs(spectrum))

    yy, xx = np.indices(magnitude.shape)
    cy, cx = np.array(magnitude.shape) // 2
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    radius = radius / (radius.max() + 1e-6)

    low_mask = radius < 0.10
    mid_mask = (radius >= 0.10) & (radius < 0.30)
    high_mask = (radius >= 0.30) & (radius < 0.50)

    low_energy = float(magnitude[low_mask].mean()) + 1e-6
    mid_energy = float(magnitude[mid_mask].mean()) + 1e-6
    high_energy = float(magnitude[high_mask].mean())
    global_high_ratio = high_energy / (0.5 * low_energy + 0.5 * mid_energy)

    patch_scores = []
    patch_peakiness = []
    for patch in iter_patches(resized, rows=6, cols=6, min_size=24):
        patch_window = np.outer(
            np.hanning(patch.shape[0]),
            np.hanning(patch.shape[1]),
        ).astype(np.float32)
        patch_spectrum = np.fft.fftshift(np.fft.fft2(patch * patch_window))
        patch_mag = np.log1p(np.abs(patch_spectrum))

        py, px = np.indices(patch_mag.shape)
        pcy, pcx = np.array(patch_mag.shape) // 2
        pr = np.sqrt((py - pcy) ** 2 + (px - pcx) ** 2)
        pr = pr / (pr.max() + 1e-6)
        patch_high = patch_mag[(pr >= 0.28) & (pr < 0.50)]
        patch_mid = patch_mag[(pr >= 0.12) & (pr < 0.28)]
        if patch_high.size < 8 or patch_mid.size < 8:
            continue

        patch_ratio = float(patch_high.mean() / (patch_mid.mean() + 1e-6))
        patch_scores.append(patch_ratio)
        patch_peakiness.append(
            float(np.percentile(patch_high, 95) / (patch_high.mean() + 1e-6))
        )

    if len(patch_scores) < 2:
        patch_inconsistency = 0.0
        patch_peak = 0.0
    else:
        patch_scores_arr = np.array(patch_scores, dtype=np.float32)
        patch_peaks_arr = np.array(patch_peakiness, dtype=np.float32)
        patch_inconsistency = normalize(
            float(np.std(patch_scores_arr) / (np.mean(patch_scores_arr) + 1e-6)),
            0.05,
            0.45,
        )
        patch_peak = normalize(float(np.median(patch_peaks_arr)), 1.05, 2.40)

    score = clamp01(
        0.45 * normalize(global_high_ratio, 0.20, 1.15)
        + 0.35 * patch_inconsistency
        + 0.20 * patch_peak
    )

    return {
        "raw_score": 10.0 * score,
        "score": score,
        "global_high_ratio": float(global_high_ratio),
        "patch_inconsistency": float(patch_inconsistency),
        "patch_peakiness": float(patch_peak),
    }


def iter_patches(image, rows: int, cols: int, min_size: int = 16) -> List[np.ndarray]:
    h, w = image.shape[:2]
    patch_h = max(h // rows, 1)
    patch_w = max(w // cols, 1)
    patches = []
    for yi in range(0, h, patch_h):
        for xi in range(0, w, patch_w):
            patch = image[yi : yi + patch_h, xi : xi + patch_w]
            if patch.shape[0] < min_size or patch.shape[1] < min_size:
                continue
            patches.append(patch)
    return patches
