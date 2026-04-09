import cv2
import numpy as np

from detector_config import get_detector_config
from detectors.base import BaseDetector, DetectorResult
from detectors.forensics_utils import (
    build_skin_mask,
    clamp01,
    compute_chroma_noise_inconsistency,
    crop_face_region,
    decode_bgr_image,
    detect_largest_face,
    ensure_gray,
    estimate_quality_metrics,
    normalize,
)
from detectors.registry import DetectorRegistry


@DetectorRegistry.register(name="AppearanceDetector", device="cuda")
class AppearanceDetector(BaseDetector):
    name = "AppearanceDetector"
    SETTINGS = get_detector_config(name)

    def _load_model(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )

    def detect(self, image_bytes: bytes) -> DetectorResult:
        bgr = decode_bgr_image(image_bytes)
        if bgr is None:
            return DetectorResult(
                name=self.name,
                features={
                    "lighting_conflict": 0.0,
                    "pose_extreme": 0.0,
                    "symmetry_break": 0.0,
                },
                meta={"face_detected": False, "decode_failed": True},
            )

        gray = ensure_gray(bgr)
        face = detect_largest_face(gray, self.face_cascade)
        face_confidence = self._face_confidence(bgr, face)
        if face is None or face_confidence < 0.52:
            quality = estimate_quality_metrics(bgr)
            return DetectorResult(
                name=self.name,
                features={
                    "lighting_conflict": 0.0,
                    "pose_extreme": 0.0,
                    "symmetry_break": 0.0,
                },
                meta={
                    "face_detected": False,
                    "unsupported_input": True,
                    "input_scope": "non_portrait",
                    "face_confidence": round(float(face_confidence), 6),
                    "quality_risk": round(float(quality.quality_risk), 6),
                    "blur_score": round(float(quality.blur_score), 6),
                    "blockiness": round(float(quality.blockiness), 6),
                    "lighting_detail": 0.0,
                    "symmetry_break": 0.0,
                    "edge_halo": 0.0,
                    "chroma_noise": 0.0,
                },
            )

        face_roi = crop_face_region(bgr, face)
        photo_texture_score = self._photo_texture_score(face_roi)
        human_face_score = self._human_face_score(face_roi, photo_texture_score)
        if human_face_score < 0.40:
            quality = estimate_quality_metrics(face_roi)
            return DetectorResult(
                name=self.name,
                features={
                    "lighting_conflict": 0.0,
                    "pose_extreme": 0.0,
                    "symmetry_break": 0.0,
                },
                meta={
                    "face_detected": True,
                    "unsupported_input": True,
                    "input_scope": "non_human_face",
                    "face_confidence": round(float(face_confidence), 6),
                    "human_face_score": round(float(human_face_score), 6),
                    "photo_texture_score": round(float(photo_texture_score), 6),
                    "quality_risk": round(float(quality.quality_risk), 6),
                    "blur_score": round(float(quality.blur_score), 6),
                    "blockiness": round(float(quality.blockiness), 6),
                    "lighting_detail": 0.0,
                    "symmetry_break": 0.0,
                    "edge_halo": 0.0,
                    "chroma_noise": round(
                        float(compute_chroma_noise_inconsistency(face_roi)),
                        6,
                    ),
                },
            )

        quality = estimate_quality_metrics(face_roi)
        symmetry_break = self._symmetry_break(face_roi)

        lighting_conflict = self._lighting_conflict(face_roi, quality.quality_risk)
        artifact_proxy = self._artifact_proxy(face_roi, quality.quality_risk)

        return DetectorResult(
            name=self.name,
            features={
                "lighting_conflict": round(float(lighting_conflict), 6),
                "pose_extreme": round(float(artifact_proxy), 6),
                "symmetry_break": round(float(symmetry_break), 6),
            },
            meta={
                "face_detected": True,
                "unsupported_input": False,
                "input_scope": "human_portrait",
                "face_confidence": round(float(face_confidence), 6),
                "human_face_score": round(float(human_face_score), 6),
                "photo_texture_score": round(float(photo_texture_score), 6),
                "quality_risk": round(float(quality.quality_risk), 6),
                "blur_score": round(float(quality.blur_score), 6),
                "blockiness": round(float(quality.blockiness), 6),
                "lighting_detail": round(
                    float(self._illumination_asymmetry(face_roi)),
                    6,
                ),
                "symmetry_break": round(float(symmetry_break), 6),
                "edge_halo": round(float(self._edge_halo(face_roi)), 6),
                "chroma_noise": round(
                    float(compute_chroma_noise_inconsistency(face_roi)),
                    6,
                ),
            },
        )

    def _face_confidence(self, image, face) -> float:
        if image is None or image.size == 0 or face is None:
            return 0.0

        h, w = image.shape[:2]
        x, y, fw, fh = face
        if fw <= 0 or fh <= 0 or h <= 0 or w <= 0:
            return 0.0

        area_ratio = float((fw * fh) / max(h * w, 1))
        aspect_ratio = float(fw / max(fh, 1))
        face_roi = image[y : y + fh, x : x + fw]
        skin_mask = build_skin_mask(face_roi)
        skin_ratio = float(np.mean(skin_mask)) if skin_mask is not None else 0.0

        area_score = normalize(area_ratio, 0.015, 0.22)
        aspect_score = clamp01(1.0 - abs(aspect_ratio - 0.95) / 0.75)
        skin_score = normalize(skin_ratio, 0.03, 0.32)
        return clamp01(0.35 * area_score + 0.25 * aspect_score + 0.40 * skin_score)

    def _human_face_score(self, roi, photo_texture_score: float | None = None) -> float:
        if roi is None or roi.size == 0:
            return 0.0

        gray = ensure_gray(roi)
        eyes = self.eye_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(10, 10),
        )
        eye_score = clamp01(len(eyes) / 2.0)

        skin_mask = build_skin_mask(roi)
        skin_ratio = float(np.mean(skin_mask)) if skin_mask is not None else 0.0
        skin_score = normalize(skin_ratio, 0.05, 0.38)

        if photo_texture_score is None:
            photo_texture_score = self._photo_texture_score(roi)
        return clamp01(
            0.45 * eye_score
            + 0.30 * skin_score
            + 0.25 * photo_texture_score
        )

    @staticmethod
    def _photo_texture_score(roi) -> float:
        if roi is None or roi.size == 0:
            return 0.0

        quality = estimate_quality_metrics(roi)
        chroma_noise = compute_chroma_noise_inconsistency(roi)
        return clamp01(
            0.45 * normalize(chroma_noise, 0.01, 0.12)
            + 0.30 * normalize(quality.noise_level, 0.05, 0.42)
            + 0.25 * normalize(quality.dynamic_range, 0.18, 0.72)
        )

    def _lighting_conflict(self, roi, quality_risk: float) -> float:
        edge_halo = self._edge_halo(roi)
        shadow_mismatch = self._shadow_gradient_mismatch(roi)
        gray = ensure_gray(roi)
        blockiness = estimate_quality_metrics(gray).blockiness if gray is not None else 0.0
        score = clamp01(
            0.50 * edge_halo
            + 0.30 * shadow_mismatch
            + 0.20 * blockiness
        )
        penalty = self.SETTINGS.quality_penalties.get("lighting_conflict", 0.25)
        return clamp01(score * (1.0 - 0.60 * penalty * quality_risk))

    def _artifact_proxy(self, roi, quality_risk: float) -> float:
        edge_halo = self._edge_halo(roi)
        symmetry_break = self._symmetry_break(roi)
        quality = estimate_quality_metrics(roi)
        score = clamp01(
            0.45 * edge_halo
            + 0.20 * quality.blockiness
            + 0.15 * quality.blur_score
            + 0.20 * (1.0 - symmetry_break)
        )
        penalty = self.SETTINGS.quality_penalties.get("artifact_proxy", 0.35)
        return clamp01(score * (1.0 - 0.55 * penalty * quality_risk))

    def _illumination_asymmetry(self, roi) -> float:
        if roi is None or roi.size == 0:
            return 0.0

        gray = ensure_gray(roi).astype(np.float32) / 255.0
        skin_mask = build_skin_mask(roi)
        if skin_mask is None or skin_mask.mean() < 0.02:
            skin_mask = np.ones_like(gray, dtype=np.uint8)

        h, w = gray.shape
        left = gray[:, : w // 2]
        right = gray[:, w // 2 :]
        top = gray[: h // 2, :]
        bottom = gray[h // 2 :, :]

        left_mask = skin_mask[:, : w // 2].astype(bool)
        right_mask = skin_mask[:, w // 2 :].astype(bool)
        top_mask = skin_mask[: h // 2, :].astype(bool)
        bottom_mask = skin_mask[h // 2 :, :].astype(bool)

        left_mean = float(left[left_mask].mean()) if np.any(left_mask) else float(left.mean())
        right_mean = float(right[right_mask].mean()) if np.any(right_mask) else float(right.mean())
        top_mean = float(top[top_mask].mean()) if np.any(top_mask) else float(top.mean())
        bottom_mean = float(bottom[bottom_mask].mean()) if np.any(bottom_mask) else float(bottom.mean())

        contrast = float(np.std(gray[skin_mask.astype(bool)])) if np.any(skin_mask) else float(np.std(gray))
        lr_asym = abs(left_mean - right_mean) / (contrast + 1e-4)
        tb_asym = abs(top_mean - bottom_mean) / (contrast + 1e-4)
        return clamp01(0.6 * normalize(lr_asym, 0.08, 1.10) + 0.4 * normalize(tb_asym, 0.10, 1.30))

    def _shadow_gradient_mismatch(self, roi) -> float:
        if roi is None or roi.size == 0:
            return 0.0

        gray = ensure_gray(roi)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

        h, w = magnitude.shape
        left_mean = float(magnitude[:, : w // 2].mean())
        right_mean = float(magnitude[:, w // 2 :].mean())
        top_mean = float(magnitude[: h // 2, :].mean())
        bottom_mean = float(magnitude[h // 2 :, :].mean())

        mismatch = (
            abs(left_mean - right_mean) / (left_mean + right_mean + 1e-5)
            + abs(top_mean - bottom_mean) / (top_mean + bottom_mean + 1e-5)
        )
        return normalize(float(mismatch), 0.10, 0.85)

    def _symmetry_break(self, roi) -> float:
        if roi is None or roi.size == 0:
            return 0.0

        gray = cv2.resize(ensure_gray(roi), (160, 160), interpolation=cv2.INTER_AREA)
        left = gray[:, :80].astype(np.float32) / 255.0
        right = cv2.flip(gray[:, 80:], 1).astype(np.float32) / 255.0
        diff = np.abs(left - right)
        return normalize(float(np.mean(diff)), 0.04, 0.18)

    def _edge_halo(self, roi) -> float:
        if roi is None or roi.size == 0:
            return 0.0

        gray = ensure_gray(roi)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blur, 60, 140)
        edge_band = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1) > 0

        lap = cv2.Laplacian(blur, cv2.CV_32F, ksize=3)
        lap_abs = np.abs(lap)
        edge_energy = float(lap_abs[edge_band].mean()) if np.any(edge_band) else float(lap_abs.mean())
        background_energy = float(lap_abs.mean()) + 1e-5
        edge_density = float(edges.mean() / 255.0)
        raw = (edge_energy / background_energy) * (0.5 + edge_density)
        return normalize(float(raw), 0.8, 2.2)
