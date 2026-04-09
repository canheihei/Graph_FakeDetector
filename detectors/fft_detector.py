import base64

import cv2
import numpy as np

from detector_config import get_detector_config
from detectors.base import BaseDetector, DetectorResult
from detectors.forensics_utils import (
    compute_spectral_profile,
    decode_bgr_image,
    ensure_gray,
    estimate_quality_metrics,
)
from detectors.registry import DetectorRegistry


@DetectorRegistry.register(name="FFTDetector", device="cuda")
class FFTDetector(BaseDetector):
    name = "FFTDetector"
    SETTINGS = get_detector_config(name)

    def _load_model(self):
        pass

    def detect(self, image_bytes: bytes) -> DetectorResult:
        bgr = decode_bgr_image(image_bytes)
        gray = ensure_gray(bgr)
        if gray is None:
            return DetectorResult(
                name=self.name,
                features={
                    "high_freq_energy": 0.0,
                    "patch_inconsistency": 0.0,
                    "blockiness": 0.0,
                },
                meta={"decode_failed": True},
            )

        quality = estimate_quality_metrics(bgr)
        spectral = compute_spectral_profile(gray)
        penalty = self.SETTINGS.quality_penalties.get("spectral_stabilizer", 0.45)
        stabilized_score = spectral["raw_score"] * (1.0 - penalty * quality.quality_risk)
        fft_visualization = self._build_fft_visualization(gray)

        return DetectorResult(
            name=self.name,
            features={
                "high_freq_energy": round(float(stabilized_score), 6),
                "patch_inconsistency": round(float(spectral["patch_inconsistency"]), 6),
                "blockiness": round(float(quality.blockiness), 6),
            },
            meta={
                "fft_spectrum": fft_visualization,
                "quality_risk": round(float(quality.quality_risk), 6),
                "blur_score": round(float(quality.blur_score), 6),
                "blockiness": round(float(quality.blockiness), 6),
                "global_high_ratio": round(float(spectral["global_high_ratio"]), 6),
                "patch_inconsistency": round(float(spectral["patch_inconsistency"]), 6),
                "patch_peakiness": round(float(spectral["patch_peakiness"]), 6),
            },
        )

    @staticmethod
    def _build_fft_visualization(gray):
        resized = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
        spectrum = np.fft.fftshift(np.fft.fft2(resized.astype(np.float32) / 255.0))
        magnitude = np.log(np.abs(spectrum) + 1.0)
        magnitude_normalized = cv2.normalize(
            magnitude,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        )
        magnitude_uint8 = magnitude_normalized.astype(np.uint8)
        heatmap = cv2.applyColorMap(magnitude_uint8, cv2.COLORMAP_JET)
        _, buffer = cv2.imencode(".png", heatmap)
        return base64.b64encode(buffer).decode("utf-8")
