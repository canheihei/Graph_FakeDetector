from __future__ import annotations

import io
from typing import Any, Dict

import numpy as np
import torch
from PIL import Image

from detector_config import get_detector_config, get_weight_path, score_from_placeholder_proxy
from detectors.base import BaseDetector, DetectorResult
from detectors.forensics_utils import (
    compute_spectral_profile,
    decode_bgr_image,
    ensure_gray,
    estimate_quality_metrics,
)
from detectors.registry import DetectorRegistry
from detectors.vision_backbones import build_vision_backbone


@DetectorRegistry.register(name="CalibratedVision", device="cuda")
class CalibratedVisionDetector(BaseDetector):
    name = "CalibratedVision"
    SETTINGS = get_detector_config(name)
    DEFAULT_WEIGHT_PATH = get_weight_path(name)

    def _load_model(self):
        self.model = None
        self.transform = None
        self.weight_ready = False
        self.placeholder_reason = None
        self._checkpoint_meta: Dict[str, Any] = {}
        self._variant = str(self.SETTINGS.runtime_params.get("variant", "compact_cnn_v1"))
        self._tta_enabled = False

        weight_path = self.DEFAULT_WEIGHT_PATH
        if weight_path is None:
            self.placeholder_reason = "missing configured weight path"
            return
        if not weight_path.exists():
            self.placeholder_reason = f"missing weight: {weight_path}"
            return

        try:
            from torchvision import transforms

            runtime = self.SETTINGS.runtime_params
            checkpoint = torch.load(
                weight_path,
                map_location="cpu",
                weights_only=False,
            )
            checkpoint_meta = checkpoint.get("meta", {}) if isinstance(checkpoint, dict) else {}
            state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

            variant = str(checkpoint_meta.get("training_version", runtime.get("variant", "compact_cnn_v1")))
            self._variant = variant
            dropout = float(checkpoint_meta.get("dropout", runtime.get("dropout", 0.35)))
            self.model = build_vision_backbone(
                variant,
                pretrained=False,
                dropout=dropout,
            )
            self.model.load_state_dict(state_dict, strict=True)
            self.model.to(self.device)
            self.model.eval()

            input_size = int(checkpoint_meta.get("input_size", runtime.get("input_size", 128)))
            resize_size = int(checkpoint_meta.get("resize_size", runtime.get("resize_size", input_size)))
            mean = checkpoint_meta.get("mean", runtime.get("mean", [0.5, 0.5, 0.5]))
            std = checkpoint_meta.get("std", runtime.get("std", [0.5, 0.5, 0.5]))
            self.transform = transforms.Compose(
                [
                    transforms.Resize((resize_size, resize_size)),
                    transforms.CenterCrop(input_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )
            self._checkpoint_meta = dict(checkpoint_meta)
            self._tta_enabled = bool(checkpoint_meta.get("tta_enabled", False))
            self.weight_ready = True
        except Exception as exc:
            self.model = None
            self.transform = None
            self.placeholder_reason = str(exc)

    def detect(self, image_bytes: bytes) -> DetectorResult:
        if self.model is None or self.transform is None:
            return self._detect_with_placeholder(image_bytes)

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        variants = [image]
        if self._tta_enabled:
            variants.append(image.transpose(Image.FLIP_LEFT_RIGHT))

        tensors = torch.stack([self.transform(item) for item in variants], dim=0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensors).view(-1)
            fake_prob = float(torch.sigmoid(logits).mean().item())

        decision_threshold = float(
            self._checkpoint_meta.get(
                "decision_threshold",
                self.SETTINGS.decision_threshold,
            )
        )
        confidence_margin = abs(fake_prob - decision_threshold)
        normalized_confidence = min(1.0, confidence_margin / max(decision_threshold, 1.0 - decision_threshold, 1e-6))

        return DetectorResult(
            name=self.name,
            features={
                "fake_probability": round(fake_prob, 6),
            },
            meta={
                "weight_ready": True,
                "placeholder_mode": False,
                "weight_path": str(self.DEFAULT_WEIGHT_PATH) if self.DEFAULT_WEIGHT_PATH else None,
                "decision_threshold": round(decision_threshold, 6),
                "decision_margin": round(float(confidence_margin), 6),
                "confidence_strength": round(float(normalized_confidence), 6),
                "training_version": self._checkpoint_meta.get("training_version", self._variant),
                "tta_enabled": self._tta_enabled,
            },
        )

    def _detect_with_placeholder(self, image_bytes: bytes) -> DetectorResult:
        bgr = decode_bgr_image(image_bytes)
        gray = ensure_gray(bgr)
        quality = estimate_quality_metrics(bgr)
        spectral = compute_spectral_profile(gray)

        appearance_proxy = 0.0
        if bgr is not None:
            appearance_proxy = float(
                np.clip(
                    0.50 * quality.blockiness + 0.30 * quality.blur_score + 0.20 * spectral["patch_inconsistency"],
                    0.0,
                    1.0,
                )
            )

        placeholder_weights = self.SETTINGS.placeholder_feature_weights
        proxy_score = np.clip(
            placeholder_weights.get("spectral", 0.45) * spectral["score"]
            + placeholder_weights.get("appearance", 0.35) * appearance_proxy
            + placeholder_weights.get("quality_risk", 0.20) * quality.quality_risk,
            0.0,
            1.0,
        )
        fake_prob = float(
            score_from_placeholder_proxy(
                float(proxy_score),
                self.SETTINGS.placeholder_score_range,
            )
        )

        return DetectorResult(
            name=self.name,
            features={
                "fake_probability": round(fake_prob, 6),
            },
            meta={
                "weight_ready": False,
                "placeholder_mode": True,
                "placeholder_reason": self.placeholder_reason,
                "decision_threshold": self.SETTINGS.decision_threshold,
                "decision_margin": round(abs(fake_prob - self.SETTINGS.decision_threshold), 6),
                "quality_risk": round(float(quality.quality_risk), 6),
                "spectral_score": round(float(spectral["score"]), 6),
                "appearance_proxy": round(float(appearance_proxy), 6),
            },
        )
