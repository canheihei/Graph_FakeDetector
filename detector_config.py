from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


@dataclass(frozen=True)
class FeatureCalibration:
    low: float
    high: float
    weight: float = 1.0


@dataclass(frozen=True)
class PlaceholderScoreRange:
    center: float = 0.5
    gain: float = 0.22
    minimum: float = 0.35
    maximum: float = 0.65


@dataclass(frozen=True)
class ReliabilityPolicy:
    quality_risk_penalty: float = 0.45
    placeholder_multiplier: float = 0.55
    missing_face_multiplier: float = 0.85


@dataclass(frozen=True)
class DetectorConfig:
    weight_path: Optional[str] = None
    ensemble_weight: float = 1.0
    decision_threshold: float = 0.5
    feature_calibrators: Mapping[str, FeatureCalibration] = field(default_factory=dict)
    quality_penalties: Mapping[str, float] = field(default_factory=dict)
    placeholder_feature_weights: Mapping[str, float] = field(default_factory=dict)
    placeholder_score_range: Optional[PlaceholderScoreRange] = None
    runtime_params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MetaEnsembleConfig:
    anomaly_threshold: float = 0.55
    reliability: ReliabilityPolicy = field(default_factory=ReliabilityPolicy)


@dataclass(frozen=True)
class AdaptiveFusionConfig:
    base_blend: float = 0.08
    max_blend: float = 0.72
    gap_weight: float = 0.65
    portrait_weight: float = 0.20
    margin_weight: float = 0.15
    margin_reference: float = 0.12


@dataclass(frozen=True)
class DetectionDecisionConfig:
    fallback_direct_weight: float = 0.78
    fallback_graph_weight: float = 0.22
    fallback_threshold: float = 0.46
    domain_threshold_profiles: Mapping[str, float] = field(default_factory=dict)
    adaptive_fusion: AdaptiveFusionConfig = field(default_factory=AdaptiveFusionConfig)


@dataclass(frozen=True)
class CandidateBenchmarkModeConfig:
    sample_per_class: int
    min_accuracy_valid: float
    min_balanced_accuracy: float


@dataclass(frozen=True)
class CandidateReviewConfig:
    dataset_profile_roots: Mapping[str, str] = field(default_factory=dict)
    quick: CandidateBenchmarkModeConfig = field(
        default_factory=lambda: CandidateBenchmarkModeConfig(
            sample_per_class=20,
            min_accuracy_valid=0.70,
            min_balanced_accuracy=0.70,
        )
    )
    formal: CandidateBenchmarkModeConfig = field(
        default_factory=lambda: CandidateBenchmarkModeConfig(
            sample_per_class=80,
            min_accuracy_valid=0.80,
            min_balanced_accuracy=0.80,
        )
    )


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS_ROOT = PROJECT_ROOT / "weights"
WEIGHTS_ROOT = Path(os.getenv("GRAPH_FAKEDETECTOR_WEIGHTS_ROOT", str(DEFAULT_WEIGHTS_ROOT)))


DETECTOR_CONFIGS: Dict[str, DetectorConfig] = {
    "FFTDetector": DetectorConfig(
        ensemble_weight=1.0,
        feature_calibrators={
            "high_freq_energy": FeatureCalibration(3.5, 7.8, 1.0),
            "patch_inconsistency": FeatureCalibration(0.30, 0.72, 0.75),
            "blockiness": FeatureCalibration(0.05, 0.20, 0.55),
        },
        quality_penalties={
            "spectral_stabilizer": 0.45,
        },
    ),
    "AppearanceDetector": DetectorConfig(
        ensemble_weight=1.05,
        feature_calibrators={
            "lighting_conflict": FeatureCalibration(0.25, 0.80, 0.45),
            "pose_extreme": FeatureCalibration(0.25, 0.85, 0.55),
            "symmetry_break": FeatureCalibration(0.28, 0.78, 0.65),
        },
        quality_penalties={
            "lighting_conflict": 0.25,
            "artifact_proxy": 0.35,
        },
    ),
    "BoundaryConsistency": DetectorConfig(
        ensemble_weight=1.05,
        feature_calibrators={
            "boundary_inconsistency": FeatureCalibration(0.20, 0.80, 1.0),
        },
    ),
    "EfficientNetB4": DetectorConfig(
        weight_path="efficientnet_b4_ff.pth",
        ensemble_weight=1.20,
        decision_threshold=0.50,
        feature_calibrators={
            "fake_probability": FeatureCalibration(0.45, 0.85, 1.0),
        },
        placeholder_feature_weights={
            "spectral": 0.50,
            "chroma": 0.25,
            "quality_risk": 0.25,
        },
        placeholder_score_range=PlaceholderScoreRange(
            center=0.5,
            gain=0.22,
            minimum=0.35,
            maximum=0.65,
        ),
    ),
    "CalibratedVision": DetectorConfig(
        weight_path="calibrated_vision_detector.pt",
        ensemble_weight=2.40,
        decision_threshold=0.38,
        feature_calibrators={
            "fake_probability": FeatureCalibration(0.20, 0.80, 1.0),
        },
        placeholder_feature_weights={
            "spectral": 0.45,
            "appearance": 0.35,
            "quality_risk": 0.20,
        },
        placeholder_score_range=PlaceholderScoreRange(
            center=0.5,
            gain=0.28,
            minimum=0.30,
            maximum=0.70,
        ),
        runtime_params={
            "variant": "efficientnet_b0_ft_v1",
            "resize_size": 256,
            "input_size": 224,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "dropout": 0.20,
        },
    ),
    "ViT": DetectorConfig(
        weight_path="vit_ff.pth",
        ensemble_weight=1.20,
        feature_calibrators={
            "vit_fake_prob": FeatureCalibration(0.45, 0.85, 1.0),
            "vit_prediction_entropy": FeatureCalibration(0.35, 0.85, 0.45),
            "vit_attention_uniformity": FeatureCalibration(0.50, 0.90, 0.55),
            "vit_cls_feature_norm": FeatureCalibration(0.35, 0.90, 0.50),
        },
        placeholder_feature_weights={
            "spectral": 0.50,
            "patch_dispersion": 0.25,
            "quality_risk": 0.25,
        },
        placeholder_score_range=PlaceholderScoreRange(
            center=0.5,
            gain=0.24,
            minimum=0.35,
            maximum=0.65,
        ),
    ),
    "FreqNet": DetectorConfig(
        weight_path="freqnet_latest.pth",
        ensemble_weight=1.10,
    ),
    "MetaEnsemble": DetectorConfig(),
}


META_ENSEMBLE_CONFIG = MetaEnsembleConfig()
DETECTION_DECISION_CONFIG = DetectionDecisionConfig(
    domain_threshold_profiles={
        # External-domain calibration defaults (2026-04-20, round7_rebalanced + sample1200/full benchmark).
        "celeb_df": 0.42,
        "celebdf": 0.42,
        "dfdc": 0.47,
        "wilddeepfake": 0.12,
    }
)
CANDIDATE_REVIEW_CONFIG = CandidateReviewConfig(
    dataset_profile_roots={
        "default": str(PROJECT_ROOT / "Datasets" / "Test"),
        "celeb_df": str(PROJECT_ROOT / "Datasets" / "Celeb-DF"),
        "celebdf": str(PROJECT_ROOT / "Datasets" / "Celeb-DF"),
        "dfdc": str(PROJECT_ROOT / "Datasets" / "DFDC"),
        "wilddeepfake": str(PROJECT_ROOT / "Datasets" / "WildDeepfake"),
    }
)


def get_detector_config(name: str) -> DetectorConfig:
    return DETECTOR_CONFIGS.get(name, DetectorConfig())


def get_weight_path(name: str) -> Optional[Path]:
    config = get_detector_config(name)
    if not config.weight_path:
        return None
    weight_path = Path(config.weight_path)
    if weight_path.is_absolute():
        return weight_path
    return WEIGHTS_ROOT / weight_path


def score_from_placeholder_proxy(proxy_score: float, score_range: Optional[PlaceholderScoreRange]) -> float:
    if score_range is None:
        return max(0.0, min(1.0, proxy_score))
    shifted = score_range.center + score_range.gain * (proxy_score - score_range.center)
    return max(score_range.minimum, min(score_range.maximum, shifted))


def get_detection_decision_config() -> DetectionDecisionConfig:
    return DETECTION_DECISION_CONFIG


def get_candidate_review_config() -> CandidateReviewConfig:
    return CANDIDATE_REVIEW_CONFIG
