"""Detector package bootstrap."""

from detectors.base import BaseDetector, DetectorResult
from detectors.appearance_detector import AppearanceDetector
from detectors.calibrated_vision_detector import CalibratedVisionDetector
from detectors.ensemble_meta_detector import MetaEnsembleDetector
from detectors.fft_detector import FFTDetector
from detectors.hub import DetectorHub
from detectors.registry import DetectorRegistry

__all__ = [
    "BaseDetector",
    "DetectorResult",
    "DetectorRegistry",
    "DetectorHub",
    "FFTDetector",
    "AppearanceDetector",
    "CalibratedVisionDetector",
    "MetaEnsembleDetector",
]
