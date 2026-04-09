from abc import ABC, abstractmethod
from typing import Any, Dict


class DetectorResult:
    """Container for detector outputs."""

    def __init__(self, name: str, features: Dict[str, float], meta: Dict[str, Any] = None):
        self.name = name
        self.features = features
        self.meta = meta or {}

    def to_dict(self):
        return {
            "detector": self.name,
            "features": self.features,
            "meta": self.meta,
        }


class BaseDetector(ABC):
    """Base contract shared by all detectors."""

    name: str = "BaseDetector"
    device: str = "cuda"
    is_meta: bool = False

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._load_model()

    @abstractmethod
    def _load_model(self):
        """Load detector resources."""

    @abstractmethod
    def detect(self, image_bytes: bytes) -> DetectorResult:
        """Run inference and return a DetectorResult."""
