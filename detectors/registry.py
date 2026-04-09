from typing import Dict, List, Type

from detectors.base import BaseDetector


class DetectorRegistry:
    """Registry for detector classes and initialized instances."""

    _detector_classes: Dict[str, Type[BaseDetector]] = {}
    _instances: List[BaseDetector] = []

    @classmethod
    def register(cls, name: str = None, device: str = "cuda"):
        def decorator(detector_cls: Type[BaseDetector]):
            detector_name = name or detector_cls.name
            cls._detector_classes[detector_name] = (detector_cls, device)
            return detector_cls

        return decorator

    @classmethod
    def init_all(cls, device: str = None):
        cls._instances.clear()
        for det_name, (det_cls, default_device) in cls._detector_classes.items():
            try:
                inst = det_cls(device=device or default_device)
                cls._instances.append(inst)
            except Exception as exc:
                print(f"[WARN] Failed to init {det_name}: {exc}")

    @classmethod
    def get_all(cls) -> List[BaseDetector]:
        if not cls._instances and cls._detector_classes:
            cls.init_all()
        return cls._instances

    @classmethod
    def clear(cls):
        cls._detector_classes.clear()
        cls._instances.clear()
