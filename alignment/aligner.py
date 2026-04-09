"""Feature-to-ontology alignment utilities."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from alignment.models import ActivatedSubDomain, FeatureMappingRule, MappingConfig
from detectors.base import DetectorResult


logger = logging.getLogger(__name__)


class FeatureOntologyAligner:
    """Map detector features into configured subdomain nodes."""

    _instance: Optional["FeatureOntologyAligner"] = None

    def __new__(cls, config_path: str = None, singleton: bool = True):
        if singleton:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
        return super().__new__(cls)

    def __init__(self, config_path: str = None, singleton: bool = True):
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._config: Optional[MappingConfig] = None
        self._rule_index: Dict[Tuple[str, str], FeatureMappingRule] = {}
        self._missing_rule_warnings: set[Tuple[str, str]] = set()

        if config_path:
            self.load_config(config_path)

        self._initialized = True

    @classmethod
    def get_instance(cls) -> "FeatureOntologyAligner":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None

    def load_config(self, config_path: str) -> None:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        self._config = MappingConfig(**payload)
        self._build_rule_index()
        logger.info("[RELOAD] loaded %s mapping rules from %s", len(self._config.rules), config_path)

    def load_config_from_dict(self, config_dict: dict) -> None:
        self._config = MappingConfig(**config_dict)
        self._build_rule_index()
        logger.info("[RELOAD] loaded mapping config from in-memory payload")

    def _build_rule_index(self) -> None:
        self._rule_index.clear()
        self._missing_rule_warnings.clear()
        if not self._config:
            return

        for rule in self._config.rules:
            self._rule_index[(rule.detector, rule.feature)] = rule

    @staticmethod
    def sigmoid(x: float, k: float = 10.0, x0: float = 0.5) -> float:
        try:
            return 1.0 / (1.0 + math.exp(-k * (x - x0)))
        except OverflowError:
            return 0.0 if x < x0 else 1.0

    @staticmethod
    def _round_score(value: float, decimals: int = 4) -> float:
        return round(value, decimals)

    def _map_feature(
        self,
        detector_name: str,
        feature_name: str,
        raw_value: float,
        feature_context: Dict[str, Dict[str, float]],
    ) -> Optional[ActivatedSubDomain]:
        rule = self._rule_index.get((detector_name, feature_name))
        if rule is None:
            key = (detector_name, feature_name)
            if key not in self._missing_rule_warnings:
                logger.warning(
                    "[WARN] missing mapping rule for detector=%s feature=%s",
                    detector_name,
                    feature_name,
                )
                self._missing_rule_warnings.add(key)
            return None

        if not rule.evidence_enabled:
            return None

        if rule.context_detector and rule.context_feature:
            context_value = (
                feature_context
                .get(rule.context_detector, {})
                .get(rule.context_feature)
            )
            if context_value is None or float(context_value) < rule.context_min_value:
                logger.debug(
                    "[WARN] context gate blocked detector=%s feature=%s context=%s:%s value=%s min=%.4f",
                    detector_name,
                    feature_name,
                    rule.context_detector,
                    rule.context_feature,
                    context_value,
                    rule.context_min_value,
                )
                return None

        mapped_score = self.sigmoid(float(raw_value), rule.sigmoid_k, rule.sigmoid_x0)
        confidence = mapped_score * rule.weight
        if confidence < rule.activation_threshold:
            logger.debug(
                "[WARN] filtered weak activation detector=%s feature=%s confidence=%.4f threshold=%.4f",
                detector_name,
                feature_name,
                confidence,
                rule.activation_threshold,
            )
            return None
        return ActivatedSubDomain(
            subdomain_id=rule.subdomain_id,
            subdomain_label=rule.subdomain_label,
            score=self._round_score(mapped_score),
            confidence=self._round_score(confidence),
            source_detector=detector_name,
            source_feature=feature_name,
            raw_value=self._round_score(float(raw_value)),
        )

    def align(self, results: List[DetectorResult]) -> List[ActivatedSubDomain]:
        if not self._config:
            raise RuntimeError("No config loaded. Call load_config() first.")

        activated: List[ActivatedSubDomain] = []
        feature_context = {result.name: dict(result.features) for result in results}
        for result in results:
            for feature_name, raw_value in result.features.items():
                subdomain = self._map_feature(
                    result.name,
                    feature_name,
                    raw_value,
                    feature_context,
                )
                if subdomain is not None:
                    activated.append(subdomain)
        return activated

    def align_from_dict(self, features_dict: Dict[str, Dict[str, float]]) -> List[ActivatedSubDomain]:
        if not self._config:
            raise RuntimeError("No config loaded. Call load_config() first.")

        activated: List[ActivatedSubDomain] = []
        for detector_name, feature_map in features_dict.items():
            for feature_name, raw_value in feature_map.items():
                subdomain = self._map_feature(
                    detector_name,
                    feature_name,
                    raw_value,
                    features_dict,
                )
                if subdomain is not None:
                    activated.append(subdomain)
        return activated
