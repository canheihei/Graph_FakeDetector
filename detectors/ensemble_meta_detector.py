from typing import List, Optional

import numpy as np

from detector_config import META_ENSEMBLE_CONFIG, get_detector_config
from detectors.base import BaseDetector, DetectorResult
from detectors.forensics_utils import clamp01, normalize
from detectors.registry import DetectorRegistry


@DetectorRegistry.register(name="MetaEnsemble", device="cuda")
class MetaEnsembleDetector(BaseDetector):
    name = "MetaEnsemble"
    is_meta = True

    def _load_model(self):
        pass

    def detect(
        self,
        image_bytes: bytes,
        previous_results: Optional[List[DetectorResult]] = None,
    ) -> DetectorResult:
        if not previous_results:
            return DetectorResult(
                self.name,
                {
                    "weighted_ensemble_score": 0.0,
                    "detector_agreement": 0.0,
                    "max_anomaly_score": 0.0,
                    "anomaly_coverage": 0.0,
                },
            )

        detector_scores = []
        detector_weights = []
        feature_scores = []
        detector_score_map = {}

        for result in previous_results:
            suspiciousness = self._aggregate_detector_suspiciousness(result)
            if suspiciousness is None:
                continue

            reliability = self._estimate_reliability(result)
            detector_config = get_detector_config(result.name)
            weight = detector_config.ensemble_weight * reliability
            detector_scores.append(suspiciousness)
            detector_weights.append(weight)
            feature_scores.extend(self._calibrated_feature_scores(result))
            detector_score_map[result.name] = round(suspiciousness, 6)

        if not detector_scores:
            return DetectorResult(
                self.name,
                {
                    "weighted_ensemble_score": 0.0,
                    "detector_agreement": 0.0,
                    "max_anomaly_score": 0.0,
                    "anomaly_coverage": 0.0,
                },
            )

        scores_arr = np.array(detector_scores, dtype=np.float32)
        weights_arr = np.array(detector_weights, dtype=np.float32)
        weighted_ensemble_score = float(np.average(scores_arr, weights=weights_arr))
        detector_agreement = float(np.clip(1.0 - 1.7 * np.std(scores_arr), 0.0, 1.0))
        max_anomaly_score = float(
            max((score for score, _ in feature_scores), default=0.0)
        )
        anomaly_coverage = float(np.mean(scores_arr >= META_ENSEMBLE_CONFIG.anomaly_threshold))

        return DetectorResult(
            name=self.name,
            features={
                "weighted_ensemble_score": round(weighted_ensemble_score, 6),
                "detector_agreement": round(detector_agreement, 6),
                "max_anomaly_score": round(max_anomaly_score, 6),
                "anomaly_coverage": round(anomaly_coverage, 6),
            },
            meta={
                "detector_scores": detector_score_map,
            },
        )

    def _aggregate_detector_suspiciousness(self, result: DetectorResult) -> Optional[float]:
        weighted_scores = self._calibrated_feature_scores(result)
        if not weighted_scores:
            return None
        values = np.array([item[0] for item in weighted_scores], dtype=np.float32)
        weights = np.array([item[1] for item in weighted_scores], dtype=np.float32)
        return float(np.average(values, weights=weights))

    def _calibrated_feature_scores(self, result: DetectorResult) -> List[tuple[float, float]]:
        calibrators = get_detector_config(result.name).feature_calibrators
        scores = []
        for feature_name, value in result.features.items():
            config = calibrators.get(feature_name)
            if config is None:
                continue
            scores.append((normalize(float(value), config.low, config.high), max(config.weight, 1e-3)))
        return scores

    @staticmethod
    def _estimate_reliability(result: DetectorResult) -> float:
        policy = META_ENSEMBLE_CONFIG.reliability
        quality_risk = float(result.meta.get("quality_risk", 0.0))
        placeholder_mode = bool(result.meta.get("placeholder_mode", False))
        face_detected = result.meta.get("face_detected", True)

        reliability = 1.0 - policy.quality_risk_penalty * quality_risk
        if placeholder_mode:
            reliability *= policy.placeholder_multiplier
        if face_detected is False:
            reliability *= policy.missing_face_multiplier
        return clamp01(reliability)
