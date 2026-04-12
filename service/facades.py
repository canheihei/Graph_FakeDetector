from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

from werkzeug.utils import secure_filename

from alignment.evolver import UnmappedFeature
from detector_config import get_detection_decision_config, get_detector_config
from detectors.forensics_utils import clamp01, normalize
from service.common_utils import (
    build_detection_response,
    get_image_base64_list,
    safe_path_name,
)
from service.decision_policy import compute_adaptive_fusion, resolve_decision_threshold
from service.detect_client_v2 import decide_v2
from service.graph_semantics import GraphSemanticGovernance
from service.llm_chain import call_qwen, match_domain, reasoning


class WorkflowError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class DetectRequest:
    image_bytes: bytes
    auto_evolve_enabled: bool
    semantic_threshold: float
    use_llm_generation: bool
    decision_profile: str | None = None
    decision_threshold_override: float | None = None


@dataclass(frozen=True)
class IterateRequest:
    prompt: str
    image_files: List
    semantic_threshold: float


@dataclass(frozen=True)
class DirectIngestRequest:
    payload: Dict


@dataclass(frozen=True)
class ManualEvolutionRequest:
    features: List[Dict]
    evolutions: List[Dict]
    semantic_threshold: float


@dataclass(frozen=True)
class SuggestDomainRequest:
    detector: str
    feature: str
    score: float
    raw_value: float


class IterationFacade:
    def __init__(self, neo4j_client, graph_writer, upload_root: str):
        self._neo4j_client = neo4j_client
        self._graph_writer = graph_writer
        self._upload_root = upload_root
        self._semantic_governor = GraphSemanticGovernance(neo4j_client)

    def execute(self, request: IterateRequest) -> Dict:
        if not request.prompt or not request.image_files:
            raise WorkflowError("Missing prompt or images", 400)

        safe_prompt_dir = safe_path_name(request.prompt)
        if not safe_prompt_dir:
            raise WorkflowError("Invalid prompt for directory name", 400)

        saved_paths = self._save_images(safe_prompt_dir, request.image_files)
        if not saved_paths:
            raise WorkflowError("No valid images uploaded", 400)

        matched_domain = self._match_domain(request.prompt)
        try:
            image_infos = self._build_image_infos(saved_paths, matched_domain)
        except Exception as exc:
            raise WorkflowError(f"Image analysis failed: {str(exc)}", 500) from exc

        if not isinstance(image_infos, dict) or not image_infos.get("results"):
            raise WorkflowError("No valid images processed", 500)

        llm_payload = self._build_llm_payload(image_infos, matched_domain)
        result = call_qwen(llm_payload)
        normalized_result = self._semantic_governor.normalize_iteration_payload(
            result,
            prompt=request.prompt,
            matched_domain=matched_domain,
            semantic_threshold=request.semantic_threshold,
        )
        if not normalized_result.get("subdomain"):
            return {
                "message": "No stable semantic subdomains were extracted; graph evolution was skipped",
                "features": [],
                "specific_domain": normalized_result.get("specific_domain"),
                "main_domain": normalized_result.get("main_domain"),
                "semantic_dedup_stats": None,
                "semantic_threshold": request.semantic_threshold,
            }

        dedup_stats = self._graph_writer.write(
            normalized_result,
            semantic_threshold=request.semantic_threshold,
        )
        return {
            "message": f"Processed {len(image_infos.get('results', []))} images",
            "features": normalized_result.get("subdomain", []),
            "specific_domain": normalized_result.get("specific_domain"),
            "main_domain": normalized_result.get("main_domain"),
            "semantic_dedup_stats": dedup_stats,
            "semantic_threshold": request.semantic_threshold,
        }

    @staticmethod
    def _build_llm_payload(image_infos: Dict, matched_domain: str) -> Dict:
        raw_results = image_infos.get("results", []) if isinstance(image_infos, dict) else []
        compact_results = []
        fake_scores: List[float] = []
        confidences: List[float] = []

        sorted_results = sorted(
            raw_results,
            key=lambda item: float(item.get("fake_score", 0.0)),
            reverse=True,
        )
        for item in sorted_results[:12]:
            fake_score = float(item.get("fake_score", 0.0))
            confidence = float(item.get("confidence", 0.0))
            is_fake = bool(item.get("is_fake", False))

            fake_scores.append(fake_score)
            confidences.append(confidence)

            compact_results.append(
                {
                    "path": item.get("path", ""),
                    "label": item.get("label", ""),
                    "is_fake": is_fake,
                    "confidence": round(confidence, 4),
                    "fake_score": round(fake_score, 4),
                }
            )

        total = len(raw_results)
        fake_count = sum(1 for item in raw_results if bool(item.get("is_fake", False)))
        avg_confidence = (sum(confidences) / len(confidences)) if confidences else 0.0
        avg_fake_score = (sum(fake_scores) / len(fake_scores)) if fake_scores else 0.0
        fake_ratio = (fake_count / total) if total else 0.0

        return {
            "domain_name": matched_domain,
            "summary": {
                "total_images": total,
                "sampled_images": len(compact_results),
                "fake_count": fake_count,
                "fake_ratio": round(fake_ratio, 4),
                "avg_confidence": round(avg_confidence, 4),
                "avg_fake_score": round(avg_fake_score, 4),
            },
            "images": compact_results,
        }

    def _save_images(self, safe_prompt_dir: str, image_files: List) -> List[str]:
        prompt_folder = os.path.join(self._upload_root, safe_prompt_dir)
        os.makedirs(prompt_folder, exist_ok=True)

        saved_paths = []
        for image_file in image_files:
            filename = secure_filename(image_file.filename)
            if not filename:
                continue
            path = os.path.join(prompt_folder, filename)
            image_file.save(path)
            saved_paths.append(path)
        return saved_paths

    def _match_domain(self, prompt: str) -> str:
        specific_domain_nodes = self._neo4j_client.get_specificdomain_nodes()
        subdomain_nodes = self._neo4j_client.get_subdomain_nodes()
        specific_domain_names = [
            item["name"] for item in specific_domain_nodes.get("data", [])
        ]
        subdomain_names = [item["name"] for item in subdomain_nodes.get("data", [])]
        return match_domain(prompt, specific_domain_names, subdomain_names)

    @staticmethod
    def _build_image_infos(saved_paths: List[str], matched_domain: str) -> Dict:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            base64_list = loop.run_until_complete(get_image_base64_list(saved_paths))
            return build_detection_response(saved_paths, base64_list, matched_domain)
        finally:
            loop.close()
            asyncio.set_event_loop(None)


class DetectionFacade:
    def __init__(self, hub, aligner, graph_evolver, evidence_builder, logger):
        self._hub = hub
        self._aligner = aligner
        self._graph_evolver = graph_evolver
        self._evidence_builder = evidence_builder
        self._logger = logger
        self._decision_config = get_detection_decision_config()

    def execute(self, request: DetectRequest) -> Dict:
        detector_results = self._hub.run(request.image_bytes)
        detector_map = {result.name: result for result in detector_results}
        content_profile = self._build_content_profile(detector_map)
        supports_graph_reasoning = bool(content_profile.get("allow_graph_reasoning", True))

        if supports_graph_reasoning:
            activated_subdomains = self._aligner.align(detector_results)
            unmapped_features = self._graph_evolver.find_unmapped_features(
                detector_results,
                self._aligner._config.rules,
            )
        else:
            activated_subdomains = []
            unmapped_features = []
            self._logger.info(
                "[WARN] detect fallback: skipping graph alignment for non-portrait input"
            )

        evolved_features = []
        if request.auto_evolve_enabled and unmapped_features:
            evolved_features = self._auto_evolve_features(
                unmapped_features=unmapped_features,
                semantic_threshold=request.semantic_threshold,
                use_llm_generation=request.use_llm_generation,
            )

        unmapped_payload = self._build_unmapped_payload(unmapped_features)
        evidence = self._evidence_builder.build(activated_subdomains) if activated_subdomains else []
        graph_decision = decide_v2(evidence)
        decision = self._merge_decisions(
            detector_results,
            graph_decision,
            content_profile,
            request=request,
        )
        gate_diagnostics = self._build_graph_gate_diagnostics(
            detector_results,
            supports_graph_reasoning=supports_graph_reasoning,
            activated_subdomains=activated_subdomains,
        )
        detector_signals = self._extract_detector_signals(detector_map, content_profile)
        reasoning_type = self._resolve_reasoning_type(
            decision=decision,
            evidence=evidence,
            content_profile=content_profile,
        )
        needs_review, review_reasons, risk_level = self._evaluate_review_flags(
            decision=decision,
            evidence=evidence,
            content_profile=content_profile,
            gate_diagnostics=gate_diagnostics,
        )
        diagnostic_chain = self._build_diagnostic_chain(
            decision=decision,
            content_profile=content_profile,
            detector_signals=detector_signals,
            evidence=evidence,
            gate_diagnostics=gate_diagnostics,
            review_reasons=review_reasons,
        )
        reasoning_payload = (
            self._build_default_reasoning(decision, content_profile)
            if not evidence
            else reasoning(evidence, decision)
        )
        reasoning_payload["diagnostic_chain"] = diagnostic_chain

        return {
            "label": decision["label"],
            "confidence": decision["confidence"],
            "decision_fake_score": decision.get("decision_fake_score"),
            "decision_threshold": decision.get("decision_threshold"),
            "decision_margin": decision.get("decision_margin"),
            "score_source": decision.get("score_source"),
            "threshold_source": decision.get("threshold_source", "default"),
            "decision_profile": decision.get("decision_profile"),
            "evidence": evidence,
            "reasoning": reasoning_payload,
            "reasoning_type": reasoning_type,
            "diagnostic_chain": diagnostic_chain,
            "needs_review": needs_review,
            "review_reasons": review_reasons,
            "risk_level": risk_level,
            "graph_gate_diagnostics": gate_diagnostics,
            "unmapped_features": unmapped_payload,
            "evolved_features": evolved_features,
            "semantic_threshold": request.semantic_threshold,
            "content_profile": content_profile,
            "visualizations": self._collect_visualizations(detector_results),
        }

    def _merge_decisions(
        self,
        detector_results,
        graph_decision: Dict,
        content_profile: Dict[str, Any],
        request: DetectRequest | None = None,
    ) -> Dict:
        detector_map = {result.name: result for result in detector_results}
        detector_signals = self._extract_detector_signals(detector_map, content_profile)
        primary_signal = next(
            (item for item in detector_signals if item["name"] == "CalibratedVision"),
            None,
        )
        decision_profile = (request.decision_profile or "").strip() if request else ""
        threshold_override = request.decision_threshold_override if request else None

        if detector_signals:
            total_weight = sum(item["weight"] for item in detector_signals)
            direct_score = sum(item["score"] * item["weight"] for item in detector_signals) / max(total_weight, 1e-6)
        else:
            direct_score = 0.0

        graph_confidence = (
            float(graph_decision.get("confidence", 0.0))
            if content_profile.get("allow_graph_reasoning", True)
            else 0.0
        )

        has_primary = bool(primary_signal and primary_signal.get("weight_ready", False))
        base_threshold = (
            float(primary_signal.get("threshold", 0.5))
            if has_primary
            else float(self._decision_config.fallback_threshold)
        )
        decision_threshold, threshold_source = resolve_decision_threshold(
            base_threshold=base_threshold,
            profile_name=decision_profile,
            override_threshold=threshold_override,
            profile_thresholds=self._decision_config.domain_threshold_profiles,
        )

        fusion_debug: Dict[str, Any] = {}
        if primary_signal and primary_signal.get("weight_ready", False):
            auxiliary_signals = [
                signal for signal in detector_signals
                if signal["name"] != "CalibratedVision"
            ]
            auxiliary_score = self._weighted_signal_score(auxiliary_signals)
            fusion_debug = compute_adaptive_fusion(
                primary_score=float(primary_signal["score"]),
                auxiliary_score=auxiliary_score,
                portrait_confidence=float(content_profile.get("portrait_confidence", 0.0)),
                decision_threshold=decision_threshold,
                config=self._decision_config.adaptive_fusion,
            )
            fused_score = clamp01(float(fusion_debug["fused_score"]))
            score_source = str(fusion_debug.get("mode", "adaptive_fusion"))
        else:
            fused_score = clamp01(
                float(self._decision_config.fallback_direct_weight) * direct_score
                + float(self._decision_config.fallback_graph_weight) * graph_confidence
            )
            score_source = "fusion"

        strong_count = sum(1 for item in detector_signals if item["score"] >= 0.55)
        moderate_count = sum(1 for item in detector_signals if item["score"] >= 0.48)

        is_fake = False
        if primary_signal and primary_signal.get("weight_ready", False):
            is_fake = fused_score >= decision_threshold
            confidence_value = fused_score if is_fake else 1.0 - fused_score
        else:
            if strong_count >= 2 and fused_score >= decision_threshold:
                is_fake = True
            elif strong_count >= 1 and moderate_count >= 2 and fused_score >= decision_threshold:
                is_fake = True
            elif graph_confidence >= 0.62 and fused_score >= decision_threshold:
                is_fake = True
            confidence_value = fused_score

        decision_margin = float(fused_score - decision_threshold)
        return {
            **graph_decision,
            "label": "FAKE" if is_fake else "REAL",
            "confidence": round(confidence_value, 3),
            "direct_score": round(direct_score, 3),
            "graph_score": round(graph_confidence, 3),
            "decision_threshold": round(decision_threshold, 3),
            "decision_fake_score": round(fused_score, 3),
            "decision_margin": round(decision_margin, 3),
            "score_source": score_source,
            "threshold_source": threshold_source,
            "decision_profile": decision_profile or None,
            "strong_signal_count": strong_count,
            "moderate_signal_count": moderate_count,
            "signal_sources": [item["name"] for item in detector_signals],
            "adaptive_blend_ratio": fusion_debug.get("blend_ratio"),
            "adaptive_shift_indicator": fusion_debug.get("shift_indicator"),
            "auxiliary_score": fusion_debug.get("auxiliary_score"),
        }

    @staticmethod
    def _weighted_signal_score(signals: List[Dict[str, Any]]) -> float | None:
        if not signals:
            return None
        total_weight = sum(float(item.get("weight", 0.0)) for item in signals)
        if total_weight <= 1e-6:
            return None
        weighted_sum = sum(
            float(item.get("score", 0.0)) * float(item.get("weight", 0.0))
            for item in signals
        )
        return clamp01(weighted_sum / total_weight)

    def _extract_detector_signals(
        self,
        detector_map: Dict,
        content_profile: Dict[str, Any],
    ) -> List[Dict]:
        signals: List[Dict] = []
        allow_face_fusion = bool(content_profile.get("allow_face_fusion", True))

        calibrated_result = detector_map.get("CalibratedVision")
        if calibrated_result is not None and allow_face_fusion:
            calibrated_score = clamp01(
                float(calibrated_result.features.get("fake_probability", 0.0))
            )
            calibrated_weight = 2.40 * self._estimate_signal_reliability(calibrated_result)
            signals.append(
                {
                    "name": "CalibratedVision",
                    "score": calibrated_score,
                    "weight": calibrated_weight,
                    "threshold": float(
                        calibrated_result.meta.get(
                            "decision_threshold",
                            get_detector_config("CalibratedVision").decision_threshold,
                        )
                    ),
                    "weight_ready": bool(calibrated_result.meta.get("weight_ready", False)),
                }
            )

        meta_result = detector_map.get("MetaEnsemble")
        if meta_result is not None and allow_face_fusion:
            meta_score = clamp01(
                0.70 * float(meta_result.features.get("weighted_ensemble_score", 0.0))
                + 0.20 * float(meta_result.features.get("max_anomaly_score", 0.0))
                + 0.10 * float(meta_result.features.get("anomaly_coverage", 0.0))
            )
            agreement = float(meta_result.features.get("detector_agreement", 0.0))
            meta_score = clamp01(meta_score * (0.92 + 0.08 * agreement))
            signals.append({"name": "MetaEnsemble", "score": meta_score, "weight": 1.85})

        fft_result = detector_map.get("FFTDetector")
        if fft_result is not None:
            fft_score = self._normalized_feature_score("FFTDetector", "high_freq_energy", fft_result.features.get("high_freq_energy", 0.0))
            fft_weight = 0.85 * self._estimate_signal_reliability(fft_result)
            signals.append({"name": "FFTDetector", "score": fft_score, "weight": fft_weight})

        appearance_result = detector_map.get("AppearanceDetector")
        if appearance_result is not None and allow_face_fusion:
            lighting = self._normalized_feature_score("AppearanceDetector", "lighting_conflict", appearance_result.features.get("lighting_conflict", 0.0))
            structure = self._normalized_feature_score("AppearanceDetector", "pose_extreme", appearance_result.features.get("pose_extreme", 0.0))
            appearance_score = clamp01(0.60 * max(lighting, structure) + 0.40 * (0.5 * lighting + 0.5 * structure))
            appearance_weight = 0.95 * self._estimate_signal_reliability(appearance_result)
            signals.append({"name": "AppearanceDetector", "score": appearance_score, "weight": appearance_weight})

        boundary_result = detector_map.get("BoundaryConsistency")
        if boundary_result is not None and allow_face_fusion:
            boundary_score = self._normalized_feature_score("BoundaryConsistency", "boundary_inconsistency", boundary_result.features.get("boundary_inconsistency", 0.0))
            signals.append({"name": "BoundaryConsistency", "score": boundary_score, "weight": 0.60})

        for detector_name, feature_name, base_weight in [
            ("ViT", "vit_fake_prob", 1.20),
            ("EfficientNetB4", "fake_probability", 1.10),
        ]:
            if not allow_face_fusion:
                continue
            result = detector_map.get(detector_name)
            if result is None:
                continue
            score = self._normalized_feature_score(detector_name, feature_name, result.features.get(feature_name, 0.0))
            reliability = self._estimate_signal_reliability(result)
            if score <= 0.0 or reliability <= 0.0:
                continue
            signals.append({"name": detector_name, "score": score, "weight": base_weight * reliability})

        return [item for item in signals if item["weight"] > 0.05]

    @staticmethod
    def _build_content_profile(detector_map: Dict[str, Any]) -> Dict[str, Any]:
        appearance_result = detector_map.get("AppearanceDetector")
        appearance_meta = appearance_result.meta if appearance_result is not None else {}
        face_detected = bool(appearance_meta.get("face_detected", False))
        unsupported_input = bool(appearance_meta.get("unsupported_input", False))
        input_scope = str(appearance_meta.get("input_scope", "non_portrait"))
        face_confidence = float(appearance_meta.get("face_confidence", 0.0))
        human_face_score = float(appearance_meta.get("human_face_score", 0.0))
        photo_texture_score = float(appearance_meta.get("photo_texture_score", 0.0))
        quality_risk = float(appearance_meta.get("quality_risk", 0.0))
        supported_input = True
        portrait_confidence = clamp01(
            0.45 * face_confidence
            + 0.35 * human_face_score
            + 0.20 * (1.0 - quality_risk)
        )
        content_scope = input_scope
        if face_detected and not unsupported_input:
            content_scope = "human_portrait"
        elif face_detected:
            content_scope = "non_standard_human"

        return {
            "supported_input": supported_input,
            "content_scope": content_scope,
            "face_detected": face_detected,
            "face_confidence": round(float(face_confidence), 3),
            "human_face_score": round(float(human_face_score), 3),
            "photo_texture_score": round(float(photo_texture_score), 3),
            "quality_risk": round(float(quality_risk), 3),
            "portrait_confidence": round(float(portrait_confidence), 3),
            "allow_graph_reasoning": True,
            "allow_face_fusion": True,
        }

    @staticmethod
    def _build_default_reasoning(
        decision: Dict[str, Any],
        content_profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not content_profile.get("supported_input", True):
            return {
                "explanations": [
                    "Input is outside the supported human-portrait deepfake scope; binary real/fake reasoning was disabled.",
                    f"Detected scope: {content_profile.get('content_scope', 'out_of_scope')}. The sample is excluded from benchmark statistics.",
                ],
                "evidence_chain": [],
            }

        decision_threshold = float(decision.get("decision_threshold", 0.5))
        decision_fake_score = float(decision.get("decision_fake_score", 0.0))

        if decision.get("label") == "REAL":
            return {
                "explanations": [
                    (
                        "Primary fake-score stayed below the decision threshold "
                        f"({decision_fake_score:.3f} < {decision_threshold:.3f})."
                    ),
                    "No graph anomaly node was activated; fallback diagnostics are attached for audit.",
                ],
                "evidence_chain": [],
            }

        if decision.get("label") == "FAKE":
            return {
                "explanations": [
                    (
                        "Detector fusion stayed above the fake threshold "
                        f"({decision_fake_score:.3f} >= {decision_threshold:.3f})."
                    ),
                    "Graph evidence was not activated; treat this decision as model-only evidence.",
                ],
                "evidence_chain": [],
            }

        return {
            "explanations": ["No significant anomalous features detected"],
            "evidence_chain": [],
        }

    def _build_graph_gate_diagnostics(
        self,
        detector_results: Iterable,
        *,
        supports_graph_reasoning: bool,
        activated_subdomains: List,
    ) -> Dict[str, Any]:
        diagnostics: Dict[str, Any] = {
            "graph_reasoning_enabled": bool(supports_graph_reasoning),
            "activated_subdomains": len(activated_subdomains),
            "enabled_rules": 0,
            "passed_rules": 0,
            "blocked_by_context": 0,
            "blocked_by_threshold": 0,
            "missing_feature": 0,
            "status": "ready",
            "calibrated_fake_probability": None,
        }

        detector_map = {result.name: result for result in detector_results}
        calibrated_result = detector_map.get("CalibratedVision")
        if calibrated_result is not None:
            diagnostics["calibrated_fake_probability"] = round(
                float(calibrated_result.features.get("fake_probability", 0.0)),
                3,
            )

        if not supports_graph_reasoning:
            diagnostics["status"] = "graph_reasoning_disabled"
            return diagnostics

        try:
            rules = getattr(getattr(self._aligner, "_config", None), "rules", []) or []
            enabled_rules = [rule for rule in rules if getattr(rule, "evidence_enabled", False)]
            diagnostics["enabled_rules"] = len(enabled_rules)
            if not enabled_rules:
                diagnostics["status"] = "no_enabled_mapping_rules"
                return diagnostics

            feature_context = {result.name: dict(result.features) for result in detector_results}
            for rule in enabled_rules:
                detector_features = feature_context.get(rule.detector, {})
                raw_value = detector_features.get(rule.feature)
                if raw_value is None:
                    diagnostics["missing_feature"] += 1
                    continue

                if rule.context_detector and rule.context_feature:
                    context_value = (
                        feature_context
                        .get(rule.context_detector, {})
                        .get(rule.context_feature)
                    )
                    if context_value is None or float(context_value) < float(rule.context_min_value):
                        diagnostics["blocked_by_context"] += 1
                        continue

                mapped_score = self._aligner.sigmoid(
                    float(raw_value),
                    rule.sigmoid_k,
                    rule.sigmoid_x0,
                )
                confidence = mapped_score * rule.weight
                if confidence < float(rule.activation_threshold):
                    diagnostics["blocked_by_threshold"] += 1
                    continue

                diagnostics["passed_rules"] += 1
        except Exception as exc:
            diagnostics["status"] = "diagnostic_degraded"
            diagnostics["error"] = str(exc)
            return diagnostics

        if diagnostics["passed_rules"] <= 0:
            diagnostics["status"] = "no_rule_activated"
        return diagnostics

    @staticmethod
    def _resolve_reasoning_type(
        *,
        decision: Dict[str, Any],
        evidence: List[Dict[str, Any]],
        content_profile: Dict[str, Any],
    ) -> str:
        label = decision.get("label")
        if not content_profile.get("supported_input", True):
            return "insufficient_evidence_out_of_scope"
        if evidence and label == "FAKE":
            return "anomaly_evidence"
        if label == "REAL":
            return "counter_evidence"
        if label == "FAKE":
            return "anomaly_model_only"
        return "diagnostic_only"

    @staticmethod
    def _evaluate_review_flags(
        *,
        decision: Dict[str, Any],
        evidence: List[Dict[str, Any]],
        content_profile: Dict[str, Any],
        gate_diagnostics: Dict[str, Any],
    ) -> Tuple[bool, List[str], str]:
        reasons: List[str] = []
        label = str(decision.get("label", ""))
        supported_input = bool(content_profile.get("supported_input", True))
        decision_fake_score = float(decision.get("decision_fake_score", decision.get("direct_score", 0.0)))
        decision_threshold = float(decision.get("decision_threshold", 0.5))
        raw_margin = decision.get("decision_margin")
        if raw_margin is None:
            raw_margin = decision_fake_score - decision_threshold
        decision_margin = float(raw_margin)
        portrait_confidence = float(content_profile.get("portrait_confidence", 0.0))

        if not supported_input:
            reasons.append("out_of_scope_input")
        if gate_diagnostics.get("status") == "no_rule_activated" and not evidence:
            reasons.append("graph_rules_not_activated")

        if label == "REAL":
            if not evidence:
                reasons.append("real_without_graph_evidence")
            if abs(decision_margin) <= 0.08:
                reasons.append("near_decision_boundary")
            if decision_fake_score >= max(0.45, decision_threshold - 0.10):
                reasons.append("elevated_fake_probability_for_real")
            if portrait_confidence < 0.62:
                reasons.append("low_portrait_confidence")

        if label == "FAKE" and not evidence:
            reasons.append("fake_without_graph_evidence")

        high_risk_markers = {
            "out_of_scope_input",
            "real_without_graph_evidence",
            "near_decision_boundary",
            "elevated_fake_probability_for_real",
        }
        medium_risk_markers = {
            "fake_without_graph_evidence",
            "graph_rules_not_activated",
            "low_portrait_confidence",
        }

        if any(marker in reasons for marker in high_risk_markers):
            risk_level = "high"
        elif any(marker in reasons for marker in medium_risk_markers):
            risk_level = "medium"
        else:
            risk_level = "none"

        return risk_level != "none", reasons, risk_level

    @staticmethod
    def _build_diagnostic_chain(
        *,
        decision: Dict[str, Any],
        content_profile: Dict[str, Any],
        detector_signals: List[Dict[str, Any]],
        evidence: List[Dict[str, Any]],
        gate_diagnostics: Dict[str, Any],
        review_reasons: List[str],
    ) -> List[str]:
        scope = content_profile.get("content_scope", "unknown")
        supported_input = bool(content_profile.get("supported_input", True))
        face_confidence = float(content_profile.get("face_confidence", 0.0))
        portrait_confidence = float(content_profile.get("portrait_confidence", 0.0))
        decision_label = decision.get("label", "UNKNOWN")
        decision_confidence = float(decision.get("confidence", 0.0))
        decision_fake_score = float(decision.get("decision_fake_score", 0.0))
        decision_threshold = float(decision.get("decision_threshold", 0.5))
        raw_margin = decision.get("decision_margin")
        if raw_margin is None:
            raw_margin = decision_fake_score - decision_threshold
        decision_margin = float(raw_margin)

        chain = [
            (
                f"Input scope={scope}, supported_input={supported_input}, "
                f"face_confidence={face_confidence:.3f}, portrait_confidence={portrait_confidence:.3f}."
            ),
            (
                f"Decision={decision_label}, confidence={decision_confidence:.3f}, "
                f"fake_score={decision_fake_score:.3f}, threshold={decision_threshold:.3f}, "
                f"margin={decision_margin:.3f}."
            ),
        ]

        if detector_signals:
            top_signals = sorted(
                detector_signals,
                key=lambda item: float(item.get("score", 0.0)) * float(item.get("weight", 0.0)),
                reverse=True,
            )[:3]
            signal_text = ", ".join(
                f"{item['name']}({float(item.get('score', 0.0)):.3f})"
                for item in top_signals
            )
            if signal_text:
                chain.append(f"Top detector signals: {signal_text}.")

        if evidence:
            chain.append(f"Graph evidence activated with {len(evidence)} linked subdomains.")
        else:
            chain.append(
                "Graph evidence empty: "
                f"enabled_rules={int(gate_diagnostics.get('enabled_rules', 0))}, "
                f"passed_rules={int(gate_diagnostics.get('passed_rules', 0))}, "
                f"blocked_by_context={int(gate_diagnostics.get('blocked_by_context', 0))}, "
                f"blocked_by_threshold={int(gate_diagnostics.get('blocked_by_threshold', 0))}, "
                f"status={gate_diagnostics.get('status', 'unknown')}."
            )

        if review_reasons:
            chain.append(f"Review required due to: {', '.join(review_reasons)}.")

        return chain

    @staticmethod
    def _estimate_signal_reliability(result) -> float:
        reliability = 1.0
        quality_risk = float(result.meta.get("quality_risk", 0.0))
        reliability *= 1.0 - 0.45 * quality_risk
        if bool(result.meta.get("placeholder_mode", False)):
            reliability *= 0.55
        if result.meta.get("face_detected", True) is False:
            reliability *= 0.75
        return clamp01(reliability)

    @staticmethod
    def _normalized_feature_score(detector_name: str, feature_name: str, value: float) -> float:
        config = get_detector_config(detector_name)
        calibrator = config.feature_calibrators.get(feature_name)
        if calibrator is None:
            return clamp01(float(value))
        return clamp01(normalize(float(value), calibrator.low, calibrator.high))

    def _auto_evolve_features(
        self,
        unmapped_features: Iterable[UnmappedFeature],
        semantic_threshold: float,
        use_llm_generation: bool,
    ) -> List[Dict]:
        evolved_features = []
        for feature in unmapped_features:
            result = self._graph_evolver.auto_evolve(
                feature,
                enable_semantic_dedup=True,
                semantic_threshold=semantic_threshold,
                use_llm_generation=use_llm_generation,
            )
            if not result:
                continue

            evolved_features.append(result)
            self._logger.info(
                "[CREATE] auto evolved feature: %s:%s -> %s",
                feature.detector,
                feature.feature,
                result["name"],
            )
        return evolved_features

    @staticmethod
    def _build_unmapped_payload(
        unmapped_features: Iterable[UnmappedFeature],
    ) -> List[Dict]:
        return [
            {
                "detector": feature.detector,
                "feature": feature.feature,
                "score": feature.score,
            }
            for feature in unmapped_features
        ]

    @staticmethod
    def _collect_visualizations(detector_results: Iterable) -> Dict:
        visualizations = {}
        for result in detector_results:
            if result.name == "FFTDetector" and "fft_spectrum" in result.meta:
                visualizations["fft_spectrum"] = result.meta["fft_spectrum"]
        return visualizations


class EvolutionFacade:
    def __init__(self, graph_evolver, graph_writer, neo4j_client):
        self._graph_evolver = graph_evolver
        self._graph_writer = graph_writer
        self._neo4j_client = neo4j_client
        self._semantic_governor = GraphSemanticGovernance(neo4j_client)

    def ingest(self, request: DirectIngestRequest) -> Dict:
        payload = request.payload
        if not payload:
            raise WorkflowError("Invalid JSON", 400)

        required_keys = {"specific_domain", "describe", "specific_id", "subdomain"}
        if not required_keys.issubset(payload.keys()):
            raise WorkflowError(
                f"Missing required fields: {required_keys - payload.keys()}",
                400,
            )

        if not isinstance(payload["subdomain"], list):
            raise WorkflowError("subdomain must be a list", 400)

        for sub in payload["subdomain"]:
            if not all(key in sub for key in ["name", "describe", "sub_id"]):
                raise WorkflowError(
                    "Each subdomain item must contain 'name', 'describe', 'sub_id'",
                    400,
                )

        resolved = self._semantic_governor.normalize_iteration_payload(
            payload,
            prompt=payload.get("specific_domain", ""),
            matched_domain=payload.get("specific_domain", ""),
            semantic_threshold=0.86,
        )
        if not resolved.get("subdomain"):
            raise WorkflowError("No valid semantic subdomains after normalization", 400)
        self._graph_writer.write(resolved)
        return {
            "status": "success",
            "message": "Feature domain and subdomains ingested into Neo4j",
        }

    def evolve(self, request: ManualEvolutionRequest) -> Dict:
        if not request.features or not request.evolutions:
            raise WorkflowError("Missing features or evolutions", 400)

        if len(request.features) != len(request.evolutions):
            raise WorkflowError("features and evolutions must have same length", 400)

        specific_domains = self._neo4j_client.list_specific_domains()
        domain_map = {domain["name"]: domain for domain in specific_domains}

        results = []
        for feature_payload, evolution in zip(request.features, request.evolutions):
            if evolution.get("skip"):
                continue

            domain_name = evolution.get("specific_domain_name")
            if domain_name not in domain_map:
                results.append(
                    {
                        "error": f"Domain '{domain_name}' not found",
                        "feature": feature_payload["feature"],
                    }
                )
                continue

            unmapped_feature = self._build_unmapped_feature(feature_payload)
            draft = self._semantic_governor.normalize_manual_subdomain(
                name=evolution["sub_name"],
                describe=evolution["sub_describe"],
                specific_domain_name=domain_name,
                fallback_prefix=f"{unmapped_feature.detector.lower()}_{unmapped_feature.feature.lower()}",
            )
            if not draft["name"]:
                results.append(
                    {
                        "error": "Normalized subdomain name is empty",
                        "feature": feature_payload["feature"],
                    }
                )
                continue
            result = self._graph_evolver.evolve(
                feature=unmapped_feature,
                specific_domain=domain_map[domain_name],
                sub_name=draft["name"],
                sub_describe=draft["describe"],
                canonical_name=draft["canonical_name"],
                enable_semantic_dedup=True,
                semantic_threshold=request.semantic_threshold,
            )
            results.append(result)

        return {
            "status": "success",
            "evolved": results,
            "message": f"Evolved {len(results)} features successfully",
            "semantic_threshold": request.semantic_threshold,
        }

    def suggest_domain(self, request: SuggestDomainRequest) -> Dict:
        feature = UnmappedFeature(
            detector=request.detector,
            feature=request.feature,
            score=request.score,
            raw_value=request.raw_value,
        )
        suggested = self._graph_evolver.suggest_domain(feature)
        return {
            "suggested_domain": suggested if suggested else None,
            "all_domains": self._neo4j_client.list_specific_domains(),
        }

    @staticmethod
    def _build_unmapped_feature(feature_payload: Dict) -> UnmappedFeature:
        return UnmappedFeature(
            detector=feature_payload["detector"],
            feature=feature_payload["feature"],
            score=feature_payload.get("score", 0),
            raw_value=feature_payload.get(
                "raw_value",
                feature_payload.get("score", 0),
            ),
        )
