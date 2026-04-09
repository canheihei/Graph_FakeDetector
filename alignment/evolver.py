"""Graph evolution helpers for unmapped detector features."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional

from service.llm_chain import match_domain
from service.graph_semantics import GraphSemanticGovernance
from service.neo_client import graph_writer, neo4j_client


@dataclass
class UnmappedFeature:
    detector: str
    feature: str
    score: float
    raw_value: float


class GraphEvolver:
    def __init__(self):
        self._specific_domains = None
        self._sub_domains = None
        self._semantic_governor = GraphSemanticGovernance(neo4j_client)

    def _load_domains(self) -> None:
        if self._specific_domains is None:
            self._specific_domains = neo4j_client.list_specific_domains(include_main_domain=True)
        if self._sub_domains is None:
            self._sub_domains = neo4j_client.list_subdomains()

    def find_unmapped_features(self, detector_results, mapping_rules) -> List[UnmappedFeature]:
        mapped_keys = {(rule.detector, rule.feature) for rule in mapping_rules}
        unmapped: List[UnmappedFeature] = []

        for result in detector_results:
            if not isinstance(result.features, dict):
                continue
            for feature_name, value in result.features.items():
                if (result.name, feature_name) in mapped_keys:
                    continue
                if not isinstance(value, (int, float)) or value <= 0.3:
                    continue
                unmapped.append(
                    UnmappedFeature(
                        detector=result.name,
                        feature=feature_name,
                        score=float(value),
                        raw_value=float(value),
                    )
                )
        return unmapped

    def suggest_domain(self, feature: UnmappedFeature) -> Optional[Dict]:
        self._load_domains()
        if not self._specific_domains:
            return None

        domain_names = [item["name"] for item in self._specific_domains]
        sub_names = [item["name"] for item in (self._sub_domains or [])]
        query_text = f"{feature.detector} {feature.feature}"
        matched_name = match_domain(query_text, domain_names, sub_names)

        if matched_name and matched_name != query_text:
            for domain in self._specific_domains:
                if domain["name"] == matched_name:
                    print(f"[RELOAD] matched domain for {feature.detector}:{feature.feature} -> {matched_name}")
                    return domain

            for subdomain in self._sub_domains or []:
                if subdomain["name"] != matched_name:
                    continue
                result = neo4j_client.get_specific_domain_by_subdomain_name(matched_name)
                if result:
                    print(f"[RELOAD] matched subdomain for {feature.detector}:{feature.feature} -> {matched_name}")
                    return result

        fallback = self._specific_domains[0]
        print(f"[WARN] fallback domain for {feature.detector}:{feature.feature} -> {fallback['name']}")
        return fallback

    def evolve(
        self,
        feature: UnmappedFeature,
        specific_domain: Dict,
        sub_name: str,
        sub_describe: str,
        canonical_name: Optional[str] = None,
        enable_semantic_dedup: bool = True,
        semantic_threshold: float = 0.80,
    ) -> Dict:
        resolved_domain = self._semantic_governor.resolve_specific_domain(
            specific_domain.get("name", ""),
            fallback_domain=specific_domain,
            threshold=max(semantic_threshold, 0.78),
        )
        if resolved_domain is None:
            raise ValueError("No specific domain available for graph evolution")

        sub_id = str(uuid.uuid4())
        normalized_name = str(sub_name or "").strip()
        normalized_describe = str(sub_describe or "").strip()
        result_data = {
            "main_domain": resolved_domain.main_domain,
            "main_describe": resolved_domain.main_describe,
            "specific_domain": resolved_domain.name,
            "describe": resolved_domain.describe or specific_domain.get("describe", ""),
            "specific_id": resolved_domain.id or specific_domain.get("id", str(uuid.uuid4())),
            "subdomain": [
                {
                    "name": normalized_name,
                    "display_name": normalized_name,
                    "canonical_name": canonical_name,
                    "describe": normalized_describe,
                    "sub_id": sub_id,
                    "source_detector": feature.detector,
                    "source_feature": feature.feature,
                }
            ],
            "semantic_source": "detect_auto_evolve" if feature.detector else "manual_evolve",
            "semantic_version": "graph_semantics_v2",
        }

        if enable_semantic_dedup:
            dedup_stats = graph_writer.write(result_data, semantic_threshold=semantic_threshold)
            print(f"[RELOAD] semantic dedup stats: {dedup_stats}")
        else:
            graph_writer.write(result_data)
            print(f"[CREATE] created graph entry without semantic dedup: {sub_name}")

        self._specific_domains = None
        self._sub_domains = None

        return {
            "sub_id": sub_id,
            "name": normalized_name,
            "display_name": normalized_name,
            "mapping_label": canonical_name,
            "describe": normalized_describe,
            "specific_domain": resolved_domain.name,
            "main_domain": resolved_domain.main_domain,
            "feature_key": f"{feature.detector}:{feature.feature}",
        }

    def batch_evolve(
        self,
        features: List[UnmappedFeature],
        evolutions: List[Dict],
        enable_semantic_dedup: bool = True,
        semantic_threshold: float = 0.80,
    ) -> List[Dict]:
        results = []
        for feature, evolution in zip(features, evolutions):
            if evolution.get("skip"):
                continue
            results.append(
                self.evolve(
                    feature=feature,
                    specific_domain=evolution["specific_domain"],
                    sub_name=evolution["sub_name"],
                    sub_describe=evolution["sub_describe"],
                    canonical_name=evolution.get("canonical_name"),
                    enable_semantic_dedup=enable_semantic_dedup,
                    semantic_threshold=semantic_threshold,
                )
            )
        return results

    def auto_evolve(
        self,
        feature: UnmappedFeature,
        update_config: bool = True,
        enable_semantic_dedup: bool = True,
        semantic_threshold: float = 0.80,
        use_llm_generation: bool = False,
    ) -> Optional[Dict]:
        suggested_domain = self.suggest_domain(feature)
        if not suggested_domain:
            return None

        if use_llm_generation:
            generated = self._generate_feature_with_llm(feature, suggested_domain)
            sub_name = generated["sub_name"]
            sub_describe = generated["sub_describe"]
            canonical_name = generated.get("canonical_name")
        else:
            generated = self._semantic_governor.build_feature_semantic_draft(
                detector=feature.detector,
                feature=feature.feature,
                score=feature.score,
                specific_domain=suggested_domain,
            )
            sub_name = generated["name"]
            sub_describe = generated["describe"]
            canonical_name = generated["canonical_name"]

        try:
            result = self.evolve(
                feature=feature,
                specific_domain=suggested_domain,
                sub_name=sub_name,
                sub_describe=sub_describe,
                canonical_name=canonical_name,
                enable_semantic_dedup=enable_semantic_dedup,
                semantic_threshold=semantic_threshold,
            )
            if update_config:
                self._update_mapping_config(feature, result)
            return result
        except Exception as exc:
            print(f"[WARN] auto evolve failed for {feature.detector}:{feature.feature}: {exc}")
            return None

    def _update_mapping_config(self, feature: UnmappedFeature, evolved_result: Dict) -> None:
        config_path = os.path.join(os.path.dirname(__file__), "mapping_config.json")
        try:
            with open(config_path, "r", encoding="utf-8") as handle:
                config = json.load(handle)

            rules = config.get("rules", [])
            if any(
                rule["detector"] == feature.detector and rule["feature"] == feature.feature
                for rule in rules
            ):
                print(f"[RELOAD] mapping already exists: {feature.detector}:{feature.feature}")
                return

            same_detector_features = [
                rule["feature"] for rule in rules if rule["detector"] == feature.detector
            ]
            if same_detector_features:
                from service.llm_chain import semantic_match

                matched = semantic_match(feature.feature, same_detector_features, threshold=0.85)
                if matched != feature.feature:
                    print(
                        f"[WARN] skipped config update due to semantic duplicate: "
                        f"{feature.detector}:{feature.feature} -> {matched}"
                    )
                    return

            rules.append(
                {
                    "detector": feature.detector,
                    "feature": feature.feature,
                    "subdomain_id": evolved_result["sub_id"],
                    "subdomain_label": evolved_result.get("mapping_label")
                    or evolved_result["name"],
                    "sigmoid_k": 6.0,
                    "sigmoid_x0": 0.5,
                    "weight": 0.7,
                }
            )
            config["rules"] = rules

            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump(config, handle, ensure_ascii=False, indent=2)

            print(
                f"[CREATE] updated mapping_config.json: "
                f"{feature.detector}:{feature.feature} -> {evolved_result['name']}"
            )
        except Exception as exc:
            print(f"[WARN] failed to update mapping_config.json: {exc}")

    @staticmethod
    def _generate_feature_name(feature: UnmappedFeature) -> str:
        detector_short = feature.detector.replace("Detector", "").replace("detector", "")
        feature_clean = feature.feature.replace("_", " ").title()
        return f"{detector_short}-{feature_clean}"

    @staticmethod
    def _generate_feature_description(feature: UnmappedFeature, domain: Dict) -> str:
        domain_name = domain.get("name", "UnknownDomain")
        return (
            f"Generated from detector '{feature.detector}' feature '{feature.feature}' "
            f"under domain '{domain_name}' with score {feature.score:.2f}."
        )

    def _generate_feature_with_llm(self, feature: UnmappedFeature, domain: Dict) -> Dict[str, str]:
        try:
            from openai import OpenAI
            from config import ALI_API_KEY, ALI_BASE_URL

            client = OpenAI(api_key=ALI_API_KEY, base_url=ALI_BASE_URL)
            prompt = (
                "你是一名 DeepFake 图谱语义工程师。"
                "请输出严格 JSON，键仅为 sub_name, canonical_name, sub_describe。"
                "其中 sub_name 必须是专业中文节点名，canonical_name 必须是英文 snake_case。"
                f"detector={feature.detector}; feature={feature.feature}; "
                f"score={feature.score:.2f}; domain={domain.get('name', '')}; "
                f"domain_describe={domain.get('describe', '')}"
            )
            response = client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {
                        "role": "system",
                        "content": "Return compact JSON only. No markdown.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            raw = response.choices[0].message.content
            parsed = json.loads(raw)
            if "sub_name" in parsed and "sub_describe" in parsed:
                print(f"[CREATE] LLM generated subdomain: {parsed['sub_name']}")
                return parsed
        except Exception as exc:
            print(f"[WARN] LLM generation failed, using fallback naming: {exc}")

        fallback = self._semantic_governor.build_feature_semantic_draft(
            detector=feature.detector,
            feature=feature.feature,
            score=feature.score,
            specific_domain=domain,
        )
        return {
            "sub_name": fallback["name"],
            "canonical_name": fallback["canonical_name"],
            "sub_describe": fallback["describe"],
        }


graph_evolver = GraphEvolver()
